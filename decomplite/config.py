"""Global paths + YAML-driven benchmark configuration."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"            # builds + channel cache (regenerable)
RESULTS_DIR = ROOT / "results"      # LLM outputs
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
GHIDRA_SCRIPTS_DIR = ROOT / "ghidra_scripts"

# The whole matrix this repo supports. One ISA, every optimization level.
ARCHS = ["x86_64"]
OPT_LEVELS = ["O0", "O1", "O2", "O3", "Os"]
COMPILERS = ["gcc"]
ARCH_DESCRIPTION = "x86-64 (System V AMD64 ABI)"

# What a pipeline can show the model, both taken from the stripped binary.
# Canonical order is the order they render in the prompt templates.
CHANNELS = ("asm", "pseudoc")

# What a pipeline may be told about its own answer between iterations.
FEEDBACK = {
    "none":      "nothing",
    "compiler":  "the compiler's diagnostics for the answer on its own",
    "interface": "the diagnostics of building the answer together with code that "
                 "calls the function (needs a test harness in the dataset)",
    "tests":     "the above, plus which of the dataset's test cases fail",
}
MAX_ITERATIONS = 10
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class Pipeline:
    """One named recipe for producing an answer. See README, "Pipelines"."""
    name: str
    input: tuple[str, ...]                    # channels shown to every role
    recovery: tuple[Optional[str], ...]       # one entry per recovery LLM;
    reviewers: tuple[Optional[str], ...]      # None = the cell's `model:`
    feedback: str
    iterations: int

    def spec(self, model: str) -> dict:
        """The recipe with models resolved: stored beside every result, so a
        result made under a different recipe is never mistaken for this one."""
        return {
            "input": list(self.input),
            "recovery": [m or model for m in self.recovery],
            "reviewers": [m or model for m in self.reviewers],
            "feedback": self.feedback,
            "iterations": self.iterations,
        }


def _roles(where: str, value, default: int, registry: dict) -> tuple[Optional[str], ...]:
    """A role is given as a count (that many LLMs, all the cell's model) or as
    a list of model keys (one LLM each)."""
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, list)):
        raise ValueError(f"{where} must be a count or a list of model keys")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{where} cannot be negative")
        return (None,) * value
    for mk in value:
        if mk not in registry:
            raise ValueError(f"{where}: unknown model '{mk}'. Known: {sorted(registry)}")
    return tuple(value)


def parse_pipeline(p: Path, name: str, raw, registry: dict) -> Pipeline:
    where = f"{p}: pipeline '{name}'"
    if not _NAME_RE.match(str(name)):
        raise ValueError(f"{where}: the name must be filename-safe")
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = set(raw) - {"input", "recovery", "reviewers", "feedback", "iterations"}
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}")

    inp = raw.get("input")
    inp = [inp] if isinstance(inp, str) else list(inp or [])
    if not inp or any(c not in CHANNELS for c in inp):
        raise ValueError(f"{where}: `input` must be one or more of {list(CHANNELS)}")
    recovery = _roles(f"{where}: `recovery`", raw.get("recovery"), 1, registry)
    reviewers = _roles(f"{where}: `reviewers`", raw.get("reviewers"), 0, registry)
    feedback = str(raw.get("feedback") or "none")
    iterations = raw.get("iterations", 1)
    if not recovery:
        raise ValueError(f"{where}: at least one recovery LLM is required")
    if feedback not in FEEDBACK:
        raise ValueError(f"{where}: unknown feedback '{feedback}'. Known: {list(FEEDBACK)}")
    if isinstance(iterations, bool) or not isinstance(iterations, int) \
            or not 1 <= iterations <= MAX_ITERATIONS:
        raise ValueError(f"{where}: `iterations` must be 1..{MAX_ITERATIONS}")
    # Both of these would spend LLM calls that cannot change the answer.
    if len(recovery) > 1 and feedback == "none" and not reviewers:
        raise ValueError(f"{where}: {len(recovery)} recovery LLMs need `feedback` or a "
                         f"reviewer to choose between their answers")
    if iterations > 1 and feedback == "none" and not reviewers:
        raise ValueError(f"{where}: more than one iteration needs `feedback` or a "
                         f"reviewer; with neither, nothing changes after the first")
    return Pipeline(name=str(name), input=tuple(c for c in CHANNELS if c in inp),
                    recovery=recovery, reviewers=reviewers,
                    feedback=feedback, iterations=iterations)


@dataclass(frozen=True)
class RepoSelection:
    name: str
    path: Optional[Path] = None           # dataset location; default datasets/<name>
    cflags: tuple[str, ...] = ()          # extra compile flags for every sample
    functions: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkConfig:
    archs: tuple[str, ...]
    opts: tuple[str, ...]
    compilers: tuple[str, ...]
    pipelines: tuple[Pipeline, ...]
    models: tuple[str, ...]               # keys into models.yaml
    repos: tuple[RepoSelection, ...]
    report_out_dir: Path
    model_registry: dict = field(compare=False, default_factory=dict)
    source_path: Optional[Path] = field(default=None, compare=False)


def load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines, existing variables win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*$", line)
        if m and not line.lstrip().startswith("#"):
            os.environ.setdefault(m.group(1), m.group(2).strip("'\""))


def _find(base: Path, rel: str) -> Path:
    """Resolve `rel` next to the config file first, then at the repo root."""
    for cand in (base / rel, ROOT / rel):
        if cand.exists():
            return cand.resolve()
    return (base / rel).resolve()


def _subset(p: Path, key: str, values, known: list[str]) -> tuple[str, ...]:
    values = [values] if isinstance(values, str) else list(values or [])
    for v in values:
        if v not in known:
            raise ValueError(f"{p}: unknown {key} entry '{v}'. Known: {known}")
    return tuple(values)


def load(path: Path | str) -> BenchmarkConfig:
    """Load and validate a benchmark.yaml."""
    p = Path(path).resolve()
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: top-level must be a mapping")

    # `functions_from:` borrows another file's `repos:` block so a large
    # function selection lives in exactly one file.
    src = raw.get("functions_from")
    if src and not raw.get("repos"):
        sp = _find(p.parent, src)
        if not sp.is_file():
            raise ValueError(f"{p}: functions_from {src!r} not found at {sp}")
        borrowed = (yaml.safe_load(sp.read_text()) or {}).get("repos")
        if not borrowed:
            raise ValueError(f"{p}: functions_from {src!r} has no `repos:` block")
        raw = dict(raw, repos=borrowed)

    archs = _subset(p, "archs", raw.get("archs") or ARCHS, ARCHS)
    opts = _subset(p, "opts", raw.get("opts"), OPT_LEVELS)
    compilers = _subset(p, "compilers", raw.get("compilers") or COMPILERS, COMPILERS)
    if not opts:
        raise ValueError(f"{p}: `opts` is required (any of {OPT_LEVELS})")

    # Models. `model:` may be one key or a list; each is looked up in the
    # registry file so an unknown key is rejected here, not at the first call.
    registry_path = _find(p.parent, raw.get("models_file") or "models.yaml")
    if not registry_path.is_file():
        raise ValueError(f"{p}: model registry not found at {registry_path}")
    registry = yaml.safe_load(registry_path.read_text()) or {}
    model = raw.get("model") or raw.get("models")
    if not model:
        raise ValueError(f"{p}: `model` is required (a key from {registry_path.name})")
    models = (model,) if isinstance(model, str) else tuple(model)
    for mk in models:
        if mk not in registry:
            raise ValueError(f"{p}: unknown model '{mk}'. Known: {sorted(registry)}. "
                             f"Add it to {registry_path.name} to enable.")

    pipes_raw = raw.get("pipelines")
    if not isinstance(pipes_raw, dict) or not pipes_raw:
        raise ValueError(f"{p}: `pipelines` is required: a mapping of name -> recipe, "
                         f"e.g. `single_pass: {{input: asm}}`")
    pipelines = tuple(parse_pipeline(p, name, spec, registry)
                      for name, spec in pipes_raw.items())

    repos_raw = raw.get("repos") or []
    if not repos_raw:
        raise ValueError(f"{p}: at least one repo entry is required "
                         f"(a `repos:` block, or `functions_from:`)")
    repos: list[RepoSelection] = []
    for r in repos_raw:
        name = r.get("name")
        if not name:
            raise ValueError(f"{p}: repo entry missing `name`")
        repos.append(RepoSelection(
            name=name,
            path=_find(p.parent, r["path"]) if r.get("path") else None,
            cflags=tuple(r.get("cflags") or []),
            functions=tuple(r.get("functions") or []),
            patterns=tuple(r.get("patterns") or []),
        ))

    report = raw.get("report") or {}
    return BenchmarkConfig(
        archs=archs, opts=opts, compilers=compilers, pipelines=pipelines,
        models=models, repos=tuple(repos),
        report_out_dir=ROOT / (report.get("out_dir") or "reports"),
        model_registry=registry, source_path=p,
    )
