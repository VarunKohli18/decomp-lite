"""The three phases: build -> decompile -> eval.

A CELL is one (repo, function, opt, model, pipeline). A pipeline is a recipe
from the config (see config.Pipeline) and runs like this:

    iteration 1    every recovery LLM writes an answer from the input
                   [feedback]  with several answers, keep those that pass the check
                   [reviewers] each reviewer in turn returns a final answer
    iteration 2+   [feedback]  check the answer; stop if it passes, else the
                               first recovery LLM repairs it from the feedback
                   [reviewers] each reviewer in turn reviews it again

The simplest pipeline -- one recovery LLM, no reviewer, no feedback, one
iteration -- is a single LLM call.
"""
from __future__ import annotations

import csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from decomplite import build as B
from decomplite import evaluate as E
from decomplite import llm
from decomplite import representations as R
from decomplite.config import (
    ARCH_DESCRIPTION, PROMPTS_DIR, RESULTS_DIR, BenchmarkConfig, Pipeline,
)
from decomplite.datasets import Sample, load_dataset, select

# A single neutral marker for any prompt slot without content.
NOT_AVAILABLE = "Not available."
_SLOT = {"asm": "ASSEMBLY", "pseudoc": "PSEUDOC"}
_SLOT_RE = re.compile(r"\{\{([A-Z_]+)\}\}")

_INFO = r"[A-Za-z0-9+#._-]*"
_FENCE_RE = re.compile(r"```" + _INFO + r"[ \t]*\r?\n(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```" + _INFO + r"[ \t]*\r?\n(.*)\Z", re.DOTALL)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", name)


@dataclass(frozen=True)
class Cell:
    sample: Sample
    opt: str
    model: str
    pipeline: Pipeline

    @property
    def out_path(self) -> Path:
        """results/<pipeline>/<model>/<repo>/<arch>-<opt>/<function>.c"""
        return (RESULTS_DIR / self.pipeline.name / _safe(self.model) / self.sample.repo
                / f"x86_64-{self.opt}" / f"{self.sample.name}.c")

    @property
    def meta_path(self) -> Path:
        return self.out_path.with_suffix(".json")

    def meta(self) -> dict:
        return json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}

    def is_current(self) -> bool:
        """An answer exists AND was made by this recipe (not by an earlier
        definition of a pipeline with the same name)."""
        return (self.out_path.exists()
                and self.meta().get("spec") == self.pipeline.spec(self.model))

    def label(self) -> str:
        return f"{self.sample.repo}/{self.sample.name} {self.opt} {self.model} {self.pipeline.name}"


def _csv(value) -> set[str] | None:
    return {s.strip() for s in value.split(",") if s.strip()} if value else None


def resolve(cfg: BenchmarkConfig, filters: dict) -> tuple[list[Sample], list[str], list[Cell]]:
    """Expand the config into (samples, opts, cells), applying --filter-*."""
    f = {k: _csv(v) for k, v in filters.items()}
    samples: list[Sample] = []
    for sel in cfg.repos:
        if f.get("repo") and sel.name not in f["repo"]:
            continue
        chosen, missing = select(sel, load_dataset(sel))
        for fn in missing:
            print(f"  [skip] {sel.name}/{fn}: not in the dataset")
        samples += [s for s in chosen if not f.get("fn") or s.name in f["fn"]]
    opts = [o for o in cfg.opts if not f.get("opt") or o in f["opt"]]
    cells = [
        Cell(s, o, m, pl)
        for s in samples for o in opts
        for m in cfg.models if not f.get("model") or m in f["model"]
        for pl in cfg.pipelines if not f.get("pipeline") or pl.name in f["pipeline"]
    ]
    return samples, opts, cells


def _map(fn, items, workers: int, what: str, log_every: int = 25) -> list:
    """Run `fn` over `items` on a thread pool, printing progress."""
    out, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, res in enumerate(pool.map(fn, items), 1):
            out.append(res)
            if i % log_every == 0 or i == len(items):
                print(f"  {what}: {i}/{len(items)}  ({time.time() - t0:.0f}s)")
    return out


def _first_line(text: str, fallback: str = "?") -> str:
    lines = (text or "").strip().splitlines()
    return lines[0][:200] if lines else fallback


# --- build -------------------------------------------------------------------

def phase_build(cfg: BenchmarkConfig, samples, opts, cells, *, force: bool, workers: int) -> None:
    """Compile every (sample, opt) and cache the inputs the pipelines need."""
    channels = sorted({c for cell in cells for c in cell.pipeline.input})
    B.toolchain()                         # fail before the pool, with one message

    def one(job):
        sample, opt = job
        b = B.build(sample, opt, force=force)
        got = R.materialize(b, channels, force=force) if b.ok else {}
        return sample, opt, b, got

    results = _map(one, [(s, o) for s in samples for o in opts], workers, "build")
    built = sum(1 for *_, b, _ in results if b.ok)
    print(f"built {built}/{len(results)} (sample, opt) pairs")
    for ch in channels:
        n = sum(1 for *_, got in results if got.get(ch))
        print(f"  {ch:<8} available for {n}/{len(results)}")
    for sample, opt, b, got in results:
        if not b.ok:
            print(f"  [build failed] {sample.repo}/{sample.name} {opt}: {_first_line(b.error)}")
            continue
        for ch in channels:
            if not got.get(ch):
                print(f"  [no {ch}] {sample.repo}/{sample.name} {opt}: "
                      f"{_first_line(R.channel_error(b, ch))}")


# --- prompts -----------------------------------------------------------------

def strip_fences(text: str) -> str:
    """Pull code out of a fenced code block if the model added one; else
    return the text trimmed."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _OPEN_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def load_prompt(name: str) -> tuple[str, str]:
    """prompts/<name>.txt as (system, user). name: recover | repair | review."""
    text = (PROMPTS_DIR / f"{name}.txt").read_text()
    if "USER:" not in text:
        raise ValueError(f"Prompt {name}.txt missing USER: separator")
    sys_part, user_part = text.split("USER:", 1)
    return sys_part.replace("SYSTEM:", "", 1).strip(), user_part.strip()


def render_prompt(template: str, channels, contents: dict, extra: dict | None = None) -> str:
    """Fill the template for one cell in a single pass, so text inserted into
    one slot is never rescanned for slots. A slot nothing fills (an input the
    pipeline does not select, or one this repo does not produce at all) reads
    "Not available."."""
    values = {"ARCH": ARCH_DESCRIPTION, **(extra or {})}
    for ch in channels:
        values[_SLOT[ch]] = (contents.get(ch) or "").strip() or NOT_AVAILABLE
    return _SLOT_RE.sub(lambda m: values.get(m.group(1), NOT_AVAILABLE), template)


def _ask(cfg: BenchmarkConfig, model: str, prompt_name: str, channels, contents: dict,
         extra_slots: dict | None = None) -> tuple[str, dict]:
    """One LLM call. Returns (code, record of the turn)."""
    system_tpl, user_tpl = load_prompt(prompt_name)
    system = render_prompt(system_tpl, channels, contents, extra_slots)
    user = render_prompt(user_tpl, channels, contents, extra_slots)
    result = llm.call(cfg.model_registry[model], system, user)
    return strip_fences(result["text"]), {
        "model": model, "prompt": prompt_name, "system_prompt": system, "user_prompt": user,
        "raw_text": result["text"], "latency_s": result["latency_s"],
        "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
        # Set when the provider reports which model actually answered.
        **({"served_by": result["served_by"]} if result.get("served_by") else {}),
    }


# --- decompile ---------------------------------------------------------------

# One lock per file a worker may be writing.
_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
# First-pass answers written by THIS run: under --force each is regenerated
# once, then shared by every pipeline of the run like any other.
_FRESH: set[Path] = set()


def _lock(path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(path, threading.Lock())


def _first_pass(cfg: BenchmarkConfig, cell: Cell, model: str, k: int, contents: dict,
                force: bool) -> tuple[str, dict]:
    """A recovery LLM's answer from the input alone: answer number `k` of
    `model` for this function, opt and input.

    Stored once and SHARED: every pipeline whose recipe starts the same way
    starts from the very same answer, so pipelines are compared cell-for-cell
    on equal footing (and the call is paid for once).
    """
    path = (RESULTS_DIR / "_first_pass" / _safe(model) / "+".join(cell.pipeline.input)
            / cell.sample.repo / f"x86_64-{cell.opt}" / f"{cell.sample.name}.{k}.json")
    with _lock(path):
        if path.exists() and (not force or path in _FRESH):
            rec = json.loads(path.read_text())
            return rec["text"], dict(rec["turn"], reused=True)
        text, turn = _ask(cfg, model, "recover", cell.pipeline.input, contents)
        path.parent.mkdir(parents=True, exist_ok=True)
        B.write_atomic(path, json.dumps({"text": text, "turn": turn}, indent=1))
        _FRESH.add(path)
        return text, turn


def _candidates(texts: list[str]) -> str:
    if len(texts) == 1:
        return texts[0]
    return "\n\n".join(f"---- Candidate {i} of {len(texts)} ----\n{t}"
                       for i, t in enumerate(texts, 1))


def run_pipeline(cfg: BenchmarkConfig, cell: Cell, *, force: bool = False) -> tuple[str, dict]:
    """Run one cell's recipe. Returns (answer, record)."""
    pl, spec = cell.pipeline, cell.pipeline.spec(cell.model)
    b = B.build(cell.sample, cell.opt)
    if not b.ok:
        raise RuntimeError(f"build failed: {b.error}")
    contents = R.materialize(b, pl.input)
    available = [c for c in pl.input if contents.get(c, "").strip()]
    if not available:
        raise RuntimeError("no input available: "
                           + "; ".join(R.channel_error(b, c) for c in pl.input))

    turns: list[dict] = []
    verdicts: dict[str, dict] = {}

    def check(text: str) -> dict:
        if text not in verdicts:
            verdicts[text] = E.feedback(pl.feedback, cell.sample, text, b.function, cell.opt)
        return verdicts[text]

    def ask(role: str, iteration: int, model: str, prompt: str, slots: dict) -> str:
        text, turn = _ask(cfg, model, prompt, pl.input, contents, slots)
        turns.append(dict(turn, role=role, iteration=iteration))
        return text

    # Iteration 1: every recovery LLM answers from the input.
    pool = []
    for i, model in enumerate(spec["recovery"]):
        text, turn = _first_pass(cfg, cell, model, spec["recovery"][:i].count(model),
                                 contents, force)
        turns.append(dict(turn, role=f"recovery {i + 1}", iteration=1))
        pool.append(text)
    if pl.feedback != "none" and len(pool) > 1:
        pool = [t for t in pool if check(t)["ok"]] or pool
    answer, stopped = pool[0], f"after iteration {pl.iterations}"

    for it in range(1, pl.iterations + 1):
        if it > 1 and pl.feedback != "none":
            verdict = check(answer)
            if verdict["ok"]:
                stopped = f"passed the {pl.feedback} check before iteration {it}"
                break
            answer = ask("repair", it, spec["recovery"][0], "repair", {
                "GENERATED_C": answer, "FEEDBACK_TITLE": verdict["title"],
                "FEEDBACK": verdict["text"] or "(the check printed nothing)"})
            turns[-1]["feedback"] = verdict
        for j, model in enumerate(spec["reviewers"]):
            # The first reviewer of the run chooses among the recovery LLMs'
            # answers; after that there is one answer.
            under_review = pool if (it == 1 and j == 0) else [answer]
            answer = ask(f"reviewer {j + 1}", it, model, "review",
                         {"GENERATED_C": _candidates(under_review)})

    record = {"selection": list(pl.input), "available": available,
              "turns": turns, "stopped": stopped}
    if pl.feedback != "none":
        final = check(answer)
        record.update(feedback_passed=final["ok"], feedback_source=final["source"])
    return answer, record


def decompile_cell(cfg: BenchmarkConfig, cell: Cell, *, force: bool) -> str:
    """Produce one cell's answer. Returns "cached" | "ok" | "stale" | "error: ..."."""
    with _lock(cell.out_path):
        if cell.out_path.exists() and not force:
            # An answer under this pipeline NAME made by a different RECIPE is
            # neither reused nor silently overwritten.
            return "cached" if cell.is_current() else "stale"
        cell.out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"repo": cell.sample.repo, "function": cell.sample.name, "arch": "x86_64",
                "opt": cell.opt, "model": cell.model, "pipeline": cell.pipeline.name,
                "spec": cell.pipeline.spec(cell.model)}
        try:
            text, record = run_pipeline(cfg, cell, force=force)
            cell.out_path.write_text(text.rstrip("\n") + "\n")
            turns = record["turns"]
            meta.update(
                status="ok", **record, llm_calls=len(turns),
                latency_s=round(sum(t["latency_s"] for t in turns), 2),
                input_tokens=sum(t["input_tokens"] for t in turns),
                output_tokens=sum(t["output_tokens"] for t in turns),
                finished_at=int(time.time()),
            )
            status = "ok"
        except B.ToolchainError:
            raise
        except Exception as e:
            # Recorded, not raised: one bad cell must not stop the sweep. No
            # output file is written, so the next run retries it.
            cell.out_path.unlink(missing_ok=True)
            meta.update(status="error", error=str(e)[-3000:])
            status = f"error: {_first_line(str(e), type(e).__name__)}"
        cell.meta_path.write_text(json.dumps(meta, indent=1))
        return status


def phase_decompile(cfg: BenchmarkConfig, cells, *, force: bool, workers: int) -> None:
    def one(cell):
        status = decompile_cell(cfg, cell, force=force)
        if status.startswith("error"):
            print(f"  [{status}] {cell.label()}")
        return status

    statuses = _map(one, cells, workers, "decompile")
    ok, cached, stale = (statuses.count(s) for s in ("ok", "cached", "stale"))
    print(f"decompiled {ok} cells, {cached} cached, {len(cells) - ok - cached - stale} failed")
    if stale:
        print(f"{stale} cells SKIPPED: results exist under the same pipeline name but were "
              f"made with different settings.\n  Rename the pipeline to keep both, or pass "
              f"--force to replace them.")


# --- eval --------------------------------------------------------------------

def _rate(rows: list, key: str) -> str:
    """Passes over the cells the metric applies to. A cell with no answer
    counts as a failure. Cells the metric cannot be measured on (no test
    harness, or nothing to run it on) are left out, not counted as failures."""
    if key != "standalone_compile":
        rows = [r for r in rows if r["harness"]]
    rows = [r for r in rows if r[key] != "" or not r["generated"]]
    if not rows:
        return "n/a"
    return f"{sum(1 for r in rows if r[key] == 1) / len(rows):.1%}"


def phase_eval(cfg: BenchmarkConfig, cells, *, workers: int) -> Path:
    """Score what is on disk and write reports/summary.csv."""
    def one(cell):
        b = B.build(cell.sample, cell.opt)
        meta = cell.meta()
        generated = cell.is_current()
        row = {
            "repo": cell.sample.repo, "function": cell.sample.name, "arch": "x86_64",
            "opt": cell.opt, "model": cell.model, "pipeline": cell.pipeline.name,
            "built": int(b.ok), "generated": int(generated),
            "harness": int(cell.sample.test_path is not None),
            "standalone_compile": "", "compile_ok": "", "test_passed": "",
            "test_pass": "", "test_total": "", "eval_status": "",
            "llm_calls": "", "input_tokens": "", "output_tokens": "", "latency_s": "",
            "stopped": "",
        }
        err = b.error or meta.get("error") or ""
        if generated:
            for k in ("llm_calls", "input_tokens", "output_tokens", "latency_s", "stopped"):
                row[k] = meta.get(k, "")
            text = cell.out_path.read_text()
            row["standalone_compile"] = int(E.standalone_compile(text, cell.opt)[0])
            h = E.harness(cell.sample, text, b.function or cell.sample.function or cell.sample.name)
            for k in ("compile_ok", "test_passed", "test_pass", "test_total"):
                row[k] = "" if h[k] is None else h[k]
            row["eval_status"] = h["status"]
            err = h["compile_error"] or (h["test_output"] if h["test_passed"] == 0 else "")
        elif cell.out_path.exists():
            err = "stale: made with different pipeline settings"
        row["error"] = _first_line(err, "")
        return row

    rows = _map(one, cells, workers, "eval")
    cfg.report_out_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.report_out_dir / "summary.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["repo"])
        w.writeheader()
        w.writerows(rows)

    print("\nRC  = standalone_compile: the answer compiles on its own"
          "\nRCI = compile_ok:         it compiles in place of the original, with its callers"
          "\nRE  = test_passed:        and then passes every test case"
          f"\n\n{'pipeline':<24} {'model':<26} {'opt':<4} {'cells':>5} {'gen':>4} "
          f"{'RC':>7} {'RCI':>7} {'RE':>7} {'calls':>6}")
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["pipeline"], r["model"], r["opt"]), []).append(r)
    for (pipeline, model, opt), rs in groups.items():
        calls = [r["llm_calls"] for r in rs if r["llm_calls"] != ""]
        print(f"{pipeline:<24} {model:<26} {opt:<4} {len(rs):>5} "
              f"{sum(r['generated'] for r in rs):>4} {_rate(rs, 'standalone_compile'):>7} "
              f"{_rate(rs, 'compile_ok'):>7} {_rate(rs, 'test_passed'):>7} "
              f"{(sum(calls) / len(calls) if calls else 0):>6.2f}")
    print(f"\n`calls` is the mean number of LLM calls per answered cell."
          f"\nper-cell rows: {out}")
    return out
