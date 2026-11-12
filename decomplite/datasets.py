"""Dataset plug-ins.

A dataset is a set of C translation units, one function of interest each. It
can be supplied three ways (first match wins, or point at one with `path:`):

  datasets/<name>/          a directory of `<function>.c` files
  datasets/<name>.jsonl     one JSON object per line: {"name", "source", ...}
  datasets/<name>.py        an adapter module defining `load()`

Optional per-sample fields (JSONL keys / adapter dict keys):
  function   symbol to decompile, when it is not the sample name
  cflags     extra compile flags for this sample
  test       source of a test driver (see below); `test_ext` is "c" or "cpp"
  io         list of {"input": ..., "output": ...} pairs for the driver

TEST HARNESS (optional; enables the compile_ok and test_passed metrics).
A driver is a C or C++ program that `#include "<name>.c"` and calls the
function the way real callers do. In a directory dataset it sits beside the
sample as `<name>_test.c` or `<name>_test.cpp`, with headers it needs under
`_include/`. Two protocols:

  no IO file      the driver runs once; exit status 0 means every test passed
  <name>_io.json  the driver runs once per pair as `driver in.json out.json`
                  and the output JSON is compared with the pair's `output`
                  (this is ExeBench's layout, so its folders work unchanged)
"""
from __future__ import annotations

import fnmatch
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from decomplite.config import DATA_DIR, ROOT, RepoSelection

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.+-]+$")


@dataclass(frozen=True)
class Sample:
    repo: str
    name: str                      # unique id within the dataset; the output filename
    source_path: Path              # the .c file that gets compiled
    function: Optional[str] = None  # symbol to decompile; None -> `name`, else auto-detect
    cflags: tuple[str, ...] = ()
    test_path: Optional[Path] = None   # the test driver, if the dataset has one
    io_path: Optional[Path] = None     # its IO pairs (.json or .json.gz), if any


def _harness(directory: Path, name: str) -> dict:
    """The test driver and IO file that sit beside `<name>.c`, if present."""
    found: dict = {}
    for ext in ("cpp", "c"):
        if (directory / f"{name}_test.{ext}").is_file():
            found["test_path"] = directory / f"{name}_test.{ext}"
            break
    for ext in ("json", "json.gz"):
        if (directory / f"{name}_io.{ext}").is_file():
            found["io_path"] = directory / f"{name}_io.{ext}"
            break
    return found


def _locate(sel: RepoSelection) -> Path:
    if sel.path is not None:
        if not sel.path.exists():
            raise FileNotFoundError(f"dataset {sel.name!r}: path {sel.path} does not exist")
        return sel.path
    base = ROOT / "datasets"
    for cand in (base / sel.name, base / f"{sel.name}.jsonl", base / f"{sel.name}.py"):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"dataset {sel.name!r} not found: expected datasets/{sel.name}/, "
        f"datasets/{sel.name}.jsonl or datasets/{sel.name}.py (or set `path:`)")


def _records_from_jsonl(path: Path):
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: not valid JSON ({e})") from None


def _records_from_adapter(path: Path):
    spec = importlib.util.spec_from_file_location(f"decomplite_dataset_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "load"):
        raise ValueError(f"{path}: an adapter must define `load()` returning sample dicts")
    return mod.load()


def _materialize(sel: RepoSelection, records) -> dict[str, Sample]:
    """Write each record's source under data/repos/<repo>/ so it has a file to
    compile, rewriting only when the text changed (keeps build caches warm)."""
    out_dir = DATA_DIR / "repos" / sel.name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: dict[str, Sample] = {}
    for rec in records:
        name, source = rec.get("name"), rec.get("source")
        if not name or source is None:
            raise ValueError(f"dataset {sel.name!r}: every sample needs `name` and `source`")
        name = str(name)
        if not _SAFE_NAME.match(name):
            raise ValueError(f"dataset {sel.name!r}: sample name {name!r} is not filename-safe")
        if name in samples:
            raise ValueError(f"dataset {sel.name!r}: duplicate sample name {name!r}")
        files = {f"{name}.c": source}
        if rec.get("test"):
            files[f"{name}_test.{rec.get('test_ext') or 'c'}"] = rec["test"]
        if rec.get("io") is not None:
            files[f"{name}_io.json"] = json.dumps(rec["io"])
        for fname, text in files.items():
            if not (out_dir / fname).exists() or (out_dir / fname).read_text() != text:
                (out_dir / fname).write_text(text)
        samples[name] = Sample(
            repo=sel.name, name=name, source_path=out_dir / f"{name}.c",
            function=rec.get("function") or None,
            cflags=sel.cflags + tuple(rec.get("cflags") or ()),
            **_harness(out_dir, name),
        )
    return samples


def load_dataset(sel: RepoSelection) -> dict[str, Sample]:
    """Every sample of one dataset, keyed by name."""
    path = _locate(sel)
    if path.is_dir():
        return {
            f.stem: Sample(repo=sel.name, name=f.stem, source_path=f, cflags=sel.cflags,
                           **_harness(path, f.stem))
            for f in sorted(path.glob("*.c"))
            # `<name>_test.c` beside `<name>.c` is that sample's driver, not a sample.
            if not (f.stem.endswith("_test") and (path / f"{f.stem[:-5]}.c").is_file())
        }
    if path.suffix == ".py":
        return _materialize(sel, _records_from_adapter(path))
    return _materialize(sel, _records_from_jsonl(path))


def select(sel: RepoSelection, samples: dict[str, Sample]) -> tuple[list[Sample], list[str]]:
    """Apply the config's `functions` (exact) and `patterns` (fnmatch). With
    neither given the whole dataset is selected. Returns (chosen, missing)."""
    if not sel.functions and not sel.patterns:
        return list(samples.values()), []
    chosen: dict[str, Sample] = {}
    missing = [fn for fn in sel.functions if fn not in samples]
    for fn in sel.functions:
        if fn in samples:
            chosen[fn] = samples[fn]
    for pat in sel.patterns:
        for name in samples:
            if fnmatch.fnmatch(name, pat):
                chosen[name] = samples[name]
    return [chosen[k] for k in sorted(chosen)], missing
