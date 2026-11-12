"""The three metrics.

  standalone_compile  RC   the answer compiles on its own
  compile_ok          RCI  the answer, put in place of the original function,
                           compiles together with code that CALLS it
  test_passed         RE   that build then passes every test case

RC needs nothing. RCI and RE need the dataset to ship a test harness for the
sample (see datasets.py); without one they are left empty, not scored 0.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from decomplite import build as B
from decomplite.config import DATA_DIR
from decomplite.datasets import Sample

# Prepended to the candidate wherever it is built for the harness. A decompilation routinely
# uses fixed-width / stddef types (uint32_t, size_t, NULL) without including
# their headers; failing it for that measures #include hygiene, not recovery.
# Applied to every arm alike, and standard headers are include-guarded.
PREAMBLE = (
    "#include <stdint.h>\n"
    "#include <stddef.h>\n"
    "#include <stdlib.h>\n"
    "#include <string.h>\n"
    "#include <stdio.h>\n"
)
#: Cap on diagnostics fed back into a prompt.
MAX_FEEDBACK_CHARS = 8000
#: Per-test-case execution timeout, and the wall clock for one cell's tests.
IO_TIMEOUT_S = 20
IO_BUDGET_S = 240.0


# --- locating and renaming C functions (tree-sitter) ---------------------------

def _parser():
    import tree_sitter_c
    from tree_sitter import Language, Parser
    return Parser(Language(tree_sitter_c.language()))


def _declared_name(node, data: bytes) -> Optional[str]:
    """The identifier a (possibly pointer/parenthesized) declarator declares."""
    for c in node.children:
        if c.type == "identifier":
            return data[c.start_byte:c.end_byte].decode(errors="ignore")
        if c.type in ("pointer_declarator", "parenthesized_declarator", "function_declarator"):
            got = _declared_name(c, data)
            if got:
                return got
    return None


def _function_defs(data: bytes) -> list[tuple[str, int, int]]:
    """(name, start_byte, end_byte) of every function definition."""
    out, stack = [], [_parser().parse(data).root_node]
    while stack:
        n = stack.pop()
        if n.type == "function_definition":
            for c in n.children:
                if c.type in ("function_declarator", "pointer_declarator",
                              "parenthesized_declarator"):
                    name = _declared_name(c, data)
                    if name:
                        out.append((name, n.start_byte, n.end_byte))
                        break
            continue
        stack.extend(n.children)
    return out


def rename_function(text: str, target: str) -> str:
    """When the answer defines exactly ONE function, rename it (and its
    self-references) to `target`. The model cannot know the original name, so
    the name is not part of what is scored. With several definitions there is
    no telling which is the function, and the text is left alone."""
    data = text.encode()
    defs = _function_defs(data)
    if len(defs) != 1 or defs[0][0] == target:
        return text
    old = defs[0][0].encode()
    spans, stack = [], [_parser().parse(data).root_node]
    while stack:
        n = stack.pop()
        if n.type == "identifier" and data[n.start_byte:n.end_byte] == old:
            spans.append((n.start_byte, n.end_byte))
        stack.extend(n.children)
    for start, end in sorted(spans, reverse=True):
        data = data[:start] + target.encode() + data[end:]
    return data.decode(errors="ignore")


# --- RC: compiles on its own -----------------------------------------------------

def standalone_compile(text: str, opt: str = "O0", *, strict: bool = False) -> tuple[bool, str]:
    """Compile the answer alone, as a translation unit. Returns (ok, diagnostics).

    The METRIC (strict=False) uses the benchmark's own flags at
    the cell's -O level, warnings off, nothing prepended. As repair FEEDBACK
    (strict=True) it is a stricter, binary-only check instead: the preamble
    is prepended and implicit declarations are errors, so the diagnostics name
    what is actually missing."""
    tc = B.toolchain()
    flags = (["-std=gnu11", "-Werror=implicit-function-declaration", "-Werror=implicit-int"]
             if strict else [f"-{opt}", *B.C_BASE_CFLAGS, "-w"])
    with tempfile.TemporaryDirectory(prefix="recomp_") as td:
        (Path(td) / "candidate.c").write_text((PREAMBLE if strict else "") + text + "\n")
        res = subprocess.run([tc.cc, "-c", "candidate.c", *flags, "-o", "candidate.o"],
                             cwd=td, capture_output=True, text=True, timeout=180)
    return res.returncode == 0, _clip((res.stderr or "").strip())


def _clip(diag: str) -> str:
    if len(diag) > MAX_FEEDBACK_CHARS:
        diag = diag[:MAX_FEEDBACK_CHARS] + "\n[... diagnostics truncated ...]"
    return diag


# --- RCI + RE: the test harness ---------------------------------------------------

def _diff_io(observed, expected) -> bool:
    """ExeBench's float-tolerant structural comparison."""
    if isinstance(expected, float) or isinstance(observed, float):
        try:
            return math.isclose(float(observed), float(expected), rel_tol=1e-3)
        except (TypeError, ValueError):
            return False
    if isinstance(observed, list) and isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _diff_io(a, b) for a, b in zip(observed, expected))
    if isinstance(observed, dict) and isinstance(expected, dict):
        return all(k in expected and _diff_io(observed[k], expected[k]) for k in observed)
    return observed == expected


def _io_dict(side) -> dict:
    """An IO-pair side as a plain {var: value} dict (the Hugging Face form of
    ExeBench stores a list of {var, value} instead)."""
    if isinstance(side, list):
        return {e["var"]: json.loads(e["value"]) for e in side}
    return side or {}


def _run(cmd, cwd: Path, timeout: float) -> Optional[int]:
    """Exit status, or None on timeout. Kills the whole process group, so a
    runaway candidate cannot outlive its test."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait()
        return None


def _run_tests(binp: Path, work: Path, io_path: Optional[Path]) -> tuple[int, int, str]:
    """(passed, total, note). Every test case runs; one failure does not stop
    the rest, so passed/total is a true pass rate."""
    if io_path is None:
        rc = _run([str(binp)], work, IO_TIMEOUT_S)
        return int(rc == 0), 1, "" if rc == 0 else ("timeout" if rc is None else f"exit {rc}")
    opener = gzip.open if io_path.suffix == ".gz" else open
    with opener(io_path, "rt") as fh:
        pairs = json.load(fh)
    inj, outj = work / "_in.json", work / "_out.json"
    passed, first, deadline = 0, "", time.monotonic() + IO_BUDGET_S
    for i, pair in enumerate(pairs):
        why = ""
        if time.monotonic() >= deadline:
            why = f"time budget of {IO_BUDGET_S:.0f}s exhausted"
        else:
            inj.write_text(json.dumps(_io_dict(pair.get("input"))))
            outj.unlink(missing_ok=True)
            if _run([str(binp), str(inj), str(outj)], work, IO_TIMEOUT_S) is None:
                why = "timeout"
            elif not outj.exists():
                why = "no output produced"
            else:
                try:
                    ok = _diff_io(json.loads(outj.read_text()), _io_dict(pair.get("output")))
                except json.JSONDecodeError:
                    ok = False
                passed += int(ok)
                why = "" if ok else "output mismatch"
        if why and not first:
            first = f"test {i}: {why}"
    return passed, len(pairs), first


def harness(sample: Sample, text: str, function: str, *, run: bool = True) -> dict:
    """Put the answer in place of the original function and build the sample's
    test harness around it (RCI); then, with `run`, execute the tests (RE).

    `status` is "ok" when the verdict is about the answer, and names the reason
    otherwise (no harness, nothing to run on, ...), in which case the metrics
    are None rather than 0.
    """
    out = {"compile_ok": None, "compile_error": "", "test_passed": None,
           "test_pass": None, "test_total": None, "test_output": "", "status": "ok"}
    if sample.test_path is None:
        return dict(out, status="no_harness")
    try:
        reference = sample.source_path.read_bytes()
        span = next(((s, e) for name, s, e in _function_defs(reference) if name == function), None)
        candidate = rename_function(text, function)
    except ImportError:
        return dict(out, status="tree_sitter_not_installed")
    if span is None:
        return dict(out, status=f"cannot locate {function} in {sample.source_path.name}")

    tc = B.toolchain()
    cpp = sample.test_path.suffix == ".cpp"
    driver = tc.cxx if cpp else tc.cc
    if shutil.which(driver) is None:
        return dict(out, status=f"{driver} not installed")

    src_dir = sample.source_path.parent
    work_root = DATA_DIR / "eval_work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="h_", dir=work_root) as td:
        work = Path(td)
        (work / sample.source_path.name).write_bytes(
            PREAMBLE.encode() + reference[:span[0]] + candidate.encode() + reference[span[1]:])
        shutil.copy(sample.test_path, work / sample.test_path.name)
        if (src_dir / "_include").is_dir():
            (work / "_include").symlink_to(src_dir / "_include", target_is_directory=True)
        binp = work / "_harness.x"
        # -O0 -w: the harness checks that the call interface holds and what the
        # function computes; warnings and optimization are not its business.
        res = subprocess.run(
            [driver, *(["-fpermissive"] if cpp else []), "-O0", "-w",
             sample.test_path.name, "-I", "_include", "-I", "_include/clib",
             "-I", str(src_dir), *sample.cflags, "-o", binp.name, "-lm"],
            cwd=work, capture_output=True, text=True, timeout=180)
        if res.returncode != 0 or not binp.exists():
            # An answer that does not build has not passed its tests: RE is 0,
            # not unmeasured.
            return dict(out, compile_ok=0, test_passed=None if not run else 0,
                        compile_error=_clip(((res.stderr or res.stdout) or "")[-2000:].strip()))
        out["compile_ok"] = 1
        if not run:
            return out
        if not tc.can_execute():
            return dict(out, status="built, but this machine cannot run x86-64 Linux binaries")
        passed, total, note = _run_tests(binp, work, sample.io_path)
        out.update(test_pass=passed, test_total=total,
                   test_passed=int(total > 0 and passed == total),
                   test_output=f"{passed}/{total} passed" + (f"; {note}" if note else ""))
    return out


def feedback(kind: str, sample: Sample, text: str, function: str, opt: str) -> dict:
    """Check an answer the way a pipeline's `feedback:` setting says, and word
    the result for the model: {ok, title, text, source}.

      compiler   the answer compiled on its own, strictly
      interface  the answer built into the sample's test harness, so a wrong
                 call interface is flagged too
      tests      as `interface`, then the test cases are run

    `interface` and `tests` use knowledge from outside the binary (how callers
    use the function; what it should compute). A sample without a harness gets
    `compiler` feedback instead; `source` records what was actually used.
    """
    if kind in ("interface", "tests") and sample.test_path is not None:
        res = harness(sample, text, function, run=(kind == "tests"))
        if res["compile_ok"] == 0:
            return {"ok": False, "title": "Compiler diagnostics",
                    "text": res["compile_error"], "source": "interface"}
        if res["compile_ok"] == 1:
            if res["test_passed"] == 0:
                return {"ok": False, "title": "Test results", "source": "tests",
                        "text": "The reconstruction compiles, but it does not behave like "
                                "the function in the binary. Running the function's test "
                                f"cases gave: {res['test_output']}."}
            return {"ok": True, "title": "", "text": "",
                    "source": "tests" if res["test_passed"] == 1 else "interface"}
    ok, diag = standalone_compile(text, opt, strict=True)
    return {"ok": ok, "title": "Compiler diagnostics", "text": diag, "source": "compiler"}
