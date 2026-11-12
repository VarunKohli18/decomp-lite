"""Compile one sample at one optimization level and derive the stripped image.

    <name>.c --cc -c--> <name>.o --cc -shared--> <name>.linked.so --strip-all--> <name>.stripped.so

The stripped shared object is what both input representations are taken from.
The unstripped twin is kept only to learn WHERE the function is (address and
size); nothing from it reaches the model.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from decomplite.config import DATA_DIR
from decomplite.datasets import Sample

# Same flags the full benchmark compiles C with. `-fPIC` lets the object link
# into a shared object; the stack-protector pin keeps the output identical
# between a distro's native gcc and its cross driver.
C_BASE_CFLAGS = [
    "-fPIC",
    "-ffunction-sections",
    "-fdata-sections",
    "-fstack-protector-strong",
    "-Wno-error",
    "-Wno-unused-function",
]

# Localize every defined symbol at link time so the target function is not
# exported into `.dynsym` (which survives stripping and would leak its name).
_LOCALIZE_ALL_VERSION_SCRIPT = "{ local: *; };\n"


_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()


def lock_for(key) -> threading.Lock:
    """One lock per key. Worker threads of different cells share a (sample, opt)
    build and its cached inputs; whoever gets there first does the work."""
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def write_atomic(path: Path, text: str) -> None:
    """Never leave a half-written file where a reader can find it."""
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(text)
    os.replace(tmp, path)


class ToolchainError(RuntimeError):
    """The x86-64 Linux toolchain is missing; nothing can be built."""


@dataclass(frozen=True)
class Toolchain:
    cc: str
    objdump: str
    objcopy: str
    cxx: str = ""                  # only needed to build C++ test harnesses

    def can_execute(self) -> bool:
        """Whether binaries this toolchain builds can run on this machine."""
        return platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")


def toolchain() -> Toolchain:
    """Native gcc on an x86-64 Linux host, the cross driver anywhere else.
    Override with DECOMPLITE_CC / _CXX / _OBJDUMP / _OBJCOPY."""
    native = platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")
    prefix = "" if native else "x86_64-linux-gnu-"
    tc = Toolchain(
        cc=os.environ.get("DECOMPLITE_CC") or prefix + "gcc",
        objdump=os.environ.get("DECOMPLITE_OBJDUMP") or prefix + "objdump",
        objcopy=os.environ.get("DECOMPLITE_OBJCOPY") or prefix + "objcopy",
        cxx=os.environ.get("DECOMPLITE_CXX") or prefix + "g++",
    )
    missing = [t for t in (tc.cc, tc.objdump, tc.objcopy) if shutil.which(t) is None]
    if missing:
        raise ToolchainError(
            f"x86-64 Linux toolchain not found (missing: {', '.join(missing)}). "
            f"Use the provided Dockerfile, or on Debian/Ubuntu install "
            f"`gcc binutils` (x86-64 host) / `gcc-x86-64-linux-gnu "
            f"binutils-x86-64-linux-gnu libc6-dev-amd64-cross` (other hosts).")
    return tc


@dataclass
class Build:
    """One (sample, opt) build. `ok` False means `error` says why."""
    repo: str
    name: str
    opt: str
    ok: bool
    error: str = ""
    obj: str = ""
    linked: str = ""               # unstripped shared object (symbol lookup only)
    stripped: str = ""             # what the model's inputs are derived from
    function: str = ""             # resolved symbol name
    section: str = ""
    offset: int = 0                # offset of the function within `section`
    vaddr: int = 0
    size: int = 0


def build_dir(repo: str, opt: str) -> Path:
    return DATA_DIR / "builds" / repo / "x86_64" / opt


def _functions(elf_path: Path) -> dict[str, dict]:
    """Defined STT_FUNC symbols of an unstripped ELF, by name."""
    from elftools.elf.elffile import ELFFile

    out: dict[str, dict] = {}
    with open(elf_path, "rb") as fh:
        elf = ELFFile(fh)
        symtab = elf.get_section_by_name(".symtab")
        if symtab is None:
            return out
        for sym in symtab.iter_symbols():
            if sym["st_info"]["type"] != "STT_FUNC" or sym["st_shndx"] == "SHN_UNDEF":
                continue
            sec = elf.get_section(sym["st_shndx"])
            out[sym.name] = {
                "bind": sym["st_info"]["bind"], "section": sec.name,
                "vaddr": int(sym["st_value"]), "size": int(sym["st_size"]),
                "offset": int(sym["st_value"]) - int(sec["sh_addr"]),
            }
    return out


def _resolve_function(sample: Sample, obj: Path) -> str:
    """Which symbol of the object is the function to decompile."""
    funcs = _functions(obj)
    want = sample.function or sample.name
    if want in funcs:
        return want
    if sample.function:
        raise LookupError(
            f"function {want!r} is not defined in {sample.source_path.name} "
            f"(inlined away, or misspelled). Defined: {sorted(funcs)}")
    # No explicit symbol and none named after the sample: unambiguous only when
    # the translation unit has a single (global) function.
    globals_ = [n for n, f in funcs.items() if f["bind"] == "STB_GLOBAL"]
    for pool in (globals_, list(funcs)):
        if len(pool) == 1:
            return pool[0]
    raise LookupError(
        f"cannot tell which function of {sample.source_path.name} to decompile: "
        f"none is named {want!r} and it defines {sorted(funcs) or 'no functions'}. "
        f"Name the file after the function, or set `function` in the dataset.")


def _run(cmd: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)


def compile_object(tc: Toolchain, src: Path, out: Path, opt: str,
                   cflags=(), cwd: Optional[Path] = None) -> tuple[bool, str]:
    """`cc -c src` under the benchmark flags; returns (ok, diagnostics)."""
    res = _run([tc.cc, "-c", str(src), *C_BASE_CFLAGS, f"-{opt}", *cflags, "-o", str(out)],
               cwd=cwd)
    ok = res.returncode == 0 and out.exists()
    return ok, (res.stderr or res.stdout or "")[-2000:]


def build(sample: Sample, opt: str, *, force: bool = False) -> Build:
    """Build one cell, or return the cached build when the source is unchanged."""
    with lock_for(("build", sample.repo, sample.name, opt)):
        return _build(sample, opt, force)


def _build(sample: Sample, opt: str, force: bool) -> Build:
    bd = build_dir(sample.repo, opt)
    bd.mkdir(parents=True, exist_ok=True)
    meta_path = bd / f"{sample.name}.build.json"
    obj = bd / f"{sample.name}.o"
    linked = bd / f"{sample.name}.linked.so"
    stripped = bd / f"{sample.name}.stripped.so"
    stamp = [sample.source_path.stat().st_mtime, list(sample.cflags), sample.function]

    if not force and meta_path.exists():
        cached = json.loads(meta_path.read_text())
        if cached.pop("stamp", None) == stamp:
            b = Build(**cached)
            if not b.ok or stripped.exists():
                return b

    b = Build(repo=sample.repo, name=sample.name, opt=opt, ok=False)
    try:
        tc = toolchain()
        # Compiled from the source's own directory so relative `#include "x.h"`
        # and `-I` flags in the dataset's cflags resolve as the author expects.
        ok, log = compile_object(tc, sample.source_path, obj, opt, sample.cflags,
                                 cwd=sample.source_path.parent)
        if not ok:
            raise RuntimeError(f"compile failed:\n{log}")
        with open(obj, "rb") as fh:
            if fh.read(4) != b"\x7fELF":
                raise ToolchainError(
                    f"{tc.cc} did not produce an ELF object (a macOS/Windows "
                    f"compiler?). Use the Dockerfile or set DECOMPLITE_CC.")
        fn = _resolve_function(sample, obj)

        with tempfile.NamedTemporaryFile("w", suffix=".map", delete=False) as vs:
            vs.write(_LOCALIZE_ALL_VERSION_SCRIPT)
        try:
            res = _run([tc.cc, "-shared", f"-Wl,--version-script={vs.name}",
                        *C_BASE_CFLAGS, "-o", str(linked), str(obj)])
        finally:
            os.unlink(vs.name)
        if res.returncode != 0 or not linked.exists():
            raise RuntimeError(f"link failed:\n{(res.stderr or '')[-2000:]}")

        sym = _functions(linked).get(fn)
        if sym is None or sym["size"] <= 0:
            raise LookupError(f"function {fn!r} has no code in the linked image")
        shutil.copy(linked, stripped)
        res = _run([tc.objcopy, "--strip-all", str(stripped)])
        if res.returncode != 0:
            raise RuntimeError(f"strip failed:\n{res.stderr[-2000:]}")

        b = Build(repo=sample.repo, name=sample.name, opt=opt, ok=True,
                  obj=str(obj), linked=str(linked), stripped=str(stripped),
                  function=fn, section=sym["section"], offset=sym["offset"],
                  vaddr=sym["vaddr"], size=sym["size"])
    except ToolchainError:
        raise                      # not a per-cell outcome; stop the run
    except Exception as e:
        b.error = str(e)
    write_atomic(meta_path, json.dumps(dict(asdict(b), stamp=stamp), indent=1))
    return b
