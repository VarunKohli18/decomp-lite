# -*- coding: utf-8 -*-
# Ghidra headless post-script: decompile a function selected by ADDRESS, not name.
# Runs inside Ghidra's Jython 2.7 — DO NOT import from the decomplite package.
#
# This is the stripped-tier counterpart of dump_function_c.py. On a stripped
# object there is no symbol table, so name matching is impossible — Ghidra names
# every function FUN_<addr>. Instead we resolve the target by its location:
#   <section_name> + <offset-within-section>
# read from the UNSTRIPPED twin's ELF symbol table before stripping (st_shndx
# gives the section, st_value gives the offset). Anchoring on the section block
# Ghidra created makes us independent of Ghidra's image base, and works for both
# `-ffunction-sections` (one `.text.<fn>` per function, offset 0) and a merged
# `.text` (offset = symbol value).
#
# Invocation (from decomplite.representations):
#   analyzeHeadless <projectDir> <projectName> -import <stripped.o> \
#       -postScript dump_function_by_addr.py <section> <offset_hex> <output_path> [rename_to] \
#       -scriptPath <this dir> -deleteProject
#
# If <rename_to> is given, the resolved function is renamed to it before getC()
# so the downstream cleaner can anchor on a known signature line; callees/data
# stay FUN_/DAT_ (genuinely stripped). If omitted, the synthetic FUN_<addr> name
# is kept and printed on the OK line so the caller can recover it.
#
# Prints "[dump_function_by_addr] OK name=<ghidra_name> wrote ..." on success.
# Exits non-zero (and writes nothing) on any failure so the caller can tell
# "Ghidra ran but found nothing" from "we got C".
#
# @category Decompiler
# @runtime Jython

import sys

from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor


def _fail(msg):
    print("[dump_function_by_addr] FAIL: %s" % msg)
    sys.exit(1)


def _resolve_addr(section, offset):
    """section block start + offset → Address, robust to Ghidra's load base."""
    mem = currentProgram.getMemory()
    block = mem.getBlock(section)
    if block is None:
        # Merged-.text fallback: some loaders fold `.text.<fn>` into `.text`.
        block = mem.getBlock(".text")
        if block is None:
            avail = [b.getName() for b in mem.getBlocks()]
            _fail("section %r not found (and no .text). Blocks: %s" % (section, avail[:30]))
    return block.getStart().add(offset)


def main():
    args = getScriptArgs()
    if len(args) < 3:
        _fail("usage: dump_function_by_addr.py <section> <offset_hex> <out_path> [rename_to]")
    section = args[0]
    offset = int(args[1], 16)
    out_path = args[2]
    rename_to = args[3] if len(args) >= 4 else None

    addr = _resolve_addr(section, offset)

    fm = currentProgram.getFunctionManager()
    target = fm.getFunctionAt(addr)
    if target is None:
        # Auto-analysis didn't carve a function at this entry (can happen for a
        # function never called within a relocatable .o). Force it: disassemble
        # from the entry, then create the function explicitly.
        disassemble(addr)
        target = createFunction(addr, None)
    if target is None:
        target = fm.getFunctionContaining(addr)
    if target is None:
        _fail("no function at %s (section=%s offset=0x%x)" % (addr, section, offset))
    if target.getEntryPoint() != addr:
        _fail("function at %s starts at %s, not the requested entry"
              % (addr, target.getEntryPoint()))

    if rename_to:
        from ghidra.program.model.symbol import SourceType
        target.setName(rename_to, SourceType.USER_DEFINED)

    ghidra_name = target.getName()

    decomp = DecompInterface()
    decomp.openProgram(currentProgram)
    res = decomp.decompileFunction(target, 60, ConsoleTaskMonitor())
    if not res.decompileCompleted():
        _fail("decompile failed: %s" % res.getErrorMessage())
    code = res.getDecompiledFunction().getC()
    if not code:
        _fail("empty decompilation")

    with open(out_path, "w") as fh:
        fh.write(code)
    print("[dump_function_by_addr] OK name=%s wrote %d bytes to %s"
          % (ghidra_name, len(code), out_path))


main()
