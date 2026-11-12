"""Command line: `decomplite benchmark --config benchmark.yaml --phase all`."""
from __future__ import annotations

import argparse
import sys

from decomplite import benchmark as bench
from decomplite import config
from decomplite.build import ToolchainError

PHASES = ("build", "decompile", "eval")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="decomplite", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("benchmark", help="run a config")
    b.add_argument("--config", default="benchmark.yaml",
                   help="path to the benchmark.yaml driving the run")
    b.add_argument("--phase", default="all", choices=(*PHASES, "all"),
                   help="build | decompile | eval | all (default)")
    b.add_argument("--dry-run", action="store_true",
                   help="print the cell plan and exit; no builds, no LLM calls")
    b.add_argument("--force", action="store_true",
                   help="redo the selected phase's work even if it is cached "
                        "(decompile: also replaces results made with other settings)")
    b.add_argument("--workers", type=int, default=1,
                   help="parallel workers (builds/Ghidra runs, concurrent LLM calls)")
    for name in ("repo", "opt", "fn", "model", "pipeline"):
        b.add_argument(f"--filter-{name}", default=None,
                       help=f"restrict to these {name}s (comma-separated)")
    args = ap.parse_args(argv)

    config.load_dotenv(config.ROOT / ".env")
    try:
        cfg = config.load(args.config)
        filters = {n: getattr(args, f"filter_{n}") for n in ("repo", "opt", "fn", "model", "pipeline")}
        samples, opts, cells = bench.resolve(cfg, filters)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"{len(samples)} functions x {len(opts)} opts x "
          f"{len({c.model for c in cells})} model(s) x "
          f"{len({c.pipeline.name for c in cells})} pipeline(s) = {len(cells)} cells")
    if args.dry_run:
        for pl in cfg.pipelines:
            n, r = len(pl.recovery), len(pl.reviewers)
            print(f"  pipeline {pl.name}: input {'+'.join(pl.input)}, {n} recovery LLM"
                  f"{'s' * (n != 1)}, {r} reviewer{'s' * (r != 1)}, feedback {pl.feedback}, "
                  f"{pl.iterations} iteration{'s' * (pl.iterations != 1)}")
        for c in cells:
            state = "  [done]" if c.is_current() else "  [stale]" if c.out_path.exists() else ""
            print(f"  {c.label()}{state}")
        return 0
    if not cells:
        print("nothing to do")
        return 0

    phases = PHASES if args.phase == "all" else (args.phase,)
    try:
        if "build" in phases:
            print("== build")
            bench.phase_build(cfg, samples, opts, cells, force=args.force, workers=args.workers)
        if "decompile" in phases:
            print("== decompile")
            # `--force` is scoped to the phase asked for: under `all` it rebuilds
            # AND re-queries; under `decompile` it only re-queries the models.
            bench.phase_decompile(cfg, cells, force=args.force, workers=args.workers)
        if "eval" in phases:
            print("== eval")
            bench.phase_eval(cfg, cells, workers=args.workers)
    except ToolchainError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
