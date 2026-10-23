"""Command line: `vke-budgets run ...` (the budgets, pipeline.run) and
`vke-budgets climatology ...` (AR statistics by month, climatology.run)."""
import argparse
from collections.abc import Sequence
from dataclasses import replace

from . import climatology, geometry, pipeline
from .budget import BudgetOptions

GEOMETRIES = {"default": geometry.DEFAULT, "reference": geometry.REFERENCE}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vke-budgets",
                                 description="KEV and VTE budgets of atmospheric rivers")
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="compute and write the budgets for a range of days",
                         description=pipeline.__doc__,
                         formatter_class=argparse.RawDescriptionHelpFormatter)
    run.add_argument("--start", required=True, help="first day, YYYY-MM-DD")
    run.add_argument("--end", required=True, help="last day, YYYY-MM-DD (inclusive)")
    run.add_argument("--root", required=True, help="MERRA-2 root, one directory per year")
    run.add_argument("--out", required=True, help="output directory")
    run.add_argument("--layout", choices=("clean", "legacy"), default="clean")
    run.add_argument("--bits", type=int, default=None,
                     help="BitRound significant bits of the month files (default: exact float32)")
    run.add_argument("--workers", type=int, default=1, help="windows computed at once")
    run.add_argument("--geometry", choices=GEOMETRIES, default="default")
    run.add_argument("--quantities", default="KEV,VTE", help="comma-separated, from KEV,VTE")
    run.add_argument("--keep-3d", action="store_true", help="also write the terms on levels")
    run.add_argument("--dyn-crosscheck", action="store_true", help="also IDOKE and IDOV")
    run.add_argument("--ncl-vte-form", action="store_true", help="also IHCV_vO and IVCV_vO")
    run.add_argument("--boundary-everywhere", action="store_true",
                     help="apply the boundary term (SPTE) at every point, as the reference")
    run.add_argument("--flux-form", action="store_true",
                     help="also the flux-form decomposition (IHFOKE ... IQPHIDIVV)")
    clim = sub.add_parser("climatology", help="AR counts and frequencies, one file per month",
                          description=climatology.__doc__,
                          formatter_class=argparse.RawDescriptionHelpFormatter)
    clim.add_argument("--start", required=True, help="first month, YYYY-MM")
    clim.add_argument("--end", required=True, help="last month, YYYY-MM (inclusive)")
    clim.add_argument("--root", required=True, help="MERRA-2 root, one directory per year")
    clim.add_argument("--out", required=True, help="output directory")
    clim.add_argument("--task", type=int, default=None,
                      help="only the task-th group of months (0 = the first), for array jobs")
    clim.add_argument("--months-per-task", type=int, default=1, help="months in each group")
    clim.add_argument("--save-steps", action="store_true",
                      help="also write the columns and masks of every time step")
    args = ap.parse_args(argv)

    if args.command == "climatology":
        if args.task is not None:
            try:
                climatology.task_months(climatology.months_between(args.start, args.end),
                                        args.task, args.months_per_task)
            except ValueError as error:
                clim.error(str(error))
        climatology.run(args.start, args.end, args.root, args.out, task=args.task,
                        months_per_task=args.months_per_task, save_steps=args.save_steps)
        return

    options = replace(BudgetOptions(), geometry=GEOMETRIES[args.geometry],
                      quantities=tuple(args.quantities.split(",")), keep_3d=args.keep_3d,
                      dyn_crosscheck=args.dyn_crosscheck, ncl_vte_form=args.ncl_vte_form,
                      flux_form=args.flux_form, boundary_everywhere=args.boundary_everywhere)
    pipeline.run(args.start, args.end, args.root, args.out, options, layout=args.layout,
                 significant_bits=args.bits, n_workers=args.workers)


if __name__ == "__main__":
    main()
