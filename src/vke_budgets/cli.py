"""Command line: `vke-budgets run --start ... --end ...`. See pipeline.run."""
import argparse
from dataclasses import replace

from . import geometry, pipeline
from .budget import BudgetOptions

GEOMETRIES = {"default": geometry.DEFAULT, "reference": geometry.REFERENCE}


def main(argv=None):
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
    args = ap.parse_args(argv)

    options = replace(BudgetOptions(), geometry=GEOMETRIES[args.geometry],
                      quantities=tuple(args.quantities.split(",")), keep_3d=args.keep_3d,
                      dyn_crosscheck=args.dyn_crosscheck, ncl_vte_form=args.ncl_vte_form,
                      flux_form=args.flux_form, boundary_everywhere=args.boundary_everywhere)
    pipeline.run(args.start, args.end, args.root, args.out, options, layout=args.layout,
                 significant_bits=args.bits, n_workers=args.workers)


if __name__ == "__main__":
    main()
