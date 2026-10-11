import argparse

from dotenv import load_dotenv

load_dotenv()

from src import energyuq
from src.machines import guess_machine
from src.programs.benchmark import ExecuteSH


def parse_args():
    parser = argparse.ArgumentParser(description="Run or resume an EnergyUQ campaign.")
    parser.add_argument(
        "benchmark",
        nargs="?",
        default="FAKEWORK",
        help="Benchmark name (HPCG, JA, PO, etc). Default: FAKEWORK",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume latest run or run from --dir",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Run directory to save to or resume from",
    )
    parser.add_argument(
        "--max-refinements",
        type=int,
        default=100,
        help="Maximum number of refinement iterations (default: 100)",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=75,
        help="Minimum number of total runs (default: 75)",
    )
    parser.add_argument(
        "--stable-mode",
        type=str,
        default="consecutive",
        choices=["consecutive", "window_mean"],
        help="Convergence mode: consecutive stable steps or mean of last stable_window relative changes (default: consecutive)",
    )
    parser.add_argument(
        "--stable-window",
        type=int,
        default=3,
        help="Window size for window_mean mode (default: 3)",
    )
    parser.add_argument(
        "--mean-tol",
        type=float,
        default=0.01,
        help="Relative mean-change tolerance (default: 0.01)",
    )
    parser.add_argument(
        "--var-tol",
        type=float,
        default=0.05,
        help="Relative variance-change tolerance (default: 0.05)",
    )
    parser.add_argument(
        "--numa",
        action="store_true",
        help="Include kernel numa balancing, if available",
    )
    parser.add_argument(
        "--force-two",
        action="store_true",
        help="Enforces at least order 2 for every dimension at the beginning",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help="Compress into tar.gz at the end",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        benchmark = next(
            b for b in ExecuteSH.__subclasses__()
            if b.__name__.lower() == args.benchmark.lower()
        )
    except StopIteration as e:
        raise ValueError(f"Unknown benchmark {args.benchmark}") from e

    mach = guess_machine()

    campaign, analysis = energyuq.create(
        benchmark,
        mach,
        dir=args.dir,
        resume=args.resume,
        numa=args.numa,
    )

    energyuq.refine_and_analyse(
        campaign,
        analysis,
        max_number_of_refinements=args.max_refinements,
        min_number_of_samples=args.min_runs,
        mean_tol=args.mean_tol,
        var_tol=args.var_tol,
        stable_mode=args.stable_mode,
        stable_window=args.stable_window,
        save_dir=args.dir,
        force_two=args.force_two
    )

    energyuq.save(campaign, analysis, name=benchmark.name, dir=args.dir, status="completed", pack=args.pack)
