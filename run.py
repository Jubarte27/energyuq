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
        help="Benchmark name (e.g. HPCG, JA, PO, FAKEWORK). Default: FAKEWORK",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume latest run or run from --dir without re-running screening",
    )
    parser.add_argument(
        "--morris-only",
        action="store_true",
        help="Execute only the Morris screening, save its files, and stop",
    )
    parser.add_argument(
        "--from-morris",
        type=str,
        default=None,
        help="Run using an existing Morris screening run from path as if it was just executed",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Run directory to save to or resume from",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=2,
        help="Periodic checkpoint saving frequency in iterations (default: 2)",
    )
    parser.add_argument(
        "--max-refinements",
        type=int,
        default=100,
        help="Maximum number of refinement iterations (default: 100)",
    )
    parser.add_argument(
        "--numa",
        action="store_true",
        help="Include kernel numa balancing, if available",
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
    if mach is None:
        raise RuntimeError("I don't know where I am at")

    campaign, analysis = energyuq.create(
        benchmark,
        mach,
        dir=args.dir,
        resume=args.resume,
        numa=args.numa,
        morris_only=args.morris_only,
        from_morris=args.from_morris,
    )

    if campaign and analysis:
        energyuq.refine_and_analyse(
            campaign,
            analysis,
            max_number_of_refinements=args.max_refinements,
            save_every=args.save_every,
            save_dir=args.dir,
        )

        energyuq.save(campaign, analysis, dir=args.dir, status="completed")
