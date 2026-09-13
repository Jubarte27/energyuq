#!/usr/bin/env python3
"""
CLI script to analyze and plot multiple experimental runs at once.
Discovers EasyVVUQ campaigns, Dakota outputs, and compilation CSVs across machines and benchmarks.
Generates comparative summary tables (CSV) and publication-quality plots.
"""

import argparse
from pathlib import Path
import sys

from matplotlib.figure import Figure, SubFigure

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
import matplotlib.pyplot as plt

from src.util.multi_run import RunCollection
from src.util import multi_plot


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and plot multiple UQ runs across machines and benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Base directory containing run folders to discover.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*/*/*",
        help="Glob pattern relative to --runs-dir to find run directories.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=None,
        help="Comma-separated list of benchmarks to include (e.g. 'HPCG,LULESH').",
    )
    parser.add_argument(
        "--machines",
        type=str,
        default=None,
        help="Comma-separated list of machines to include (e.g. 'cei,hype,sirius').",
    )
    parser.add_argument(
        "--qoi",
        type=str,
        default="energy_uj",
        help="Primary quantity of interest to evaluate (e.g. 'energy_uj', 'time', 'power_w').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Directory where summary CSVs and generated plots will be saved.",
    )
    parser.add_argument(
        "--plots",
        type=str,
        default="all",
        help="Plots to generate: comma-separated list of 'sobols,convergence,pareto,distribution,best,effects,dashboard' or 'all' or 'none'.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Image format for saved figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for saved figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after generating.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-essential progress output.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    # Parse list filters
    benchmarks = [b.strip() for b in args.benchmarks.split(",")] if args.benchmarks else None
    machines = [m.strip() for m in args.machines.split(",")] if args.machines else None

    if not args.quiet:
        print(f"Discovering runs in '{args.runs_dir}' (pattern: '{args.pattern}')...")

    runs = RunCollection.discover(
        base_dir=args.runs_dir,
        pattern=args.pattern,
        benchmarks=benchmarks,
        machines=machines,
    )

    if len(runs) == 0:
        print(f"No runs found matching criteria in '{args.runs_dir}'. Exiting.")
        sys.exit(1)

    if not args.quiet:
        print(f"Successfully loaded {len(runs)} runs:")
        print(f"  Benchmarks: {', '.join(runs.benchmarks)}")
        print(f"  Machines:   {', '.join(runs.machines)}")
        print()

    # Output directory
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary tables
    if not args.quiet:
        print(f"Exporting summary tables to '{out_dir}'...")

    runs.export_csv(out_dir)
    summary_df = runs.summary_table()

    if not args.quiet:
        print("\nRuns Summary Overview:")
        cols_to_print = [c for c in ["benchmark", "machine", "sample_count", "min_energy_j", "min_time_s", "best_energy_N_THREADS", "best_energy_CLK"] if c in summary_df.columns]
        if cols_to_print:
            print(summary_df[cols_to_print].to_string(index=False))
        print()

    # 2. Plots
    if args.plots.lower() != "none":
        plot_choices = set(p.strip().lower() for p in args.plots.split(","))
        generate_all = "all" in plot_choices

        figs: dict[str, Figure | SubFigure] = {}

        # Sobol sensitivity
        if generate_all or "sobols" in plot_choices:
            if not args.quiet:
                print("Generating Sobol sensitivity plots...")
            figs["sobol_bars"] = multi_plot.plot_multi_sobols(runs, qoi=args.qoi, mode="grouped_bar")
            figs["sobol_heatmap"] = multi_plot.plot_multi_sobols(runs, qoi=args.qoi, mode="heatmap")

        # Convergence
        if generate_all or "convergence" in plot_choices:
            if not args.quiet:
                print("Generating convergence history plot...")
            figs["convergence_by_benchmark"] = multi_plot.plot_multi_convergence(runs, metric="adaptation_error", facet_by="benchmark")

        # Energy vs Time Pareto
        if generate_all or "pareto" in plot_choices:
            if not args.quiet:
                print("Generating Pareto frontier plot...")
            figs["pareto_energy_time"] = multi_plot.plot_multi_energy_time_pareto(runs, qoi_x="time", qoi_y=args.qoi)

        # QoI Distributions
        if generate_all or "distribution" in plot_choices:
            if not args.quiet:
                print("Generating QoI distribution boxplots...")
            figs["distribution_by_machine"] = multi_plot.plot_multi_qoi_distribution(runs, qoi=args.qoi, group_by="machine")
            figs["distribution_by_benchmark"] = multi_plot.plot_multi_qoi_distribution(runs, qoi=args.qoi, group_by="benchmark")

        # Best configurations
        if generate_all or "best" in plot_choices:
            if not args.quiet:
                print("Generating optimal configurations plot...")
            figs["best_configurations"] = multi_plot.plot_multi_best_configurations(runs, qoi=args.qoi)

        # Parameter effects
        if generate_all or "effects" in plot_choices:
            if not args.quiet:
                print("Generating parameter effects plots...")
            figs["param_effects_threads"] = multi_plot.plot_multi_parameter_effects(runs, param="N_THREADS", qoi=args.qoi)

        # Dashboard
        if generate_all or "dashboard" in plot_choices:
            if not args.quiet:
                print("Generating comprehensive summary dashboard...")
            figs["multi_run_dashboard"] = multi_plot.plot_multi_dashboard(runs, qoi=args.qoi)

        # Save all figures
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        for name, fig in figs.items():
            if isinstance(fig, SubFigure):
                continue
            fig_path = plots_dir / f"{name}.{args.format}"
            fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
            if not args.quiet:
                print(f"  Saved: {fig_path}")

        if args.show:
            plt.show()

        for _, fig in figs.items():
            if isinstance(fig, SubFigure):
                continue
            plt.close(fig)

    print(f"\nAnalysis complete! Results saved in '{out_dir}'.")


if __name__ == "__main__":
    main()

