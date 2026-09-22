#!/usr/bin/env python3
"""
Compile multi-run and individual run plots from a RunCollection into a single,
structured, indexed, and bookmarked PDF report.

Features:
- Groups plots strictly by benchmark (with optional global cross-benchmark summary).
- Multi-run comparative plots for each benchmark:
  * plot_multi_dashboard
  * plot_multi_sobols (grouped bar & heatmap)
  * plot_multi_energy_time_pareto
  * plot_multi_convergence
- Complete individual run analyses from analyze_each (all single-run plots per machine).
- Publication-quality cover page, section dividers, and visual Table of Contents.
- Clickable PDF link annotations on the visual index.
- Full interactive PDF bookmarks outline hierarchy for PDF viewers.
- Running headers and page numbering (Page X of Y).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import io
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence, Union

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Link

from src.util.multi_run import RunCollection, RunData, discover_runs
from src.util import multi_plot


# Standard landscape page size (US Letter: 11 x 8.5 inches, 792 x 612 points)
PAGE_WIDTH_INCHES = 11.0
PAGE_HEIGHT_INCHES = 8.5
PAGE_WIDTH_PT = PAGE_WIDTH_INCHES * 72.0
PAGE_HEIGHT_PT = PAGE_HEIGHT_INCHES * 72.0


def _format_qoi_name(qoi: str) -> str:
    """Format QoI key to human-readable label."""
    mapping = {
        "energy_uj": "Energy (μJ)",
        "energy_j": "Energy (J)",
        "energy_scaled": "Scaled Energy",
        "time": "Execution Time (s)",
        "power_w": "Power (W)",
        "EDP": "Energy-Delay Product (J·s)",
    }
    return mapping.get(qoi, qoi.replace("_", " ").title())


def _format_individual_plot_title(name: str) -> str:
    """Return a descriptive, human-readable title for individual analysis figures."""
    mapping = {
        "grid_2d_best_energy_uj": "2D Grid Best Evaluations — Energy (μJ)",
        "sorted_evaluations_energy_uj": "Sorted Evaluations — Energy (μJ)",
        "single_dimension_projections_energy_uj": "Single Dimension Projections — Energy (μJ)",
        "boxplot_per_dimension_energy_uj": "Parameter Boxplots — Energy (μJ)",
        "sobol_indices_energy_uj": "First-Order Sobol Sensitivity Indices — Energy (μJ)",
        "sobol_treemap_energy_uj": "Sobol Sensitivity Treemap — Energy (μJ)",
        "grid_2d_best_time": "2D Grid Best Evaluations — Execution Time (s)",
        "sorted_evaluations_time": "Sorted Evaluations — Execution Time (s)",
        "single_dimension_projections_time": "Single Dimension Projections — Execution Time (s)",
        "boxplot_per_dimension_time": "Parameter Boxplots — Execution Time (s)",
        "sobol_indices_time": "First-Order Sobol Sensitivity Indices — Execution Time (s)",
        "sobol_treemap_time": "Sobol Sensitivity Treemap — Execution Time (s)",
        "adaptation_error_history": "Adaptation Surplus Error Convergence History",
        "statistical_moments_convergence": "Statistical Moments Convergence History",
        "adaptation_histogram": "Surplus Error Distribution Histogram",
        "adaptation_table": "Surrogate Adaptation Iteration Table",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _create_cover_page(
    runs: RunCollection,
    total_pages: int,
    qoi: str,
    date_str: str,
) -> Figure:
    """Create a cover/title page with executive summary and metadata cards."""
    fig = plt.figure(figsize=(PAGE_WIDTH_INCHES, PAGE_HEIGHT_INCHES), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Background gradient / decorative bands
    ax.fill_between([0, 1], 0.88, 1.0, color="#1A365D", transform=ax.transAxes)
    ax.fill_between([0, 1], 0.865, 0.88, color="#2B6CB0", transform=ax.transAxes)
    ax.fill_between([0, 1], 0.0, 0.05, color="#0F172A", transform=ax.transAxes)

    # Document Header
    ax.text(
        0.06, 0.94,
        "ENERGYUQ EXPERIMENTAL REPORT",
        fontsize=13,
        fontweight="bold",
        color="#90CDF4",
        transform=ax.transAxes,
        va="center",
    )
    ax.text(
        0.94, 0.94,
        date_str,
        fontsize=11,
        color="#E2E8F0",
        transform=ax.transAxes,
        ha="right",
        va="center",
    )

    # Main Title
    ax.text(
        0.06, 0.78,
        "Energy & Uncertainty Quantification Analysis",
        fontsize=26,
        fontweight="bold",
        color="#1E293B",
        transform=ax.transAxes,
    )
    ax.text(
        0.06, 0.725,
        "Comprehensive Multi-Run Comparative Benchmark Report & Single-Run Diagnostics",
        fontsize=14,
        color="#475569",
        transform=ax.transAxes,
    )

    # Divider line
    ax.plot([0.06, 0.94], [0.69, 0.69], color="#CBD5E1", lw=1.5, transform=ax.transAxes)

    # Summary Statistics Cards
    n_benchmarks = len(runs.benchmarks)
    n_machines = len(runs.machines)
    n_runs = len(runs)
    total_samples = sum(r.sample_count for r in runs)

    cards = [
        ("Total Runs", f"{n_runs}", "#2B6CB0"),
        ("Benchmarks", f"{n_benchmarks}", "#2C7A7B"),
        ("Architectures", f"{n_machines}", "#C05621"),
        ("Evaluated Samples", f"{total_samples:,}", "#6B46C1"),
        ("Total Report Pages", f"{total_pages}", "#1A365D"),
    ]

    card_w = 0.16
    card_spacing = (0.88 - (len(cards) * card_w)) / (len(cards) - 1)
    card_y = 0.53
    card_h = 0.12

    for idx, (label, val, accent) in enumerate(cards):
        cx = 0.06 + idx * (card_w + card_spacing)
        # Card background box
        rect = plt.Rectangle(
            (cx, card_y), card_w, card_h,
            facecolor="#F8FAFC", edgecolor="#E2E8F0", lw=1.2,
            transform=ax.transAxes, zorder=1,
        )
        ax.add_patch(rect)
        # Top color stripe
        stripe = plt.Rectangle(
            (cx, card_y + card_h - 0.015), card_w, 0.015,
            facecolor=accent, edgecolor="none",
            transform=ax.transAxes, zorder=2,
        )
        ax.add_patch(stripe)
        # Card text
        ax.text(
            cx + card_w / 2.0, card_y + 0.055, val,
            fontsize=20, fontweight="bold", color="#1E293B",
            ha="center", va="center", transform=ax.transAxes, zorder=3,
        )
        ax.text(
            cx + card_w / 2.0, card_y + 0.022, label,
            fontsize=9.5, fontweight="semibold", color="#64748B",
            ha="center", va="center", transform=ax.transAxes, zorder=3,
        )

    # Metadata & Scope Overview Box
    meta_box = plt.Rectangle(
        (0.06, 0.10), 0.88, 0.38,
        facecolor="#FFFFFF", edgecolor="#E2E8F0", lw=1.2,
        transform=ax.transAxes, zorder=1,
    )
    ax.add_patch(meta_box)

    ax.text(
        0.09, 0.44, "Report Scope & Parameters",
        fontsize=13, fontweight="bold", color="#1E293B",
        transform=ax.transAxes,
    )

    bench_str = ", ".join(runs.benchmarks)
    mach_str = ", ".join(runs.machines)
    qoi_label = _format_qoi_name(qoi)

    info_items = [
        ("Evaluated Benchmarks:", bench_str),
        ("Target Machines:", mach_str),
        ("Primary Quantity of Interest:", f"{qoi} ({qoi_label})"),
        ("Secondary Quantity of Interest:", "time (Execution Time in seconds)"),
        ("Multi-Run Diagnostics:", "Dashboard, Sobol Indices (Bar & Heatmap), Pareto Frontiers, Convergence"),
        ("Single-Run Diagnostics:", "2D Grids, Sorted Evaluations, Projections, Boxplots, Treemaps, UQ Tables"),
    ]

    y_pos = 0.395
    for title_lbl, desc_lbl in info_items:
        ax.text(0.09, y_pos, title_lbl, fontsize=10, fontweight="bold", color="#334155", transform=ax.transAxes)
        ax.text(0.35, y_pos, desc_lbl, fontsize=10, color="#475569", transform=ax.transAxes)
        y_pos -= 0.045

    # Footer note
    ax.text(
        0.50, 0.025,
        "EnergyUQ Multi-Run Automated Compilation Report  •  Includes interactive navigation index and bookmarks",
        fontsize=9, color="#94A3B8", ha="center", va="center",
        transform=ax.transAxes,
    )

    return fig


def _create_section_divider_page(
    benchmark_name: str,
    bench_runs: RunCollection,
    qoi: str,
    start_page: int,
) -> Figure:
    """Create a prominent section divider page for a benchmark."""
    fig = plt.figure(figsize=(PAGE_WIDTH_INCHES, PAGE_HEIGHT_INCHES), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Decorative header
    ax.fill_between([0, 1], 0.85, 1.0, color="#1E293B", transform=ax.transAxes)
    ax.fill_between([0, 1], 0.835, 0.85, color="#3B82F6", transform=ax.transAxes)

    # Section tag
    ax.text(
        0.06, 0.94,
        f"SECTION: BENCHMARK EVALUATION  •  STARTING AT PAGE {start_page}",
        fontsize=11, fontweight="bold", color="#93C5FD",
        transform=ax.transAxes, va="center",
    )
    # Benchmark Name
    ax.text(
        0.06, 0.89,
        f"Benchmark: {benchmark_name}",
        fontsize=26, fontweight="bold", color="#FFFFFF",
        transform=ax.transAxes, va="center",
    )

    # Benchmark Description / Overview
    ax.text(
        0.06, 0.77,
        f"Multi-run comparative analysis and individual architecture diagnostic reports for {benchmark_name}.",
        fontsize=13, color="#334155", transform=ax.transAxes,
    )
    ax.text(
        0.06, 0.73,
        f"Total architectures analyzed: {len(bench_runs)} ({', '.join(bench_runs.machines)}).",
        fontsize=11, color="#64748B", transform=ax.transAxes,
    )

    # Machine overview table
    summary_df = bench_runs.summary_table()
    table_y = 0.68
    rect_tbl = plt.Rectangle(
        (0.06, 0.20), 0.88, 0.48,
        facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.2,
        transform=ax.transAxes, zorder=1,
    )
    ax.add_patch(rect_tbl)

    ax.text(
        0.08, 0.64,
        f"{benchmark_name} Run Summary across Architectures",
        fontsize=13, fontweight="bold", color="#1E293B",
        transform=ax.transAxes, zorder=2,
    )

    # Table columns
    col_headers = ["Machine", "Samples", "Iterations", "Min Energy (J)", "Mean Energy (J)", "Min Time (s)", "Best Threads", "Best Clock"]
    col_x = [0.08, 0.20, 0.28, 0.38, 0.50, 0.62, 0.74, 0.84]

    # Draw header row
    ax.fill_between([0.06, 0.94], 0.575, 0.615, color="#F1F5F9", transform=ax.transAxes, zorder=2)
    for ch, cx in zip(col_headers, col_x):
        ax.text(cx, 0.595, ch, fontsize=9.5, fontweight="bold", color="#475569", va="center", transform=ax.transAxes, zorder=3)

    row_y = 0.54
    for _, r_data in summary_df.iterrows():
        m_name = str(r_data.get("machine", "N/A"))
        samples = str(r_data.get("sample_count", "N/A"))
        iters = str(r_data.get("iterations", "N/A"))

        min_e = f"{r_data['min_energy_j']:.3f}" if "min_energy_j" in r_data and pd.notna(r_data["min_energy_j"]) else "N/A"
        mean_e = f"{r_data['mean_energy_j']:.3f}" if "mean_energy_j" in r_data and pd.notna(r_data["mean_energy_j"]) else "N/A"
        min_t = f"{r_data['min_time_s']:.3f}" if "min_time_s" in r_data and pd.notna(r_data["min_time_s"]) else "N/A"
        best_th = str(r_data.get("best_energy_N_THREADS", "N/A"))
        best_clk = str(r_data.get("best_energy_CLK", "N/A"))

        row_vals = [m_name, samples, iters, min_e, mean_e, min_t, best_th, best_clk]
        for cv, cx in zip(row_vals, col_x):
            ax.text(cx, row_y, cv, fontsize=9.5, color="#1E293B", va="center", transform=ax.transAxes, zorder=3)

        # Subtle separator line
        ax.plot([0.06, 0.94], [row_y - 0.02, row_y - 0.02], color="#F1F5F9", lw=1, transform=ax.transAxes, zorder=2)
        row_y -= 0.05

    # Section Content Roadmap Card
    ax.fill_between([0.06, 0.94], 0.08, 0.17, color="#EFF6FF", transform=ax.transAxes, zorder=1)
    rect_bot = plt.Rectangle(
        (0.06, 0.08), 0.88, 0.09,
        facecolor="none", edgecolor="#BFDBFE", lw=1.2,
        transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(rect_bot)
    ax.text(
        0.08, 0.14,
        "Section Layout:  1. Benchmark Multi-Run Comparative Plots (Dashboard, Sobols, Pareto, Convergence)",
        fontsize=9.5, fontweight="semibold", color="#1E40AF",
        transform=ax.transAxes, zorder=3,
    )
    ax.text(
        0.08, 0.105,
        "                 2. Individual Machine Diagnostic Reports (2D Grids, Projections, Boxplots, Treemaps, UQ Tables)",
        fontsize=9.5, color="#1E40AF",
        transform=ax.transAxes, zorder=3,
    )

    return fig


def _create_toc_pages(
    toc_data: list[dict[str, Any]],
    total_pages: int,
) -> list[tuple[Figure, list[dict[str, Any]]]]:
    """
    Generate visual Table of Contents / Index pages with clickable link areas.
    Returns list of (figure, list_of_link_rects).
    """
    ENTRIES_PER_PAGE = 24
    pages_output = []

    # Chunk entries into pages
    chunks = [toc_data[i:i + ENTRIES_PER_PAGE] for i in range(0, len(toc_data), ENTRIES_PER_PAGE)]
    if not chunks:
        chunks = [[]]

    total_toc_pages = len(chunks)

    for toc_idx, chunk in enumerate(chunks):
        fig = plt.figure(figsize=(PAGE_WIDTH_INCHES, PAGE_HEIGHT_INCHES), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        # Top banner
        ax.fill_between([0, 1], 0.90, 1.0, color="#1E293B", transform=ax.transAxes)
        ax.fill_between([0, 1], 0.885, 0.90, color="#3B82F6", transform=ax.transAxes)

        ax.text(
            0.06, 0.955,
            "REPORT INDEX & NAVIGATION DIRECTORY",
            fontsize=11, fontweight="bold", color="#93C5FD",
            transform=ax.transAxes, va="center",
        )
        ax.text(
            0.06, 0.925,
            f"Table of Contents (Part {toc_idx + 1} of {total_toc_pages})",
            fontsize=18, fontweight="bold", color="#FFFFFF",
            transform=ax.transAxes, va="center",
        )
        ax.text(
            0.94, 0.94,
            f"Click any row to jump directly to page",
            fontsize=10, color="#CBD5E1", ha="right", va="center",
            transform=ax.transAxes,
        )

        # Header bar for table
        ax.fill_between([0.06, 0.94], 0.835, 0.87, color="#F1F5F9", transform=ax.transAxes)
        ax.text(0.08, 0.852, "Section / Benchmark", fontsize=9.5, fontweight="bold", color="#475569", va="center", transform=ax.transAxes)
        ax.text(0.30, 0.852, "Diagnostic / Plot Category", fontsize=9.5, fontweight="bold", color="#475569", va="center", transform=ax.transAxes)
        ax.text(0.72, 0.852, "Architecture", fontsize=9.5, fontweight="bold", color="#475569", va="center", transform=ax.transAxes)
        ax.text(0.92, 0.852, "Page", fontsize=9.5, fontweight="bold", color="#475569", ha="right", va="center", transform=ax.transAxes)

        page_links = []
        y_pos = 0.805
        row_height = 0.030

        for item in chunk:
            item_type = item["type"]
            bench = item["benchmark"]
            title = item["title"]
            machine = item.get("machine", "All")
            p_num = item["page_num"]

            # Visual styling based on item type
            if item_type == "section_header":
                bg_col = "#EFF6FF"
                ax.fill_between([0.06, 0.94], y_pos - row_height * 0.4, y_pos + row_height * 0.6, color=bg_col, transform=ax.transAxes)
                ax.text(0.08, y_pos, f"SECTION: {bench}", fontsize=10, fontweight="bold", color="#1D4ED8", va="center", transform=ax.transAxes)
                ax.text(0.30, y_pos, title, fontsize=9.5, fontweight="semibold", color="#1E40AF", va="center", transform=ax.transAxes)
                ax.text(0.72, y_pos, machine, fontsize=9.5, color="#1E40AF", va="center", transform=ax.transAxes)
                ax.text(0.92, y_pos, f"p. {p_num}", fontsize=10, fontweight="bold", color="#1D4ED8", ha="right", va="center", transform=ax.transAxes)
            elif item_type == "multi_plot":
                ax.text(0.08, y_pos, f"{bench}", fontsize=9, color="#475569", va="center", transform=ax.transAxes)
                ax.text(0.30, y_pos, f"• {title}", fontsize=9, fontweight="medium", color="#0F172A", va="center", transform=ax.transAxes)
                ax.text(0.72, y_pos, machine, fontsize=9, color="#64748B", va="center", transform=ax.transAxes)
                ax.text(0.92, y_pos, f"{p_num}", fontsize=9, color="#2563EB", ha="right", va="center", transform=ax.transAxes)
            else:  # individual run summary
                ax.text(0.08, y_pos, f"{bench}", fontsize=9, color="#64748B", va="center", transform=ax.transAxes)
                ax.text(0.30, y_pos, f"  └ {title}", fontsize=9, color="#334155", va="center", transform=ax.transAxes)
                ax.text(0.72, y_pos, machine, fontsize=9, fontweight="medium", color="#1E293B", va="center", transform=ax.transAxes)
                ax.text(0.92, y_pos, f"{p_num}", fontsize=9, color="#2563EB", ha="right", va="center", transform=ax.transAxes)

            # Draw subtle divider line
            ax.plot([0.06, 0.94], [y_pos - row_height * 0.45, y_pos - row_height * 0.45], color="#E2E8F0", lw=0.6, transform=ax.transAxes)

            # Record link coordinates in PDF points (72 pt per inch)
            # rect: [x1, y1, x2, y2] where (0,0) is bottom-left
            rect_pt = (
                0.06 * PAGE_WIDTH_PT,
                (y_pos - row_height * 0.45) * PAGE_HEIGHT_PT,
                0.94 * PAGE_WIDTH_PT,
                (y_pos + row_height * 0.55) * PAGE_HEIGHT_PT,
            )
            page_links.append({
                "rect": rect_pt,
                "target_page": p_num - 1,  # 0-indexed in pypdf
            })

            y_pos -= row_height

        # Footer
        ax.text(
            0.50, 0.025,
            f"Index Page {toc_idx + 1} of {total_toc_pages}  •  Report Total: {total_pages} Pages",
            fontsize=8.5, color="#94A3B8", ha="center", va="center", transform=ax.transAxes,
        )

        pages_output.append((fig, page_links))

    return pages_output


def _create_footer_stamp(page_num: int, total_pages: int, subtitle: str) -> PdfReader:
    """Create a transparent footer stamp page for page numbering."""
    fig = plt.figure(figsize=(PAGE_WIDTH_INCHES, PAGE_HEIGHT_INCHES))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Footer rule
    ax.plot([0.05, 0.95], [0.035, 0.035], color="#CBD5E1", lw=0.8, transform=ax.transAxes)
    ax.text(
        0.05, 0.018,
        subtitle,
        fontsize=8, color="#64748B", va="center", transform=ax.transAxes,
    )
    ax.text(
        0.95, 0.018,
        f"Page {page_num} of {total_pages}",
        fontsize=8, fontweight="bold", color="#334155", ha="right", va="center", transform=ax.transAxes,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return PdfReader(buf)


def compile_plots_pdf(
    runs: Union[RunCollection, Sequence[RunData], str, Path],
    output_path: Union[str, Path] = "reports/energyuq_compiled_report.pdf",
    qoi: str = "energy_uj",
    qois: Sequence[str] = ("energy_uj", "time"),
    benchmarks: Union[str, Sequence[str], None] = None,
    machines: Union[str, Sequence[str], None] = None,
    include_global: bool = False,
    sobol_modes: Sequence[str] = ("grouped_bar", "heatmap"),
    dpi: int = 150,
    quiet: bool = False,
) -> Path:
    """
    Compile all multi-run and individual run plots into a master PDF report.

    Parameters
    ----------
    runs : RunCollection, list of RunData, or path to run directory.
    output_path : Destination file path for compiled PDF.
    qoi : Primary QoI for multi-run evaluations (e.g. 'energy_uj').
    qois : List of QoIs for individual evaluations (e.g. ['energy_uj', 'time']).
    benchmarks : Benchmark filter (comma-separated or list).
    machines : Machine filter (comma-separated or list).
    include_global : Include a global cross-benchmark section at the start if >1 benchmark.
    sobol_modes : Sobol visualization modes ('grouped_bar', 'heatmap', 'stacked_bar').
    dpi : Resolution for raster components in figures.
    quiet : Suppress progress print statements.

    Returns
    -------
    Path to generated PDF file.
    """
    t_start = time.time()
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Resolve RunCollection
    if isinstance(runs, (str, Path)):
        if not quiet:
            print(f"Discovering runs from '{runs}'...")
        collection = RunCollection.discover(base_dir=runs, benchmarks=benchmarks, machines=machines)
    elif isinstance(runs, RunCollection):
        collection = runs.filter(benchmarks=benchmarks, machines=machines) if (benchmarks or machines) else runs
    else:
        collection = RunCollection(runs).filter(benchmarks=benchmarks, machines=machines)

    if len(collection) == 0:
        raise ValueError("No matching runs found to compile into PDF.")

    active_benchmarks = collection.benchmarks
    active_machines = collection.machines

    if not quiet:
        print(f"\n==================================================")
        print(f"  EnergyUQ Master PDF Report Compiler")
        print(f"==================================================")
        print(f"Loaded {len(collection)} runs across {len(active_benchmarks)} benchmarks & {len(active_machines)} machines.")
        print(f"Benchmarks: {', '.join(active_benchmarks)}")
        print(f"Machines:   {', '.join(active_machines)}")
        print(f"Output PDF: {out_file}")
        print(f"Primary QoI: {qoi} | QoIs: {list(qois)}")
        print(f"==================================================\n")

    # We will buffer content pages into an in-memory PDF, recording exact metadata
    content_buf = io.BytesIO()
    content_entries = []  # List of page metadata dicts

    with PdfPages(content_buf) as pdf:
        # ----------------------------------------------------
        # Optional: Global Cross-Benchmark Summary
        # ----------------------------------------------------
        if include_global and len(active_benchmarks) > 1:
            if not quiet:
                print("Generating Global Cross-Benchmark Section...")

            # Global Dashboard
            try:
                fig_gd = multi_plot.plot_multi_dashboard(collection, qoi=qoi, show_title=True)
                fig_gd.suptitle(f"Global Cross-Benchmark Comparative Dashboard ({_format_qoi_name(qoi)})", fontsize=14, fontweight="bold")
                pdf.savefig(fig_gd, dpi=dpi, bbox_inches="tight")
                plt.close(fig_gd)
                content_entries.append({
                    "type": "multi_plot",
                    "benchmark": "Global",
                    "machine": "All",
                    "title": "Global Multi-Run Dashboard",
                    "outline_level": "global",
                })
            except Exception as e:
                if not quiet:
                    print(f"  [Warning] Global dashboard skipped: {e}")

            # Global Pareto
            try:
                fig_gp = multi_plot.plot_multi_energy_time_pareto(collection, qoi_y=qoi, show_title=True)
                fig_gp.suptitle(f"Global Energy vs Time Pareto Frontiers across All Benchmarks", fontsize=13, fontweight="bold")
                pdf.savefig(fig_gp, dpi=dpi, bbox_inches="tight")
                plt.close(fig_gp)
                content_entries.append({
                    "type": "multi_plot",
                    "benchmark": "Global",
                    "machine": "All",
                    "title": "Global Pareto Frontiers",
                    "outline_level": "global",
                })
            except Exception as e:
                if not quiet:
                    print(f"  [Warning] Global Pareto skipped: {e}")

            # Global Convergence
            try:
                fig_gc = multi_plot.plot_multi_convergence(collection, facet_by="benchmark", show_title=True)
                fig_gc.suptitle("Global Convergence History Faceted by Benchmark", fontsize=13, fontweight="bold")
                pdf.savefig(fig_gc, dpi=dpi, bbox_inches="tight")
                plt.close(fig_gc)
                content_entries.append({
                    "type": "multi_plot",
                    "benchmark": "Global",
                    "machine": "All",
                    "title": "Global Convergence History",
                    "outline_level": "global",
                })
            except Exception as e:
                if not quiet:
                    print(f"  [Warning] Global convergence skipped: {e}")

        # ----------------------------------------------------
        # Main Body: Grouped strictly by Benchmark
        # ----------------------------------------------------
        for b_idx, bench in enumerate(active_benchmarks, start=1):
            bench_runs = collection.filter(benchmarks=[bench])
            if not quiet:
                print(f"[{b_idx}/{len(active_benchmarks)}] Compiling Benchmark: {bench} ({len(bench_runs)} runs)...")

            # 1. Section Divider Page (placeholder for now, will re-render with true page number or render now)
            fig_div = _create_section_divider_page(
                benchmark_name=bench,
                bench_runs=bench_runs,
                qoi=qoi,
                start_page=len(content_entries) + 1,  # will be offset later
            )
            pdf.savefig(fig_div, dpi=dpi, bbox_inches="tight")
            plt.close(fig_div)
            content_entries.append({
                "type": "section_header",
                "benchmark": bench,
                "machine": f"{len(bench_runs)} Machines",
                "title": f"Benchmark Overview & Machine Comparison",
                "outline_level": "benchmark",
            })

            # 2. Benchmark Multi-Run Plots
            # 2.1 Dashboard
            try:
                fig_dash = multi_plot.plot_multi_dashboard(bench_runs, qoi=qoi, show_title=True)
                fig_dash.suptitle(f"Benchmark: {bench} — Multi-Run Comparative Dashboard ({_format_qoi_name(qoi)})", fontsize=14, fontweight="bold")
                pdf.savefig(fig_dash, dpi=dpi, bbox_inches="tight")
                plt.close(fig_dash)
                content_entries.append({
                    "type": "multi_plot",
                    "benchmark": bench,
                    "machine": "All Machines",
                    "title": "Multi-Run Comparative Dashboard",
                    "outline_level": "multi_plot",
                })
            except Exception as e:
                if not quiet:
                    print(f"  [{bench}] Multi-dashboard skipped: {e}")

            # 2.2 Sobols
            for smode in sobol_modes:
                try:
                    mode_title = "Grouped" if smode == "grouped_bar" else smode.replace("_", " ").title()
                    fig_sob = multi_plot.plot_multi_sobols(bench_runs, qoi=qoi, mode=smode, show_title=True)
                    fig_sob.suptitle(f"Benchmark: {bench} — Sobol Sensitivity Comparison ({mode_title})", fontsize=13, fontweight="bold")
                    pdf.savefig(fig_sob, dpi=dpi, bbox_inches="tight")
                    plt.close(fig_sob)
                    content_entries.append({
                        "type": "multi_plot",
                        "benchmark": bench,
                        "machine": "All Machines",
                        "title": f"Sobol Sensitivities ({mode_title})",
                        "outline_level": "multi_plot",
                    })
                except Exception as e:
                    if not quiet:
                        print(f"  [{bench}] Sobols ({smode}) skipped: {e}")

            # 2.3 Pareto
            try:
                fig_par = multi_plot.plot_multi_energy_time_pareto(bench_runs, qoi_x="time", qoi_y=qoi, show_title=True)
                fig_par.suptitle(f"Benchmark: {bench} — Energy vs Time Pareto Frontiers Across Machines", fontsize=13, fontweight="bold")
                pdf.savefig(fig_par, dpi=dpi, bbox_inches="tight")
                plt.close(fig_par)
                content_entries.append({
                    "type": "multi_plot",
                    "benchmark": bench,
                    "machine": "All Machines",
                    "title": "Energy vs Time Pareto Frontier",
                    "outline_level": "multi_plot",
                })
            except Exception as e:
                if not quiet:
                    print(f"  [{bench}] Pareto skipped: {e}")

            # 2.4 Convergence
            try:
                fig_conv = multi_plot.plot_multi_convergence(bench_runs, metric="adaptation_error", show_title=True)
                fig_conv.suptitle(f"Benchmark: {bench} — Multi-Run Convergence History (Adaptation Error)", fontsize=13, fontweight="bold")
                pdf.savefig(fig_conv, dpi=dpi, bbox_inches="tight")
                plt.close(fig_conv)
                content_entries.append({
                    "type": "multi_plot",
                    "benchmark": bench,
                    "machine": "All Machines",
                    "title": "Convergence History (Adaptation Error)",
                    "outline_level": "multi_plot",
                })
            except Exception as e:
                if not quiet:
                    print(f"  [{bench}] Convergence skipped: {e}")

            # 3. Individual Run Analyses (analyze_each plots)
            for r in bench_runs:
                if not quiet:
                    print(f"    • Generating individual diagnostic plots for {r.tag}...")
                rep = r.analyze_individually(
                    output_dir=None,
                    qois=qois,
                    save_plots=False,
                    quiet=True,
                    show=False,
                )
                figures = rep.get("figures", {})

                first_plot_for_run = True
                for fig_key, fig_obj in figures.items():
                    if fig_obj is None:
                        continue
                    disp_title = _format_individual_plot_title(fig_key)
                    # Apply descriptive suptitle
                    fig_obj.suptitle(f"Benchmark: {bench}  |  Machine: {r.machine_name}\n{disp_title}", fontsize=12, fontweight="bold")
                    pdf.savefig(fig_obj, dpi=dpi, bbox_inches="tight")
                    plt.close(fig_obj)

                    content_entries.append({
                        "type": "individual_plot",
                        "benchmark": bench,
                        "machine": r.machine_name,
                        "tag": r.tag,
                        "title": disp_title,
                        "plot_key": fig_key,
                        "outline_level": "individual_plot",
                        "is_run_start": first_plot_for_run,
                    })
                    first_plot_for_run = False

    # Read rendered content pages
    content_buf.seek(0)
    content_reader = PdfReader(content_buf)
    n_content_pages = len(content_reader.pages)

    if not quiet:
        print(f"\nGenerated {n_content_pages} content pages. Preparing Table of Contents and Cover...")

    # Calculate Table of Contents entries for visual TOC:
    # We include: Section headers, Multi-plots, and Individual run starting entries
    toc_display_items = []
    for idx, entry in enumerate(content_entries):
        if entry["type"] == "section_header":
            toc_display_items.append(dict(entry, content_idx=idx))
        elif entry["type"] == "multi_plot":
            toc_display_items.append(dict(entry, content_idx=idx))
        elif entry.get("is_run_start", False):
            run_tag = entry.get("tag", f"{entry['benchmark']}_{entry['machine']}")
            toc_display_items.append({
                "type": "individual_summary",
                "benchmark": entry["benchmark"],
                "machine": entry["machine"],
                "title": f"Single-Run Diagnostics ({run_tag})",
                "content_idx": idx,
            })

    # Estimate number of TOC pages needed
    ENTRIES_PER_TOC_PAGE = 24
    n_toc_pages = max(1, (len(toc_display_items) + ENTRIES_PER_TOC_PAGE - 1) // ENTRIES_PER_TOC_PAGE)
    cover_pages = 1
    offset = cover_pages + n_toc_pages
    total_doc_pages = offset + n_content_pages

    # Assign final 1-indexed document page numbers to TOC display items
    for item in toc_display_items:
        item["page_num"] = item["content_idx"] + offset + 1

    # Also assign page numbers to all content entries
    for idx, entry in enumerate(content_entries):
        entry["page_num"] = idx + offset + 1

    # Generate Cover Page
    date_str = datetime.now().strftime("%B %d, %Y")
    fig_cover = _create_cover_page(collection, total_doc_pages, qoi, date_str)
    cover_buf = io.BytesIO()
    fig_cover.savefig(cover_buf, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig_cover)
    cover_buf.seek(0)
    cover_reader = PdfReader(cover_buf)

    # Generate TOC Pages
    toc_pages_data = _create_toc_pages(toc_display_items, total_doc_pages)
    toc_readers_and_links = []
    for fig_t, links_t in toc_pages_data:
        t_buf = io.BytesIO()
        fig_t.savefig(t_buf, format="pdf", dpi=dpi, bbox_inches="tight")
        plt.close(fig_t)
        t_buf.seek(0)
        toc_readers_and_links.append((PdfReader(t_buf), links_t))

    # ----------------------------------------------------
    # Assemble Final Document with Bookmarks & Link Annotations
    # ----------------------------------------------------
    if not quiet:
        print("Assembling final PDF document with outline hierarchy and link annotations...")

    writer = PdfWriter()

    # 1. Add Cover Page
    writer.add_page(cover_reader.pages[0])

    # 2. Add TOC Pages & Attach Link Annotations
    for toc_idx, (r_toc, links_t) in enumerate(toc_readers_and_links):
        toc_page = writer.add_page(r_toc.pages[0])
        current_page_idx = 1 + toc_idx
        for lnk in links_t:
            target_idx = lnk["target_page"]
            annot = Link(rect=lnk["rect"], target_page_index=target_idx)
            writer.add_annotation(page_number=current_page_idx, annotation=annot)

    # 3. Add Content Pages
    for p in content_reader.pages:
        writer.add_page(p)

    # 4. Stamp Running Footers
    if not quiet:
        print("Applying running page number stamps (Page X of Y)...")

    footer_cache: dict[int, PdfReader] = {}
    for p_idx in range(len(writer.pages)):
        p_num = p_idx + 1
        sub_text = "EnergyUQ Multi-Run Automated Compilation Report"
        if p_idx not in footer_cache:
            footer_cache[p_idx] = _create_footer_stamp(p_num, total_doc_pages, sub_text)
        stamp_page = footer_cache[p_idx].pages[0]
        writer.pages[p_idx].merge_page(stamp_page)

    # 5. Build Hierarchical Bookmarks Outline
    if not quiet:
        print("Building interactive PDF navigation bookmarks...")

    # Root bookmarks for Cover and TOC
    writer.add_outline_item("Title & Document Overview", page_number=0)
    writer.add_outline_item("Table of Contents / Index", page_number=1)

    bench_parents: dict[str, Any] = {}
    multi_parents: dict[str, Any] = {}
    indiv_run_parents: dict[str, Any] = {}

    for entry in content_entries:
        level = entry["outline_level"]
        bench = entry["benchmark"]
        page_idx = entry["page_num"] - 1  # 0-indexed for pypdf

        if level == "global":
            if "Global" not in bench_parents:
                bench_parents["Global"] = writer.add_outline_item("🌐 Global Cross-Benchmark Comparison", page_number=page_idx)
            writer.add_outline_item(entry["title"], page_number=page_idx, parent=bench_parents["Global"])

        elif level == "benchmark":
            bench_parent = writer.add_outline_item(f"📁 Benchmark: {bench}", page_number=page_idx)
            bench_parents[bench] = bench_parent
            writer.add_outline_item(f"Overview & Machine Summary ({bench})", page_number=page_idx, parent=bench_parent)

        elif level == "multi_plot":
            b_parent = bench_parents.get(bench)
            if bench not in multi_parents:
                multi_parents[bench] = writer.add_outline_item("📊 Multi-Run Comparative Plots", page_number=page_idx, parent=b_parent)
            writer.add_outline_item(entry["title"], page_number=page_idx, parent=multi_parents[bench])

        elif level == "individual_plot":
            b_parent = bench_parents.get(bench)
            run_key = f"{bench}_{entry['machine']}"
            if run_key not in indiv_run_parents:
                # Top individual run node for this machine
                run_title = f"💻 Run: {bench} ({entry['machine']})"
                indiv_run_parents[run_key] = writer.add_outline_item(run_title, page_number=page_idx, parent=b_parent)

            writer.add_outline_item(entry["title"], page_number=page_idx, parent=indiv_run_parents[run_key])

    # Write output PDF
    with open(out_file, "wb") as f_out:
        writer.write(f_out)

    elapsed = time.time() - t_start
    file_size_mb = out_file.stat().st_size / (1024 * 1024)

    if not quiet:
        print(f"\n==================================================")
        print(f"  ✓ PDF Compilation Finished Successfully!")
        print(f"==================================================")
        print(f"Output File:     {out_file}")
        print(f"Total Pages:     {total_doc_pages}")
        print(f"File Size:       {file_size_mb:.2f} MB")
        print(f"Elapsed Time:    {elapsed:.1f} s")
        print(f"Cover Pages:     {cover_pages}")
        print(f"Index Pages:     {n_toc_pages}")
        print(f"Content Pages:   {n_content_pages}")
        print(f"==================================================\n")

    return out_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile multi-run and individual run plots from RunCollection into a single PDF report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs/NTHREADS_CLK",
        help="Base directory containing run folders to discover.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*/*/*",
        help="Glob pattern relative to --runs-dir to find run directories.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/energyuq_compiled_report.pdf",
        help="Output path for the compiled PDF document.",
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
        help="Comma-separated list of machines to include (e.g. 'cei,hype').",
    )
    parser.add_argument(
        "--qoi",
        type=str,
        default="energy_uj",
        help="Primary QoI for multi-run evaluations (e.g. 'energy_uj', 'time', 'power_w').",
    )
    parser.add_argument(
        "--qois",
        type=str,
        default="energy_uj,time",
        help="Comma-separated list of QoIs for single-run evaluations.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI resolution for rendered figure components.",
    )
    parser.add_argument(
        "--include-global",
        action="store_true",
        help="Include a global cross-benchmark comparison section at the start if multiple benchmarks are present.",
    )
    parser.add_argument(
        "--sobol-modes",
        type=str,
        default="grouped_bar,heatmap",
        help="Comma-separated list of modes for plot_multi_sobols ('grouped_bar', 'heatmap', 'stacked_bar').",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",")] if args.benchmarks else None
    machines = [m.strip() for m in args.machines.split(",")] if args.machines else None
    qois = [q.strip() for q in args.qois.split(",")] if args.qois else ["energy_uj", "time"]
    sobol_modes = [s.strip() for s in args.sobol_modes.split(",")] if args.sobol_modes else ["grouped_bar", "heatmap"]

    compile_plots_pdf(
        runs=args.runs_dir,
        output_path=args.output,
        qoi=args.qoi,
        qois=qois,
        benchmarks=benchmarks,
        machines=machines,
        include_global=args.include_global,
        sobol_modes=sobol_modes,
        dpi=args.dpi,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
