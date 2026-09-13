#!/usr/bin/env python3
"""
Compile multi-run and individual run plots from a RunCollection into an
interactive HTML report with a companion static JSON metadata file.

Uses the persistent template assets in the 'view/' folder (index.html, style.css, app.js).
Generates multi-plots for all machines combined and for each individual machine,
supporting toggle/swap navigation in the interface.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Sequence, Union

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
import pandas as pd
import numpy as np

from src.util.multi_run import RunCollection, RunData, discover_runs, load_run, _to_json_serializable
from src.util import multi_plot


def _format_qoi_name(qoi: str) -> str:
    """Format QoI key to human-readable label."""
    mapping = {
        "energy_uj": "Energy (μJ)",
        "energy_j": "Energy (J)",
        "energy_scaled": "Scaled Energy",
        "time": "Execution Time (s)",
        "power_w": "Power (W)",
        "edp_j_s": "Energy-Delay Product (J·s)",
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


def _get_plot_category(name: str) -> str:
    """Classify plot key into a functional category."""
    if "sobol" in name:
        return "Sensitivity"
    if "grid_2d" in name or "single_dimension" in name:
        return "Projections"
    if "boxplot" in name or "sorted" in name:
        return "Distributions"
    if "adaptation" in name or "moments" in name or "convergence" in name:
        return "Convergence"
    if "param_effects" in name:
        return "Parameter Effects"
    return "Diagnostics"


def _get_plot_description(name: str) -> str:
    """Return an intuitive description of what the plot visualizes."""
    descriptions = {
        "grid_2d_best_energy_uj": "Interpolated 2D surface projection showing optimal energy levels across two dominant input dimensions.",
        "sorted_evaluations_energy_uj": "All evaluated configuration samples ranked monotonically by measured energy consumption.",
        "single_dimension_projections_energy_uj": "Marginal response curves showing the effect of varying each parameter individually on energy.",
        "boxplot_per_dimension_energy_uj": "Spread and dispersion of energy consumption across distinct values of each control knob.",
        "sobol_indices_energy_uj": "Variance-based first-order Sobol sensitivity decomposition quantifying the relative influence of each parameter on energy.",
        "sobol_treemap_energy_uj": "Hierarchical treemap visualizing variance contribution of control knobs to energy variability.",
        "grid_2d_best_time": "Interpolated 2D surface projection showing optimal execution times across two dominant input dimensions.",
        "sorted_evaluations_time": "All evaluated configuration samples ranked monotonically by measured execution time.",
        "single_dimension_projections_time": "Marginal response curves showing the effect of varying each parameter individually on execution time.",
        "boxplot_per_dimension_time": "Spread and dispersion of execution time across distinct values of each control knob.",
        "sobol_indices_time": "Variance-based first-order Sobol sensitivity decomposition for execution time.",
        "sobol_treemap_time": "Hierarchical treemap visualizing variance contribution of control knobs to execution time variability.",
        "adaptation_error_history": "Convergence of surrogate adaptation surplus error across iterations on a logarithmic scale.",
        "statistical_moments_convergence": "Evolution of surrogate mean and standard deviation estimates as new points are adaptively sampled.",
        "adaptation_histogram": "Frequency distribution of adaptation surplus errors across the parameter domain.",
        "adaptation_table": "Tabular log of candidate evaluation points, surrogate predictions, and error indicators per adaptation step.",
        "dashboard": "4-panel executive summary combining Sobol sensitivities, Pareto trade-offs, convergence histories, and optimal parameter configurations.",
        "sobol_grouped_bar": "Side-by-side comparative bar chart of first-order Sobol sensitivity indices across evaluated machines.",
        "sobol_heatmap": "Color-encoded matrix of parameter sensitivity intensities across evaluated machines.",
        "sobol_stacked_bar": "Stacked representation of relative parameter importance proportions across machines.",
        "pareto": "Multi-objective energy vs. execution time trade-off highlighting non-dominated Pareto frontiers for each machine.",
        "convergence": "Surrogate adaptation error trajectory comparing convergence rates across machines.",
        "param_effects_energy": "Response curves and scatter points showing the effect of parameter variations on Energy.",
        "param_effects_time": "Response curves and scatter points showing the effect of parameter variations on Execution Time.",
        "param_effects_edp": "Response curves and scatter points showing the effect of parameter variations on Energy-Delay Product (EDP).",
    }
    for k, v in descriptions.items():
        if k in name:
            return v
    return "Analytical diagnostic figure generated during campaign evaluation."


def _save_figure(fig: Figure | SubFigure, out_file: Path, dpi: int = 150):
    """Save matplotlib figure and close it cleanly."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(fig, Figure):
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    elif isinstance(fig, SubFigure):
        fig.get_figure().savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close("all")


def _build_multi_plot_defs(
    qoi: str,
    sobol_modes: Sequence[str],
    bench_params: Sequence[str],
) -> list[dict[str, Any]]:
    """Build list of multi-plot definitions and builders for a benchmark."""
    multi_plot_defs: list[dict[str, Any]] = [
        {
            "key": "dashboard",
            "tab_label": "Dashboard",
            "title_template": "Multi-Run Comparative Dashboard",
            "category": "Overview",
            "builder": lambda r_set, bench_name, sc_name: multi_plot.plot_multi_dashboard(r_set, qoi=qoi, show_title=True),
        },
        {
            "key": "pareto",
            "tab_label": "Pareto Frontiers",
            "title_template": "Energy vs Time Pareto Frontiers",
            "category": "Trade-offs",
            "builder": lambda r_set, bench_name, sc_name: multi_plot.plot_multi_energy_time_pareto(r_set, qoi_x="time", qoi_y=qoi, show_title=True),
        },
        {
            "key": "convergence",
            "tab_label": "Convergence",
            "title_template": "Convergence History (Adaptation Error)",
            "category": "Convergence",
            "builder": lambda r_set, bench_name, sc_name: multi_plot.plot_multi_convergence(r_set, metric="adaptation_error", show_title=True),
        },
    ]

    # Sobol modes
    for smode in sobol_modes:
        mode_title = "Grouped Bar" if smode == "grouped_bar" else smode.replace("_", " ").title()
        multi_plot_defs.append({
            "key": f"sobols_{smode}",
            "tab_label": f"Sobols ({mode_title})",
            "title_template": f"Sobol Sensitivities ({mode_title})",
            "category": "Sensitivity",
            "builder": (lambda m_mode: lambda r_set, bench_name, sc_name: multi_plot.plot_multi_sobols(r_set, qoi=qoi, mode=m_mode, show_title=True))(smode),
        })

    # Parameter Effects for Energy, Time, and EDP ((df["energy_uj"] * 1e-6) * df["time"])
    edp_callable = lambda df: (df["energy_uj"] * 1e-6) * df["time"]

    for param in bench_params:
        # Energy
        multi_plot_defs.append({
            "key": f"param_effects_energy_{param}",
            "tab_label": f"Effects: Energy ({param})",
            "title_template": f"Parameter Effects: Energy vs {param}",
            "category": "Parameter Effects",
            "builder": (lambda p: lambda r_set, bench_name, sc_name: multi_plot.plot_multi_parameter_effects(
                r_set, param=p, qoi="energy_uj", qoi_label="Energy (J)", facet_by="benchmark", show_title=True
            ))(param),
        })
        # Time
        multi_plot_defs.append({
            "key": f"param_effects_time_{param}",
            "tab_label": f"Effects: Time ({param})",
            "title_template": f"Parameter Effects: Time vs {param}",
            "category": "Parameter Effects",
            "builder": (lambda p: lambda r_set, bench_name, sc_name: multi_plot.plot_multi_parameter_effects(
                r_set, param=p, qoi="time", qoi_label="Execution Time (s)", facet_by="benchmark", show_title=True
            ))(param),
        })
        # EDP ((df["energy_uj"] * 1e-6) * df["time"])
        multi_plot_defs.append({
            "key": f"param_effects_edp_{param}",
            "tab_label": f"Effects: EDP ({param})",
            "title_template": f"Parameter Effects: EDP vs {param}",
            "category": "Parameter Effects",
            "builder": (lambda p: lambda r_set, bench_name, sc_name: multi_plot.plot_multi_parameter_effects(
                r_set, param=p, qoi=edp_callable, qoi_label="Energy-Delay Product (J·s)", facet_by="benchmark", show_title=True
            ))(param),
        })

    return multi_plot_defs


def _render_scope_multi_plots(
    r_subset: RunCollection,
    bench: str,
    scope_id: str,
    scope_title: str,
    scope_dir: Path,
    out_dir: Path,
    bench_params: Sequence[str],
    qoi: str,
    sobol_modes: Sequence[str],
    fmt: str = "png",
    dpi: int = 150,
    quiet: bool = True,
) -> dict[str, dict[str, str]]:
    """Render all multi-plots for a given scope, save to scope_dir, and return scope metadata."""
    scope_dir.mkdir(parents=True, exist_ok=True)
    defs = _build_multi_plot_defs(qoi=qoi, sobol_modes=sobol_modes, bench_params=bench_params)
    scopes: dict[str, dict[str, str]] = {}

    for p_def in defs:
        p_key = p_def["key"]
        out_fig_file = scope_dir / f"{p_key}.{fmt}"
        disp_title = f"Benchmark: {bench} — {p_def['title_template']} ({scope_title})"

        try:
            fig = p_def["builder"](r_subset, bench, scope_title)
            if getattr(fig, "suptitle", None):
                fig.suptitle(disp_title, fontsize=13, fontweight="bold")
            _save_figure(fig, out_fig_file, dpi=dpi)

            scopes[p_key] = {
                "file": str(out_fig_file.relative_to(out_dir)),
                "title": disp_title,
                "scope_name": scope_title,
            }
        except Exception as e:
            if not quiet:
                print(f"  [{bench} - {scope_id}] {p_key} skipped: {e}")

    return scopes


def _render_individual_run(
    r: RunData,
    bench: str,
    mach_plot_dir: Path,
    out_dir: Path,
    qois: Sequence[str],
    fmt: str = "png",
    dpi: int = 150,
) -> tuple[str, dict[str, Any]]:
    """Analyze a single run, generate its figures, and return (machine_name, run_record)."""
    mach_plot_dir.mkdir(parents=True, exist_ok=True)

    rep = r.analyze_individually(
        output_dir=None,
        qois=qois,
        save_plots=False,
        quiet=True,
        show=False,
    )

    figures = rep.get("figures", {})
    metrics = rep.get("metrics", {})

    run_plots: list[dict[str, Any]] = []

    for fig_key, fig_obj in figures.items():
        if fig_obj is None:
            continue
        disp_title = _format_individual_plot_title(fig_key)
        f_path = mach_plot_dir / f"{fig_key}.{fmt}"
        _save_figure(fig_obj, f_path, dpi=dpi)

        fig_qoi = None
        for q in qois:
            if fig_key.endswith(f"_{q}"):
                fig_qoi = q
                break

        tab_label = disp_title.split("—")[0].strip()
        if "First-Order Sobol" in tab_label:
            tab_label = "Sobol Indices"

        run_plots.append({
            "key": fig_key,
            "title": disp_title,
            "tab_label": tab_label,
            "category": _get_plot_category(fig_key),
            "qoi": fig_qoi,
            "file": str(f_path.relative_to(out_dir)),
            "description": _get_plot_description(fig_key),
        })

    samples_preview = []
    if not r.df.empty:
        preview_df = r.df.head(20).copy()
        samples_preview = _to_json_serializable(preview_df.to_dict(orient="records"))

    run_record: dict[str, Any] = {
        "tag": r.tag,
        "benchmark": bench,
        "machine": r.machine_name,
        "sample_count": r.sample_count,
        "iterations": r.iteration_count,
        "input_params": r.input_params,
        "qois_data": _to_json_serializable(metrics.get("qois", {})),
        "convergence_history": _to_json_serializable(r.get_convergence_history()),
        "plots": run_plots,
        "samples_preview": samples_preview,
    }

    return r.machine_name, run_record


def _worker_task_global(
    run_paths: list[str],
    global_plots_dir_str: str,
    out_dir_str: str,
    qoi: str,
    fmt: str = "png",
    dpi: int = 150,
    quiet: bool = True,
) -> dict[str, Any]:
    """Worker task to generate global cross-benchmark summary plots."""
    global_plots_dir = Path(global_plots_dir_str)
    out_dir = Path(out_dir_str)
    global_plots_dir.mkdir(parents=True, exist_ok=True)

    runs = [load_run(p) for p in run_paths]
    collection = RunCollection(runs)
    g_plots: dict[str, Any] = {}

    # Global Dashboard
    try:
        fig_gd = multi_plot.plot_multi_dashboard(collection, qoi=qoi, show_title=True)
        fig_gd.suptitle(f"Global Cross-Benchmark Comparative Dashboard ({_format_qoi_name(qoi)})", fontsize=14, fontweight="bold")
        file_gd = global_plots_dir / f"multi_dashboard.{fmt}"
        _save_figure(fig_gd, file_gd, dpi=dpi)
        g_plots["dashboard"] = {
            "key": "dashboard",
            "title": "Global Multi-Run Dashboard",
            "tab_label": "Dashboard",
            "file": str(file_gd.relative_to(out_dir)),
            "description": _get_plot_description("dashboard"),
            "scopes": {
                "all": {
                    "file": str(file_gd.relative_to(out_dir)),
                    "title": "Global Multi-Run Dashboard",
                }
            }
        }
    except Exception as e:
        if not quiet:
            print(f"  [Global] Dashboard skipped: {e}")

    # Global Pareto
    try:
        fig_gp = multi_plot.plot_multi_energy_time_pareto(collection, qoi_y=qoi, show_title=True)
        fig_gp.suptitle("Global Energy vs Time Pareto Frontiers across All Benchmarks", fontsize=13, fontweight="bold")
        file_gp = global_plots_dir / f"multi_pareto.{fmt}"
        _save_figure(fig_gp, file_gp, dpi=dpi)
        g_plots["pareto"] = {
            "key": "pareto",
            "title": "Global Pareto Frontiers",
            "tab_label": "Pareto Frontiers",
            "file": str(file_gp.relative_to(out_dir)),
            "description": _get_plot_description("pareto"),
            "scopes": {
                "all": {
                    "file": str(file_gp.relative_to(out_dir)),
                    "title": "Global Pareto Frontiers",
                }
            }
        }
    except Exception as e:
        if not quiet:
            print(f"  [Global] Pareto skipped: {e}")

    # Global Convergence
    try:
        fig_gc = multi_plot.plot_multi_convergence(collection, facet_by="benchmark", show_title=True)
        fig_gc.suptitle("Global Convergence History Faceted by Benchmark", fontsize=13, fontweight="bold")
        file_gc = global_plots_dir / f"multi_convergence.{fmt}"
        _save_figure(fig_gc, file_gc, dpi=dpi)
        g_plots["convergence"] = {
            "key": "convergence",
            "title": "Global Convergence History",
            "tab_label": "Convergence",
            "file": str(file_gc.relative_to(out_dir)),
            "description": _get_plot_description("convergence"),
            "scopes": {
                "all": {
                    "file": str(file_gc.relative_to(out_dir)),
                    "title": "Global Convergence History",
                }
            }
        }
    except Exception as e:
        if not quiet:
            print(f"  [Global] Convergence skipped: {e}")

    return {
        "type": "global",
        "g_plots": g_plots,
    }


def _worker_task_bench_all(
    bench: str,
    run_paths: list[str],
    bench_dir_str: str,
    out_dir_str: str,
    bench_params: list[str],
    qoi: str,
    sobol_modes: list[str],
    fmt: str = "png",
    dpi: int = 150,
    quiet: bool = True,
) -> dict[str, Any]:
    """Worker task to generate scope 'all' multi-plots for a benchmark."""
    bench_dir = Path(bench_dir_str)
    out_dir = Path(out_dir_str)
    scope_dir = bench_dir / "multi" / "all"

    runs = [load_run(p) for p in run_paths]
    collection = RunCollection(runs)

    scopes = _render_scope_multi_plots(
        r_subset=collection,
        bench=bench,
        scope_id="all",
        scope_title="All Machines",
        scope_dir=scope_dir,
        out_dir=out_dir,
        bench_params=bench_params,
        qoi=qoi,
        sobol_modes=sobol_modes,
        fmt=fmt,
        dpi=dpi,
        quiet=quiet,
    )

    return {
        "type": "bench_all",
        "benchmark": bench,
        "scopes": scopes,
    }


def _worker_task_bench_machine(
    bench: str,
    machine: str,
    run_paths: list[str],
    bench_dir_str: str,
    out_dir_str: str,
    bench_params: list[str],
    qoi: str,
    qois: list[str],
    sobol_modes: list[str],
    fmt: str = "png",
    dpi: int = 150,
    quiet: bool = True,
) -> dict[str, Any]:
    """Worker task to generate scope <machine> multi-plots and individual run plots for (bench, machine)."""
    bench_dir = Path(bench_dir_str)
    out_dir = Path(out_dir_str)

    runs = [load_run(p) for p in run_paths]
    collection = RunCollection(runs)

    # Multi-plots for scope <machine>
    scope_dir = bench_dir / "multi" / machine
    scopes = _render_scope_multi_plots(
        r_subset=collection,
        bench=bench,
        scope_id=machine,
        scope_title=f"Machine: {machine}",
        scope_dir=scope_dir,
        out_dir=out_dir,
        bench_params=bench_params,
        qoi=qoi,
        sobol_modes=sobol_modes,
        fmt=fmt,
        dpi=dpi,
        quiet=quiet,
    )

    # Individual run analyses for runs of this machine
    run_records: dict[str, Any] = {}
    for r in runs:
        mach_plot_dir = bench_dir / r.machine_name
        _, rec = _render_individual_run(
            r=r,
            bench=bench,
            mach_plot_dir=mach_plot_dir,
            out_dir=out_dir,
            qois=qois,
            fmt=fmt,
            dpi=dpi,
        )
        run_records[r.machine_name] = rec

    return {
        "type": "bench_machine",
        "benchmark": bench,
        "machine": machine,
        "scopes": scopes,
        "run_records": run_records,
    }


def _worker_task_benchmark(
    bench: str,
    machine_run_paths: dict[str, list[str]],
    bench_dir_str: str,
    out_dir_str: str,
    bench_params: list[str],
    qoi: str,
    qois: list[str],
    sobol_modes: list[str],
    fmt: str = "png",
    dpi: int = 150,
    quiet: bool = True,
) -> dict[str, Any]:
    """Worker task for benchmark-level granularity: computes 'all' scope and all machines sequentially within the worker."""
    all_bench_paths = [p for paths in machine_run_paths.values() for p in paths]
    bench_all_res = _worker_task_bench_all(
        bench=bench,
        run_paths=all_bench_paths,
        bench_dir_str=bench_dir_str,
        out_dir_str=out_dir_str,
        bench_params=bench_params,
        qoi=qoi,
        sobol_modes=sobol_modes,
        fmt=fmt,
        dpi=dpi,
        quiet=quiet,
    )
    machine_results = []
    for machine, paths in machine_run_paths.items():
        m_res = _worker_task_bench_machine(
            bench=bench,
            machine=machine,
            run_paths=paths,
            bench_dir_str=bench_dir_str,
            out_dir_str=out_dir_str,
            bench_params=bench_params,
            qoi=qoi,
            qois=qois,
            sobol_modes=sobol_modes,
            fmt=fmt,
            dpi=dpi,
            quiet=quiet,
        )
        machine_results.append(m_res)

    return {
        "type": "benchmark_combined",
        "benchmark": bench,
        "bench_all_res": bench_all_res,
        "machine_results": machine_results,
    }


def _apply_task_result(metadata: dict[str, Any], res: dict[str, Any]) -> None:
    """Merge a worker task result dictionary into the main report metadata."""
    res_type = res.get("type")
    if res_type == "global":
        metadata["global_section"]["has_global"] = True
        metadata["global_section"]["multi_plots"] = res.get("g_plots", {})

    elif res_type == "bench_all":
        bench = res["benchmark"]
        b_entry = metadata["benchmark_sections"].get(bench)
        if b_entry:
            for p_key, scope_info in res.get("scopes", {}).items():
                if p_key in b_entry["multi_plots"]:
                    b_entry["multi_plots"][p_key]["scopes"]["all"] = scope_info
                    b_entry["multi_plots"][p_key]["file"] = scope_info["file"]

    elif res_type == "bench_machine":
        bench = res["benchmark"]
        machine = res["machine"]
        b_entry = metadata["benchmark_sections"].get(bench)
        if b_entry:
            for p_key, scope_info in res.get("scopes", {}).items():
                if p_key in b_entry["multi_plots"]:
                    b_entry["multi_plots"][p_key]["scopes"][machine] = scope_info
                    if "file" not in b_entry["multi_plots"][p_key]:
                        b_entry["multi_plots"][p_key]["file"] = scope_info["file"]
            for m_name, rec in res.get("run_records", {}).items():
                b_entry["runs"][m_name] = rec

    elif res_type == "benchmark_combined":
        _apply_task_result(metadata, res["bench_all_res"])
        for m_res in res["machine_results"]:
            _apply_task_result(metadata, m_res)


def compile_plots_html(
    runs: Union[str, Path, RunCollection, Sequence[RunData]] = "runs",
    output_path: Union[str, Path] = "reports/index.html",
    json_name: str = "report_metadata.json",
    qoi: str = "energy_uj",
    qois: Sequence[str] = ("energy_uj", "time"),
    benchmarks: Sequence[str] | None = None,
    machines: Sequence[str] | None = None,
    include_global: bool = True,
    sobol_modes: Sequence[str] = ("grouped_bar", "heatmap", "stacked_bar"),
    fmt: str = "png",
    dpi: int = 150,
    view_source_dir: Union[str, Path] = "view",
    workers: int | None = None,
    parallel_mode: str = "pair",
    quiet: bool = False,
) -> Path:
    """
    Compile all multi-run and single-run plots from a RunCollection into an
    interactive single-page HTML report with a companion static JSON metadata file.

    Uses persistent template assets (index.html, style.css, app.js) from the 'view' folder.
    Generates multi-plots for all machines combined and for each individual machine,
    including parameter effects for energy, time, and EDP ((df['energy_uj'] * 1e-6) * df['time']).

    Supports multi-process parallelism across pairs (benchmark, machine) or benchmarks.
    """
    t_start = time.time()
    out_file = Path(output_path).resolve()
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_root = out_dir / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)

    view_src = Path(view_source_dir).resolve()
    if not view_src.exists():
        # Fallback to repo_root / view
        view_src = (repo_root / "view").resolve()

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
        raise ValueError("No matching runs found to compile into HTML report.")

    active_benchmarks = collection.benchmarks
    active_machines = collection.machines

    if not quiet:
        print(f"\n==================================================")
        print(f"  EnergyUQ Interactive HTML Report Compiler")
        print(f"==================================================")
        print(f"Loaded {len(collection)} runs across {len(active_benchmarks)} benchmarks & {len(active_machines)} machines.")
        print(f"Benchmarks: {', '.join(active_benchmarks)}")
        print(f"Machines:   {', '.join(active_machines)}")
        print(f"Output HTML: {out_file}")
        print(f"Metadata:    {out_dir / json_name}")
        print(f"View source: {view_src}")
        print(f"Primary QoI: {qoi} | QoIs: {list(qois)}")
        print(f"Parallel:    mode={parallel_mode}, workers={workers or 'auto'}")
        print(f"==================================================\n")

    metadata: dict[str, Any] = {
        "title": "EnergyUQ Master Interactive Run Report",
        "generated_at": datetime.now().isoformat(),
        "generated_at_human": datetime.now().strftime("%B %d, %Y - %H:%M"),
        "benchmarks": active_benchmarks,
        "machines": active_machines,
        "primary_qoi": qoi,
        "qois": list(qois),
        "global_section": {
            "has_global": False,
            "multi_plots": {},
        },
        "benchmark_sections": {},
    }

    # 2. Prepare Benchmark Section Schemas and Task Inputs
    all_run_paths = [str(r.path) for r in collection]
    bench_data: dict[str, dict[str, Any]] = {}

    for bench in active_benchmarks:
        bench_runs = collection.filter(benchmarks=bench)
        if len(bench_runs) == 0:
            continue

        bench_machines = sorted(list({r.machine_name for r in bench_runs}))
        bench_params: list[str] = []
        for r in bench_runs:
            for p in r.input_params:
                if p not in bench_params:
                    bench_params.append(p)
        if not bench_params:
            bench_params = ["N_THREADS", "CLK"]

        bench_entry: dict[str, Any] = {
            "name": bench,
            "machines": bench_machines,
            "multi_plots": {},
            "runs": {},
        }

        # Initialize multi_plots metadata templates for this benchmark
        multi_plot_defs = _build_multi_plot_defs(qoi=qoi, sobol_modes=sobol_modes, bench_params=bench_params)
        for p_def in multi_plot_defs:
            bench_entry["multi_plots"][p_def["key"]] = {
                "key": p_def["key"],
                "title": p_def["title_template"],
                "tab_label": p_def["tab_label"],
                "category": p_def["category"],
                "description": _get_plot_description(p_def["key"]),
                "scopes": {},
            }

        metadata["benchmark_sections"][bench] = bench_entry

        machine_paths: dict[str, list[str]] = {
            m: [str(r.path) for r in bench_runs if r.machine_name == m]
            for m in bench_machines
        }

        bench_data[bench] = {
            "bench_dir": str(plots_root / bench),
            "bench_params": bench_params,
            "all_paths": [str(r.path) for r in bench_runs],
            "machine_paths": machine_paths,
            "machines": bench_machines,
        }

    # 3. Build Task List
    tasks: list[tuple[str, Callable, tuple]] = []

    # 3.1 Global cross-benchmark task
    if include_global and len(active_benchmarks) > 1:
        tasks.append((
            "Global Cross-Benchmark Summary",
            _worker_task_global,
            (
                all_run_paths,
                str(plots_root / "global"),
                str(out_dir),
                qoi,
                fmt,
                dpi,
                quiet,
            )
        ))

    # 3.2 Benchmark tasks
    if parallel_mode == "benchmark":
        for bench, b_info in bench_data.items():
            tasks.append((
                f"Benchmark: {bench}",
                _worker_task_benchmark,
                (
                    bench,
                    b_info["machine_paths"],
                    b_info["bench_dir"],
                    str(out_dir),
                    b_info["bench_params"],
                    qoi,
                    list(qois),
                    list(sobol_modes),
                    fmt,
                    dpi,
                    quiet,
                )
            ))
    else:
        # 'pair' mode (or fallback to per-pair breakdown)
        for bench, b_info in bench_data.items():
            # Task for scope 'all' of this benchmark
            tasks.append((
                f"[{bench}] Scope: all",
                _worker_task_bench_all,
                (
                    bench,
                    b_info["all_paths"],
                    b_info["bench_dir"],
                    str(out_dir),
                    b_info["bench_params"],
                    qoi,
                    list(sobol_modes),
                    fmt,
                    dpi,
                    quiet,
                )
            ))
            # Task per machine for this benchmark
            for m in b_info["machines"]:
                tasks.append((
                    f"[{bench}] Machine: {m}",
                    _worker_task_bench_machine,
                    (
                        bench,
                        m,
                        b_info["machine_paths"][m],
                        b_info["bench_dir"],
                        str(out_dir),
                        b_info["bench_params"],
                        qoi,
                        list(qois),
                        list(sobol_modes),
                        fmt,
                        dpi,
                        quiet,
                    )
                ))

    # 4. Execute Tasks (Sequential or Multi-Process)
    total_tasks = len(tasks)
    if parallel_mode == "none":
        effective_workers = 1
    elif workers is not None:
        effective_workers = max(1, workers)
    else:
        effective_workers = max(1, min(os.cpu_count() or 4, total_tasks))

    if effective_workers <= 1:
        if not quiet:
            print(f"Running {total_tasks} compilation tasks sequentially in main process...\n")
        for idx, (label, fn, args) in enumerate(tasks, 1):
            if not quiet:
                print(f"  [{idx}/{total_tasks}] Processing: {label}...")
            res = fn(*args)
            _apply_task_result(metadata, res)
    else:
        if not quiet:
            print(f"Dispatching {total_tasks} compilation tasks across {effective_workers} worker processes (mode: {parallel_mode})...\n")

        start_method = "fork" if "fork" in mp.get_all_start_methods() else None
        ctx = mp.get_context(start_method) if start_method else None

        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as executor:
            future_to_label = {
                executor.submit(fn, *args): label
                for label, fn, args in tasks
            }
            completed_count = 0
            for future in as_completed(future_to_label):
                label = future_to_label[future]
                completed_count += 1
                try:
                    res = future.result()
                    _apply_task_result(metadata, res)
                    if not quiet:
                        print(f"  [{completed_count}/{total_tasks}] Finished: {label}")
                except Exception as e:
                    if not quiet:
                        print(f"  [{completed_count}/{total_tasks}] ERROR in {label}: {e}")

    # 5. Write static JSON metadata file in the same directory as HTML
    json_path = out_dir / json_name
    if not quiet:
        print(f"\nWriting static JSON metadata to '{json_path}'...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(metadata), f, indent=2)

    # 6. Copy persistent view assets (index.html, style.css, app.js) to out_dir
    if view_src.exists() and view_src != out_dir:
        if not quiet:
            print(f"Deploying persistent view assets from '{view_src}' to '{out_dir}'...")
        for asset in ["index.html", "style.css", "app.js"]:
            src_file = view_src / asset
            if src_file.exists():
                shutil.copy2(src_file, out_dir / asset)
            else:
                if not quiet:
                    print(f"Warning: {src_file} not found in persistent view folder.")

    elapsed = time.time() - t_start
    if not quiet:
        print(f"\n==================================================")
        print(f"  Interactive HTML Report Successfully Generated!")
        print(f"==================================================")
        print(f"HTML Shell: {out_file}")
        print(f"Metadata:   {json_path}")
        print(f"View Folder:{view_src}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"==================================================\n")

    return out_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile multi-run and individual run plots into an interactive HTML report using persistent view assets."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Base directory containing run campaigns (default: 'runs').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="reports/index.html",
        help="Target output HTML file path (default: 'reports/index.html').",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default="report_metadata.json",
        help="Filename for static JSON metadata file in the same directory (default: 'report_metadata.json').",
    )
    parser.add_argument(
        "--view-dir",
        type=str,
        default="view",
        help="Path to persistent view folder containing index.html, style.css, app.js (default: 'view').",
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
        help="Comma-separated list of machines to include (e.g. 'sirius,cei').",
    )
    parser.add_argument(
        "--qoi",
        type=str,
        default="energy_uj",
        help="Primary QoI for multi-run evaluations (default: 'energy_uj').",
    )
    parser.add_argument(
        "--qois",
        type=str,
        default="energy_uj,time",
        help="Comma-separated list of QoIs for single-run evaluations (default: 'energy_uj,time').",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI resolution for rendered figure components (default: 150).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "svg"],
        help="Plot image format ('png' or 'svg', default: 'png').",
    )
    parser.add_argument(
        "--include-global",
        action="store_true",
        default=True,
        help="Include global cross-benchmark summary if multiple benchmarks present.",
    )
    parser.add_argument(
        "--sobol-modes",
        type=str,
        default="grouped_bar,heatmap,stacked_bar",
        help="Comma-separated list of modes for plot_multi_sobols (default: 'grouped_bar,heatmap,stacked_bar').",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=None,
        help="Number of worker processes to use (default: auto, up to CPU count). Set to 1 for sequential execution.",
    )
    parser.add_argument(
        "--parallel-mode",
        type=str,
        default="pair",
        choices=["pair", "benchmark", "none"],
        help="Parallel execution granularity ('pair' [default], 'benchmark', or 'none').",
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
    sobol_modes = [s.strip() for s in args.sobol_modes.split(",")] if args.sobol_modes else ["grouped_bar", "heatmap", "stacked_bar"]

    compile_plots_html(
        runs=args.runs_dir,
        output_path=args.output,
        json_name=args.json_name,
        qoi=args.qoi,
        qois=qois,
        benchmarks=benchmarks,
        machines=machines,
        include_global=args.include_global,
        sobol_modes=sobol_modes,
        fmt=args.format,
        dpi=args.dpi,
        view_source_dir=args.view_dir,
        workers=args.workers,
        parallel_mode=args.parallel_mode,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
