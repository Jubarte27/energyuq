#!/usr/bin/env python3
"""
Compile multi-run and individual run plots from a RunCollection into an
interactive HTML report with a companion static JSON metadata file.

Uses the persistent template assets in the 'view/' folder (index.html, style.css, app.js).
Generates multi-plots for all machines combined and for each individual machine,
supporting toggle/swap navigation in the interface.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from src.util import multi_plot
from src.util.data import to_serializable_primitive
from src.util.multi_run import (
    RunCollection,
    RunData,
    load_run,
)


def _json_ready(obj: Any) -> Any:
    """Coerce ``obj`` into strictly valid JSON (non-finite floats become ``null``)."""
    obj = to_serializable_primitive(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    return obj


def _format_individual_plot_title(name: str) -> str:
    """Return a descriptive, human-readable title for individual analysis figures."""
    mapping = {
        "sobol_indices_energy_uj": "First-Order Sobol Sensitivity Indices — Energy (μJ)",
        "sobol_treemap_energy_uj": "Sobol Sensitivity Treemap — Energy (μJ)",
        "sobol_indices_time": "First-Order Sobol Sensitivity Indices — Execution Time (s)",
        "sobol_treemap_time": "Sobol Sensitivity Treemap — Execution Time (s)",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _get_plot_category(name: str) -> str:
    """Classify plot key into a functional category."""
    return "Sensitivity" if "sobol" in name else "Diagnostics"


def _get_plot_description(name: str) -> str:
    """Return an intuitive description of what the plot visualizes."""
    descriptions = {
        "sobol_indices_energy_uj": "Variance-based first-order Sobol sensitivity decomposition quantifying the relative influence of each parameter on energy.",
        "sobol_treemap_energy_uj": "Hierarchical treemap visualizing variance contribution of control knobs to energy variability.",
        "sobol_indices_time": "Variance-based first-order Sobol sensitivity decomposition for execution time.",
        "sobol_treemap_time": "Hierarchical treemap visualizing variance contribution of control knobs to execution time variability.",
        "sobol_grouped_bar": "Side-by-side comparative bar chart of first-order Sobol sensitivity indices across evaluated machines.",
        "sobol_heatmap": "Color-encoded matrix of parameter sensitivity intensities across evaluated machines.",
        "sobol_stacked_bar": "Stacked representation of relative parameter importance proportions across machines.",
    }
    for k, v in descriptions.items():
        if k in name:
            return v
    return "Analytical diagnostic figure generated during campaign evaluation."


def _save_figure(fig: Figure, out_file: Path, dpi: int = 150):
    """Save a matplotlib figure and close it cleanly."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close("all")


#: Human-readable labels for the supported Sobol visualisation modes.
SOBOL_MODE_LABELS: dict[str, str] = {
    "grouped_bar": "Grouped Bar",
    "heatmap": "Heatmap",
    "stacked_bar": "Stacked Bar",
}


def _build_multi_plot_defs(qoi: str, sobol_modes: Sequence[str]) -> list[dict[str, Any]]:
    """Build the Sobol plot definitions for a benchmark, one per requested mode."""
    return [
        {
            "key": f"sobols_{mode}",
            "tab_label": f"Sobols ({SOBOL_MODE_LABELS.get(mode, mode.replace('_', ' ').title())})",
            "title_template": f"Sobol Sensitivities ({SOBOL_MODE_LABELS.get(mode, mode.replace('_', ' ').title())})",
            "category": "Sensitivity",
            "builder": (lambda m: lambda r_set: multi_plot.plot_multi_sobols(
                r_set, qoi=qoi, mode=m, show_title=True,
            ))(mode),
        }
        for mode in sobol_modes
    ]


def _render_scope_multi_plots(
    r_subset: RunCollection,
    bench: str,
    scope_id: str,
    scope_title: str,
    scope_dir: Path,
    out_dir: Path,
    qoi: str,
    sobol_modes: Sequence[str],
    fmt: str = "png",
    dpi: int = 150,
    quiet: bool = True,
) -> dict[str, dict[str, str]]:
    """Render all multi-plots for a given scope, save to scope_dir, and return scope metadata."""
    scope_dir.mkdir(parents=True, exist_ok=True)
    scopes: dict[str, dict[str, str]] = {}

    for p_def in _build_multi_plot_defs(qoi=qoi, sobol_modes=sobol_modes):
        p_key = p_def["key"]
        out_fig_file = scope_dir / f"{p_key}.{fmt}"
        disp_title = f"Benchmark: {bench} — {p_def['title_template']} ({scope_title})"

        try:
            fig = p_def["builder"](r_subset)
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
        figures=["sobols"],
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

        tab_label = "Sobol Treemap" if fig_key.startswith("sobol_treemap") else "Sobol Indices"

        run_plots.append({
            "key": fig_key,
            "title": disp_title,
            "tab_label": tab_label,
            "category": _get_plot_category(fig_key),
            "qoi": fig_qoi,
            "file": str(f_path.relative_to(out_dir)),
            "description": _get_plot_description(fig_key),
        })

    run_record: dict[str, Any] = {
        "tag": r.tag,
        "benchmark": bench,
        "machine": r.machine_name,
        "sample_count": r.sample_count,
        "iterations": r.iteration_count,
        "input_params": r.input_params,
        "qois_data": metrics.get("qois", {}),
        "plots": run_plots,
        # The whole metadata tree is made strictly JSON-safe in one pass before dumping.
        "samples_preview": r.df.head(20).to_dict(orient="records") if not r.df.empty else [],
    }

    return r.machine_name, run_record


def _worker_task_global(
    run_paths: list[str],
    global_plots_dir_str: str,
    out_dir_str: str,
    qoi: str,
    sobol_modes: list[str],
    fmt: str = "png",
    dpi: int = 150,
    quiet: bool = True,
) -> dict[str, Any]:
    """Worker task to generate cross-benchmark Sobol summary plots."""
    global_plots_dir = Path(global_plots_dir_str)
    out_dir = Path(out_dir_str)
    global_plots_dir.mkdir(parents=True, exist_ok=True)

    collection = RunCollection([load_run(p) for p in run_paths])
    g_plots: dict[str, Any] = {}

    for p_def in _build_multi_plot_defs(qoi=qoi, sobol_modes=sobol_modes):
        p_key = p_def["key"]
        title = f"{p_def['title_template']} across All Benchmarks"
        try:
            fig = p_def["builder"](collection)
            fig.suptitle(title, fontsize=13, fontweight="bold")
            out_fig = global_plots_dir / f"multi_{p_key}.{fmt}"
            _save_figure(fig, out_fig, dpi=dpi)
            rel = str(out_fig.relative_to(out_dir))
            g_plots[p_key] = {
                "key": p_key,
                "title": title,
                "tab_label": p_def["tab_label"],
                "file": rel,
                "description": _get_plot_description(p_key),
                "scopes": {"all": {"file": rel, "title": title}},
            }
        except Exception as e:
            if not quiet:
                print(f"  [Global] {p_def['tab_label']} skipped: {e}")

    return {
        "type": "global",
        "g_plots": g_plots,
    }


def _worker_task_bench_all(
    bench: str,
    run_paths: list[str],
    bench_dir_str: str,
    out_dir_str: str,
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
        qoi=qoi,
        sobol_modes=sobol_modes,
        fmt=fmt,
        dpi=dpi,
        quiet=quiet,
    )

    return {
        "type": "bench_all",
        "benchmark": bench,
        "scope": "all",
        "scopes": scopes,
    }


def _worker_task_bench_machine(
    bench: str,
    machine: str,
    run_paths: list[str],
    bench_dir_str: str,
    out_dir_str: str,
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
        "scope": machine,
        "scopes": scopes,
        "run_records": run_records,
    }


def _apply_task_result(metadata: dict[str, Any], res: dict[str, Any]) -> None:
    """Merge a worker task result dictionary into the main report metadata."""
    if res["type"] == "global":
        metadata["global_section"]["has_global"] = True
        metadata["global_section"]["multi_plots"] = res["g_plots"]
        return

    b_entry = metadata["benchmark_sections"].get(res["benchmark"])
    if b_entry is None:
        return

    for p_key, scope_info in res.get("scopes", {}).items():
        plot = b_entry["multi_plots"].get(p_key)
        if plot is None:
            continue
        plot["scopes"][res.get("scope", "all")] = scope_info
        # Point the default view at the first scope that produced a file.
        plot.setdefault("file", scope_info["file"])

    for m_name, rec in res.get("run_records", {}).items():
        b_entry["runs"][m_name] = rec


def compile_plots_html(
    runs: str | Path | RunCollection | Sequence[RunData] = "runs",
    output_path: str | Path = "reports/index.html",
    json_name: str = "report_metadata.json",
    qoi: str = "energy_uj",
    qois: Sequence[str] = ("energy_uj", "time"),
    benchmarks: Sequence[str] | None = None,
    machines: Sequence[str] | None = None,
    include_global: bool = True,
    sobol_modes: Sequence[str] = ("grouped_bar", "heatmap", "stacked_bar"),
    fmt: str = "png",
    dpi: int = 150,
    view_source_dir: str | Path = "view",
    workers: int | None = None,
    quiet: bool = False,
) -> Path:
    """
    Compile all multi-run and single-run plots from a RunCollection into an
    interactive single-page HTML report with a companion static JSON metadata file.

    Uses persistent template assets (index.html, style.css, app.js) from the 'view' folder.
    Renders Sobol sensitivity plots: for every benchmark, one multi-run set covering all
    machines plus a set per individual machine, plus per-run Sobol indices and a
    cross-benchmark comparison when more than one benchmark is present.

    Supports multi-process parallelism across (benchmark, machine) scopes.
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
        print("\n==================================================")
        print("  EnergyUQ Interactive HTML Report Compiler")
        print("==================================================")
        print(f"Loaded {len(collection)} runs across {len(active_benchmarks)} benchmarks & {len(active_machines)} machines.")
        print(f"Benchmarks: {', '.join(active_benchmarks)}")
        print(f"Machines:   {', '.join(active_machines)}")
        print(f"Output HTML: {out_file}")
        print(f"Metadata:    {out_dir / json_name}")
        print(f"View source: {view_src}")
        print(f"Primary QoI: {qoi} | QoIs: {list(qois)}")
        print(f"Plots:       Sobol modes {list(sobol_modes)}")
        print("==================================================\n")

    metadata: dict[str, Any] = {
        "title": "EnergyUQ Master Interactive Run Report",
        "generated_at": datetime.now().isoformat(),
        "generated_at_human": datetime.now().strftime("%B %d, %Y - %H:%M"),
        "benchmarks": active_benchmarks,
        "machines": active_machines,
        "primary_qoi": qoi,
        "qois": list(qois),
        "qoi_labels": dict(multi_plot.QOI_LABELS),
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

        bench_machines = sorted({r.machine_name for r in bench_runs})

        # Seed the multi_plots metadata templates for this benchmark.
        bench_entry: dict[str, Any] = {
            "name": bench,
            "machines": bench_machines,
            "multi_plots": {
                p_def["key"]: {
                    "key": p_def["key"],
                    "title": p_def["title_template"],
                    "tab_label": p_def["tab_label"],
                    "category": p_def["category"],
                    "description": _get_plot_description(p_def["key"]),
                    "scopes": {},
                }
                for p_def in _build_multi_plot_defs(qoi=qoi, sobol_modes=sobol_modes)
            },
            "runs": {},
        }

        metadata["benchmark_sections"][bench] = bench_entry

        bench_data[bench] = {
            "bench_dir": str(plots_root / bench),
            "all_paths": [str(r.path) for r in bench_runs],
            "machine_paths": {
                m: [str(r.path) for r in bench_runs if r.machine_name == m]
                for m in bench_machines
            },
            "machines": bench_machines,
        }

    # 3. Build Task List
    tasks: list[tuple[str, Callable, tuple]] = []

    # 3.1 Global cross-benchmark Sobol comparison
    if include_global and len(active_benchmarks) > 1:
        tasks.append((
            "Global Cross-Benchmark Summary",
            _worker_task_global,
            (
                all_run_paths,
                str(plots_root / "global"),
                str(out_dir),
                qoi,
                list(sobol_modes),
                fmt,
                dpi,
                quiet,
            )
        ))

    # 3.2 Benchmark tasks
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
    effective_workers = 1 if workers == 1 else (workers or max(1, min(os.cpu_count() or 4, total_tasks)))

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
            print(f"Dispatching {total_tasks} compilation tasks across {effective_workers} worker processes...\n")

        start_method = "fork" if "fork" in mp.get_all_start_methods() else None
        ctx = mp.get_context(start_method) if start_method else None

        with ProcessPoolExecutor(max_workers=effective_workers, mp_context=ctx) as executor:
            future_to_label = {
                executor.submit(fn, *args): label
                for label, fn, args in tasks
            }
            for completed_count, future in enumerate(as_completed(future_to_label), 1):
                label = future_to_label[future]
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
        json.dump(_json_ready(metadata), f, indent=2)

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
        print("\n==================================================")
        print("  Interactive HTML Report Successfully Generated!")
        print("==================================================")
        print(f"HTML Shell: {out_file}")
        print(f"Metadata:   {json_path}")
        print(f"View Folder:{view_src}")
        print(f"Total time: {elapsed:.1f}s")
        print("==================================================\n")

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
        "--no-global",
        dest="include_global",
        action="store_false",
        default=True,
        help="Skip the global cross-benchmark summary section.",
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
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
