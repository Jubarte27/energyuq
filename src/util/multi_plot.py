import inspect
from typing import Any, Callable, Sequence, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator

from .multi_run import RunData, RunCollection


RunsInput = Union[RunCollection, Sequence[RunData]]


def _to_run_list(runs: RunsInput) -> list[RunData]:
    """Convert input to a list of RunData."""
    if isinstance(runs, RunCollection):
        return runs.runs
    return list(runs)


def _format_qoi_label(qoi: str | Callable[..., Any]) -> str:
    """Format QoI column name or callable into a clean publication label."""
    if callable(qoi):
        if hasattr(qoi, "name") and isinstance(qoi.name, str) and qoi.name: # type: ignore
            return _format_qoi_label(qoi.name) # type: ignore
        name = getattr(qoi, "__name__", "")
        if not name or name == "<lambda>":
            return "Computed Metric"
        return _format_qoi_label(str(name))
    mapping = {
        "energy_uj": r"Energy ($\mu$J)",
        "energy_j": "Energy (J)",
        "energy_scaled": "Scaled Energy",
        "EDP": "Energy-Delay Product (J·s)",
        "time": "Execution Time (s)",
        "power_w": "Power (W)",
    }
    return mapping.get(qoi, qoi.replace("_", " ").title())


def plot_multi_sobols(
    runs: RunsInput,
    qoi: str = "energy_uj",
    mode: str = "grouped_bar",  # "grouped_bar", "stacked_bar", or "heatmap"
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    show_title: bool = False,
    units: dict[str, str | None] | None = None,
) -> Figure | SubFigure:
    """
    Plot first-order Sobol sensitivity indices across multiple runs.
    
    Parameters
    ----------
    runs : RunCollection or list of RunData
    qoi : Quantity of interest, e.g. "energy_uj" or "time"
    mode : "grouped_bar", "stacked_bar", or "heatmap"
    figsize : Optional figure dimensions
    ax : Optional matplotlib Axes to draw into (not used for heatmap)
    show_title : Whether to display figure/axis titles (default False)
    units : Optional parameter units dictionary
    """
    run_list = _to_run_list(runs)
    if not run_list:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
        if show_title:
            empty_ax.set_title("No runs provided")
        return fig

    from ..plotting.plot import get_axis_label
    single_bench = len({r.benchmark_name for r in run_list}) == 1
    # Collect Sobol data
    records = []
    for r in run_list:
        sobols = r.get_sobols(qoi)
        for param, val in sobols.items():
            records.append({
                "benchmark": r.benchmark_name,
                "machine": r.machine_name,
                "tag": f"{r.machine_name}" if single_bench else f"{r.benchmark_name}\n({r.machine_name})",
                "parameter": get_axis_label(param, units),
                "sobol": val
            })

    if not records:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
        if show_title:
            empty_ax.set_title(f"No Sobol indices found for QoI '{qoi}'")
        return fig

    df_sobol = pd.DataFrame(records)
    params = sorted(df_sobol["parameter"].unique())
    tags = list(dict.fromkeys(df_sobol["tag"]))

    # 1. Heatmap Mode
    if mode == "heatmap":
        benchmarks = sorted(df_sobol["benchmark"].unique())
        machines = sorted(df_sobol["machine"].unique())

        n_params = len(params)
        fig_size = figsize or (5 * n_params, 4)
        fig, axes = plt.subplots(1, n_params, figsize=fig_size, squeeze=False, layout="constrained")

        for idx, param in enumerate(params):
            cur_ax = axes[0, idx]
            matrix = np.full((len(benchmarks), len(machines)), np.nan)
            sub_df = df_sobol[df_sobol["parameter"] == param]

            for _, row in sub_df.iterrows():
                b_idx = benchmarks.index(row["benchmark"])
                m_idx = machines.index(row["machine"])
                matrix[b_idx, m_idx] = row["sobol"]

            cax = cur_ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
            cur_ax.set_xticks(range(len(machines)))
            cur_ax.set_xticklabels(machines, rotation=45, ha="right")
            cur_ax.set_yticks(range(len(benchmarks)))
            cur_ax.set_yticklabels(benchmarks)
            cur_ax.set_title(f"Sensitivity: {param}")

            # Annotate numbers
            for i in range(len(benchmarks)):
                for j in range(len(machines)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        text_color = "white" if val > 0.6 else "black"
                        cur_ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold")

            fig.colorbar(cax, ax=cur_ax, label=r"$S_i$")

        if show_title:
            fig.suptitle(f"First-Order Sobol Sensitivity Indices for {_format_qoi_label(qoi)}")
        return fig

    # 2. Bar Modes (grouped or stacked)
    if ax is None:
        fig_size = figsize or (max(8.0, len(tags) * 0.8), 5.5)
        fig, ax = plt.subplots(figsize=fig_size, layout="constrained")
    else:
        fig = ax.get_figure()

    x = np.arange(len(tags))
    colors = plt.cm.tab10.colors # type: ignore

    if mode == "stacked_bar":
        bottom = np.zeros(len(tags))
        for p_idx, param in enumerate(params):
            vals = []
            for tag in tags:
                match = df_sobol[(df_sobol["tag"] == tag) & (df_sobol["parameter"] == param)]
                vals.append(match["sobol"].values[0] if not match.empty else 0.0)
            ax.bar(x, vals, bottom=bottom, label=param, color=colors[p_idx % len(colors)], edgecolor="white", width=0.6)
            bottom += np.array(vals)
        ax.set_ylim(0, 1.1)
    else:
        # Grouped bar
        n_p = len(params)
        width = 0.8 / n_p
        for p_idx, param in enumerate(params):
            vals = []
            for tag in tags:
                match = df_sobol[(df_sobol["tag"] == tag) & (df_sobol["parameter"] == param)]
                vals.append(match["sobol"].values[0] if not match.empty else 0.0)
            offset = (p_idx - (n_p - 1) / 2) * width
            ax.bar(x + offset, vals, width=width, label=param, color=colors[p_idx % len(colors)], edgecolor="white")
        ax.set_ylim(0, 1.05)

    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(r"First-Order Sobol Index ($S_i$)", fontsize=11)
    if show_title:
        ax.set_title(f"Parameter Sensitivity Comparison across Runs ({_format_qoi_label(qoi)})", fontsize=12)
    ax.legend(title="Parameter", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    assert fig is not None
    return fig


def plot_multi_convergence(
    runs: RunsInput,
    metric: str = "adaptation_error",  # "adaptation_error", "mean", or "std"
    facet_by: str | None = None,       # None or "benchmark"
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    show_title: bool = False,
) -> Figure | SubFigure:
    """
    Plot convergence histories (adaptation errors, mean, or standard deviation) across iterations.
    """
    run_list = _to_run_list(runs)
    if not run_list:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
        if show_title:
            empty_ax.set_title("No runs provided")
        return fig

    if facet_by == "benchmark":
        benchmarks = sorted(list({r.benchmark_name for r in run_list}))
        n_b = len(benchmarks)
        fig_size = figsize or (5.5 * n_b, 4.5)
        fig, axes = plt.subplots(1, n_b, figsize=fig_size, sharey=True, layout="constrained")
        if n_b == 1:
            axes = [axes]

        for b_idx, bench in enumerate(benchmarks):
            cur_ax = axes[b_idx]
            bench_runs = [r for r in run_list if r.benchmark_name == bench]
            for r in bench_runs:
                hist = r.get_convergence_history().get(metric, [])
                if hist:
                    iters = np.arange(1, len(hist) + 1)
                    cur_ax.plot(iters, hist, marker="o", label=r.machine_name)

            cur_ax.set_title(f"Benchmark: {bench}")
            cur_ax.set_xlabel("Iteration")
            if metric == "adaptation_error":
                cur_ax.set_yscale("log")
            cur_ax.grid(True, linestyle="--", alpha=0.5)
            cur_ax.legend(title="Machine")

        axes[0].set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        if show_title:
            fig.suptitle(f"Convergence History by Benchmark ({metric})", fontsize=13)
        return fig

    # Single axis mode
    if ax is None:
        fig_size = figsize or (8.5, 5.0)
        fig, ax = plt.subplots(figsize=fig_size, layout="constrained")
    else:
        fig = ax.get_figure()

    for r in run_list:
        hist = r.get_convergence_history().get(metric, [])
        if hist:
            iters = np.arange(1, len(hist) + 1)
            ax.plot(iters, hist, marker="o", label=r.tag, alpha=0.8)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    if metric == "adaptation_error":
        ax.set_yscale("log")
    if show_title:
        ax.set_title(f"Multi-Run Convergence History ({metric})", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)

    assert fig is not None
    return fig


def plot_multi_energy_time_pareto(
    runs: RunsInput,
    qoi_x: str = "time",
    qoi_y: str = "energy_uj",
    show_all_samples: bool = True,
    show_frontier: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    show_title: bool = False,
) -> Figure | SubFigure:
    """
    Plot energy vs execution time trade-off for multiple runs, highlighting Pareto frontiers.
    """
    run_list = _to_run_list(runs)
    if not run_list:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
        if show_title:
            empty_ax.set_title("No runs provided")
        return fig

    if ax is None:
        fig_size = figsize or (9.0, 6.0)
        fig, ax = plt.subplots(figsize=fig_size, layout="constrained")
    else:
        fig = ax.get_figure()

    # Distinct markers for benchmarks, colors for machines
    benchmarks = sorted(list({r.benchmark_name for r in run_list}))
    machines = sorted(list({r.machine_name for r in run_list}))

    marker_choices = ["o", "s", "^", "D", "v", "P", "*"]
    bench_markers = {b: marker_choices[i % len(marker_choices)] for i, b in enumerate(benchmarks)}

    cmap = plt.get_cmap("tab10")
    mach_colors = {m: cmap(i % 10) for i, m in enumerate(machines)}

    for r in run_list:
        df = r.df
        if df.empty or qoi_x not in df.columns or qoi_y not in df.columns:
            continue

        color = mach_colors[r.machine_name]
        marker = bench_markers[r.benchmark_name]

        # Convert to Joules if energy_uj for readable scale
        scale_y = 1e-6 if qoi_y == "energy_uj" else 1.0
        x_vals = df[qoi_x].to_numpy()
        y_vals = df[qoi_y].to_numpy() * scale_y

        if show_all_samples:
            ax.scatter(x_vals, y_vals, color=color, marker=marker, alpha=0.35, s=35)

        # Compute and plot Pareto frontier
        if show_frontier:
            front = r.pareto_front(qoi_x, qoi_y)
            if not front.empty:
                fx = front[qoi_x].to_numpy()
                fy = front[qoi_y].to_numpy() * scale_y
                ax.plot(fx, fy, color=color, linestyle="-", linewidth=1.8, label=r.tag)
                ax.scatter(fx, fy, color=color, marker=marker, s=50, edgecolor="black", zorder=5)

    y_label = "Energy (J)" if qoi_y == "energy_uj" else _format_qoi_label(qoi_y)
    ax.set_xlabel(_format_qoi_label(qoi_x), fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    if show_title:
        ax.set_title("Energy vs Time Pareto Frontier Comparison", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Custom legends for Machines and Benchmarks
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=mach_colors[m], lw=2, label=m) for m in machines
    ]
    if len(benchmarks) > 1:
        legend_elements += [
            Line2D([0], [0], marker=bench_markers[b], color="gray", label=b, linestyle="none", markersize=8)
            for b in benchmarks
        ]
    ax.legend(handles=legend_elements, title="Machine / Benchmark", bbox_to_anchor=(1.04, 1), loc="upper left")

    assert fig is not None
    return fig


def plot_multi_qoi_distribution(
    runs: RunsInput,
    qoi: str = "energy_uj",
    group_by: str = "machine",  # "machine" or "benchmark"
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    show_title: bool = False,
) -> Figure | SubFigure:
    """
    Boxplot comparison of QoI distribution across machines or benchmarks.
    """
    run_list = _to_run_list(runs)
    if not run_list:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
        if show_title:
            empty_ax.set_title("No runs provided")
        return fig

    # Build long DataFrame
    records = []
    scale = 1e-6 if qoi == "energy_uj" else 1.0
    for r in run_list:
        if not r.df.empty and qoi in r.df.columns:
            for val in r.df[qoi].dropna():
                records.append({
                    "benchmark": r.benchmark_name,
                    "machine": r.machine_name,
                    "val": val * scale
                })

    if not records:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
        if show_title:
            empty_ax.set_title(f"No data for {qoi}")
        return fig

    df_plot = pd.DataFrame(records)

    primary_col = "machine" if group_by == "machine" else "benchmark"
    secondary_col = "benchmark" if group_by == "machine" else "machine"

    primary_groups = sorted(df_plot[primary_col].unique())
    secondary_groups = sorted(df_plot[secondary_col].unique())

    if ax is None:
        fig_size = figsize or (max(7.0, len(primary_groups) * 1.5), 5.0)
        fig, ax = plt.subplots(figsize=fig_size, layout="constrained")
    else:
        fig = ax.get_figure()

    colors = plt.cm.Set2.colors # type: ignore
    n_sec = len(secondary_groups)
    width = 0.8 / max(n_sec, 1)

    for s_idx, sec_val in enumerate(secondary_groups):
        sub = df_plot[df_plot[secondary_col] == sec_val]
        data_to_plot = []
        positions = []
        for p_idx, prim_val in enumerate(primary_groups):
            match_vals = sub[sub[primary_col] == prim_val]["val"].values
            if len(match_vals) > 0:
                data_to_plot.append(match_vals)
                offset = (s_idx - (n_sec - 1) / 2) * width
                positions.append(p_idx + offset)

        if data_to_plot:
            color = colors[s_idx % len(colors)]
            bp = ax.boxplot(
                data_to_plot,
                positions=positions,
                widths=width * 0.85,
                patch_artist=True,
                showmeans=True,
                meanprops={"marker": "x", "markeredgecolor": "black", "markersize": 6},
                boxprops={"facecolor": color, "alpha": 0.7, "edgecolor": "black"},
                medianprops={"color": "black", "linewidth": 1.5},
            )

    # Setup axes
    ax.set_xticks(range(len(primary_groups)))
    ax.set_xticklabels(primary_groups, fontsize=10)
    ax.set_xlabel(primary_col.title(), fontsize=11)

    y_label = "Energy (J)" if qoi == "energy_uj" else _format_qoi_label(qoi)
    ax.set_ylabel(y_label, fontsize=11)
    if show_title:
        ax.set_title(f"Distribution of {y_label} by {primary_col.title()}", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[i % len(colors)], alpha=0.7, edgecolor="black", label=sec_val)
        for i, sec_val in enumerate(secondary_groups)
    ]
    ax.legend(handles=legend_elements, title=secondary_col.title(), bbox_to_anchor=(1.04, 1), loc="upper left")

    assert fig is not None
    return fig


def plot_multi_best_configurations(
    runs: RunsInput,
    qoi: str = "energy_uj",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    show_title: bool = False,
    units: dict[str, str | None] | None = None,
) -> Figure | SubFigure:
    """
    Plot the best configuration (N_THREADS and CLK) found in each run for a given QoI.
    """
    run_list = _to_run_list(runs)
    single_bench = len({r.benchmark_name for r in run_list}) == 1
    if not run_list:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
        if show_title:
            empty_ax.set_title("No runs provided")
        return fig

    from ..plotting.plot import to_real_clk, get_unit_converter
    rows = []
    has_real_clk = False
    for r in run_list:
        best = r.best_config(qoi, mode="min")
        if best:
            clk_val = best.get("CLK", best.get("CLK_LEVEL", np.nan))
            if r.machine and hasattr(r.machine, "freq") and r.machine.freq and not pd.isna(clk_val):
                clk_val = to_real_clk(clk_val, r.machine, units)
                has_real_clk = True
            rows.append({
                "benchmark": r.benchmark_name,
                "machine": r.machine_name,
                "tag": r.machine_name if single_bench else r.tag,
                "threads": best.get("N_THREADS", best.get("THREADS", np.nan)),
                "clk": clk_val,
                "val": best.get(qoi, np.nan)
            })

    if not rows:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
        if show_title:
            empty_ax.set_title(f"No best configurations found for {qoi}")
        return fig

    df_best = pd.DataFrame(rows)

    if ax is None:
        fig_size = figsize or (8.0, 5.5)
        fig, ax = plt.subplots(figsize=fig_size, layout="constrained")
    else:
        fig = ax.get_figure()

    benchmarks = sorted(df_best["benchmark"].unique())
    machines = sorted(df_best["machine"].unique())

    marker_choices = ["o", "s", "^", "D", "v", "P", "*"]
    bench_markers = {b: marker_choices[i % len(marker_choices)] for i, b in enumerate(benchmarks)}
    cmap = plt.get_cmap("tab10")
    mach_colors = {m: cmap(i % 10) for i, m in enumerate(machines)}

    for _, row in df_best.iterrows():
        color = mach_colors[row["machine"]]
        marker = bench_markers[row["benchmark"]]
        ax.scatter(row["threads"], row["clk"], color=color, marker=marker, s=120, edgecolor="black", zorder=4)
        ax.annotate(row["tag"], (row["threads"], row["clk"]), textcoords="offset points", xytext=(5, 5), fontsize=8)

    clk_target_unit, _ = get_unit_converter("CLK", units)
    if clk_target_unit and str(clk_target_unit).strip().lower() not in ("", "none"):
        clk_label = f"Optimal Clock Frequency ({str(clk_target_unit).strip()})"
    else:
        clk_label = "Optimal Clock Frequency (CLK)" if has_real_clk else "Optimal Clock Frequency Index (CLK)"

    threads_target_unit, _ = get_unit_converter("N_THREADS", units)
    if threads_target_unit and str(threads_target_unit).strip().lower() not in ("", "none"):
        threads_label = f"Optimal Threads ({str(threads_target_unit).strip()})"
    else:
        threads_label = "Optimal Threads (N_THREADS)"

    ax.set_xlabel(threads_label, fontsize=11)
    ax.set_ylabel(clk_label, fontsize=11)
    if show_title:
        ax.set_title(f"Optimal Configurations Found Across Runs (Min {_format_qoi_label(qoi)})", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    clk_series = df_best["clk"].dropna()
    is_clk_int = True
    if not clk_series.empty:
        try:
            arr = clk_series.to_numpy(dtype=float)
            is_clk_int = np.all(np.isclose(arr, np.round(arr)))
        except Exception:
            is_clk_int = False
    ax.yaxis.set_major_locator(MaxNLocator(integer=bool(is_clk_int)))
    ax.grid(True, linestyle="--", alpha=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=mach_colors[m], marker="o", lw=0, label=m, markersize=8) for m in machines
    ]
    if not single_bench:
        legend_elements += [
            Line2D([0], [0], marker=bench_markers[b], color="gray", label=b, linestyle="none", markersize=8)
            for b in benchmarks
        ]
    ax.legend(handles=legend_elements, title="Machine / Benchmark", bbox_to_anchor=(1.04, 1), loc="upper left")

    assert fig is not None
    return fig


def _evaluate_qoi_callable(
    qoi_fn: Callable[..., Any],
    run: RunData,
    df: pd.DataFrame,
) -> pd.Series | None:
    """
    Evaluate a custom QoI callable for a single run and return a numeric Series matching df.index.

    Supports callables taking:
    - (df): Vectorized DataFrame operation, e.g. lambda df: df["energy_uj"] / df["time"]
    - (run): RunData object, e.g. lambda r: r.df["energy_uj"] * 1e-6
    - (row): Row-wise function, e.g. lambda row: row["energy_uj"] if row["N_THREADS"] > 2 else 0
    - (df, run) or (run, df): Both DataFrame and RunData
    """
    if df.empty:
        return None

    def _validate_result(res: Any) -> pd.Series | None:
        if res is None:
            return None
        if isinstance(res, pd.DataFrame):
            if res.shape[1] == 1:
                res = res.iloc[:, 0]
            else:
                return None
        if isinstance(res, np.ndarray):
            res = np.squeeze(res)
            if res.ndim != 1:
                return None
        if isinstance(res, (pd.Series, np.ndarray, list, tuple)):
            if len(res) == len(df):
                raw_vals = res.values if hasattr(res, "values") else np.asarray(res) # type: ignore
                s = pd.to_numeric(pd.Series(raw_vals, index=df.index), errors="coerce")
                return s.replace([np.inf, -np.inf], np.nan)
        elif len(df) == 1 and isinstance(res, (int, float, np.number)):
            return pd.Series([float(res)], index=df.index)
        return None

    # Inspect signature to optimize invocation order
    order = ["df", "run", "row"]
    try:
        sig = inspect.signature(qoi_fn)
        params = list(sig.parameters.values())
        if len(params) == 1:
            p_name = params[0].name.lower()
            p_anno = str(params[0].annotation).lower()
            if p_name in ("r", "run", "rundata", "run_data") or "rundata" in p_anno:
                order = ["run", "df", "row"]
            elif p_name in ("row", "record", "sample"):
                order = ["row", "df", "run"]
            else:
                order = ["df", "run", "row"]
        elif len(params) == 2:
            p0_name = params[0].name.lower()
            if p0_name in ("r", "run", "rundata", "run_data"):
                order = ["run_df", "df_run", "df", "run", "row"]
            else:
                order = ["df_run", "run_df", "df", "run", "row"]
    except Exception:
        order = ["df", "run", "row"]

    for mode in order:
        try:
            if mode == "df":
                val = _validate_result(qoi_fn(df))
            elif mode == "run":
                val = _validate_result(qoi_fn(run))
            elif mode == "row":
                val = _validate_result(df.apply(qoi_fn, axis=1))
            elif mode == "df_run":
                val = _validate_result(qoi_fn(df, run))
            elif mode == "run_df":
                val = _validate_result(qoi_fn(run, df))
            else:
                val = None

            if val is not None:
                return val
        except Exception:
            continue

    return None


def plot_multi_parameter_effects(
    runs: RunsInput,
    param: str = "N_THREADS",
    qoi: str | Callable[..., Any] = "energy_uj",
    facet_by: str = "benchmark",  # "benchmark" or "machine"
    figsize: tuple[float, float] | None = None,
    show_title: bool = False,
    units: dict[str, str | None] | None = None,
    qoi_label: str | None = None,
    scale: float | None = None,
) -> Figure:
    """
    Plot the effect / response curve of varying a specific parameter on the QoI across runs.

    Parameters
    ----------
    runs : RunCollection or Sequence[RunData]
        The experimental runs to plot.
    param : str, default "N_THREADS"
        The parameter column name to vary on the x-axis.
    qoi : str or Callable, default "energy_uj"
        The Quantity of Interest to display on the y-axis. Can be:
        - A string representing a column in run DataFrames (e.g. "energy_uj", "time", "power_w")
        - A callable/function computing the metric. The function may accept:
            * `df: pd.DataFrame` returning a Series or 1D array of values for each row
            * `run: RunData` returning a Series or 1D array
            * `row: pd.Series` returning a scalar value evaluated row-wise
    facet_by : str, default "benchmark"
        Dimension to facet across columns ("benchmark" or "machine").
    figsize : tuple[float, float], optional
        Figure dimensions. Defaults to (5.5 * n_groups, 4.5).
    show_title : bool, default False
        Whether to show the suptitle across the figure.
    units : dict, optional
        Unit conversions dictionary for parameters.
    qoi_label : str, optional
        Custom label for the y-axis and title. If omitted, inferred from the column name
        or function name.
    scale : float, optional
        Multiplicative scale factor for QoI values. If omitted, defaults to 1e-6 if
        qoi == "energy_uj" and 1.0 otherwise.
    """
    run_list = _to_run_list(runs)
    eff_scale = scale if scale is not None else (1e-6 if qoi == "energy_uj" else 1.0)

    group_col = facet_by
    group_values = sorted(list({getattr(r, f"{group_col}_name") for r in run_list}))
    n_groups = len(group_values)

    fig_size = figsize or (5.5 * n_groups, 4.5)
    fig, axes = plt.subplots(1, n_groups, figsize=fig_size, sharey=True, squeeze=False, layout="constrained")

    other_col = "machine" if facet_by == "benchmark" else "benchmark"

    for g_idx, g_val in enumerate(group_values):
        cur_ax = axes[0, g_idx]
        cur_runs = [r for r in run_list if getattr(r, f"{group_col}_name") == g_val]
        all_x_vals = []

        for r in cur_runs:
            if param not in r.df.columns or r.df.empty:
                continue

            r_df = r.df.copy()
            if callable(qoi):
                metric_series = _evaluate_qoi_callable(qoi, r, r_df)
                if metric_series is None:
                    continue
                target_col = "_computed_qoi"
                r_df[target_col] = metric_series
            else:
                if qoi not in r_df.columns:
                    continue
                target_col = qoi

            if str(param).upper() in ("CLK", "CLK_LEVEL") and r.machine and hasattr(r.machine, "freq") and r.machine.freq:
                from ..plotting.plot import to_real_clk
                r_df[param] = r_df[param].map(lambda v: to_real_clk(v, r.machine, units))
            else:
                from ..plotting.plot import get_unit_converter
                _, p_conv = get_unit_converter(param, units)
                if p_conv is not None:
                    r_df[param] = r_df[param].map(lambda v: p_conv(float(v)) if not pd.isna(v) else v) # type: ignore

            r_df = r_df.dropna(subset=[param, target_col])
            if r_df.empty:
                continue

            all_x_vals.extend(r_df[param].tolist())

            cur_ax.scatter(
                r_df[param],
                r_df[target_col] * eff_scale,
                alpha=0.7,
                marker="o",
                label=getattr(r, f"{other_col}_name")
            )

        cur_ax.set_title(f"{group_col.title()}: {g_val}")
        from ..plotting.plot import get_axis_label
        param_label = get_axis_label(param, units)
        cur_ax.set_xlabel(param_label)

        if all_x_vals:
            try:
                arr = np.array(all_x_vals, dtype=float)
                is_int = np.all(np.isclose(arr, np.round(arr)))
                cur_ax.xaxis.set_major_locator(MaxNLocator(integer=bool(is_int)))
            except Exception:
                cur_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            cur_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        cur_ax.grid(True, linestyle="--", alpha=0.5)
        handles, labels = cur_ax.get_legend_handles_labels()
        if handles:
            cur_ax.legend(handles, labels, title=other_col.title(), fontsize=8)

    if qoi_label is not None:
        y_label = qoi_label
    elif qoi == "energy_uj":
        y_label = "Energy (J)"
    else:
        y_label = _format_qoi_label(qoi)

    axes[0, 0].set_ylabel(y_label, fontsize=11)
    if show_title:
        from ..plotting.plot import get_axis_label
        param_label = get_axis_label(param, units)
        fig.suptitle(f"Effect of {param_label} on {y_label} across {facet_by.title()}s", fontsize=13)

    return fig


def plot_multi_dashboard(
    runs: RunsInput,
    qoi: str = "energy_uj",
    figsize: tuple[float, float] = (15.0, 11.0),
    show_title: bool = False,
    units: dict[str, str | None] | None = None,
) -> Figure:
    """
    Generate a 4-panel dashboard summarizing cross-run findings:
    1. Sobol sensitivity comparison (stacked bars)
    2. Energy vs Time Pareto trade-off
    3. Convergence histories
    4. Optimal configuration comparison
    """
    run_list = _to_run_list(runs)
    fig = plt.figure(figsize=figsize, layout="constrained")
    if show_title:
        fig.suptitle(f"Multi-Run Comparative Analysis Dashboard ({_format_qoi_label(qoi)})", fontsize=15, fontweight="bold")

    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # 1. Sobol sensitivities
    plot_multi_sobols(run_list, qoi=qoi, mode="stacked_bar", ax=ax1, units=units)

    # 2. Pareto frontiers
    plot_multi_energy_time_pareto(run_list, qoi_x="time", qoi_y=qoi, ax=ax2)

    # 3. Convergence
    plot_multi_convergence(run_list, metric="adaptation_error", ax=ax3)

    # 4. Best configurations
    plot_multi_best_configurations(run_list, qoi=qoi, ax=ax4, units=units)

    return fig
