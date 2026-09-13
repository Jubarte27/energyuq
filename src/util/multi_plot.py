from typing import Sequence, Union
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


def _format_qoi_label(qoi: str) -> str:
    """Format QoI column name into a clean publication label."""
    mapping = {
        "energy_uj": r"Energy ($\mu$J)",
        "energy_j": "Energy (J)",
        "energy_scaled": "Scaled Energy",
        "time": "Execution Time (s)",
        "power_w": "Power (W)",
        "edp_j_s": "Energy-Delay Product (J·s)",
    }
    return mapping.get(qoi, qoi.replace("_", " ").title())


def plot_multi_sobols(
    runs: RunsInput,
    qoi: str = "energy_uj",
    mode: str = "grouped_bar",  # "grouped_bar", "stacked_bar", or "heatmap"
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
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
    """
    run_list = _to_run_list(runs)
    if not run_list:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
        empty_ax.set_title("No runs provided")
        return fig

    # Collect Sobol data
    records = []
    for r in run_list:
        sobols = r.get_sobols(qoi)
        for param, val in sobols.items():
            records.append({
                "benchmark": r.benchmark_name,
                "machine": r.machine_name,
                "tag": f"{r.benchmark_name}\n({r.machine_name})",
                "parameter": param,
                "sobol": val
            })

    if not records:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
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
) -> Figure | SubFigure:
    """
    Plot convergence histories (adaptation errors, mean, or standard deviation) across iterations.
    """
    run_list = _to_run_list(runs)
    if not run_list:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
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
) -> Figure | SubFigure:
    """
    Plot energy vs execution time trade-off for multiple runs, highlighting Pareto frontiers.
    """
    run_list = _to_run_list(runs)
    if not run_list:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
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
    ax.set_title(f"Energy vs Time Pareto Frontier Comparison", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Custom legends for Machines and Benchmarks
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=mach_colors[m], lw=2, label=m) for m in machines
    ] + [
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
) -> Figure | SubFigure:
    """
    Boxplot comparison of QoI distribution across machines or benchmarks.
    """
    run_list = _to_run_list(runs)
    if not run_list:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
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
) -> Figure |SubFigure:
    """
    Plot the best configuration (N_THREADS and CLK) found in each run for a given QoI.
    """
    run_list = _to_run_list(runs)
    if not run_list:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
        empty_ax.set_title("No runs provided")
        return fig

    rows = []
    for r in run_list:
        best = r.best_config(qoi, mode="min")
        if best:
            rows.append({
                "benchmark": r.benchmark_name,
                "machine": r.machine_name,
                "tag": r.tag,
                "threads": best.get("N_THREADS", best.get("THREADS", np.nan)),
                "clk": best.get("CLK", best.get("CLK_LEVEL", np.nan)),
                "val": best.get(qoi, np.nan)
            })

    if not rows:
        fig, empty_ax = plt.subplots(figsize=figsize or (6, 4))
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

    ax.set_xlabel("Optimal Threads (N_THREADS)", fontsize=11)
    ax.set_ylabel("Optimal Clock Frequency Index (CLK)", fontsize=11)
    ax.set_title(f"Optimal Configurations Found Across Runs (Min {_format_qoi_label(qoi)})", fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=mach_colors[m], marker="o", lw=0, label=m, markersize=8) for m in machines
    ] + [
        Line2D([0], [0], marker=bench_markers[b], color="gray", label=b, linestyle="none", markersize=8)
        for b in benchmarks
    ]
    ax.legend(handles=legend_elements, title="Machine / Benchmark", bbox_to_anchor=(1.04, 1), loc="upper left")

    assert fig is not None
    return fig


def plot_multi_parameter_effects(
    runs: RunsInput,
    param: str = "N_THREADS",
    qoi: str = "energy_uj",
    facet_by: str = "benchmark",  # "benchmark" or "machine"
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """
    Plot the effect / response curve of varying a specific parameter on the QoI across runs.
    """
    run_list = _to_run_list(runs)
    scale = 1e-6 if qoi == "energy_uj" else 1.0

    group_col = facet_by
    group_values = sorted(list({getattr(r, f"{group_col}_name") for r in run_list}))
    n_groups = len(group_values)

    fig_size = figsize or (5.5 * n_groups, 4.5)
    fig, axes = plt.subplots(1, n_groups, figsize=fig_size, sharey=True, squeeze=False, layout="constrained")

    other_col = "machine" if facet_by == "benchmark" else "benchmark"

    for g_idx, g_val in enumerate(group_values):
        cur_ax = axes[0, g_idx]
        cur_runs = [r for r in run_list if getattr(r, f"{group_col}_name") == g_val]

        for r in cur_runs:
            if param in r.df.columns and qoi in r.df.columns:
                # Group by parameter value and compute mean response
                grouped = r.df.groupby(param)[qoi].agg(["mean", "std"]).reset_index()
                cur_ax.errorbar(
                    grouped[param],
                    grouped["mean"] * scale,
                    yerr=grouped["std"] * scale,
                    marker="o",
                    capsize=3,
                    label=getattr(r, f"{other_col}_name")
                )

        cur_ax.set_title(f"{group_col.title()}: {g_val}")
        cur_ax.set_xlabel(param)
        cur_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        cur_ax.grid(True, linestyle="--", alpha=0.5)
        cur_ax.legend(title=other_col.title(), fontsize=8)

    y_label = "Energy (J)" if qoi == "energy_uj" else _format_qoi_label(qoi)
    axes[0, 0].set_ylabel(f"Mean {y_label}", fontsize=11)
    fig.suptitle(f"Effect of {param} on {y_label} across {facet_by.title()}s", fontsize=13)

    return fig


def plot_multi_dashboard(
    runs: RunsInput,
    qoi: str = "energy_uj",
    figsize: tuple[float, float] = (15.0, 11.0),
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
    fig.suptitle(f"Multi-Run Comparative Analysis Dashboard ({_format_qoi_label(qoi)})", fontsize=15, fontweight="bold")

    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # 1. Sobol sensitivities
    plot_multi_sobols(run_list, qoi=qoi, mode="stacked_bar", ax=ax1)

    # 2. Pareto frontiers
    plot_multi_energy_time_pareto(run_list, qoi_x="time", qoi_y=qoi, ax=ax2)

    # 3. Convergence
    plot_multi_convergence(run_list, metric="adaptation_error", ax=ax3)

    # 4. Best configurations
    plot_multi_best_configurations(run_list, qoi=qoi, ax=ax4)

    return fig
