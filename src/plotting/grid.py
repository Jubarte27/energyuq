from collections.abc import Sequence
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors

from ..machines.machine import Machine
from ..util.data import EasyResult, Result, limit
from .layout import (
    mostly_square_grid,
    pad_to_even_and_split,
    get_machine,
)
from .units import _is_integer_range


class PlotterGridMixin:
    """Mixin providing 2D and 1D projection and distribution plots for Plotter."""

    machine: Machine | None
    labels: np.ndarray
    values: np.ndarray

    def init(self, mach: Machine, units: dict[str, Any] | None = None, active_params: list[str] | None = None): ...
    def _ensure_real_clk_limits(self, mach: Machine | None = None, units: dict[str, Any] | None = None): ...
    def get_result_params(self, result: Any, df: DataFrame | None = None) -> list[str]: ...
    def get_axis_label(self, param: str, units: dict[str, Any] | None = None) -> str: ...
    def to_real_clk(self, val: Any, mach: Machine | None = None, units: dict[str, Any] | None = None) -> Any: ...
    def convert_clk_df(self, df: DataFrame, mach: Machine | None = None, units: dict[str, Any] | None = None) -> DataFrame: ...
    def get_axis_bounds(self, val_low: float, val_high: float, data: Any) -> tuple[float, float]: ...
    def key_for(self, result: Result, qoi: str) -> str | tuple[str, int]: ...
    def colors_for(self, qoi: str) -> dict[str, tuple[str, str, str, str]]: ...

    def _prepare_context(
        self,
        result: Any,
        units: dict[str, Any] | None = None,
        qoi: str | None = None,
    ) -> tuple[Machine | None, str, Any, DataFrame]:
        mach = get_machine(result) or self.machine
        if self.machine is None and mach is not None:
            self.init(mach, units=units)
        self._ensure_real_clk_limits(mach, units)

        resolved_qoi = qoi if qoi is not None else (result.qois[0] if hasattr(result, "qois") and result.qois else "")
        key = self.key_for(result, resolved_qoi) if resolved_qoi and hasattr(self, "key_for") else resolved_qoi
        raw_df = getattr(result, "df", None)
        df = self.convert_clk_df(raw_df, mach, units) if raw_df is not None else DataFrame()
        return mach, resolved_qoi, key, df

    def _get_param_bounds(self, lbl: str) -> tuple[float, float]:
        if len(self.labels) > 0 and lbl in self.labels:
            lim = self.values[list(self.labels).index(lbl)]
            return float(lim.lower), float(lim.upper)
        return 0.0, 1.0

    def _setup_pairwise_grid_layout(
        self, cur_labels: Sequence[str], units: dict[str, Any] | None = None
    ) -> tuple[int, int, tuple[float, float], list[int], np.ndarray, np.ndarray]:
        cur_values = np.array(
            [limit(*self._get_param_bounds(lbl)) for lbl in cur_labels],
            dtype=limit,
        )
        cur_L = (len(cur_labels) + 1) // 2
        (cur_C, cur_R), cur_fig_size = mostly_square_grid(cur_L, 6, 2)
        cur_full_rows = cur_L // cur_C if cur_C > 0 else 0
        rem = cur_L % cur_C if cur_C > 0 else 0
        row_col_counts = [cur_C] * cur_full_rows + ([rem] if rem > 0 else [])

        cur_nd_values = pad_to_even_and_split(cur_values, value=limit(lower=0, upper=1))
        axis_labels = np.array([self.get_axis_label(lbl, units) for lbl in cur_labels], dtype=str)
        cur_nd_labels = pad_to_even_and_split(axis_labels, value="")
        return cur_L, cur_R, cur_fig_size, row_col_counts, cur_nd_values, cur_nd_labels

    def _setup_subgrid_dims(self, L: int) -> tuple[int, int]:
        if L <= 1:
            return 1, 1
        if L == 2:
            return 2, 1
        return 2, int(np.ceil(L / 2))

    def plot_grid_2D(self, result: EasyResult, units: dict[str, str | None] | None = None) -> Figure:
        """Plot chosen sampling grid in 2D pairwise projections."""
        analysis = result.analysis
        sampler = result.sampler

        mach, _, _, _ = self._prepare_context(result, units=units)

        cur_labels = self.get_result_params(result)
        cur_L, cur_R, cur_fig_size, row_col_counts, cur_nd_values, cur_nd_labels = self._setup_pairwise_grid_layout(
            cur_labels, units
        )

        raw_grid = sampler.generate_grid(analysis.l_norm).astype(object)
        if raw_grid.ndim == 2:
            if raw_grid.shape[1] == len(cur_labels):
                for col_idx, lbl in enumerate(cur_labels):
                    if str(lbl).upper() in ("CLK", "CLK_LEVEL") and mach is not None and hasattr(mach, "freq"):
                        raw_grid[:, col_idx] = [self.to_real_clk(v, mach, units) for v in raw_grid[:, col_idx]]
            elif raw_grid.shape[0] == len(cur_labels):
                for row_idx, lbl in enumerate(cur_labels):
                    if str(lbl).upper() in ("CLK", "CLK_LEVEL") and mach is not None and hasattr(mach, "freq"):
                        raw_grid[row_idx, :] = [self.to_real_clk(v, mach, units) for v in raw_grid[row_idx, :]]

        accepted_grid = pad_to_even_and_split(raw_grid, value=0)

        fig = plt.figure(figsize=cur_fig_size, layout="constrained")
        fig.supylabel("Configurations chosen")

        ax: list[Axes] = []
        i = 0
        for r, cols in enumerate(row_col_counts):
            for c in range(cols):
                subplot_num = r * cols + c + 1
                xs_data = accepted_grid[:, 0, i] if accepted_grid.ndim == 3 and i < accepted_grid.shape[2] else None
                ys_data = accepted_grid[:, 1, i] if accepted_grid.ndim == 3 and i < accepted_grid.shape[2] else None

                xlim = self.get_axis_bounds(cur_nd_values[0, i].lower, cur_nd_values[0, i].upper, xs_data)
                ylim = self.get_axis_bounds(cur_nd_values[1, i].lower, cur_nd_values[1, i].upper, ys_data)

                ax.append(
                    fig.add_subplot(
                        cur_R,
                        cols,
                        subplot_num,
                        xlim=list(xlim),
                        ylim=list(ylim),
                        xlabel=cur_nd_labels[0, i],
                        ylabel=cur_nd_labels[1, i],
                    )
                )
                x_is_int = _is_integer_range(cur_nd_values[0, i].lower, cur_nd_values[0, i].upper)
                y_is_int = _is_integer_range(cur_nd_values[1, i].lower, cur_nd_values[1, i].upper)
                ax[-1].xaxis.set_major_locator(MaxNLocator(integer=x_is_int))
                ax[-1].yaxis.set_major_locator(MaxNLocator(integer=y_is_int))
                ax[-1].set_box_aspect(1)
                ax[-1].set_anchor('N')
                i += 1

        for ic in range(cur_L):
            ax[ic].plot(accepted_grid[:, 0, ic], accepted_grid[:, 1, ic], 'o', alpha=0.25)

        return fig

    def plot_grid_2D_best(
        self,
        result: Result,
        qoi: str | None = None,
        order_focus: bool = False,
        subfig: SubFigure | None = None,
        units: dict[str, str | None] | None = None,
    ) -> Figure | SubFigure:
        """Plot pairwise 2D parameter evaluations colored by QoI cost."""
        mach, qoi, key, df = self._prepare_context(result, units=units, qoi=qoi)
        pretty_colors = self.colors_for(qoi)

        cur_labels = self.get_result_params(result, df)
        cur_L, cur_R, cur_fig_size, row_col_counts, cur_nd_values, cur_nd_labels = self._setup_pairwise_grid_layout(
            cur_labels, units
        )

        if subfig is None:
            fig = plt.figure(figsize=cur_fig_size, layout="constrained")
            fig.supylabel(f"Configurations evaluated by {qoi}")
        else:
            fig = subfig

        try:
            if df.empty:
                ax_empty = fig.add_subplot(1, 1, 1)
                ax_empty.text(0.5, 0.5, "No evaluation data available", ha="center", va="center", transform=ax_empty.transAxes)
                return fig

            dataframe: DataFrame = df.sort_values(by=key, ascending=False)
            column = dataframe[qoi]
            if isinstance(column, pd.DataFrame):
                column = column.iloc[:, 0]

            Q1 = column.quantile(0.25)
            Q3 = column.quantile(0.75)
            IQR = Q3 - Q1

            high_outlier_mask = column > (Q3 + 1.5 * IQR)
            low_outlier_mask = column < (Q1 - 1.5 * IQR)
            outlier_mask = high_outlier_mask | low_outlier_mask
            inlier_mask = ~outlier_mask

            inlier_column = column[inlier_mask]
            inlier_min = inlier_column.min() if not inlier_column.empty else column.min()
            inlier_max = inlier_column.max() if not inlier_column.empty else column.max()

            if hasattr(inlier_min, "item"):
                inlier_min = inlier_min.item()
            if hasattr(inlier_max, "item"):
                inlier_max = inlier_max.item()

            if inlier_max == inlier_min:
                norm_vals = pd.Series(0.5, index=dataframe.index)
            else:
                norm_vals = (column - inlier_min) / (inlier_max - inlier_min)

            dataframe[f"{qoi}_norm"] = norm_vals.clip(0, 1)

            custom_handles = []
            legend_labels = []

            if order_focus:
                cmap = mcolors.LinearSegmentedColormap.from_list("ba", list(pretty_colors['highest_lowest'][::2]))
                c_low, c_high = cmap(0.0), cmap(1.0)
                custom_handles.append(Line2D([], [], color='w', marker='o', markerfacecolor=c_low, markersize=8))
                legend_labels.append("Lower ranked")
                custom_handles.append(Line2D([], [], color='w', marker='o', markerfacecolor=c_high, markersize=8))
                legend_labels.append("Higher ranked")
            else:
                cmap = mcolors.LinearSegmentedColormap.from_list("ba", list(pretty_colors['high_low'][::2]))
                c_low, c_high = cmap(0.0), cmap(1.0)
                custom_handles.append(Line2D([], [], color='w', marker='o', markerfacecolor=c_low, markersize=8))
                legend_labels.append(f"Lesser {qoi}")
                custom_handles.append(Line2D([], [], color='w', marker='o', markerfacecolor=c_high, markersize=8))
                legend_labels.append(f"Higher {qoi}")

            if high_outlier_mask.any():
                custom_handles.append(Line2D([], [], color='w', marker='o', markerfacecolor='magenta', markersize=8))
                legend_labels.append("High outlier")

            if low_outlier_mask.any():
                custom_handles.append(Line2D([], [], color='w', marker='o', markerfacecolor='cyan', markersize=8))
                legend_labels.append("Low outlier")

            colors = []
            for idx in dataframe.index:
                if high_outlier_mask[idx]:
                    colors.append("magenta")
                elif low_outlier_mask[idx]:
                    colors.append("cyan")
                else:
                    colors.append(cmap(dataframe.loc[idx, f"{qoi}_norm"]))

            ax: list[Axes] = []
            i = 0
            for r, cols in enumerate(row_col_counts):
                for c in range(cols):
                    subplot_num = r * cols + c + 1
                    col_x = cur_labels[i * 2]
                    xs = dataframe[col_x].to_numpy().flatten()
                    if i * 2 + 1 < len(cur_labels):
                        col_y = cur_labels[i * 2 + 1]
                        ylabel_text = cur_nd_labels[1, i]
                    else:
                        col_y = cur_labels[0]
                        ylabel_text = cur_nd_labels[0, 0]
                    ys = dataframe[col_y].to_numpy().flatten()
                    y_low = cur_nd_values[1, i].lower
                    y_high = cur_nd_values[1, i].upper
                    ylim = self.get_axis_bounds(y_low, y_high, ys)

                    xlim = self.get_axis_bounds(cur_nd_values[0, i].lower, cur_nd_values[0, i].upper, xs)

                    ax.append(
                        fig.add_subplot(
                            cur_R,
                            cols,
                            subplot_num,
                            xlim=list(xlim),
                            ylim=list(ylim),
                            xlabel=cur_nd_labels[0, i],
                            ylabel=ylabel_text,
                        )
                    )
                    x_is_int = _is_integer_range(cur_nd_values[0, i].lower, cur_nd_values[0, i].upper)
                    y_is_int = _is_integer_range(cur_nd_values[1, i].lower, cur_nd_values[1, i].upper)
                    ax[-1].xaxis.set_major_locator(MaxNLocator(integer=x_is_int))
                    ax[-1].yaxis.set_major_locator(MaxNLocator(integer=y_is_int))
                    ax[-1].set_box_aspect(1)
                    ax[-1].set_anchor('N')

                    ax[-1].legend(
                        handles=custom_handles,
                        labels=legend_labels,
                        draggable=True,
                        fontsize='x-small',
                        ncols=2,
                        bbox_to_anchor=(1, 1.1),
                        loc='upper right',
                    )
                    ax[-1].scatter(xs, ys, c=colors)
                    i += 1

            return fig
        except Exception:
            if subfig is None:
                plt.close(fig)
            raise

    def plot_sorted(
        self,
        result: Result,
        qoi: str | None = None,
        subfig: SubFigure | None = None,
        title: str | None = None,
    ) -> Figure | SubFigure:
        """Plot evaluations sorted monotonically by QoI cost."""
        if qoi is None:
            qoi = result.qois[0]
        key = self.key_for(result, qoi)
        dataframe = result.df

        if not subfig:
            fig, ax = plt.subplots(figsize=(12, 12), layout="constrained")
        else:
            fig = subfig
            ax = fig.subplots()

        try:
            if dataframe.empty:
                ax.text(0.5, 0.5, "No evaluation data available", ha="center", va="center", transform=ax.transAxes)
                return fig

            sorted_dataframe: DataFrame = dataframe.sort_values(by=key, ascending=False)
            ax.semilogy(sorted_dataframe[key].to_numpy())
            ax.set_box_aspect(1)
            ax.set_anchor('N')
            if not subfig and title:
                ax.set_title(title)
            return fig
        except Exception:
            if not subfig:
                plt.close(fig)
            raise

    def plot_2D_single_dimension(
        self,
        result: Result,
        qoi: str | None = None,
        subfig: SubFigure | None = None,
        units: dict[str, str | None] | None = None,
    ) -> Figure | SubFigure:
        """Plot 2D projections of each parameter against the QoI."""
        mach, qoi, key, df = self._prepare_context(result, units=units, qoi=qoi)
        cur_labels = self.get_result_params(result, df)
        L = len(cur_labels)
        C, R = self._setup_subgrid_dims(L)

        if subfig:
            fig = subfig
        else:
            fig = plt.figure(figsize=(12, max(4.0, 12 / C * R)), layout="constrained")
            fig.supylabel(f"For {qoi}")

        try:
            if df.empty:
                ax_empty = fig.add_subplot(1, 1, 1)
                ax_empty.text(0.5, 0.5, "No evaluation data available", ha="center", va="center", transform=ax_empty.transAxes)
                return fig

            ax_grid = fig.subplots(R, C, sharey=False)
            ax = list(ax_grid.flatten()) if isinstance(ax_grid, np.ndarray) else [ax_grid]

            for i in range(L):
                lbl = cur_labels[i]
                xs = df[lbl].to_numpy().flatten()
                ys = df[key].to_numpy().flatten()

                val_low, val_high = self._get_param_bounds(lbl)
                xlim = self.get_axis_bounds(val_low, val_high, xs)

                ax[i].set_xlim(xlim)
                ax[i].set_xlabel(xlabel=self.get_axis_label(lbl, units))
                ax[i].set_ylabel(qoi)
                is_int = _is_integer_range(val_low, val_high)
                ax[i].xaxis.set_major_locator(MaxNLocator(integer=is_int))
                ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))
                ax[i].set_box_aspect(1)
                ax[i].set_anchor('N')
                ax[i].scatter(xs, ys)

            for j in range(L, len(ax)):
                fig.delaxes(ax[j])

            return fig
        except Exception:
            if not subfig:
                plt.close(fig)
            raise

    def plot_boxplot(
        self,
        result: Result,
        qoi: str | None = None,
        units: dict[str, str | None] | None = None,
    ) -> Figure:
        """Plot boxplots per discrete parameter level against the QoI."""
        mach, qoi, key, df = self._prepare_context(result, units=units, qoi=qoi)
        cur_labels = self.get_result_params(result, df)
        L = len(cur_labels)
        C, R = self._setup_subgrid_dims(L)

        fig = plt.figure(figsize=(12, max(4.0, 12 / C * R)), layout="constrained")
        fig.supylabel(f"For {qoi}")

        try:
            if df.empty:
                ax_empty = fig.add_subplot(1, 1, 1)
                ax_empty.text(0.5, 0.5, "No evaluation data available", ha="center", va="center", transform=ax_empty.transAxes)
                return fig

            ax: list[Axes] = []
            for i in range(L):
                lbl = cur_labels[i]
                val_low, val_high = self._get_param_bounds(lbl)
                xlim = self.get_axis_bounds(val_low, val_high, df[lbl])
                ax.append(
                    fig.add_subplot(
                        R,
                        C,
                        i + 1,
                        xlim=list(xlim),
                        xlabel=self.get_axis_label(lbl, units),
                        ylabel=qoi,
                    )
                )
                is_int = _is_integer_range(val_low, val_high)
                ax[-1].xaxis.set_major_locator(MaxNLocator(integer=is_int))
                ax[-1].yaxis.set_major_locator(MaxNLocator(integer=True))

            def col_to_numpy(x: DataFrame):
                return x[key].to_numpy()

            for i in range(L):
                lbl = cur_labels[i]
                label_c = (lbl, 0) if isinstance(key, tuple) else lbl
                box_frame = df[[key, label_c]].groupby(by=label_c)[[key]].apply(col_to_numpy)

                positions = np.array(box_frame.index.to_list(), dtype=float)
                diffs = np.diff(np.sort(positions))
                if len(diffs) > 0 and np.min(diffs) > 0:
                    width = 0.5 * float(np.min(diffs))
                else:
                    width = 0.5

                ax[i].boxplot(
                    box_frame.to_numpy(),
                    positions=positions,
                    widths=width,
                    patch_artist=True,
                    boxprops={"facecolor": "lightblue", "edgecolor": "C0", "linewidth": 1.5},
                    medianprops={"color": "darkblue", "linewidth": 2},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5},
                )

            return fig
        except Exception:
            plt.close(fig)
            raise


# Standalone module-level functions delegating via Plotter.from_result
def plot_grid_2D(result: EasyResult, units: dict[str, str | None] | None = None) -> Figure:
    from .plotter import Plotter
    plotter = Plotter.from_result(result, units=units)
    return plotter.plot_grid_2D(result, units=units)


def plot_grid_2D_best(
    result: Result,
    qoi: str | None = None,
    order_focus: bool = False,
    subfig: SubFigure | None = None,
    units: dict[str, str | None] | None = None,
) -> Figure | SubFigure:
    from .plotter import Plotter
    plotter = Plotter.from_result(result, units=units)
    return plotter.plot_grid_2D_best(result, qoi=qoi, order_focus=order_focus, subfig=subfig, units=units)


def plot_sorted(
    result: Result,
    qoi: str | None = None,
    subfig: SubFigure | None = None,
    title: str | None = None,
) -> Figure | SubFigure:
    from .plotter import Plotter
    plotter = Plotter.from_result(result)
    return plotter.plot_sorted(result, qoi=qoi, subfig=subfig, title=title)


def plot_2D_single_dimension(
    result: Result,
    qoi: str | None = None,
    subfig: SubFigure | None = None,
    units: dict[str, str | None] | None = None,
) -> Figure | SubFigure:
    from .plotter import Plotter
    plotter = Plotter.from_result(result, units=units)
    return plotter.plot_2D_single_dimension(result, qoi=qoi, subfig=subfig, units=units)


def plot_boxplot(
    result: Result,
    qoi: str | None = None,
    units: dict[str, str | None] | None = None,
) -> Figure:
    from .plotter import Plotter
    plotter = Plotter.from_result(result, units=units)
    return plotter.plot_boxplot(result, qoi=qoi, units=units)

