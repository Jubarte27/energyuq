from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from pandas import DataFrame

from ..machines.machine import Machine
from ..util.data import EasyResult, Result, limit
from .layout import (
    get_machine,
    mostly_square_grid,
    pad_to_even_and_split,
)
from .units import is_integer_range

SQUARE_GRID_BLOCK_SIZE = 8

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
        self, param_labels: Sequence[str], units: dict[str, Any] | None = None, width: int | None = None
    ) -> tuple[int, int, tuple[float, float], list[int], np.ndarray, np.ndarray]:
        """Compute grid dimensions and paired bounds/labels for pairwise 2D plots."""
        bounds = np.array(
            [limit(*self._get_param_bounds(lbl)) for lbl in param_labels],
            dtype=limit,
        )
        if width is None:
            width = 16
        num_pairs = (len(param_labels) + 1) // 2
        (grid_cols, grid_rows), fig_size = mostly_square_grid(num_pairs, width, SQUARE_GRID_BLOCK_SIZE)

        full_rows = num_pairs // grid_cols if grid_cols > 0 else 0
        remainder = num_pairs % grid_cols if grid_cols > 0 else 0
        cols_per_row = [grid_cols] * full_rows + ([remainder] if remainder > 0 else [])

        paired_bounds = pad_to_even_and_split(bounds, value=limit(lower=0, upper=1))
        axis_labels = np.array([self.get_axis_label(lbl, units) for lbl in param_labels], dtype=str)
        paired_labels = pad_to_even_and_split(axis_labels, value="")
        return num_pairs, grid_rows, fig_size, cols_per_row, paired_bounds, paired_labels

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

        param_names = self.get_result_params(result)
        num_pairs, grid_rows, fig_size, cols_per_row, paired_bounds, paired_labels = (
            self._setup_pairwise_grid_layout(param_names, units)
        )

        raw_grid = sampler.generate_grid(analysis.l_norm).astype(object)
        if raw_grid.ndim == 2:
            if raw_grid.shape[1] == len(param_names):
                for col_idx, lbl in enumerate(param_names):
                    if str(lbl).upper() in ("CLK", "CLK_LEVEL") and mach is not None and hasattr(mach, "freq"):
                        raw_grid[:, col_idx] = [self.to_real_clk(v, mach, units) for v in raw_grid[:, col_idx]]
            elif raw_grid.shape[0] == len(param_names):
                for row_idx, lbl in enumerate(param_names):
                    if str(lbl).upper() in ("CLK", "CLK_LEVEL") and mach is not None and hasattr(mach, "freq"):
                        raw_grid[row_idx, :] = [self.to_real_clk(v, mach, units) for v in raw_grid[row_idx, :]]

        split_grid = pad_to_even_and_split(raw_grid, value=0)

        fig = plt.figure(figsize=fig_size, layout="constrained")
        fig.supylabel("Configurations chosen")

        axes: list[Axes] = []
        pair_idx = 0
        for row, n_cols in enumerate(cols_per_row):
            for col in range(n_cols):
                subplot_pos = row * n_cols + col + 1
                xs = split_grid[:, 0, pair_idx] if split_grid.ndim == 3 and pair_idx < split_grid.shape[2] else None
                ys = split_grid[:, 1, pair_idx] if split_grid.ndim == 3 and pair_idx < split_grid.shape[2] else None

                x_bounds = paired_bounds[0, pair_idx]
                y_bounds = paired_bounds[1, pair_idx]
                xlim = self.get_axis_bounds(x_bounds.lower, x_bounds.upper, xs)
                ylim = self.get_axis_bounds(y_bounds.lower, y_bounds.upper, ys)

                ax = fig.add_subplot(
                    grid_rows, n_cols, subplot_pos,
                    xlim=list(xlim), ylim=list(ylim),
                    xlabel=paired_labels[0, pair_idx], ylabel=paired_labels[1, pair_idx],
                )
                ax.xaxis.set_major_locator(MaxNLocator(integer=is_integer_range(x_bounds.lower, x_bounds.upper)))
                ax.yaxis.set_major_locator(MaxNLocator(integer=is_integer_range(y_bounds.lower, y_bounds.upper)))
                ax.set_box_aspect(1)
                ax.set_anchor('N')
                axes.append(ax)
                pair_idx += 1

        for pair_idx in range(num_pairs):
            axes[pair_idx].plot(split_grid[:, 0, pair_idx], split_grid[:, 1, pair_idx], 'o', alpha=0.25)

        return fig

    def _classify_outliers(self, series: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Return (high_outlier_mask, low_outlier_mask) using the IQR method."""
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return series > (q3 + 1.5 * iqr), series < (q1 - 1.5 * iqr)

    def _normalize_to_inliers(self, series: pd.Series, is_outlier: pd.Series) -> pd.Series:
        """Normalize *series* to [0, 1] based on inlier min/max, clamping outliers."""
        inliers = series[~is_outlier]
        lo = (inliers.min() if not inliers.empty else series.min())
        hi = (inliers.max() if not inliers.empty else series.max())
        lo = lo.item() if hasattr(lo, "item") else lo
        hi = hi.item() if hasattr(hi, "item") else hi
        if hi == lo:
            return pd.Series(0.5, index=series.index)
        return ((series - lo) / (hi - lo)).clip(0, 1)

    def _build_legend_and_colors(
        self, qoi: str, order_focus: bool, high_mask: pd.Series, low_mask: pd.Series,
        normalized: pd.Series,
    ) -> tuple[list[Line2D], list[str], list]:
        """Build legend handles and per-point colors from outlier masks and normalized values."""
        pretty = self.colors_for(qoi)

        def _marker(facecolor):
            return Line2D([], [], color='w', marker='o', markerfacecolor=facecolor, markersize=8)

        style_key = 'highest_lowest' if order_focus else 'high_low'
        cmap = mcolors.LinearSegmentedColormap.from_list("ba", list(pretty[style_key][::2]))

        labels_low, labels_high = (
            ("Lower ranked", "Higher ranked") if order_focus
            else (f"Lesser {qoi}", f"Higher {qoi}")
        )
        handles = [_marker(cmap(0.0)), _marker(cmap(1.0))]
        labels = [labels_low, labels_high]

        if high_mask.any():
            handles.append(_marker('magenta'))
            labels.append("High outlier")
        if low_mask.any():
            handles.append(_marker('cyan'))
            labels.append("Low outlier")

        point_colors = [
            "magenta" if high_mask[i] else "cyan" if low_mask[i] else cmap(normalized[i])
            for i in normalized.index
        ]
        return handles, labels, point_colors

    def plot_grid_2D_best(
        self,
        result: Result,
        qoi: str | None = None,
        order_focus: bool = False,
        subfig: SubFigure | None = None,
        units: dict[str, str | None] | None = None,
    ) -> Figure | SubFigure:
        """Plot pairwise 2D parameter evaluations colored by QoI cost."""
        _, qoi, key, df = self._prepare_context(result, units=units, qoi=qoi)
        param_names = self.get_result_params(result, df)
        width = subfig.bbox_relative.width * subfig.figure.get_size_inches()[0] if subfig else None
        _, grid_rows, fig_size, row_col_counts, paired_bounds, paired_labels = (
            self._setup_pairwise_grid_layout(param_names, units, width=width)
        )

        fig = subfig or plt.figure(figsize=fig_size, layout="constrained")
        if subfig is None:
            fig.supylabel(f"Configurations evaluated by {qoi}")

        try:
            if df.empty:
                ax_empty = fig.add_subplot(1, 1, 1)
                ax_empty.text(0.5, 0.5, "No evaluation data available",
                              ha="center", va="center", transform=ax_empty.transAxes)
                return fig

            sorted_df: DataFrame = df.sort_values(by=key, ascending=False)
            qoi_series = sorted_df[qoi]
            if isinstance(qoi_series, pd.DataFrame):
                qoi_series = qoi_series.iloc[:, 0]

            high_mask, low_mask = self._classify_outliers(qoi_series)
            normalized = self._normalize_to_inliers(qoi_series, high_mask | low_mask)
            handles, labels, point_colors = self._build_legend_and_colors(
                qoi, order_focus, high_mask, low_mask, normalized,
            )

            pair_idx = 0
            for row, n_cols in enumerate(row_col_counts):
                for col in range(n_cols):
                    x_param = param_names[pair_idx * 2]
                    if pair_idx * 2 + 1 < len(param_names):
                        y_param = param_names[pair_idx * 2 + 1]
                        y_label = paired_labels[1, pair_idx]
                    else:
                        y_param = param_names[0]
                        y_label = paired_labels[0, 0]

                    xs = sorted_df[x_param].to_numpy().flatten()
                    ys = sorted_df[y_param].to_numpy().flatten()

                    x_bounds = paired_bounds[0, pair_idx]
                    y_bounds = paired_bounds[1, pair_idx]
                    xlim = self.get_axis_bounds(x_bounds.lower, x_bounds.upper, xs)
                    ylim = self.get_axis_bounds(y_bounds.lower, y_bounds.upper, ys)

                    subplot_pos = row * n_cols + col + 1
                    ax = fig.add_subplot(
                        grid_rows, n_cols, subplot_pos,
                        xlim=list(xlim), ylim=list(ylim),
                        xlabel=paired_labels[0, pair_idx], ylabel=y_label,
                    )
                    ax.xaxis.set_major_locator(MaxNLocator(integer=is_integer_range(x_bounds.lower, x_bounds.upper)))
                    ax.yaxis.set_major_locator(MaxNLocator(integer=is_integer_range(y_bounds.lower, y_bounds.upper)))
                    ax.set_box_aspect(1)
                    ax.set_anchor('N')
                    ax.legend(handles=handles, labels=labels, draggable=True,
                              fontsize='x-small', ncols=2, bbox_to_anchor=(1, 1.1), loc='upper right')
                    ax.scatter(xs, ys, c=point_colors)
                    pair_idx += 1

            return fig
        finally:
            if subfig is None:
                plt.close(fig)

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
        finally:
            if subfig is None:
                plt.close(fig)

    def plot_2D_single_dimension(
        self,
        result: Result,
        qoi: str | None = None,
        subfig: SubFigure | None = None,
        units: dict[str, str | None] | None = None,
    ) -> Figure | SubFigure:
        """Plot 2D projections of each parameter against the QoI."""
        _, qoi, key, df = self._prepare_context(result, units=units, qoi=qoi)
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
                is_int = is_integer_range(val_low, val_high)
                ax[i].xaxis.set_major_locator(MaxNLocator(integer=is_int))
                ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))
                ax[i].set_box_aspect(1)
                ax[i].set_anchor('N')
                ax[i].scatter(xs, ys)

            for j in range(L, len(ax)):
                fig.delaxes(ax[j])

            return fig
        finally:
            if subfig is None:
                plt.close(fig)

    def plot_boxplot(
        self,
        result: Result,
        qoi: str | None = None,
        units: dict[str, str | None] | None = None,
    ) -> Figure:
        """Plot boxplots per discrete parameter level against the QoI."""
        _, qoi, key, df = self._prepare_context(result, units=units, qoi=qoi)
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
                is_int = is_integer_range(val_low, val_high)
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
        finally:
            plt.close(fig)


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

