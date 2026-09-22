from collections.abc import Callable
from math import floor, ceil, sqrt
from typing import Any, Sequence

from matplotlib.figure import Figure, SubFigure
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

import pandas as pd
from pandas import DataFrame

from .. import energyuq
from ..machines.machine import Machine
from .data import EasyResult, Result, limit


def pad_to_even_and_split(arr: np.ndarray, value=None) -> np.ndarray:
    arr = np.asarray(arr)
    pad_by = arr.shape[-1] % 2
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, pad_by)]
    padded = np.pad(arr, pad_width, mode='constant', constant_values=value)
    new_shape = [*padded.shape[:-1], -1, 2]
    return padded.reshape(new_shape).swapaxes(-2, -1)


def mostly_square_grid(blocks: int, total_width: float, min_block_width: float):
    max_col = floor(total_width / min_block_width)
    cols = min(ceil(sqrt(blocks)), max_col)
    rows = ceil(blocks / cols)

    total_height = rows * total_width
    return (cols, rows), (total_width, total_height)


SI_PREFIX_FACTORS: dict[str, float] = {
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "μ": 1e-6,
    "m": 1e-3,
    "c": 1e-2,
    "d": 1e-1,
    "": 1.0,
    "k": 1e3,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
}


def parse_unit_spec(spec: Any) -> tuple[str | None, str | None, float | Callable[[float], float] | None]:
    """
    Parse a unit specification into (from_unit, to_unit, explicit_scale_or_func).
    Supported formats:
        - 'GHz' -> (None, 'GHz', None)
        - 'Hz to GHz' or 'Hz->GHz' -> ('Hz', 'GHz', None)
        - ('Hz', 'GHz') -> ('Hz', 'GHz', None)
        - ('GHz', 1e-6) -> (None, 'GHz', 1e-6)
        - ('GHz', func) -> (None, 'GHz', func)
        - {'from': 'Hz', 'to': 'GHz'} -> ('Hz', 'GHz', None)
        - {'unit': 'GHz', 'scale': 1e-6} -> (None, 'GHz', 1e-6)
        - func -> (None, None, func)
        - None -> (None, None, None)
    """
    if spec is None:
        return None, None, None
    if callable(spec):
        return None, None, spec
    if isinstance(spec, (tuple, list)):
        if len(spec) == 2:
            second = spec[1]
            if isinstance(second, (int, float)) or callable(second):
                u = str(spec[0]).strip() if spec[0] is not None else None
                return None, u, second
            else:
                from_u = str(spec[0]).strip() if spec[0] is not None else None
                to_u = str(spec[1]).strip() if spec[1] is not None else None
                return from_u, to_u, None
        elif len(spec) == 1:
            u = str(spec[0]).strip() if spec[0] is not None else None
            return None, u, None
    if isinstance(spec, dict):
        scale = spec.get("scale")
        to_u = spec.get("to") or spec.get("unit")
        from_u = spec.get("from")
        return from_u, to_u, scale
    if isinstance(spec, str):
        import re
        s = spec.strip()
        if not s or s.lower() == "none":
            return None, None, None
        if "->" in s:
            parts = s.split("->", 1)
            return parts[0].strip(), parts[1].strip(), None
        if re.search(r'\s+to\s+', s, flags=re.IGNORECASE):
            parts = re.split(r'\s+to\s+', s, flags=re.IGNORECASE)
            return parts[0].strip(), parts[1].strip(), None
        return None, s, None
    return None, str(spec).strip(), None


DEFAULT_PARAM_UNITS: dict[str, Any] = {"CLK": "Hz"}


def _is_integer_range(lower: Any, upper: Any) -> bool:
    try:
        f_low = float(lower)
        f_up = float(upper)
        if f_up - f_low < 1:
            return False
        return f_low.is_integer() and f_up.is_integer()
    except Exception:
        return False


def get_machine(result: Any = None) -> Machine | None:
    """Retrieve Machine instance associated with result or campaign."""
    if result is not None:
        if hasattr(result, "machine") and isinstance(result.machine, Machine):
            return result.machine
        if hasattr(result, "sampler") and hasattr(result.sampler, "machine") and isinstance(result.sampler.machine, Machine):
            return result.sampler.machine
        if hasattr(result, "campaign") and hasattr(result.campaign, "machine") and isinstance(result.campaign.machine, Machine):
            return result.campaign.machine
    return None


class Plotter:
    """
    Self-contained plotter for single-run and multi-dimensional EnergyUQ evaluation results.
    
    Encapsulates all visualization logic, machine parameter scaling, axis bound calculations,
    and figure formatting without relying on global state.
    """

    def __init__(
        self,
        machine: Machine | None = None,
        units: dict[str, Any] | None = None,
        active_params: list[str] | None = None,
    ):
        self.machine: Machine | None = machine
        self.units: dict[str, Any] = dict(DEFAULT_PARAM_UNITS)
        if units:
            self.units.update(units)
        self.active_params: list[str] | None = list(active_params) if active_params else None

        self.labels: np.ndarray = np.array([], dtype=str)
        self.values: np.ndarray = np.array([], dtype=limit)
        self.nd_labels: np.ndarray = np.array([])
        self.nd_values: np.ndarray = np.array([])
        self.grid_fig_size: tuple[float, float] = (6.0, 6.0)
        self.L: int = 0
        self.C: int = 1
        self.R: int = 1
        self.full_rows: int = 0
        self.legend_handles: list[Line2D] = []

        if machine is not None:
            self.init(machine, units=units, active_params=active_params)

    @classmethod
    def from_result(cls, result: Any, units: dict[str, Any] | None = None) -> "Plotter":
        """Create a Plotter configured from the machine and parameters inside result."""
        mach = get_machine(result)
        active_params = None
        if hasattr(result, "sampler") and hasattr(result.sampler, "vary"):
            if hasattr(result.sampler.vary, "get_keys"):
                active_params = list(result.sampler.vary.get_keys())
            elif isinstance(result.sampler.vary, dict):
                active_params = list(result.sampler.vary.keys())
        return cls(machine=mach, units=units, active_params=active_params)

    def init(
        self,
        mach: Machine,
        units: dict[str, Any] | None = None,
        active_params: list[str] | None = None,
    ):
        """Initialize parameter boundaries, labels, and grid layout for the given machine."""
        self.machine = mach
        if units:
            self.units.update(units)
        if active_params is not None:
            self.active_params = list(active_params)

        _, vary = energyuq.default_params(mach, active_params=self.active_params)

        _, conv_fn = self.get_unit_converter("CLK", self.units)
        min_freq = min(mach.freq) if (hasattr(mach, "freq") and mach.freq) else 0
        max_freq = max(mach.freq) if (hasattr(mach, "freq") and mach.freq) else 0
        if conv_fn is not None:
            min_freq = conv_fn(min_freq)
            max_freq = conv_fn(max_freq)

        limits = {}
        for k, v in vary.items():
            if str(k).upper() in ("CLK", "CLK_LEVEL") and hasattr(mach, "freq") and mach.freq:
                limits[k] = limit(lower=int(min_freq), upper=int(max_freq))
            else:
                v_low = int(np.asarray(v.lower).item()) if hasattr(v.lower, "__len__") else int(v.lower)
                v_up = int(np.asarray(v.upper).item()) if hasattr(v.upper, "__len__") else int(v.upper)
                limits[k] = limit(lower=v_low, upper=v_up)

        self.labels = np.array(list(limits.keys()), dtype=str)
        self.values = np.array(list(limits.values()), dtype=limit)

        self.L = (self.labels.size + 1) // 2
        (self.C, self.R), self.grid_fig_size = mostly_square_grid(self.L, 6, 2)
        self.full_rows = self.L // self.C if self.C > 0 else 0

        self.nd_values = pad_to_even_and_split(self.values, value=limit(lower=0, upper=1))
        axis_labels = np.array([self.get_axis_label(lbl, self.units) for lbl in self.labels], dtype=str)
        self.nd_labels = pad_to_even_and_split(axis_labels, value="")

        more_red = Line2D([0], [0], color='red', lw=2, marker="o", linestyle='')
        more_blue = Line2D([0], [0], color='blue', lw=2, marker="o", linestyle='')
        self.legend_handles = [more_red, more_blue]

    def set_units(self, units: dict[str, Any]):
        """Set unit configurations and recompute axis labels."""
        self.units.update(units)
        if len(self.labels) > 0:
            axis_labels = np.array([self.get_axis_label(lbl, self.units) for lbl in self.labels], dtype=str)
            self.nd_labels = pad_to_even_and_split(axis_labels, value="")

    def get_axis_bounds(
        self,
        val_low: float,
        val_high: float,
        data: Any,
    ) -> tuple[float, float]:
        """
        Dynamically compute safe, padded [min, max] bounds ensuring all data points
        are strictly visible in the plot.
        """
        if data is not None:
            arr = np.asarray(data, dtype=float)
            arr = arr[~np.isnan(arr)]
        else:
            arr = np.array([], dtype=float)

        if len(arr) > 0:
            d_min = float(np.min(arr))
            d_max = float(np.max(arr))
            if val_high > val_low and val_low <= d_min and d_max <= val_high:
                low = float(val_low)
                high = float(val_high)
            elif val_high > val_low:
                low = float(min(val_low, d_min))
                high = float(max(val_high, d_max))
            else:
                low = d_min
                high = d_max
        else:
            if val_high > val_low:
                low = float(val_low)
                high = float(val_high)
            else:
                low = 0.0
                high = 1.0

        if np.isclose(low, high):
            low -= 0.5
            high += 0.5

        span = high - low
        return low - span / 10.0, high + span / 10.0

    def get_result_params(self, result: Any, df: DataFrame | None = None) -> list[str]:
        """Identify which parameter names belong to the given result and are present in df."""
        if df is None and hasattr(result, "df"):
            df = result.df

        df_cols: list[str] = []
        if df is not None and hasattr(df, "columns"):
            df_cols = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        sampler_params: list[str] = []
        if hasattr(result, "sampler") and hasattr(result.sampler, "vary"):
            if hasattr(result.sampler.vary, "get_keys"):
                sampler_params = list(result.sampler.vary.get_keys())
            elif isinstance(result.sampler.vary, dict):
                sampler_params = list(result.sampler.vary.keys())

        candidates = sampler_params or df_cols
        if len(self.labels) > 0:
            matching = [lbl for lbl in self.labels if lbl in candidates and (not df_cols or lbl in df_cols)]
            if matching:
                return matching

        if df_cols:
            known = ["N_THREADS", "CLK", "CLK_LEVEL", "PLACES", "BINDING", "BOOST", "NUMA"]
            matching = [c for c in df_cols if c in known or c in candidates]
            if matching:
                return matching

        return list(self.labels) if len(self.labels) > 0 else ["N_THREADS", "CLK"]

    def get_unit(self, param: str, units: dict[str, Any] | None = None) -> Any:
        def _find_in_dict(p: str, d: dict[str, Any]) -> tuple[bool, Any]:
            if p in d:
                return True, d[p]
            for k, v in d.items():
                if k.upper() == p.upper():
                    return True, v
            if p.upper() == "CLK_LEVEL":
                for k, v in d.items():
                    if k.upper() == "CLK":
                        return True, v
            if p.upper() == "CLK":
                for k, v in d.items():
                    if k.upper() == "CLK_LEVEL":
                        return True, v
            return False, None

        if units is not None:
            found, val = _find_in_dict(param, units)
            if found:
                return val
        found, val = _find_in_dict(param, self.units)
        if found:
            return val
        return None

    def get_unit_converter(
        self,
        param: str,
        units: dict[str, Any] | None = None,
    ) -> tuple[str | None, Callable[[float], float] | None]:
        spec = self.get_unit(param, units)
        if spec is None:
            return None, None

        from_u, to_u, explicit_scale = parse_unit_spec(spec)

        if callable(explicit_scale):
            return to_u, explicit_scale

        if isinstance(explicit_scale, (int, float)):
            factor = float(explicit_scale)
            return to_u, (lambda v: v * factor)

        target_unit = to_u or from_u
        if target_unit is None or target_unit.lower() in ("none", ""):
            return None, None

        if param.upper() in ("CLK", "CLK_LEVEL"):
            target_lower = target_unit.lower()
            if target_lower == "ghz":
                return target_unit, (lambda v: v * 1e-6)
            elif target_lower == "mhz":
                return target_unit, (lambda v: v * 1e-3)
            elif target_lower == "khz":
                return target_unit, (lambda v: v * 1.0)
            elif target_lower == "hz":
                if from_u and from_u.lower() == "khz":
                    return target_unit, (lambda v: v * 1e3)
                return target_unit, None
            else:
                return target_unit, None

        if from_u and to_u and from_u != to_u:
            for p_from, f_from in SI_PREFIX_FACTORS.items():
                for p_to, f_to in SI_PREFIX_FACTORS.items():
                    if from_u.startswith(p_from) and to_u.startswith(p_to):
                        base_from = from_u[len(p_from):]
                        base_to = to_u[len(p_to):]
                        if base_from == base_to:
                            factor = f_from / f_to
                            return target_unit, (lambda v: v * factor)

        return target_unit, None

    def get_axis_label(self, param: str, units: dict[str, Any] | None = None) -> str:
        target_unit, _ = self.get_unit_converter(param, units)
        if target_unit is not None and str(target_unit).strip().lower() not in ("", "none"):
            return f"{param} ({str(target_unit).strip()})"
        return str(param)

    def to_real_clk(self, val: Any, mach: Machine | None = None, units: dict[str, Any] | None = None) -> Any:
        if mach is None:
            mach = self.machine
        if mach is None or not hasattr(mach, "freq") or not mach.freq:
            return val
        try:
            if pd.isna(val):
                return val
            f_val = float(val)
            _, conv_fn = self.get_unit_converter("CLK", units)
            scale_fn = conv_fn if conv_fn is not None else (lambda v: v)

            if not np.isclose(f_val, round(f_val)):
                return val

            min_raw = min(mach.freq) if mach.freq else 1e5
            if f_val >= min_raw * 0.5:
                return scale_fn(f_val if conv_fn is not None else int(round(f_val)))

            if f_val >= len(mach.freq):
                return val

            i_val = int(round(f_val))
            if 0 <= i_val < len(mach.freq):
                real_val = mach.freq[i_val]
                return scale_fn(real_val)

            return val
        except Exception:
            pass
        return val

    def convert_clk_series(
        self,
        series: pd.Series,
        mach: Machine | None = None,
        units: dict[str, Any] | None = None,
    ) -> pd.Series:
        if mach is None:
            mach = self.machine
        if mach is None or not hasattr(mach, "freq") or not mach.freq:
            return series
        clean = series.dropna()
        if clean.empty:
            return series

        if not np.all(np.isclose(clean, np.round(clean))):
            return series

        min_raw = min(mach.freq) if mach.freq else 1e5
        if clean.min() >= len(mach.freq) and clean.max() < min_raw * 0.5:
            return series

        return series.map(lambda v: self.to_real_clk(v, mach, units))

    def convert_clk_df(
        self,
        df: DataFrame,
        mach: Machine | None = None,
        units: dict[str, Any] | None = None,
    ) -> DataFrame:
        if mach is None:
            mach = self.machine
        if mach is None or not hasattr(mach, "freq") or not mach.freq:
            return df
        df_copy = df.copy()
        for col in df_copy.columns:
            col_name = col[0] if isinstance(col, tuple) else col
            if str(col_name).upper() in ("CLK", "CLK_LEVEL"):
                df_copy[col] = self.convert_clk_series(df_copy[col], mach, units)
        return df_copy

    def _ensure_real_clk_limits(self, mach: Machine | None = None, units: dict[str, Any] | None = None):
        if mach is None:
            mach = self.machine
        if mach is None or not hasattr(mach, "freq") or not mach.freq:
            return
        if len(self.labels) == 0:
            return
        _, conv_fn = self.get_unit_converter("CLK", units)
        min_val = min(mach.freq)
        max_val = max(mach.freq)
        if conv_fn is not None:
            min_val = conv_fn(min_val)
            max_val = conv_fn(max_val)

        new_values = []
        for lbl, val in zip(self.labels, self.values):
            if str(lbl).upper() in ("CLK", "CLK_LEVEL"):
                new_values.append(limit(lower=int(min_val), upper=int(max_val)))
            else:
                new_values.append(val)
        self.values = np.array(new_values, dtype=limit)
        self.nd_values = pad_to_even_and_split(self.values, value=limit(lower=0, upper=1))

    def colors_for(self, qoi: str) -> dict[str, tuple[str, str, str, str]]:
        return {
            'high_low': ('#0000ff', f'lower {qoi}', '#ff0000', f'higher {qoi}'),
            'highest_lowest': ('#0000ff00', f'lowest {qoi}', '#0000ff', f'highest {qoi}'),
        }

    def key_for(self, result: Result, qoi: str) -> str | tuple[str, int]:
        if hasattr(result, "df") and hasattr(result.df, "columns"):
            if isinstance(result.df.columns, pd.MultiIndex) and (qoi, 0) in result.df.columns:
                return (qoi, 0)
            if qoi in result.df.columns:
                return qoi
        if isinstance(result, EasyResult):
            return (qoi, 0)
        return qoi

    def plot_grid_2D(self, result: EasyResult, units: dict[str, str | None] | None = None) -> Figure:
        """Plot chosen sampling grid in 2D pairwise projections."""
        analysis = result.analysis
        sampler = result.sampler

        mach = get_machine(result) or self.machine
        if self.machine is None and mach is not None:
            self.init(mach, units=units)
        self._ensure_real_clk_limits(mach, units)

        cur_labels = self.get_result_params(result)
        cur_values = np.array(
            [self.values[list(self.labels).index(lbl)] if (len(self.labels) > 0 and lbl in self.labels) else limit(lower=0, upper=1) for lbl in cur_labels],
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
        if qoi is None:
            qoi = result.qois[0]
        key = self.key_for(result, qoi)

        mach = get_machine(result) or self.machine
        if self.machine is None and mach is not None:
            self.init(mach, units=units)
        self._ensure_real_clk_limits(mach, units)

        df = self.convert_clk_df(result.df, mach, units)
        pretty_colors = self.colors_for(qoi)

        cur_labels = self.get_result_params(result, df)
        cur_values = np.array(
            [self.values[list(self.labels).index(lbl)] if (len(self.labels) > 0 and lbl in self.labels) else limit(lower=0, upper=1) for lbl in cur_labels],
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
        if qoi is None:
            qoi = result.qois[0]
        key = self.key_for(result, qoi)

        mach = get_machine(result) or self.machine
        if self.machine is None and mach is not None:
            self.init(mach, units=units)
        self._ensure_real_clk_limits(mach, units)

        df = self.convert_clk_df(result.df, mach, units)
        cur_labels = self.get_result_params(result, df)
        L = len(cur_labels)

        if L <= 1:
            C, R = 1, 1
        elif L == 2:
            C, R = 2, 1
        else:
            C = 2
            R = int(np.ceil(L / C))

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

                val_low, val_high = (
                    (self.values[list(self.labels).index(lbl)].lower, self.values[list(self.labels).index(lbl)].upper)
                    if (len(self.labels) > 0 and lbl in self.labels)
                    else (0, 1)
                )
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
        if qoi is None:
            qoi = result.qois[0]
        key = self.key_for(result, qoi)

        mach = get_machine(result) or self.machine
        if self.machine is None and mach is not None:
            self.init(mach, units=units)
        self._ensure_real_clk_limits(mach, units)

        df = self.convert_clk_df(result.df, mach, units)
        cur_labels = self.get_result_params(result, df)
        L = len(cur_labels)

        if L <= 1:
            C, R = 1, 1
        elif L == 2:
            C, R = 2, 1
        else:
            C = 2
            R = int(np.ceil(L / C))

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
                val_low, val_high = (
                    (self.values[list(self.labels).index(lbl)].lower, self.values[list(self.labels).index(lbl)].upper)
                    if (len(self.labels) > 0 and lbl in self.labels)
                    else (0, 1)
                )
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

    def plot_sobols1(
        self,
        result: Result,
        qoi: str | None = None,
        subfig: SubFigure | None = None,
        title: str | None = None,
        units: dict[str, str | None] | None = None,
    ) -> Figure | SubFigure:
        """Plot first-order Sobol sensitivity indices."""
        results = getattr(result, "results", None)
        if results is None:
            raise ValueError("No analysis results available for Sobol indices.")
        if qoi is None:
            qoi = result.qois[0]

        sobol_dict = results.sobols_first(qoi)
        param_names = list(sobol_dict.keys())
        sobols_first = np.array([
            float(np.asarray(v).ravel()[0]) if np.asarray(v).size > 0 else 0.0
            for v in sobol_dict.values()
        ])
        d = len(param_names)

        fig = subfig if subfig is not None else plt.figure(layout="constrained")
        try:
            if title:
                ax = fig.add_subplot(title=title, ylim=[0, 1])
            else:
                ax = fig.add_subplot(ylim=[0, 1])
            ax.set_ylabel(r'$S_i$', fontsize=14)

            ax.bar(0, np.sum(sobols_first), color='salmon')
            ax.bar(np.arange(1, d + 1), sobols_first.flatten(), color='dodgerblue')

            ax.set_xticks(np.arange(d + 1))
            formatted_labels = [self.get_axis_label(lbl, units) for lbl in param_names]
            ax.set_xticklabels(['Total first order', *formatted_labels], rotation=90)
            return fig
        except Exception:
            if subfig is None:
                plt.close(fig)
            raise

    def draw_gradients(self, *color_list) -> Figure:
        """Render horizontal gradient swatches."""
        n_items = len(color_list)
        fig, ax = plt.subplots(figsize=(6, 0.9 * n_items + 0.4))
        ax.set_facecolor('#ffffff')
        for spine in ax.spines.values():
            spine.set_color('#cccccc')

        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        for i, colors in enumerate(color_list):
            cmap = mcolors.LinearSegmentedColormap.from_list(f'cmap_{i}', colors)
            ax.imshow(gradient, extent=[0, 10, i, i + 0.6], cmap=cmap, aspect='auto')

        ax.set_xlim(0, 10)
        ax.set_ylim(-0.2, n_items)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig

    def plot_stat_convergence(self, result: Any, title: str | None = None) -> Figure | None:
        """Generate EasyVVUQ statistical moments convergence plot."""
        from unittest.mock import patch
        analysis = getattr(result, "analysis", result)
        if not hasattr(analysis, "plot_stat_convergence"):
            return None
        plt.close("stat_conv")
        with patch("matplotlib.pyplot.show", lambda *args, **kwargs: None):
            analysis.plot_stat_convergence()
        fig = plt.figure("stat_conv")
        if len(fig.axes) == 0:
            plt.close("stat_conv")
            return None
        if title:
            fig.suptitle(title, fontsize=11)
        else:
            if getattr(fig, "_suptitle", None) is not None:
                fig._suptitle.set_text("")
            for ax in fig.axes:
                ax.set_title("")
        return fig

    def plot_adaptation_histogram(self, result: Any, title: str | None = None) -> Figure | None:
        """Generate EasyVVUQ adaptation histogram plot."""
        from unittest.mock import patch
        analysis = getattr(result, "analysis", result)
        if not hasattr(analysis, "adaptation_histogram"):
            return None
        plt.close("adapt_hist")
        with patch("matplotlib.pyplot.show", lambda *args, **kwargs: None):
            analysis.adaptation_histogram()
        fig = plt.figure("adapt_hist")
        if len(fig.axes) == 0:
            plt.close("adapt_hist")
            return None
        if title:
            fig.suptitle(title, fontsize=11)
        else:
            if getattr(fig, "_suptitle", None) is not None:
                fig._suptitle.set_text("")
            for ax in fig.axes:
                ax.set_title("")
        return fig

    def plot_adaptation_table(self, result: Any, title: str | None = None) -> Figure | None:
        """Generate EasyVVUQ adaptation table plot."""
        from unittest.mock import patch
        analysis = getattr(result, "analysis", result)
        if not hasattr(analysis, "adaptation_table"):
            return None
        with patch("matplotlib.pyplot.show", lambda *args, **kwargs: None):
            analysis.adaptation_table()
        fig = plt.gcf()
        if len(fig.axes) == 0:
            plt.close(fig)
            return None
        if title:
            fig.suptitle(title, fontsize=11)
        else:
            if getattr(fig, "_suptitle", None) is not None:
                fig._suptitle.set_text("")
            for ax in fig.axes:
                ax.set_title("")
        return fig


# Helper function for standalone palette lookup
def colors_for(qoi: str) -> dict[str, tuple[str, str, str, str]]:
    return Plotter().colors_for(qoi)


# Re-export multi-run plotting functions
from .multi_plot import (
    plot_multi_sobols,
    plot_multi_convergence,
    plot_multi_energy_time_pareto,
    plot_multi_qoi_distribution,
    plot_multi_best_configurations,
    plot_multi_parameter_effects,
    plot_multi_dashboard,
)
