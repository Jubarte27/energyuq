from math import floor, ceil, sqrt
from typing import Sequence

from matplotlib.figure import SubFigure
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
from .data import *

def pad_to_even_and_split(arr: np.ndarray, value=None) -> np.ndarray:
    pad_by = arr.shape[-1] % 2
    new_shape = [*arr.shape[:-1], 2, -1]
    return np.pad(arr, (0, pad_by), mode='constant', constant_values=value).reshape(new_shape)

def mostly_square_grid(blocks: int, total_width: float, min_block_width: float):
    max_col = floor(total_width / min_block_width)
    cols = min(ceil(sqrt(blocks)), max_col)
    rows = ceil(blocks / cols)

    total_height = rows * total_width
    return (cols, rows), (total_width, total_height)

current_machine: Machine | None = None

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
param_units: dict[str, Any] = dict(DEFAULT_PARAM_UNITS)

def set_param_units(units: dict[str, Any]):
    global param_units, nd_labels
    param_units = dict(units)
    if "labels" in globals() and labels is not None and len(labels) > 0:
        axis_labels = np.array([get_axis_label(lbl) for lbl in labels], dtype=str)
        nd_labels = pad_to_even_and_split(axis_labels)

set_units = set_param_units

def get_unit(param: str, units: dict[str, Any] | None = None) -> Any:
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
    found, val = _find_in_dict(param, param_units)
    if found:
        return val
    return None

def get_unit_converter(param: str, units: dict[str, Any] | None = None) -> tuple[str | None, Callable[[float], float] | None]:
    """
    Returns (display_unit, convert_function) for the specified parameter.
    If no conversion is needed, convert_function is None.
    """
    spec = get_unit(param, units)
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

    # Frequency / CLK handling: machine.freq is in kHz (e.g. 1400000 = 1.4 GHz)
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

    # General SI prefix conversion
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

def get_axis_label(param: str, units: dict[str, Any] | None = None) -> str:
    target_unit, _ = get_unit_converter(param, units)
    if target_unit is not None and str(target_unit).strip().lower() not in ("", "none"):
        return f"{param} ({str(target_unit).strip()})"
    return str(param)

def to_real_clk(val, mach: Machine | None = None, units: dict[str, Any] | None = None):
    if mach is None:
        mach = current_machine
    if mach is None or not hasattr(mach, "freq") or not mach.freq:
        return val
    try:
        if pd.isna(val):
            return val
        f_val = float(val)
        _, conv_fn = get_unit_converter("CLK", units)
        scale_fn = conv_fn if conv_fn is not None else (lambda v: v)

        # 1. Non-integer float with fractional part (e.g. 1.4, 2.3):
        # Indices are strictly integers (0, 1, 2...), so a fractional value
        # is already a converted frequency (e.g. in GHz).
        if not np.isclose(f_val, round(f_val)):
            return val

        # 2. Raw frequency in kHz (e.g. 1400000):
        min_raw = min(mach.freq) if mach.freq else 1e5
        if f_val >= min_raw * 0.5:
            return scale_fn(f_val if conv_fn is not None else int(round(f_val)))

        # 3. Frequency already converted to MHz (e.g. 1400):
        # If f_val > len(mach.freq) and f_val < min_raw * 0.5:
        # It's larger than any valid index, but smaller than raw kHz.
        if f_val >= len(mach.freq):
            return val

        # 4. Discrete index (0 <= i_val < len(mach.freq)):
        i_val = int(round(f_val))
        if 0 <= i_val < len(mach.freq):
            real_val = mach.freq[i_val]
            return scale_fn(real_val)

        return val
    except Exception:
        pass
    return val

def convert_clk_series(series: pd.Series, mach: Machine | None = None, units: dict[str, Any] | None = None) -> pd.Series:
    if mach is None:
        mach = current_machine
    if mach is None or not hasattr(mach, "freq") or not mach.freq:
        return series
    clean = series.dropna()
    if clean.empty:
        return series

    # If series already has fractional numbers, it is already converted (e.g. to GHz)
    if not np.all(np.isclose(clean, np.round(clean))):
        return series

    # If values are larger than index count but smaller than raw frequency, already converted (e.g. to MHz)
    min_raw = min(mach.freq) if mach.freq else 1e5
    if clean.min() >= len(mach.freq) and clean.max() < min_raw * 0.5:
        return series

    return series.map(lambda v: to_real_clk(v, mach, units))

def convert_clk_df(df: DataFrame, mach: Machine | None = None, units: dict[str, Any] | None = None) -> DataFrame:
    if mach is None:
        mach = current_machine
    if mach is None or not hasattr(mach, "freq") or not mach.freq:
        return df
    df_copy = df.copy()
    for col in df_copy.columns:
        col_name = col[0] if isinstance(col, tuple) else col
        if str(col_name).upper() in ("CLK", "CLK_LEVEL"):
            df_copy[col] = convert_clk_series(df_copy[col], mach, units)
    return df_copy

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
    if result is not None:
        if hasattr(result, "machine") and isinstance(result.machine, Machine):
            return result.machine
        if hasattr(result, "sampler") and hasattr(result.sampler, "machine") and isinstance(result.sampler.machine, Machine):
            return result.sampler.machine
    return current_machine

def _ensure_real_clk_limits(mach: Machine | None = None, units: dict[str, Any] | None = None):
    global values, nd_values
    if mach is None:
        mach = current_machine
    if mach is None or not hasattr(mach, "freq") or not mach.freq:
        return
    if "labels" not in globals() or labels is None:
        return
    _, conv_fn = get_unit_converter("CLK", units)
    min_val = min(mach.freq)
    max_val = max(mach.freq)
    if conv_fn is not None:
        min_val = conv_fn(min_val)
        max_val = conv_fn(max_val)

    new_values = []
    for lbl, val in zip(labels, values):
        if str(lbl).upper() in ("CLK", "CLK_LEVEL"):
            new_values.append(limit(lower=min_val, upper=max_val))
        else:
            new_values.append(val)
    values = np.array(new_values, dtype=limit)
    nd_values = pad_to_even_and_split(values)

def init(mach: Machine, units: dict[str, Any] | None = None):
    global current_machine
    current_machine = mach
    _, vary = energyuq.default_params(mach)
    global labels, values, grid_fig_size, L, C, R, full_rows, nd_labels, nd_values, legend_handles

    _, conv_fn = get_unit_converter("CLK", units)
    min_freq = min(mach.freq) if (hasattr(mach, "freq") and mach.freq) else 0
    max_freq = max(mach.freq) if (hasattr(mach, "freq") and mach.freq) else 0
    if conv_fn is not None:
        min_freq = conv_fn(min_freq)
        max_freq = conv_fn(max_freq)

    limits = {}
    for k, v in vary.items():
        if str(k).upper() in ("CLK", "CLK_LEVEL") and hasattr(mach, "freq") and mach.freq:
            limits[k] = limit(lower=min_freq, upper=max_freq)
        else:
            limits[k] = limit(lower=int(v.lower), upper=int(v.upper))

    labels = np.array(list(limits.keys()), dtype=str)
    values = np.array(list(limits.values()), dtype=limit)

    L = (labels.size+1)//2
    (C, R), grid_fig_size = mostly_square_grid(L, 6, 2)
    full_rows = L // C

    nd_values = pad_to_even_and_split(values)
    axis_labels = np.array([get_axis_label(lbl, units) for lbl in labels], dtype=str)
    nd_labels = pad_to_even_and_split(axis_labels)

    more_red = Line2D([0], [0], color='red', lw=2, marker="o", linestyle='')
    more_blue = Line2D([0], [0], color='blue', lw=2, marker="o", linestyle='')

    legend_handles = [more_red, more_blue]

def colors_for(qoi) -> dict[str, tuple[str,str,str,str]]:
    return {
        'high_low': ('#0000ff', f'lower {qoi}', '#ff0000', f'higher {qoi}'),
        'highest_lowest': ( '#0000ff00', f'lowest {qoi}', '#0000ff', f'highest {qoi}'),
    }

def key_for(result, qoi) -> str | tuple[str, int]:
    if hasattr(result, "df") and hasattr(result.df, "columns"):
        if isinstance(result.df.columns, pd.MultiIndex) and (qoi, 0) in result.df.columns:
            return (qoi, 0)
        if qoi in result.df.columns:
            return qoi
    if isinstance(result, EasyResult):
        return (qoi, 0)
    return qoi



def plot_grid_2D(result: EasyResult, units: dict[str, str | None] | None = None):
    analysis = result.analysis
    sampler = result.sampler

    mach = get_machine(result)
    _ensure_real_clk_limits(mach, units)

    fig = plt.figure(figsize=grid_fig_size, layout="constrained")
    fig.supylabel("Configurations chosen")

    ax=[]
    i = 0
    cols = C
    index = lambda: i + 1
    axis_labels = np.array([get_axis_label(lbl, units) for lbl in labels], dtype=str)
    cur_nd_labels = pad_to_even_and_split(axis_labels)
    for _ in range(full_rows):
        if R > full_rows:
            cols = L % C
            index = lambda: R * cols - (R * C - i)
        for _ in range(cols):
            xd = nd_values[0, i].upper - nd_values[0, i].lower
            yd = nd_values[1, i].upper - nd_values[1, i].lower
            ax.append(fig.add_subplot(R, cols, index(),
                                    xlim=[nd_values[0, i].lower - xd/10, nd_values[0, i].upper + xd/10],
                                    ylim=[nd_values[1, i].lower - yd/10, nd_values[1, i].upper + yd/10], 
                                    xlabel=cur_nd_labels[0, i], ylabel=cur_nd_labels[1, i])
                    )
            x_is_int = _is_integer_range(nd_values[0, i].lower, nd_values[0, i].upper)
            y_is_int = _is_integer_range(nd_values[1, i].lower, nd_values[1, i].upper)
            ax[-1].xaxis.set_major_locator(MaxNLocator(integer=x_is_int))
            ax[-1].yaxis.set_major_locator(MaxNLocator(integer=y_is_int))
            i += 1

    raw_grid = sampler.generate_grid(analysis.l_norm).astype(object)
    if raw_grid.ndim == 2:
        if raw_grid.shape[1] == len(labels):
            for col_idx, lbl in enumerate(labels):
                if str(lbl).upper() in ("CLK", "CLK_LEVEL") and mach is not None and hasattr(mach, "freq"):
                    raw_grid[:, col_idx] = [to_real_clk(v, mach, units) for v in raw_grid[:, col_idx]]
        elif raw_grid.shape[0] == len(labels):
            for row_idx, lbl in enumerate(labels):
                if str(lbl).upper() in ("CLK", "CLK_LEVEL") and mach is not None and hasattr(mach, "freq"):
                    raw_grid[row_idx, :] = [to_real_clk(v, mach, units) for v in raw_grid[row_idx, :]]

    accepted_grid = pad_to_even_and_split(raw_grid)
    ic=0
    for i in range(L):
        ax[i].plot(accepted_grid[:,0, ic], accepted_grid[:,1,ic], 'o', alpha=0.25)
        ic += 1
    # plt.tight_layout()
    return fig


def plot_grid_2D_best(result: Result, qoi=None, order_focus=False, subfig=None, units: dict[str, str | None] | None = None):
    if qoi is None:
        qoi = result.qois[0]
    key = key_for(result, qoi)
    mach = get_machine(result)
    _ensure_real_clk_limits(mach, units)
    df = convert_clk_df(result.df, mach, units)

    pretty_colors = colors_for(qoi)

    if subfig is None:
        fig = plt.figure(figsize=grid_fig_size, layout="constrained")
        fig.supylabel(f"Configurations evaluated by {qoi}")
    else:
        fig = subfig
    
    try:
        ax=[]
        i = 0
        cols = C
        index = lambda: i + 1
        ax: list[Axes]=[]
        axis_labels = np.array([get_axis_label(lbl, units) for lbl in labels], dtype=str)
        cur_nd_labels = pad_to_even_and_split(axis_labels)
        for _ in range(full_rows):
            if R > full_rows:
                cols = L % C
                index = lambda: R * cols - (R * C - i)
            for _ in range(cols):
                xd = nd_values[0, i].upper - nd_values[0, i].lower
                yd = nd_values[1, i].upper - nd_values[1, i].lower
                ax.append(fig.add_subplot(R, cols, index(),
                                        xlim=[nd_values[0, i].lower - xd/10, nd_values[0, i].upper + xd/10],
                                        ylim=[nd_values[1, i].lower - yd/10, nd_values[1, i].upper + yd/10], 
                                        xlabel=cur_nd_labels[0, i], ylabel=cur_nd_labels[1, i])
                        )
                x_is_int = _is_integer_range(nd_values[0, i].lower, nd_values[0, i].upper)
                y_is_int = _is_integer_range(nd_values[1, i].lower, nd_values[1, i].upper)
                ax[-1].xaxis.set_major_locator(MaxNLocator(integer=x_is_int))
                ax[-1].yaxis.set_major_locator(MaxNLocator(integer=y_is_int))
                ax[-1].set_box_aspect(1)
                ax[-1].set_anchor('N')
                i += 1

        dataframe: DataFrame = df.sort_values(by=key, ascending=False)
        column = dataframe[qoi]
        if isinstance(column, pd.DataFrame):
            column = column.iloc[:, 0]

        # Detect outliers using the Interquartile Range (IQR) method
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        
        high_outlier_mask = column > (Q3 + 1.5 * IQR)
        low_outlier_mask = column < (Q1 - 1.5 * IQR)
        outlier_mask = high_outlier_mask | low_outlier_mask
        inlier_mask = ~outlier_mask

        # Compute normalization bounds exclusively using non-outlier (inlier) values
        inlier_column = column[inlier_mask]
        inlier_min = inlier_column.min()
        inlier_max = inlier_column.max()

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

        for i in range(L):
            xs = dataframe[labels[i*2]].to_numpy()
            ys = dataframe[labels[i*2 + 1]].to_numpy()
            ax[i].legend(handles=custom_handles, labels=legend_labels, draggable=True, fontsize='x-small', ncols=2, bbox_to_anchor=(1, 1.1), loc='upper right')
            ax[i].scatter(xs, ys, c=colors)
        
        return fig
    except Exception:
        if subfig is None:
            plt.close(fig)
        raise


def plot_sobols1(result: Result, qoi=None, subfig: SubFigure | None=None, title: str | None = None, units: dict[str, str | None] | None = None):
    results = result.results
    if results is None:
        raise ValueError("No analysis results available for Sobol indices.")
    if qoi is None:
        qoi = result.qois[0]
    
    # Retrieve sobol values first before opening a matplotlib figure
    sobols_first = np.array(list(results.sobols_first(qoi).values()))
    d = len(labels)
    
    fig = subfig if subfig is not None else plt.figure(layout="constrained")
    try:
        if title:
            ax = fig.add_subplot(title=title, ylim=[0,1])
        else:
            ax = fig.add_subplot(ylim=[0,1])
        ax.set_ylabel(r'$S_i$', fontsize=14)
        
        ax.bar(0, np.sum(sobols_first), color='salmon')
        ax.bar(np.arange(1, d+1), sobols_first.flatten(), color='dodgerblue')

        ax.set_xticks(np.arange(d+1))
        formatted_labels = [get_axis_label(lbl, units) for lbl in labels]
        ax.set_xticklabels(['Total first order', *formatted_labels], rotation=90)
        return fig
    except Exception:
        if subfig is None:
            plt.close(fig)
        raise


def draw_gradients(*color_list):
    n_items = len(color_list)
    
    fig, ax = plt.subplots(figsize=(6, 0.9 * n_items + 0.4))
    
    ax.set_facecolor('#ffffff')
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(1.0)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.2, n_items)

    gradient_base = np.linspace(0, 1, 256).reshape(1, -1)

    for i, (color_a, label_a, color_b, label_b) in enumerate(reversed(color_list)):
        y_bottom = i * 1.0
        y_top = y_bottom + 0.28
    
        cmap = mcolors.LinearSegmentedColormap.from_list(f"legend_cmap_{i}", [color_a, color_b])
        
        ax.imshow(gradient_base, aspect='auto', cmap=cmap, extent=(0, 1, y_bottom, y_top))
        ax.text(0.0, y_top + 0.05, label_a, ha='left', va='bottom', fontsize=10, color='#333333')
        ax.text(1.0, y_top + 0.05, label_b, ha='right', va='bottom', fontsize=10, color='#333333')
        
    # plt.tight_layout()
    return fig

def plot_sorted(result: Result, qoi=None, subfig: SubFigure | None=None, title: str | None = None):
    if qoi is None:
        qoi = result.qois[0]
    key = key_for(result, qoi)
    dataframe = result.df
    
    if not subfig:
        fig, ax = plt.subplots(figsize=(12,12), layout="constrained")
    else:
        fig = subfig
        ax = fig.subplots()

    try:
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


def get_confidence_intervals(samples, conf=0.9):
    """
    Compute the confidence intervals given an array of samples

    Parameters
    ----------
    samples : array
        Samples on which to compute the intervals.
    conf : float, optional, must be in [0, 1].
        The confidence interval percentage. The default is 0.9.

    Returns
    -------
    lower : array
        The lower confidence bound..
    upper : array
        The upper confidence bound.

    """

    # ake sure conf is in [0, 1]
    if conf < 0.0 or conf > 1.0:
        print('conf must be specified within [0, 1]')
        return

    # lower bound = alpha, upper bound = 1 - alpha
    alpha = 0.5 * (1.0 - conf)

    # arrays for lower and upper bound of the interval
    n_samples = samples.shape[0]
    N_qoi = samples.shape[1]
    lower = np.zeros(N_qoi)
    upper = np.zeros(N_qoi)

    # the probabilities of the ecdf
    prob = np.linspace(0, 1, n_samples)
    # the closest locations in prob that correspond to the interval bounds
    idx0 = np.where(prob <= alpha)[0][-1]
    idx1 = np.where(prob <= 1.0 - alpha)[0][-1]

    # for every location of qoi compute the ecdf-based confidence interval
    for i in range(N_qoi):
        # the sorted surrogate samples at the current location
        samples_sorted = np.sort(samples[:, i])
        # the corresponding confidence interval
        lower[i] = samples_sorted[idx0]
        upper[i] = samples_sorted[idx1]

    return lower, upper

def plot_2D_single_dimension(result: Result, qoi=None, subfig=None, units: dict[str, str | None] | None = None):
    if qoi is None:
        qoi = result.qois[0]
    key = key_for(result, qoi)
    mach = get_machine(result)
    _ensure_real_clk_limits(mach, units)
    df = convert_clk_df(result.df, mach, units)

    L = len(labels)
    C = 2
    R = int(np.ceil(L / C))

    if subfig:
        fig = subfig
    else:
        fig = plt.figure(figsize=(12,12/C*R), layout="constrained")
        fig.supylabel(f"For {qoi}")

    try:
        ax = []
        i=0
        
        ax = fig.subplots(R, C, sharey=False)
        if isinstance(ax, np.ndarray):
            ax = ax.flatten()

        for i in range(L):
            xd = values[i].upper - values[i].lower
            ax[i].set_xlim((values[i].lower - xd/10, values[i].upper + xd/10))
            ax[i].set_xlabel(xlabel=get_axis_label(labels[i], units))
            ax[i].set_ylabel(qoi)
            is_int = _is_integer_range(values[i].lower, values[i].upper)
            ax[i].xaxis.set_major_locator(MaxNLocator(integer=is_int))
            ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))
            ax[i].set_box_aspect(1)
            ax[i].set_anchor('N')

        for i in range(L):
            ax[i].scatter(df[labels[i]], df[key])
        
        return fig
    except Exception:
        if not subfig:
            plt.close(fig)
        raise

def plot_boxplot(result: Result, qoi=None, units: dict[str, str | None] | None = None):
    if qoi is None:
        qoi = result.qois[0]
    key = key_for(result, qoi)
    mach = get_machine(result)
    _ensure_real_clk_limits(mach, units)
    df = convert_clk_df(result.df, mach, units)

    L = len(labels)
    C = int(np.ceil(np.sqrt((10+1)//2)))
    R = int(np.ceil(L / C))

    fig = plt.figure(figsize=(12,12/C*R), layout="constrained")
    fig.supylabel(f"For {qoi}")
    
    try:
        ax: list[Axes]=[]
        for i in range(L):
            xd = values[i].upper - values[i].lower
            ax.append(fig.add_subplot(R, C, i+1,
                                      xlim=[values[i].lower - xd/10, values[i].upper + xd/10],
                                      xlabel=get_axis_label(labels[i], units), ylabel=qoi
                        )
                     )
            is_int = _is_integer_range(values[i].lower, values[i].upper)
            ax[-1].xaxis.set_major_locator(MaxNLocator(integer=is_int))
            ax[-1].yaxis.set_major_locator(MaxNLocator(integer=True))

        def col_to_numpy(x: DataFrame):
            return x[key].to_numpy()

        for i in range(L):
            label_c = (labels[i], 0) if isinstance(key, tuple) else labels[i]
            box_frame = df[[key, label_c]].groupby(by=label_c)[[key]].apply(col_to_numpy) # pyright: ignore[reportArgumentType, reportCallIssue]

            positions = np.array(box_frame.index.to_list(), dtype=float)
            diffs = np.diff(np.sort(positions))
            xd = values[i].upper - values[i].lower
            if len(diffs) > 0 and np.min(diffs) > 0:
                width = 0.5 * np.min(diffs)
            elif xd > 0:
                width = 0.05 * xd
            else:
                width = 0.5

            VP = ax[i].boxplot(box_frame.to_numpy(),
                                positions=positions,
                                widths=width,
                                patch_artist=True,
                                showmeans=False, showfliers=False, manage_ticks = False,
                                medianprops={"color": "white", "linewidth": 0.5},
                                boxprops={"facecolor": "C0", "edgecolor": "white", "linewidth": 0.5},
                                whiskerprops={"color": "C0", "linewidth": 1.5},
                                capprops={"color": "C0", "linewidth": 1.5}
                            )

        return fig
    except Exception:
        plt.close(fig)
        raise
    return fig


def plot_stat_convergence(result: Any, title: str | None = None) -> Figure | None:
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


def plot_adaptation_histogram(result: Any, title: str | None = None) -> Figure | None:
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


def plot_adaptation_table(result: Any, title: str | None = None) -> Figure | None:
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
