from __future__ import annotations

from math import ceil, floor, sqrt
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from ..machines.machine import Machine
from ..util.data import EasyResult, Result


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


def get_sampler_params(result: Any) -> list[str]:
    """Extract varied parameter names from a result, analysis, campaign, or sampler."""
    sampler = None
    if hasattr(result, "sampler"):
        sampler = result.sampler
    elif hasattr(result, "analysis") and hasattr(result.analysis, "sampler"):
        sampler = result.analysis.sampler
    elif hasattr(result, "vary"):
        sampler = result

    if sampler is not None and hasattr(sampler, "vary"):
        vary = sampler.vary
        if hasattr(vary, "get_keys"):
            return list(vary.get_keys())
        elif isinstance(vary, dict):
            return list(vary.keys())
    return []


class PlotterLayoutMixin:
    """Mixin providing layout calculations, parameter inspection, and axis bounds."""

    labels: np.ndarray

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

        sampler_params = get_sampler_params(result)
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


# Standalone delegators
_default_layout = PlotterLayoutMixin()
_default_layout.labels = np.array([], dtype=str)


def get_axis_bounds(val_low: float, val_high: float, data: Any) -> tuple[float, float]:
    return _default_layout.get_axis_bounds(val_low, val_high, data)


def get_result_params(result: Any, df: DataFrame | None = None) -> list[str]:
    return _default_layout.get_result_params(result, df)


def colors_for(qoi: str) -> dict[str, tuple[str, str, str, str]]:
    return _default_layout.colors_for(qoi)


def key_for(result: Result, qoi: str) -> str | tuple[str, int]:
    return _default_layout.key_for(result, qoi)

