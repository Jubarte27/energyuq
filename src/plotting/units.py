from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from ..machines.machine import Machine
from ..util.data import limit
from .layout import pad_to_even_and_split

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

DEFAULT_PARAM_UNITS: dict[str, Any] = {"CLK": "Hz"}


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


def _is_integer_range(lower: Any, upper: Any) -> bool:
    try:
        f_low = float(lower)
        f_up = float(upper)
        if f_up - f_low < 1:
            return False
        return f_low.is_integer() and f_up.is_integer()
    except (ValueError, TypeError, OverflowError):
        return False


class PlotterUnitsMixin:
    """Mixin providing unit management and machine clock conversions for Plotter."""

    machine: Machine | None
    units: dict[str, Any]
    labels: np.ndarray
    values: np.ndarray
    nd_labels: np.ndarray
    nd_values: np.ndarray

    def set_units(self, units: dict[str, Any]):
        """Set unit configurations and recompute axis labels."""
        self.units.update(units)
        if len(self.labels) > 0:
            axis_labels = np.array([self.get_axis_label(lbl, self.units) for lbl in self.labels], dtype=str)
            self.nd_labels = pad_to_even_and_split(axis_labels, value="")

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
                            return target_unit, (lambda v, factor=factor: v * factor)

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

        if pd.isna(val):
            return val
        f_val = float(val)
        _, conv_fn = self.get_unit_converter("CLK", units)
        scale_fn = conv_fn if conv_fn is not None else (lambda v: v)

        if not np.isclose(f_val, round(f_val)):
            return val

        min_raw = min(mach.freq) if mach.freq else 1e5
        if f_val >= min_raw * 0.5:
            return scale_fn(f_val if conv_fn is not None else round(f_val))

        if f_val >= len(mach.freq):
            return val

        i_val = round(f_val)
        if 0 <= i_val < len(mach.freq):
            real_val = mach.freq[i_val]
            return scale_fn(real_val)

        return val
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


# Standalone functions delegating to PlotterUnitsMixin default behavior
_default_units_handler = PlotterUnitsMixin()
_default_units_handler.machine = None
_default_units_handler.units = dict(DEFAULT_PARAM_UNITS)
_default_units_handler.labels = np.array([], dtype=str)
_default_units_handler.values = np.array([], dtype=limit)
_default_units_handler.nd_labels = np.array([])
_default_units_handler.nd_values = np.array([])


def get_unit(param: str, units: dict[str, Any] | None = None) -> Any:
    return _default_units_handler.get_unit(param, units)


def get_unit_converter(
    param: str,
    units: dict[str, Any] | None = None,
) -> tuple[str | None, Callable[[float], float] | None]:
    return _default_units_handler.get_unit_converter(param, units)


def get_axis_label(param: str, units: dict[str, Any] | None = None) -> str:
    return _default_units_handler.get_axis_label(param, units)


def to_real_clk(val: Any, mach: Machine | None = None, units: dict[str, Any] | None = None) -> Any:
    return _default_units_handler.to_real_clk(val, mach=mach, units=units)


def convert_clk_series(
    series: pd.Series,
    mach: Machine | None = None,
    units: dict[str, Any] | None = None,
) -> pd.Series:
    return _default_units_handler.convert_clk_series(series, mach=mach, units=units)


def convert_clk_df(
    df: DataFrame,
    mach: Machine | None = None,
    units: dict[str, Any] | None = None,
) -> DataFrame:
    return _default_units_handler.convert_clk_df(df, mach=mach, units=units)

