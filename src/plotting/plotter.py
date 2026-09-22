from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.lines import Line2D

from .. import energyuq
from ..machines.machine import Machine
from ..util.data import limit
from .diagnostics import PlotterDiagnosticsMixin
from .grid import PlotterGridMixin
from .layout import (
    PlotterLayoutMixin,
    get_machine,
    get_sampler_params,
    mostly_square_grid,
    pad_to_even_and_split,
)
from .sobol import PlotterSobolMixin
from .units import DEFAULT_PARAM_UNITS, PlotterUnitsMixin


class Plotter(
    PlotterUnitsMixin,
    PlotterLayoutMixin,
    PlotterGridMixin,
    PlotterSobolMixin,
    PlotterDiagnosticsMixin,
):
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
    def from_result(cls, result: Any, units: dict[str, Any] | None = None) -> Plotter:
        """Create a Plotter configured from the machine and parameters inside result."""
        mach = get_machine(result)
        sampler_params = get_sampler_params(result)
        active_params = sampler_params if sampler_params else None
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

