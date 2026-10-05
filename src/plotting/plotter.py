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
    ):
        self.machine: Machine | None = machine
        self.units: dict[str, Any] = dict(DEFAULT_PARAM_UNITS)
        if units:
            self.units.update(units)

        self.labels: np.ndarray = np.array([], dtype=str)
        self.values: np.ndarray = np.array([], dtype=limit)
        if machine is not None:
            self.init(machine, units=units)

    @classmethod
    def from_result(cls, result: Any, units: dict[str, Any] | None = None) -> "Plotter":
        """Create a Plotter configured from the machine and parameters inside result."""
        mach = get_machine(result)
        return cls(machine=mach, units=units)

    def init(
        self,
        mach: Machine,
        units: dict[str, Any] | None = None,
    ):
        """Initialize parameter boundaries, labels, and grid layout for the given machine."""
        self.machine = mach
        if units:
            self.units.update(units)

        _, vary = energyuq.default_params(mach)

        _, convert = self.get_unit_converter("CLK", self.units)
        min_freq = min(mach.freq) if (hasattr(mach, "freq") and mach.freq) else 0
        max_freq = max(mach.freq) if (hasattr(mach, "freq") and mach.freq) else 0
        if convert is not None:
            min_freq = convert(min_freq)
            max_freq = convert(max_freq)

        limits = {}
        for k, v in vary.items():
            if str(k).upper() == "CLK" and hasattr(mach, "freq") and mach.freq:
                limits[k] = limit(lower=int(min_freq), upper=int(max_freq))
            else:
                v_low = int(np.asarray(v.lower).item()) if hasattr(v.lower, "__len__") else int(v.lower)
                v_up = int(np.asarray(v.upper).item()) if hasattr(v.upper, "__len__") else int(v.upper)
                limits[k] = limit(lower=v_low, upper=v_up)

        self.labels = np.array(list(limits.keys()), dtype=str)
        self.values = np.array(list(limits.values()), dtype=limit)
