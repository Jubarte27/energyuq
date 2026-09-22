from typing import Any

from . import Plotter

_default_plotter = Plotter()


def init(mach: Any, units: Any = None, active_params: Any = None) -> Plotter:
    """Initialize default plotter module instance for the given machine."""
    _default_plotter.init(mach, units=units, active_params=active_params)
    return _default_plotter

