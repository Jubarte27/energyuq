from typing import Any
from . import *  # noqa: F403
from . import __all__ as _plotting_all
from ..util.multi_plot import (
    plot_multi_sobols,
    plot_multi_convergence,
    plot_multi_energy_time_pareto,
    plot_multi_qoi_distribution,
    plot_multi_best_configurations,
    plot_multi_parameter_effects,
    plot_multi_dashboard,
)

_default_plotter = Plotter()  # noqa: F405


def init(mach: Any, units: Any = None, active_params: Any = None) -> Plotter:  # noqa: F405
    """Initialize default plotter module instance for the given machine."""
    _default_plotter.init(mach, units=units, active_params=active_params)
    return _default_plotter


_multi_plot_all = [
    "plot_multi_sobols",
    "plot_multi_convergence",
    "plot_multi_energy_time_pareto",
    "plot_multi_qoi_distribution",
    "plot_multi_best_configurations",
    "plot_multi_parameter_effects",
    "plot_multi_dashboard",
]

__all__ = list(_plotting_all) + ["init"] + _multi_plot_all
