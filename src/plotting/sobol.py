from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.patches import Patch

from ..util.data import Result


class SobolOrderResult(dict):
    """
    Dictionary mapping parameter/interaction label to its Sobol sensitivity index.

    Attributes
    ----------
    n : int
        The smallest order such that higher orders have total influence < k%.
    order : int
        Alias for n.
    k : float
        The threshold percentage used (e.g. 5.0 for 5%).
    total_influence : float
        Sum of Sobol indices of orders <= n.
    higher_order_influence : float
        Sum of Sobol indices of orders > n.
    higher_order_pct : float
        Percentage of total Sobol influence in orders > n (< k%).
    total_sobol : float
        Sum of all computed Sobol indices.
    order_influences : dict[int, float]
        Total influence per order {1: ..., 2: ...}.
    order_terms : dict[int, dict[str, float]]
        Sobol terms grouped by order {1: {label: val}, 2: {label: val}}.
    total_variance : float | None
        Total PCE variance D.
    mean : float | None
        PCE mean.
    raw_sobols : dict[tuple[int, ...], float]
        Raw Sobol indices keyed by multi-index tuple of parameter positions.
    tuple_sobols : dict[tuple[str, ...], float]
        Sobol indices keyed by parameter name tuples.
    formatted_labels : dict[tuple[int, ...], str]
        Mapping from multi-index tuple to formatted display label.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n: int = 1
        self.order: int = 1
        self.k: float = 5.0
        self.total_influence: float = 0.0
        self.higher_order_influence: float = 0.0
        self.higher_order_pct: float = 0.0
        self.total_sobol: float = 0.0
        self.order_influences: dict[int, float] = {}
        self.order_terms: dict[int, dict[str, float]] = {}
        self.total_variance: float | None = None
        self.mean: float | None = None
        self.raw_sobols: dict[tuple[int, ...], float] = {}
        self.tuple_sobols: dict[tuple[str, ...], float] = {}
        self.formatted_labels: dict[tuple[int, ...], str] = {}

    def __repr__(self) -> str:
        return (
            f"SobolOrderResult(n={self.n}, k={self.k:.1f}%, "
            f"included_terms={len(self)}, "
            f"total_influence={self.total_influence:.4f}, "
            f"higher_order_influence={self.higher_order_influence:.4f} "
            f"({self.higher_order_pct:.2f}%))"
        )

    def __getitem__(self, key: Any) -> float:
        if super().__contains__(key):
            return super().__getitem__(key)
        if hasattr(self, "tuple_sobols"):
            if isinstance(key, tuple) and key in self.tuple_sobols:
                return self.tuple_sobols[key]
            if isinstance(key, str):
                if (key,) in self.tuple_sobols:
                    return self.tuple_sobols[(key,)]
                # Match interaction string like "CLK × N_THREADS"
                parts = tuple(p.strip().split(" (")[0] for p in key.replace("*", "×").split("×"))
                if parts in self.tuple_sobols:
                    return self.tuple_sobols[parts]
                for k, v in self.items():
                    if k.split(" (")[0] == key:
                        return v
        return super().__getitem__(key)

    def __contains__(self, key: Any) -> bool:
        if super().__contains__(key):
            return True
        if hasattr(self, "tuple_sobols"):
            if isinstance(key, tuple) and key in self.tuple_sobols:
                return True
            if isinstance(key, str):
                if (key,) in self.tuple_sobols:
                    return True
                parts = tuple(p.strip().split(" (")[0] for p in key.replace("*", "×").split("×"))
                if parts in self.tuple_sobols:
                    return True
                for k in self.keys():
                    if k.split(" (")[0] == key:
                        return True
        return False

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class PlotterSobolMixin:
    """Mixin providing Sobol sensitivity calculations and bar charts for Plotter."""

    def get_axis_label(self, param: str, units: dict[str, Any] | None = None) -> str: ...
    def get_result_params(self, result: Any, df: Any = None) -> list[str]: ...

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

    def get_sobols_up_to_order(
        self,
        result: Any,
        qoi: str | None = None,
        k: float = 5.0,
        order: int | None = None,
        units: dict[str, Any] | None = None,
        **kwargs,
    ) -> SobolOrderResult:
        """
        Compute Sobol sensitivity indices of up to order n, where n is the smallest
        such that sobols of higher orders have total influence lower than k%.

        Uses SCAnalysis.get_pce_sobol_indices.
        """
        # Resolve QoI
        if qoi is None:
            if hasattr(result, "qois") and result.qois:
                qoi = result.qois[0]
            elif hasattr(result, "qoi_cols") and result.qoi_cols:
                qoi = result.qoi_cols[0]
            else:
                analysis_obj = getattr(result, "analysis", None)
                if analysis_obj is not None and hasattr(analysis_obj, "qoi_cols") and analysis_obj.qoi_cols:
                    qoi = analysis_obj.qoi_cols[0]
                else:
                    qoi = "energy_uj"

        # Resolve analysis object
        analysis = None
        if hasattr(result, "analysis") and result.analysis is not None:
            analysis = result.analysis
        elif hasattr(result, "get_pce_sobol_indices"):
            analysis = result
        elif hasattr(result, "results") and hasattr(result.results, "analysis"):
            analysis = result.results.analysis

        if analysis is None or not hasattr(analysis, "get_pce_sobol_indices"):
            raise ValueError(
                "SCAnalysis object with 'get_pce_sobol_indices' is required. "
                "Pass an EasyResult, RunData, or SCAnalysis instance."
            )

        # Call SCAnalysis.get_pce_sobol_indices
        ret = analysis.get_pce_sobol_indices(qoi, typ="all", **kwargs)
        if isinstance(ret, tuple) and len(ret) == 4:
            mean, D, D_u, S_u = ret
        elif isinstance(ret, tuple) and len(ret) == 3:
            mean, D, S_u = ret
            D_u = {}
        elif isinstance(ret, dict):
            mean, D, D_u, S_u = None, None, {}, ret
        else:
            raise ValueError(f"Unexpected return format from get_pce_sobol_indices: {type(ret)}")

        mean_val = float(np.asarray(mean).ravel()[0]) if (mean is not None and np.asarray(mean).size > 0) else None
        d_val = float(np.asarray(D).ravel()[0]) if (D is not None and np.asarray(D).size > 0) else None

        # Resolve parameter names
        param_names: list[str] = []
        if hasattr(analysis, "sampler") and hasattr(analysis.sampler, "vary"):
            if hasattr(analysis.sampler.vary, "get_keys"):
                param_names = list(analysis.sampler.vary.get_keys())
            elif isinstance(analysis.sampler.vary, dict):
                param_names = list(analysis.sampler.vary.keys())
        if not param_names and hasattr(result, "sampler") and hasattr(result.sampler, "vary"):
            if hasattr(result.sampler.vary, "get_keys"):
                param_names = list(result.sampler.vary.get_keys())
            elif isinstance(result.sampler.vary, dict):
                param_names = list(result.sampler.vary.keys())
        if not param_names and hasattr(self, "get_result_params"):
            param_names = self.get_result_params(result)

        max_idx = max((max(u) for u in S_u.keys() if u), default=-1)
        N = getattr(analysis, "N", max_idx + 1)
        if len(param_names) < N:
            param_names = param_names + [f"X{i}" for i in range(len(param_names), N)]

        # Convert values to float >= 0
        sobol_values: dict[tuple[int, ...], float] = {}
        for u, v in S_u.items():
            if not u:
                continue
            u_tuple = tuple(int(i) for i in u)
            v_arr = np.asarray(v)
            val = float(v_arr.ravel()[0]) if v_arr.size > 0 else 0.0
            if np.isnan(val) or val < 0.0:
                val = 0.0
            sobol_values[u_tuple] = val

        total_sobol = sum(sobol_values.values())

        # Determine threshold percentage: k is the percentage in k%.
        if "threshold_ratio" in kwargs and kwargs["threshold_ratio"] is not None:
            k_pct = float(kwargs["threshold_ratio"]) * 100.0
        else:
            k_pct = float(k)

        max_order = max((len(u) for u in sobol_values.keys()), default=1)
        if order is not None:
            n = max(1, min(int(order), max_order))
        else:
            # Smallest n such that sobols of higher orders have total influence lower than k%
            n = max_order
            for candidate_n in range(1, max_order + 1):
                higher_order_sum = sum(
                    v for u, v in sobol_values.items() if len(u) > candidate_n
                )
                higher_order_pct = (
                    (higher_order_sum / total_sobol * 100.0) if total_sobol > 0 else 0.0
                )
                if higher_order_pct < k_pct:
                    n = candidate_n
                    break

        order_influences = {
            m: sum(v for u, v in sobol_values.items() if len(u) == m)
            for m in range(1, max_order + 1)
        }
        higher_order_influence = sum(
            v for u, v in sobol_values.items() if len(u) > n
        )
        higher_order_pct = (
            (higher_order_influence / total_sobol * 100.0) if total_sobol > 0 else 0.0
        )
        total_influence = sum(
            v for u, v in sobol_values.items() if len(u) <= n
        )

        res = SobolOrderResult()
        res.n = n
        res.order = n
        res.k = k_pct
        res.total_influence = total_influence
        res.higher_order_influence = higher_order_influence
        res.higher_order_pct = higher_order_pct
        res.total_sobol = total_sobol
        res.order_influences = order_influences
        res.total_variance = d_val
        res.mean = mean_val
        res.raw_sobols = {u: sobol_values[u] for u in sobol_values if len(u) <= n}
        res.tuple_sobols = {}
        res.order_terms = {m: {} for m in range(1, n + 1)}
        res.formatted_labels = {}

        for u in sorted(sobol_values.keys(), key=lambda x: (len(x), x)):
            if len(u) <= n:
                raw_tuple = tuple(param_names[i] for i in u)
                if len(u) == 1:
                    label = param_names[u[0]]
                    fmt_label = self.get_axis_label(label, units)
                else:
                    fmt_label = " × ".join(self.get_axis_label(param_names[i], units) for i in u)

                val = sobol_values[u]
                res[fmt_label] = val
                res.tuple_sobols[raw_tuple] = val
                res.formatted_labels[u] = fmt_label
                res.order_terms[len(u)][fmt_label] = val

        return res

    def plot_sobols(
        self,
        result: Any,
        qoi: str | None = None,
        k: float = 5.0,
        order: int | None = None,
        subfig: SubFigure | None = None,
        title: str | None = None,
        units: dict[str, str | None] | None = None,
        include_higher_orders: bool = True,
        include_total: bool = True,
        **kwargs,
    ) -> Figure | SubFigure:
        """
        Plot Sobol sensitivity indices of up to order n, where n is the smallest
        such that sobols of higher orders have total influence lower than k%.

        Uses SCAnalysis.get_pce_sobol_indices.
        """
        res = self.get_sobols_up_to_order(result, qoi=qoi, k=k, order=order, units=units, **kwargs)

        ORDER_COLORS = {
            1: "dodgerblue",
            2: "mediumseagreen",
            3: "coral",
            4: "mediumpurple",
            5: "goldenrod",
        }

        labels: list[str] = []
        heights: list[float] = []
        colors: list[str] = []

        legend_handles: list[Any] = []

        if include_total:
            total_lbl = f"Total (orders 1..{res.n})" if res.n > 1 else "Total first order"
            labels.append(total_lbl)
            heights.append(res.total_influence)
            colors.append("salmon")
            legend_handles.append(Patch(facecolor="salmon", label=f"{total_lbl} ({res.total_influence:.1%})"))

        for m in range(1, res.n + 1):
            c = ORDER_COLORS.get(m, "teal")
            order_items = res.order_terms.get(m, {})
            for term_lbl, val in order_items.items():
                labels.append(term_lbl)
                heights.append(val)
                colors.append(c)
            if order_items:
                legend_handles.append(
                    Patch(facecolor=c, label=f"Order {m} total ({res.order_influences.get(m, 0.0):.1%})")
                )

        if include_higher_orders and res.higher_order_influence > 0:
            higher_lbl = f"Higher orders (> {res.n})"
            labels.append(higher_lbl)
            heights.append(res.higher_order_influence)
            colors.append("silver")
            legend_handles.append(
                Patch(facecolor="silver", label=f"{higher_lbl} ({res.higher_order_pct:.1f}%)")
            )

        n_bars = len(labels)
        fig_w = max(7.0, 0.65 * n_bars + 1.5)
        fig = subfig if subfig is not None else plt.figure(figsize=(fig_w, 5.0), layout="constrained")

        try:
            resolved_qoi = qoi or (result.qois[0] if hasattr(result, "qois") and result.qois else "energy_uj")
            if title:
                ax = fig.add_subplot(title=title)
            else:
                default_title = (
                    f"Sobol Sensitivity Indices up to Order {res.n} ({resolved_qoi}) "
                    f"[Higher Orders < {res.k:.1f}%]"
                )
                ax = fig.add_subplot(title=default_title)

            x_pos = np.arange(n_bars)
            ax.bar(x_pos, heights, color=colors, edgecolor="none", width=0.65)
            ax.set_ylabel(r"$S_u$", fontsize=14)

            max_h = max(heights) if heights else 1.0
            ax.set_ylim(0, max(1.0, max_h * 1.08))
            ax.set_xticks(x_pos)

            rot = 90 if n_bars > 4 else 45
            ha = "center" if rot == 90 else "right"
            ax.set_xticklabels(labels, rotation=rot, ha=ha)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

            if legend_handles:
                ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=9)

            fig._sobol_result = res
            ax._sobol_result = res
            return fig
        except Exception:
            if subfig is None:
                plt.close(fig)
            raise

    # Aliases on Plotter
    sobols_up_to_order = get_sobols_up_to_order
    sobols_of_up_to_order_n = get_sobols_up_to_order
    plot_sobols_up_to_order = plot_sobols
    plot_sobols_order_n = plot_sobols
    plot_sobols_n = plot_sobols


def get_sobols_up_to_order(
    result: Any,
    qoi: str | None = None,
    k: float = 5.0,
    order: int | None = None,
    units: dict[str, Any] | None = None,
    **kwargs,
) -> SobolOrderResult:
    """
    Compute Sobol sensitivity indices of up to order n, where n is the smallest
    such that sobols of higher orders have total influence lower than k%.

    Uses SCAnalysis.get_pce_sobol_indices.
    """
    from .plotter import Plotter
    plotter = Plotter.from_result(result, units=units)
    return plotter.get_sobols_up_to_order(result, qoi=qoi, k=k, order=order, units=units, **kwargs)


def plot_sobols(
    result: Any,
    qoi: str | None = None,
    k: float = 5.0,
    order: int | None = None,
    subfig: SubFigure | None = None,
    title: str | None = None,
    units: dict[str, str | None] | None = None,
    include_higher_orders: bool = True,
    include_total: bool = True,
    **kwargs,
) -> Figure | SubFigure:
    """
    Plot Sobol sensitivity indices of up to order n, where n is the smallest
    such that sobols of higher orders have total influence lower than k%.

    Uses SCAnalysis.get_pce_sobol_indices.
    """
    from .plotter import Plotter
    plotter = Plotter.from_result(result, units=units)
    return plotter.plot_sobols(
        result,
        qoi=qoi,
        k=k,
        order=order,
        subfig=subfig,
        title=title,
        units=units,
        include_higher_orders=include_higher_orders,
        include_total=include_total,
        **kwargs,
    )


def plot_sobols1(
    result: Result,
    qoi: str | None = None,
    subfig: SubFigure | None = None,
    title: str | None = None,
    units: dict[str, str | None] | None = None,
) -> Figure | SubFigure:
    """Plot first-order Sobol sensitivity indices."""
    from .plotter import Plotter
    plotter = Plotter.from_result(result, units=units)
    return plotter.plot_sobols1(result, qoi=qoi, subfig=subfig, title=title, units=units)


# Aliases
sobols_up_to_order = get_sobols_up_to_order
sobols_of_up_to_order_n = get_sobols_up_to_order
plot_sobols_up_to_order = plot_sobols
plot_sobols_order_n = plot_sobols
plot_sobols_n = plot_sobols

