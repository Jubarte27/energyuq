from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors


class PlotterDiagnosticsMixin:
    """Mixin providing EasyVVUQ diagnostic plots and gradient visualizers for Plotter."""

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

    def _capture_analysis_plot(
        self,
        result: Any,
        method_name: str,
        fig_name: str | None = None,
        title: str | None = None,
    ) -> Figure | None:
        """Capture and clean an EasyVVUQ analytical plot rendered via pyplot."""
        from unittest.mock import patch

        analysis = getattr(result, "analysis", result)
        if not hasattr(analysis, method_name):
            return None

        if fig_name:
            plt.close(fig_name)

        with patch("matplotlib.pyplot.show", lambda *args, **kwargs: None):
            getattr(analysis, method_name)()

        fig = plt.figure(fig_name) if fig_name else plt.gcf()
        if len(fig.axes) == 0:
            if fig_name:
                plt.close(fig_name)
            else:
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

    def plot_stat_convergence(self, result: Any, title: str | None = None) -> Figure | None:
        """Generate EasyVVUQ statistical moments convergence plot."""
        return self._capture_analysis_plot(result, "plot_stat_convergence", fig_name="stat_conv", title=title)

    def plot_adaptation_histogram(self, result: Any, title: str | None = None) -> Figure | None:
        """Generate EasyVVUQ adaptation histogram plot."""
        return self._capture_analysis_plot(result, "adaptation_histogram", fig_name="adapt_hist", title=title)

    def plot_adaptation_table(self, result: Any, title: str | None = None) -> Figure | None:
        """Generate EasyVVUQ adaptation table plot."""
        return self._capture_analysis_plot(result, "adaptation_table", fig_name=None, title=title)


# Standalone module-level delegators
_default_diagnostics = PlotterDiagnosticsMixin()


def draw_gradients(*color_list) -> Figure:
    return _default_diagnostics.draw_gradients(*color_list)


def plot_stat_convergence(result: Any, title: str | None = None) -> Figure | None:
    return _default_diagnostics.plot_stat_convergence(result, title=title)


def plot_adaptation_histogram(result: Any, title: str | None = None) -> Figure | None:
    return _default_diagnostics.plot_adaptation_histogram(result, title=title)


def plot_adaptation_table(result: Any, title: str | None = None) -> Figure | None:
    return _default_diagnostics.plot_adaptation_table(result, title=title)

