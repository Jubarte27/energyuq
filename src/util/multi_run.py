import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .. import programs
from ..machines.machine import NONE as NONE_MACHINE
from ..machines.machine import Machine, load_machine
from ..plotting.plotter import Plotter
from ..programs.program import Program
from .data import to_serializable_primitive


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns if present (e.g. ('energy_uj', 0) -> 'energy_uj')."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else str(col) for col in df.columns]
    return df


def _compute_derived_qois(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived metrics such as energy in Joules, power in Watts, and EDP."""
    df = df.copy()
    if "energy_uj" in df.columns and "energy_j" not in df.columns:
        df["energy_j"] = df["energy_uj"] * 1e-6

    if "energy_uj" in df.columns and "time" in df.columns:
        # Avoid division by zero
        safe_time = df["time"].replace(0, np.nan)
        if "power_w" not in df.columns:
            df["power_w"] = (df["energy_uj"] * 1e-6) / safe_time
        if "edp_j_s" not in df.columns:
            df["edp_j_s"] = (df["energy_uj"] * 1e-6) * df["time"]

    return df


def _compute_pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Find the 2D Pareto frontier minimizing both x_col and y_col."""
    if x_col not in df.columns or y_col not in df.columns:
        return pd.DataFrame()

    sorted_df = df.sort_values(by=[x_col, y_col], ascending=[True, True]).copy()
    pareto_indices = []
    current_min_y = float("inf")

    for idx, row in sorted_df.iterrows():
        y_val = row[y_col]
        if y_val < current_min_y:
            pareto_indices.append(idx)
            current_min_y = y_val

    return sorted_df.loc[pareto_indices].sort_values(by=x_col)


#: QoIs produced by the campaign pipeline, in reporting order.
MEASURED_QOIS: tuple[str, ...] = ("energy_uj", "time", "EDP", "energy_scaled")

#: Figure families :meth:`RunData.analyze_individually` knows how to build.
#: Pass a subset as ``figures=`` to build only what a caller will render.
FIGURE_FAMILIES: tuple[str, ...] = (
    "projections",     # grid_2d_best, single_dimension_projections
    "distributions",   # sorted_evaluations, boxplot_per_dimension
    "sobols",          # sobol_indices, sobol_treemap
    "convergence",     # adaptation_error_history, moments, histogram, table
)


@dataclass
class RunData:
    """Encapsulates data and analysis state for a single experimental run.

    ``qois`` and ``input_params`` default to ``None``, meaning "detect from the
    data". Pass an explicit list to pin them down instead.
    """
    path: Path
    benchmark_name: str
    machine_name: str
    machine: Machine | None = None
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    qois: list[str] | None = None
    input_params: list[str] | None = None
    campaign: Any = None
    analysis: Any = None
    results: Any = None
    sampler: Any = None
    tag: str = ""

    def __post_init__(self):
        if not self.tag:
            self.tag = f"{self.benchmark_name}_{self.machine_name}"

        # Flatten and compute derived QoIs
        if not self.df.empty:
            self.df = _flatten_columns(self.df)
            self.df = _compute_derived_qois(self.df)

            existing_cols = set(self.df.columns)
            if self.qois is None:
                self.qois = [q for q in MEASURED_QOIS if q in existing_cols]
            if self.input_params is None:
                self.input_params = [
                    p for p in ("N_THREADS", "THREADS", "CLK", "CLK_LEVEL", "POWER_CAP", "PLACE_WIDE", "AFF_DISTANCE")
                    if p in existing_cols
                ] or ["N_THREADS", "CLK"]

        if self.qois is None:
            self.qois = ["energy_uj", "time"]
        if self.input_params is None:
            self.input_params = ["N_THREADS", "CLK"]

    @property
    def sample_count(self) -> int:
        return len(self.df)

    @property
    def iteration_count(self) -> int:
        if "iteration" in self.df.columns:
            return int(self.df["iteration"].max()) + 1
        if self.analysis and hasattr(self.analysis, "adaptation_errors"):
            return len(self.analysis.adaptation_errors)
        return 0

    def best_config(self, qoi: str = "energy_uj", mode: str = "min") -> dict[str, Any]:
        """Return row dictionary of the optimal configuration evaluated."""
        if self.df.empty or qoi not in self.df.columns:
            return {}
        idx = self.df[qoi].idxmin() if mode == "min" else self.df[qoi].idxmax()
        row: dict[str, Any] = cast(dict[str, Any], self.df.loc[idx].to_dict())
        row["run_tag"] = self.tag
        row["benchmark"] = self.benchmark_name
        row["machine"] = self.machine_name
        return row

    def get_sobols(self, qoi: str = "energy_uj") -> dict[str, float]:
        """Extract first-order Sobol indices for the specified QoI."""
        sobols: dict[str, float] = {}
        if self.results is not None and hasattr(self.results, "sobols_first"):
            try:
                raw_sobols = self.results.sobols_first(qoi)
                for k, v in raw_sobols.items():
                    val = float(v[0]) if isinstance(v, (np.ndarray, list)) else float(v)
                    sobols[k] = val
            except Exception:
                pass

        if not sobols and self.analysis is not None and hasattr(self.analysis, "get_sobol_indices"):
            try:
                indices = self.analysis.get_sobol_indices(qoi)
                # Map (0,) -> first param, (1,) -> second param
                for perm, val in indices.items():
                    if len(perm) == 1:
                        dim = perm[0]
                        param_name = self.input_params[dim] if dim < len(self.input_params) else f"param_{dim}"
                        v = float(val[0]) if isinstance(val, (np.ndarray, list)) else float(val)
                        sobols[param_name] = v
            except Exception:
                pass

        return sobols

    def get_convergence_history(self) -> dict[str, list[float]]:
        """Return history of adaptation surplus errors, mean, and std across refinement steps."""
        history: dict[str, list[float]] = {
            "adaptation_error": [],
            "mean": [],
            "std": []
        }
        if self.analysis is not None:
            if hasattr(self.analysis, "adaptation_errors"):
                history["adaptation_error"] = [float(x) for x in self.analysis.adaptation_errors]
            if hasattr(self.analysis, "mean_history"):
                history["mean"] = [float(np.linalg.norm(m)) for m in self.analysis.mean_history]
            if hasattr(self.analysis, "std_history"):
                history["std"] = [float(np.linalg.norm(s)) for s in self.analysis.std_history]
        return history

    def get_statistics(self, qoi: str = "energy_uj") -> dict[str, float]:
        """Compute basic statistical summary for a QoI."""
        if self.df.empty or qoi not in self.df.columns:
            return {}
        col = self.df[qoi].dropna()
        return {
            "mean": float(col.mean()),
            "std": float(col.std()),
            "min": float(col.min()),
            "max": float(col.max()),
            "median": float(col.median()),
            "iqr": float(col.quantile(0.75) - col.quantile(0.25)),
            "count": int(col.count()),
        }

    def pareto_front(self, qoi_x: str = "time", qoi_y: str = "energy_uj") -> pd.DataFrame:
        """Return Pareto frontier between two QoIs."""
        return _compute_pareto_front(self.df, qoi_x, qoi_y)

    def plotter(self) -> Plotter:
        """Return a Plotter configured for this run's machine."""
        return Plotter(self.machine)

    def _sobol_treemap(self, qoi: str) -> Any:
        """Render the EasyVVUQ Sobol treemap headlessly and return the current figure."""
        with patch("matplotlib.pyplot.show", lambda *a, **k: None), \
                patch.object(Figure, "show", lambda self: None):
            self.results.plot_sobols_treemap(qoi)
        fig = plt.gcf()
        if not fig.axes:
            plt.close(fig)
            return None
        for ax in fig.axes:
            ax.set_title("")
        return fig

    def analyze_individually(
        self,
        output_dir: str | Path | None = None,
        qois: Sequence[str] | None = None,
        figures: Sequence[str] | None = None,
        save_plots: bool = True,
        fmt: str = "png",
        dpi: int = 200,
        quiet: bool = False,
        show: bool = False,
    ) -> dict[str, Any]:
        """
        Perform individual analysis and generate single-run plots.

        Parameters
        ----------
        output_dir : Optional folder to save plots, metrics JSON, and summary tables.
        qois : List of QoIs to evaluate (defaults to ["energy_uj", "time"]).
        figures : Subset of :data:`FIGURE_FAMILIES` to build (default: all of them).
        save_plots : Whether to save plots to output_dir / "plots".
        fmt : Figure image format ('png', 'pdf', 'svg').
        dpi : Resolution for saved figures.
        quiet : Suppress print statements.
        show : Whether to display plots interactively.

        Returns
        -------
        dict with run metadata, metrics, and figure objects.
        """
        out_path = Path(output_dir) if output_dir is not None else None
        plots_dir = (out_path / "plots") if (out_path and save_plots) else None
        if plots_dir:
            plots_dir.mkdir(parents=True, exist_ok=True)

        if not quiet:
            print(f"[{self.tag}] Running individual analysis...")

        plotter = self.plotter()

        target_qois = [q for q in (qois or ["energy_uj", "time"]) if q in self.df.columns]
        if not target_qois and not self.df.empty:
            target_qois = [self.df.columns[-1]]

        wanted = set(figures) if figures is not None else set(FIGURE_FAMILIES)
        unknown = wanted - set(FIGURE_FAMILIES)
        if unknown:
            raise ValueError(f"Unknown figure families {sorted(unknown)}; expected {list(FIGURE_FAMILIES)}")

        figs: dict[str, Any] = {}

        def add(key: str, make: Callable[[], Any]) -> None:
            """Build one figure, recording it (or the failure reason) under ``key``."""
            try:
                fig = make()
            except Exception as e:
                if not quiet:
                    print(f"[{self.tag}] {key} skipped: {e}")
                return
            if fig is not None:
                figs[key] = fig

        # 1. Per-QoI figures
        for qoi in target_qois:
            if "projections" in wanted:
                add(f"grid_2d_best_{qoi}", lambda q=qoi: plotter.plot_grid_2D_best(self, qoi=q))
                add(f"single_dimension_projections_{qoi}", lambda q=qoi: plotter.plot_2D_single_dimension(self, qoi=q))
            if "distributions" in wanted:
                add(f"sorted_evaluations_{qoi}", lambda q=qoi: plotter.plot_sorted(self, qoi=q))
                add(f"boxplot_per_dimension_{qoi}", lambda q=qoi: plotter.plot_boxplot(self, qoi=q))

            # Sobol-based figures need analysis results for this QoI.
            results_qois = getattr(self.results, "qois", None)
            if "sobols" in wanted and self.results is not None \
                    and hasattr(self.results, "sobols_first") \
                    and (results_qois is None or qoi in results_qois):
                add(f"sobol_indices_{qoi}", lambda q=qoi: plotter.plot_sobols1(self, qoi=q))
                if hasattr(self.results, "plot_sobols_treemap"):
                    add(f"sobol_treemap_{qoi}", lambda q=qoi: self._sobol_treemap(q))

        # 2. Convergence & UQ plots
        if "convergence" in wanted:
            errors = self.get_convergence_history().get("adaptation_error", [])
            if errors:
                def _adaptation_error_history() -> Figure:
                    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
                    ax.semilogy(range(1, len(errors) + 1), errors, marker="o", color="crimson")
                    ax.set_xlabel("Iteration", fontsize=11)
                    ax.set_ylabel("Adaptation Surplus Error", fontsize=11)
                    ax.grid(True, linestyle="--", alpha=0.5)
                    return fig

                add("adaptation_error_history", _adaptation_error_history)

            if self.analysis is not None:
                add("statistical_moments_convergence", lambda: plotter.plot_stat_convergence(self, title=None))
                add("adaptation_histogram", lambda: plotter.plot_adaptation_histogram(self, title=None))
                add("adaptation_table", lambda: plotter.plot_adaptation_table(self, title=None))

        # Save figures if requested
        if plots_dir:
            for fig_name, fig_obj in figs.items():
                if getattr(fig_obj, "_suptitle", None) is not None:
                    fig_obj._suptitle.set_text("")
                for ax in getattr(fig_obj, "axes", []):
                    ax.set_title("")
                fig_obj.savefig(plots_dir / f"{fig_name}.{fmt}", dpi=dpi, bbox_inches="tight")
                if not show:
                    plt.close(fig_obj)

        if not show:
            plt.close("all")
        else:
            for fig_obj in figs.values():
                plt.figure(fig_obj.number)
                plt.show()

        # 3. Compute metrics
        hist = self.get_convergence_history()
        metrics: dict[str, Any] = {
            "run_tag": self.tag,
            "benchmark": self.benchmark_name,
            "machine": self.machine_name,
            "path": str(self.path),
            "sample_count": self.sample_count,
            "iterations": self.iteration_count,
            "input_params": self.input_params,
            "qois": {},
            "convergence": hist,
        }

        # Calculate relative differences in mean and std history if available
        if self.analysis is not None and hasattr(self.analysis, "mean_history") and len(self.analysis.mean_history) > 1:
            mean_h = self.analysis.mean_history
            std_h = self.analysis.std_history
            rel_mean_diff = []
            rel_std_diff = []
            for i in range(1, len(mean_h)):
                d_m = float(np.linalg.norm(mean_h[i] - mean_h[i - 1], np.inf)) / (float(np.linalg.norm(mean_h[i - 1], np.inf)) + 1e-12)
                d_s = float(np.linalg.norm(std_h[i] - std_h[i - 1], np.inf)) / (float(np.linalg.norm(std_h[i - 1], np.inf)) + 1e-12)
                rel_mean_diff.append(d_m)
                rel_std_diff.append(d_s)
            metrics["relative_mean_diff_history"] = rel_mean_diff
            metrics["relative_std_diff_history"] = rel_std_diff

        for qoi in target_qois:
            qoi_dict: dict[str, Any] = {
                "sample_statistics": self.get_statistics(qoi),
                "sobol_indices": self.get_sobols(qoi),
                "best_config_min": self.best_config(qoi, mode="min"),
                "best_config_max": self.best_config(qoi, mode="max"),
            }
            if self.analysis is not None and hasattr(self.analysis, "get_moments"):
                try:
                    mean_f, var_f = self.analysis.get_moments(qoi)
                    m_val = float(mean_f[0]) if isinstance(mean_f, (np.ndarray, list)) else float(mean_f)
                    v_val = float(var_f[0]) if isinstance(var_f, (np.ndarray, list)) else float(var_f)
                    s_val = float(np.sqrt(max(v_val, 0.0)))
                    cv_val = s_val / m_val if m_val != 0 else 0.0
                    qoi_dict["surrogate_moments"] = {
                        "mean": m_val,
                        "var": v_val,
                        "std": s_val,
                        "cv": cv_val,
                    }
                except Exception:
                    pass

            metrics["qois"][qoi] = qoi_dict

        # Save files to output_dir
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)
            if not self.df.empty:
                self.df.to_csv(out_path / "evaluated_samples.csv", index=False)

            with open(out_path / "metrics.json", "w") as f:
                json.dump(to_serializable_primitive(metrics), f, indent=2)

            # Summary text report
            summary_lines = [
                f"=== Individual Run Report: {self.tag} ===",
                f"Benchmark:   {self.benchmark_name}",
                f"Machine:     {self.machine_name}",
                f"Directory:   {self.path}",
                f"Samples:     {self.sample_count}",
                f"Iterations:  {self.iteration_count}",
                "",
                "--- Quantities of Interest ---",
            ]
            for qoi, q_data in metrics["qois"].items():
                stats = q_data.get("sample_statistics", {})
                best_min = q_data.get("best_config_min", {})
                summary_lines.append(f"QoI: {qoi}")
                summary_lines.append(f"  Min: {stats.get('min', 0):.4e}, Mean: {stats.get('mean', 0):.4e}, Max: {stats.get('max', 0):.4e}")
                summary_lines.append(f"  Optimal Min Config: {', '.join(f'{k}={best_min.get(k)}' for k in self.input_params if k in best_min)}")
                if "surrogate_moments" in q_data:
                    sm = q_data["surrogate_moments"]
                    summary_lines.append(f"  Surrogate: Mean={sm.get('mean', 0):.4e}, Std={sm.get('std', 0):.4e}, CV={sm.get('cv', 0):.4f}")
                if "sobol_indices" in q_data:
                    sobols_str = ", ".join(f"{k}={v:.3f}" for k, v in q_data["sobol_indices"].items())
                    summary_lines.append(f"  Sobol First-Order: {sobols_str}")
                summary_lines.append("")

            with open(out_path / "summary.txt", "w") as f:
                f.write("\n".join(summary_lines) + "\n")

            if not quiet:
                print(f"[{self.tag}] Saved individual report to '{out_path}'.")

        return {
            "run_tag": self.tag,
            "benchmark": self.benchmark_name,
            "machine": self.machine_name,
            "metrics": metrics,
            "figures": figs,
            "output_dir": out_path,
        }


#: Benchmark names recognised when inferring a run's benchmark from its path.
KNOWN_PATH_BENCHMARKS: tuple[str, ...] = ("HPCG", "JA", "LULESH", "FFT", "ST", "MW", "PO", "FLETCHER", "FAKEWORK")

#: Files/entries whose presence marks a directory as a loadable run.
RUN_MARKERS: tuple[str, ...] = (
    "campaign.db",
    "campaign/campaign.db",
    "analysis",
    "compilation.csv",
    "machine.msgpack",
    "machine.pkl",
)


def _resolve_benchmark_class(name: str) -> type[Program]:
    """Find matching benchmark class in programs module, or return NONE."""
    name_clean = name.strip().upper()
    if hasattr(programs, name_clean):
        obj = getattr(programs, name_clean)
        if isinstance(obj, type):
            return cast(type[Program], obj)

    for attr in dir(programs):
        obj = getattr(programs, attr)
        if isinstance(obj, type) and (
            attr.upper() == name_clean or getattr(obj, "name", "").upper() == name_clean
        ):
            return cast(type[Program], obj)
    return programs.NONE


def load_run(
    path: str | Path,
    benchmark: str | type[Program] | None = None,
    machine: Machine | None = None,
    campaign_name: str = "energy",
    quiet: bool = True,
) -> RunData:
    """
    Load a single run directory containing an EasyVVUQ campaign, Dakota results, or compilation CSV.
    """
    p = Path(path).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Run directory '{p}' does not exist.")

    active_machine = machine if machine is not None else (load_machine(p) or NONE_MACHINE)
    machine_name = active_machine.name

    from ..programs.benchmark import ExecuteSH

    if benchmark is None:
        #gambiarra
        benches = {b.__name__.lower() for b in ExecuteSH.__subclasses__()}
        benchmark_name = next((part for part in reversed(p.parts) if part.lower() in benches), None)
        if benchmark_name is None:
            raise RuntimeError(f"Don't know the benchmark for {p}")
    elif isinstance(benchmark, str):
        benchmark_name = benchmark
    else:
        benchmark_name = getattr(benchmark, "name", benchmark.__name__)

    from .. import energyuq

    camp = None
    anal = None
    last_res = None
    sampler = None
    df = pd.DataFrame()

    if (p / "campaign" / "campaign.db").exists() or (p / "campaign.db").exists():
        try:
            camp, anal, active_machine = energyuq.load(
                _resolve_benchmark_class(benchmark_name), active_machine, campaign_name, dir=str(p)
            )
            df = camp.get_collation_result()
            last_res = camp.get_last_analysis()
            sampler = camp.sampler
        except Exception as e:
            if not quiet:
                print(f"Warning: campaign load failed for '{p}': {e}")
    elif (p / "compilation.csv").exists():
        df = pd.read_csv(p / "compilation.csv")

    # 5. Build and return RunData
    return RunData(
        path=p,
        benchmark_name=benchmark_name,
        machine_name=machine_name,
        machine=active_machine,
        df=df,
        campaign=camp,
        analysis=anal,
        results=last_res,
        sampler=sampler,
    )


def _is_run_dir(candidate: Path) -> bool:
    """True when ``candidate`` carries any of the recognised run markers."""
    return candidate.name != "campaign" and any((candidate / m).exists() for m in RUN_MARKERS)


def discover_runs(
    base_dir: str | Path = "runs",
    pattern: str = "*/*",
    benchmark_filter: str | Sequence[str] | None = None,
    machine_filter: str | Sequence[str] | None = None,
) -> list[RunData]:
    """
    Search recursively for all valid run directories under base_dir.
    A directory qualifies when it contains any of :data:`RUN_MARKERS`.
    """
    base = Path(base_dir).resolve()
    if not base.exists():
        return []

    # Normalize filters to comparable, case-normalized lists
    benchmarks = _normalize_filter(benchmark_filter, upper=True)
    machines = _normalize_filter(machine_filter)

    candidate_dirs = {c for c in base.glob(pattern) if c.is_dir() and _is_run_dir(c)}

    # Also pick up nested layouts that the glob pattern does not reach
    for marker in ("campaign.db", "compilation.csv", "analysis"):
        for hit in base.glob(f"**/{marker}"):
            run_folder = hit.parent.parent if hit.parent.name == "campaign" else hit.parent
            if _is_run_dir(run_folder):
                candidate_dirs.add(run_folder)

    runs: list[RunData] = []
    for candidate in sorted(candidate_dirs):
        try:
            run_data = load_run(candidate)
        except Exception as e:
            print(f"Warning: Failed to load run from {candidate}: {e}")
            continue
        if benchmarks and run_data.benchmark_name.upper() not in benchmarks:
            continue
        if machines and run_data.machine_name.lower() not in machines:
            continue
        runs.append(run_data)

    return runs


def _normalize_filter(value: str | Sequence[str] | None, upper: bool = False) -> list[str]:
    """Normalize a string-or-sequence filter into a list of comparable names."""
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)
    return [str(v).upper() if upper else str(v).lower() for v in items]


class RunCollection:
    """Collection wrapper for managing, querying, and analyzing multiple runs at once."""

    def __init__(self, runs: Sequence[RunData] | None = None):
        self.runs: list[RunData] = list(runs) if runs else []

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def __getitem__(self, idx: int | str | slice) -> "RunData | RunCollection":
        if isinstance(idx, slice):
            return RunCollection(self.runs[idx])
        if isinstance(idx, int):
            return self.runs[idx]
        for run in self.runs:
            if run.tag == idx or run.machine_name == idx or run.benchmark_name == idx:
                return run
        raise KeyError(f"Run with tag/name '{idx}' not found.")

    @classmethod
    def discover(
        cls,
        base_dir: str | Path = "runs",
        pattern: str = "*/*/*",
        benchmarks: str | Sequence[str] | None = None,
        machines: str | Sequence[str] | None = None,
    ) -> "RunCollection":
        runs = discover_runs(base_dir, pattern, benchmark_filter=benchmarks, machine_filter=machines)
        return cls(runs)

    def add(self, run: RunData):
        self.runs.append(run)

    @property
    def benchmarks(self) -> list[str]:
        return sorted(list({r.benchmark_name for r in self.runs}))

    @property
    def machines(self) -> list[str]:
        return sorted(list({r.machine_name for r in self.runs}))

    def filter(
        self,
        benchmarks: str | Sequence[str] | None = None,
        machines: str | Sequence[str] | None = None,
        predicate: Callable[[RunData], bool] | None = None,
    ) -> "RunCollection":
        """Return a new filtered RunCollection."""
        wanted_benchmarks = _normalize_filter(benchmarks, upper=True)
        wanted_machines = _normalize_filter(machines)

        return RunCollection([
            r for r in self.runs
            if (not wanted_benchmarks or r.benchmark_name.upper() in wanted_benchmarks)
            and (not wanted_machines or r.machine_name.lower() in wanted_machines)
            and (predicate is None or predicate(r))
        ])

    def summary_table(self) -> pd.DataFrame:
        """
        Generate comprehensive summary table comparing all runs.
        Includes sample count, iteration count, min/mean/max energy, time, power, and optimal configs.
        """
        rows = []
        for r in self.runs:
            row: dict[str, Any] = {
                "benchmark": r.benchmark_name,
                "machine": r.machine_name,
                "sample_count": r.sample_count,
                "iterations": r.iteration_count,
            }

            # Energy stats
            if "energy_uj" in r.df.columns:
                row["min_energy_uj"] = r.df["energy_uj"].min()
                row["mean_energy_uj"] = r.df["energy_uj"].mean()
                row["max_energy_uj"] = r.df["energy_uj"].max()
                row["min_energy_j"] = row["min_energy_uj"] * 1e-6
                row["mean_energy_j"] = row["mean_energy_uj"] * 1e-6

            # Time stats
            if "time" in r.df.columns:
                row["min_time_s"] = r.df["time"].min()
                row["mean_time_s"] = r.df["time"].mean()
                row["max_time_s"] = r.df["time"].max()

            # Power stats
            if "power_w" in r.df.columns:
                row["min_power_w"] = r.df["power_w"].min()
                row["mean_power_w"] = r.df["power_w"].mean()
                row["max_power_w"] = r.df["power_w"].max()

            # Optimal config for energy
            best_energy = r.best_config("energy_uj", mode="min")
            for param in r.input_params:
                if param in best_energy:
                    row[f"best_energy_{param}"] = best_energy[param]

            # Optimal config for time
            best_time = r.best_config("time", mode="min")
            for param in r.input_params:
                if param in best_time:
                    row[f"best_time_{param}"] = best_time[param]

            # First-order Sobol indices for energy
            sobols = r.get_sobols("energy_uj")
            for param, val in sobols.items():
                row[f"sobol_energy_{param}"] = val

            rows.append(row)

        return pd.DataFrame(rows)

    def sobols_table(self, qoi: str = "energy_uj") -> pd.DataFrame:
        """Matrix of first-order Sobol indices: rows=runs, cols=parameters."""
        rows = []
        for r in self.runs:
            sobols = r.get_sobols(qoi)
            row = {
                "benchmark": r.benchmark_name,
                "machine": r.machine_name,
                "tag": r.tag,
                **sobols
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def best_configurations_table(self, qoi: str = "energy_uj", mode: str = "min") -> pd.DataFrame:
        """Table of optimal configurations across all runs for a given QoI."""
        rows = [r.best_config(qoi, mode=mode) for r in self.runs if not r.df.empty]
        return pd.DataFrame(rows)

    def convergence_table(self) -> pd.DataFrame:
        """Long-format DataFrame of adaptation errors per iteration for all runs."""
        rows = []
        for r in self.runs:
            hist = r.get_convergence_history()
            errors = hist.get("adaptation_error", [])
            for iteration, err in enumerate(errors):
                rows.append({
                    "benchmark": r.benchmark_name,
                    "machine": r.machine_name,
                    "tag": r.tag,
                    "iteration": iteration + 1,
                    "adaptation_error": err
                })
        return pd.DataFrame(rows)

    def export_csv(self, output_dir: str | Path):
        """Export summary, Sobols, and best configuration tables to CSV files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary_df = self.summary_table()
        if not summary_df.empty:
            summary_df.to_csv(out / "runs_summary.csv", index=False)

        sobol_df = self.sobols_table("energy_uj")
        if not sobol_df.empty:
            sobol_df.to_csv(out / "sobol_indices_energy.csv", index=False)

        best_df = self.best_configurations_table("energy_uj")
        if not best_df.empty:
            best_df.to_csv(out / "best_configurations_energy.csv", index=False)

        conv_df = self.convergence_table()
        if not conv_df.empty:
            conv_df.to_csv(out / "convergence_history.csv", index=False)

        print(f"Exported run analyses to {out}")

    def analyze_each(
        self,
        output_dir: str | Path = "individual_analyses",
        qois: Sequence[str] | None = None,
        save_plots: bool = True,
        fmt: str = "png",
        dpi: int = 200,
        quiet: bool = False,
        show: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """
        Perform individual analysis in bulk across all runs in this collection.
        
        Parameters
        ----------
        output_dir : Root directory where each run will have its own subfolder.
        qois : List of QoIs to evaluate (defaults to ["energy_uj", "time"]).
        save_plots : Whether to save individual plot images.
        fmt : Plot image format ('png', 'pdf', 'svg').
        dpi : Resolution for saved plots.
        quiet : Suppress non-essential output.
        show : Whether to display plots interactively.
        
        Returns
        -------
        dict mapping run tag to each run's individual report dictionary.
        """
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        if not quiet:
            print(f"Starting bulk individual analysis for {len(self.runs)} runs...")

        reports = {}
        index_rows = []

        for r in self.runs:
            run_sub = out_root / f"{r.benchmark_name}_{r.machine_name}"
            rep = r.analyze_individually(
                output_dir=run_sub,
                qois=qois,
                save_plots=save_plots,
                fmt=fmt,
                dpi=dpi,
                quiet=quiet,
                show=show,
            )
            reports[r.tag] = rep

            row: dict[str, Any] = {
                "tag": r.tag,
                "benchmark": r.benchmark_name,
                "machine": r.machine_name,
                "sample_count": r.sample_count,
                "iterations": r.iteration_count,
                "report_dir": str(run_sub),
            }
            # Add QoI summary stats
            for qoi in (qois or ["energy_uj", "time"]):
                if qoi in rep["metrics"]["qois"]:
                    q_data = rep["metrics"]["qois"][qoi]
                    stats = q_data.get("sample_statistics", {})
                    row[f"{qoi}_min"] = stats.get("min")
                    row[f"{qoi}_mean"] = stats.get("mean")
                    if "surrogate_moments" in q_data:
                        row[f"{qoi}_surrogate_mean"] = q_data["surrogate_moments"].get("mean")
                        row[f"{qoi}_surrogate_std"] = q_data["surrogate_moments"].get("std")
            index_rows.append(row)

        # Save master index table
        if index_rows:
            df_index = pd.DataFrame(index_rows)
            df_index.to_csv(out_root / "individual_analysis_index.csv", index=False)

        if not quiet:
            print(f"Completed bulk individual analysis for {len(self.runs)} runs. Master index saved to '{out_root / 'individual_analysis_index.csv'}'.")

        return reports

    def compile_html(
        self,
        output_path: str | Path = "reports/index.html",
        qoi: str = "energy_uj",
        qois: Sequence[str] = ("energy_uj", "time"),
        include_global: bool = True,
        sobol_modes: Sequence[str] = ("grouped_bar", "heatmap", "stacked_bar"),
        dpi: int = 150,
        quiet: bool = False,
        **kwargs,
    ) -> Path:
        """
        Compile all multi-run and individual run plots into the interactive HTML report.

        Thin wrapper around :func:`scripts.compile_plots_html.compile_plots_html`, which
        is the single entry point for report generation.

        Parameters
        ----------
        output_path : Destination path for the HTML shell.
        qoi : Primary QoI for multi-run evaluations (e.g. 'energy_uj').
        qois : Sequence of QoIs for individual single-run evaluations.
        include_global : Whether to include a global cross-benchmark section.
        sobol_modes : Sobol visualization modes ('grouped_bar', 'heatmap', 'stacked_bar').
        dpi : DPI resolution for rendered figures.
        quiet : Suppress progress logging.

        Returns
        -------
        Path to the compiled HTML report.
        """
        from scripts.compile_plots_html import compile_plots_html

        return compile_plots_html(
            runs=self,
            output_path=output_path,
            qoi=qoi,
            qois=qois,
            include_global=include_global,
            sobol_modes=sobol_modes,
            dpi=dpi,
            quiet=quiet,
            **kwargs,
        )


def analyze_runs_individually(
    runs: RunCollection | Sequence[RunData],
    output_dir: str | Path = "individual_analyses",
    qois: Sequence[str] | None = None,
    save_plots: bool = True,
    fmt: str = "png",
    dpi: int = 200,
    quiet: bool = False,
    show: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Perform individual analysis in bulk for a collection or list of runs.
    """
    collection = RunCollection(runs) if not isinstance(runs, RunCollection) else runs
    return collection.analyze_each(
        output_dir=output_dir,
        qois=qois,
        save_plots=save_plots,
        fmt=fmt,
        dpi=dpi,
        quiet=quiet,
        show=show,
    )
