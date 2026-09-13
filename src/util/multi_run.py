import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, Union, cast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..machines.machine import Machine, NONE as NONE_MACHINE
from .. import programs
from ..programs.program import Program


def _to_json_serializable(obj: Any) -> Any:
    """Convert numpy and pandas types to standard JSON-serializable types."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, Path):
        return str(obj)
    elif pd.isna(obj):
        return None
    return obj


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns if present (e.g. ('energy_uj', 0) -> 'energy_uj')."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else str(col) for col in df.columns]
    return df


def _compute_derived_qois(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived metrics such as energy in Joules, power in Watts, and EDP."""
    df = df.copy()
    if "energy_uj" in df.columns:
        if "energy_j" not in df.columns:
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


@dataclass
class RunData:
    """Encapsulates data and analysis state for a single experimental run."""
    path: Path
    benchmark_name: str
    machine_name: str
    machine: Machine | None = None
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    qois: list[str] = field(default_factory=lambda: ["energy_uj", "time"])
    input_params: list[str] = field(default_factory=lambda: ["N_THREADS", "CLK"])
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

            # Auto-detect QoIs and input params if defaults aren't fully matching
            existing_cols = set(self.df.columns)
            potential_qois = ["energy_uj", "energy_j", "energy_scaled", "time", "power_w", "edp_j_s"]
            self.qois = [q for q in potential_qois if q in existing_cols]

            potential_inputs = ["N_THREADS", "CLK", "THREADS", "CLK_LEVEL", "POWER_CAP", "PLACE_WIDE", "AFF_DISTANCE"]
            detected_inputs = [p for p in potential_inputs if p in existing_cols]
            if detected_inputs:
                self.input_params = detected_inputs

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

    def as_easy_result(self, qois: Sequence[str] | None = None) -> Any:
        """Return this run wrapped in EasyResult for compatibility with single-run plot functions."""
        from .data import EasyResult
        use_qois = list(qois) if qois else self.qois
        return EasyResult(
            df=self.df,
            qois=use_qois,
            analysis=self.analysis,
            campaign=self.campaign,
            sampler=self.sampler,
            results=self.results,
        )

    def analyze_individually(
        self,
        output_dir: Union[str, Path, None] = None,
        qois: Sequence[str] | None = None,
        save_plots: bool = True,
        fmt: str = "png",
        dpi: int = 200,
        quiet: bool = False,
        show: bool = False,
    ) -> dict[str, Any]:
        """
        Perform complete individual analysis and generate single-run plots.
        
        Parameters
        ----------
        output_dir : Optional folder to save plots, metrics JSON, and summary tables.
        qois : List of QoIs to evaluate (defaults to ["energy_uj", "time"]).
        save_plots : Whether to save plots to output_dir / "plots".
        fmt : Figure image format ('png', 'pdf', 'svg').
        dpi : Resolution for saved figures.
        quiet : Suppress print statements.
        show : Whether to display plots interactively.
        
        Returns
        -------
        dict with run metadata, metrics, and figure objects.
        """
        from . import plot

        out_path = Path(output_dir) if output_dir is not None else None
        plots_dir = (out_path / "plots") if (out_path and save_plots) else None
        if plots_dir:
            plots_dir.mkdir(parents=True, exist_ok=True)

        if not quiet:
            print(f"[{self.tag}] Running individual analysis...")

        # Initialize plot settings for this machine
        if self.machine is not None and self.machine.name != "NONE":
            try:
                plot.init(self.machine)
            except Exception as e:
                if not quiet:
                    print(f"[{self.tag}] Note: plot.init encountered: {e}")

        target_qois = [q for q in (qois or ["energy_uj", "time"]) if q in self.df.columns]
        if not target_qois and not self.df.empty:
            target_qois = [self.df.columns[-1]]

        easy_res = self.as_easy_result(target_qois)
        figures: dict[str, Any] = {}

        # 1. Generate individual plots for each QoI
        for qoi in target_qois:
            # 2D Grid best evaluations
            try:
                fig = plot.plot_grid_2D_best(easy_res, qoi=qoi)
                figures[f"grid_2d_best_{qoi}"] = fig
            except Exception as e:
                if not quiet:
                    print(f"[{self.tag}] grid_2D_best ({qoi}) skipped: {e}")

            # Sorted evaluations
            try:
                fig = plot.plot_sorted(easy_res, qoi=qoi)
                figures[f"sorted_evaluations_{qoi}"] = fig
            except Exception as e:
                if not quiet:
                    print(f"[{self.tag}] plot_sorted ({qoi}) skipped: {e}")

            # 2D Single dimension projection
            try:
                fig = plot.plot_2D_single_dimension(easy_res, qoi=qoi)
                figures[f"single_dimension_projections_{qoi}"] = fig
            except Exception as e:
                if not quiet:
                    print(f"[{self.tag}] plot_2D_single_dimension ({qoi}) skipped: {e}")

            # Boxplot per dimension
            try:
                fig = plot.plot_boxplot(easy_res, qoi=qoi)
                figures[f"boxplot_per_dimension_{qoi}"] = fig
            except Exception as e:
                if not quiet:
                    print(f"[{self.tag}] plot_boxplot ({qoi}) skipped: {e}")

            # First-order Sobol bar chart
            if self.results is not None and hasattr(self.results, "sobols_first"):
                try:
                    if not hasattr(self.results, "qois") or qoi in self.results.qois:
                        fig = plot.plot_sobols1(easy_res, qoi=qoi)
                        figures[f"sobol_indices_{qoi}"] = fig
                except Exception as e:
                    if not quiet:
                        print(f"[{self.tag}] plot_sobols1 ({qoi}) skipped: {e}")

            # Sobol treemap if available
            if self.results is not None and hasattr(self.results, "plot_sobols_treemap"):
                try:
                    if not hasattr(self.results, "qois") or qoi in self.results.qois:
                        from unittest.mock import patch
                        with patch("matplotlib.pyplot.show", lambda *args, **kwargs: None), \
                             patch.object(plt.Figure, "show", lambda self: None):
                            self.results.plot_sobols_treemap(qoi)
                        fig_tm = plt.gcf()
                        if len(fig_tm.axes) > 0:
                            for ax in fig_tm.axes:
                                ax.set_title("")
                            figures[f"sobol_treemap_{qoi}"] = fig_tm
                        else:
                            plt.close(fig_tm)
                except Exception as e:
                    if not quiet:
                        print(f"[{self.tag}] plot_sobols_treemap ({qoi}) skipped: {e}")

        # 2. Convergence & UQ plots
        hist = self.get_convergence_history()
        errors = hist.get("adaptation_error", [])
        if errors:
            try:
                fig_err, ax_err = plt.subplots(figsize=(7, 4.5), layout="constrained")
                ax_err.semilogy(range(1, len(errors) + 1), errors, marker="o", color="crimson")
                ax_err.set_xlabel("Iteration", fontsize=11)
                ax_err.set_ylabel("Adaptation Surplus Error", fontsize=11)
                ax_err.grid(True, linestyle="--", alpha=0.5)
                figures["adaptation_error_history"] = fig_err
            except Exception as e:
                if not quiet:
                    print(f"[{self.tag}] adaptation_errors plot skipped: {e}")

        if self.analysis is not None:
            try:
                fig_sc = plot.plot_stat_convergence(self, title=None)
                if fig_sc is not None:
                    figures["statistical_moments_convergence"] = fig_sc
            except Exception as e:
                if not quiet:
                    print(f"[{self.tag}] plot_stat_convergence skipped: {e}")

            try:
                fig_ah = plot.plot_adaptation_histogram(self, title=None)
                if fig_ah is not None:
                    figures["adaptation_histogram"] = fig_ah
            except Exception as e:
                if not quiet:
                    print(f"[{self.tag}] plot_adaptation_histogram skipped: {e}")

            try:
                fig_at = plot.plot_adaptation_table(self, title=None)
                if fig_at is not None:
                    figures["adaptation_table"] = fig_at
            except Exception as e:
                if not quiet:
                    print(f"[{self.tag}] plot_adaptation_table skipped: {e}")

        # Save figures if requested
        if plots_dir:
            for fig_name, fig_obj in figures.items():
                if fig_obj is not None:
                    if getattr(fig_obj, "_suptitle", None) is not None:
                        fig_obj._suptitle.set_text("")
                    for ax in getattr(fig_obj, "axes", []):
                        ax.set_title("")
                    fig_file = plots_dir / f"{fig_name}.{fmt}"
                    fig_obj.savefig(fig_file, dpi=dpi, bbox_inches="tight")
                    if not show:
                        plt.close(fig_obj)

        if not show:
            plt.close("all")
        else:
            for fig_obj in figures.values():
                if fig_obj is not None:
                    plt.figure(fig_obj.number)
                    plt.show()

        # 3. Compute metrics
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
                json.dump(_to_json_serializable(metrics), f, indent=2)

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
            "figures": figures,
            "output_dir": out_path,
        }


def _resolve_benchmark_class(name: str) -> type[Program]:
    """Find matching benchmark class in programs module, or return NONE."""
    name_clean = name.strip().upper()
    if hasattr(programs, name_clean):
        obj = getattr(programs, name_clean)
        if isinstance(obj, type):
            return cast(type[Program], obj)

    for attr in dir(programs):
        obj = getattr(programs, attr)
        if isinstance(obj, type):
            if attr.upper() == name_clean or getattr(obj, "name", "").upper() == name_clean:
                return cast(type[Program], obj)
    return programs.NONE


def load_run(
    path: Union[str, Path],
    benchmark: Union[str, type[Program], None] = None,
    machine: Union[Machine, None] = None,
    campaign_name: str = "energy",
) -> RunData:
    """
    Load a single run directory containing an EasyVVUQ campaign, Dakota results, or compilation CSV.
    """
    p = Path(path).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Run directory '{p}' does not exist.")

    # 1. Infer benchmark and machine names from path if not provided
    parts = p.parts
    path_bench = None
    path_mach = None

    # Check parts against known programs
    for part in reversed(parts):
        for b_name in ["HPCG", "JA", "LULESH", "FFT", "FAKEWORK", "FLETCHER"]:
            if part.upper() == b_name:
                path_bench = b_name
                break
        if path_bench:
            break

    if benchmark is None:
        benchmark_name = path_bench or (parts[-2] if len(parts) >= 2 else "UNKNOWN_BENCH")
    elif isinstance(benchmark, str):
        benchmark_name = benchmark
    else:
        benchmark_name = getattr(benchmark, "name", benchmark.__name__)

    # 2. Load machine.pkl if available
    loaded_machine = None
    machine_file = p / "machine.pkl"
    if machine_file.exists():
        try:
            with open(machine_file, "rb") as f:
                loaded_machine = pickle.load(f)
        except Exception:
            pass

    if machine is not None:
        active_machine = machine
    elif loaded_machine is not None:
        active_machine = loaded_machine
    else:
        active_machine = NONE_MACHINE

    # Infer machine name
    if loaded_machine and loaded_machine.name != "NONE":
        machine_name = p.name  # Use folder name for clean label (e.g. 'cei' rather than 'cei6')
    else:
        machine_name = p.name

    # 3. Attempt loading EasyVVUQ campaign and analysis
    from .. import energyuq
    from easyvvuq.analysis.sc_analysis import SCAnalysisResults

    camp = None
    anal = None
    last_res = None
    sampler = None
    df = pd.DataFrame()

    campaign_db = p / "campaign" / "campaign.db"
    alt_campaign_db = p / "campaign.db"

    if campaign_db.exists() or alt_campaign_db.exists():
        bench_cls = _resolve_benchmark_class(benchmark_name)
        try:
            camp, anal, active_machine = energyuq.load(
                bench_cls, active_machine, campaign_name, dir=str(p)
            )
            df = camp.get_collation_result()
            last_res = cast(SCAnalysisResults, camp.get_last_analysis())
            sampler = energyuq.get_sampler(camp)
        except Exception as e:
            # Fallback if campaign load fails
            pass

    # 4. Fallback to compilation.csv if campaign loading didn't produce df
    if df.empty and (p / "compilation.csv").exists():
        try:
            df = pd.read_csv(p / "compilation.csv")
        except Exception:
            pass

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


def discover_runs(
    base_dir: Union[str, Path] = "runs",
    pattern: str = "*/*/*",
    benchmark_filter: Union[str, Sequence[str], None] = None,
    machine_filter: Union[str, Sequence[str], None] = None,
) -> list[RunData]:
    """
    Search recursively for all valid run directories under base_dir.
    Matches directories containing campaign.db, analysis, machine.pkl, or compilation.csv.
    """
    base = Path(base_dir).resolve()
    if not base.exists():
        return []

    # Normalize filters
    if isinstance(benchmark_filter, str):
        benchmark_filter = [benchmark_filter.upper()]
    elif benchmark_filter is not None:
        benchmark_filter = [b.upper() for b in benchmark_filter]

    if isinstance(machine_filter, str):
        machine_filter = [machine_filter.lower()]
    elif machine_filter is not None:
        machine_filter = [m.lower() for m in machine_filter]

    # Find potential run directories
    candidate_dirs = set()

    # Match by glob pattern
    for candidate in base.glob(pattern):
        if candidate.is_dir() and candidate.name != "campaign":
            if (candidate / "campaign" / "campaign.db").exists() or \
               (candidate / "campaign.db").exists() or \
               (candidate / "compilation.csv").exists() or \
               (candidate / "machine.pkl").exists() or \
               (candidate / "analysis").exists():
                candidate_dirs.add(candidate)

    # Also search any subdirectory containing campaign.db
    for db_path in base.glob("**/campaign.db"):
        run_folder = db_path.parent.parent if db_path.parent.name == "campaign" else db_path.parent
        if run_folder.name != "campaign":
            candidate_dirs.add(run_folder)

    for comp_path in base.glob("**/compilation.csv"):
        if comp_path.parent.name != "campaign":
            candidate_dirs.add(comp_path.parent)

    runs: list[RunData] = []
    for candidate in sorted(candidate_dirs):
        try:
            run_data = load_run(candidate)
            if benchmark_filter and run_data.benchmark_name.upper() not in benchmark_filter:
                continue
            if machine_filter and run_data.machine_name.lower() not in machine_filter:
                continue
            runs.append(run_data)
        except Exception as e:
            print(f"Warning: Failed to load run from {candidate}: {e}")

    return runs


class RunCollection:
    """Collection wrapper for managing, querying, and analyzing multiple runs at once."""

    def __init__(self, runs: Sequence[RunData] | None = None):
        self.runs: list[RunData] = list(runs) if runs else []

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def __getitem__(self, idx: int | str | slice) -> Union[RunData, "RunCollection"]:
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
        base_dir: Union[str, Path] = "runs",
        pattern: str = "*/*/*",
        benchmarks: Union[str, Sequence[str], None] = None,
        machines: Union[str, Sequence[str], None] = None,
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
        benchmarks: Union[str, Sequence[str], None] = None,
        machines: Union[str, Sequence[str], None] = None,
        predicate: Union[Callable[[RunData], bool], None] = None,
    ) -> "RunCollection":
        """Return a new filtered RunCollection."""
        if isinstance(benchmarks, str):
            benchmarks = [benchmarks.upper()]
        elif benchmarks is not None:
            benchmarks = [b.upper() for b in benchmarks]

        if isinstance(machines, str):
            machines = [machines.lower()]
        elif machines is not None:
            machines = [m.lower() for m in machines]

        filtered: list[RunData] = []
        for r in self.runs:
            if benchmarks and r.benchmark_name.upper() not in benchmarks:
                continue
            if machines and r.machine_name.lower() not in machines:
                continue
            if predicate and not predicate(r):
                continue
            filtered.append(r)

        return RunCollection(filtered)

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

    def export_csv(self, output_dir: Union[str, Path]):
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
        output_dir: Union[str, Path] = "individual_analyses",
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

    def compile_pdf(
        self,
        output_path: Union[str, Path] = "reports/energyuq_compiled_report.pdf",
        qoi: str = "energy_uj",
        qois: Sequence[str] = ("energy_uj", "time"),
        include_global: bool = False,
        sobol_modes: Sequence[str] = ("grouped_bar", "heatmap"),
        dpi: int = 150,
        quiet: bool = False,
        **kwargs,
    ) -> Path:
        """
        Compile all multi-run and individual run plots into a structured, indexed, and bookmarked master PDF report.

        Parameters
        ----------
        output_path : Destination path for the output PDF.
        qoi : Primary QoI for multi-run evaluations (e.g. 'energy_uj').
        qois : Sequence of QoIs for individual single-run evaluations.
        include_global : Whether to include a global cross-benchmark comparison section at the start.
        sobol_modes : Sobol visualization modes ('grouped_bar', 'heatmap', 'stacked_bar').
        dpi : DPI resolution for rendered figures.
        quiet : Suppress progress logging.

        Returns
        -------
        Path to compiled PDF report.
        """
        from scripts.compile_plots_pdf import compile_plots_pdf
        return compile_plots_pdf(
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
    runs: Union[RunCollection, Sequence[RunData]],
    output_dir: Union[str, Path] = "individual_analyses",
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
