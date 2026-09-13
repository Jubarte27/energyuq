import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, Union, cast
import numpy as np
import pandas as pd

from ..machines.machine import Machine, NONE as NONE_MACHINE
from .. import programs
from ..programs.program import Program


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

    def __getitem__(self, idx: int | str) -> RunData:
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
