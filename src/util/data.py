import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, cast

import dill
import numpy as np
import pandas as pd
from easyvvuq.analysis.sc_analysis import SCAnalysis, SCAnalysisResults
from easyvvuq.campaign import Campaign
from easyvvuq.sampling.stochastic_collocation import SCSampler
from pandas import DataFrame

from src.util.constants import QOI, QOIS

from ..machines.machine import Machine


def compute_edp(energy_uj: Any, time: Any) -> Any:
    """Calculate Energy-Delay Product from energy in μJ and time in seconds."""
    return (energy_uj * 1e-6) * time

_TYPE_HANDLERS = {
        (np.integer, int): int,
        (np.floating, float): float,
        np.ndarray: lambda o: o.tolist(),
        (set, tuple, list): lambda o: [to_serializable_primitive(x) for x in o],
        dict: lambda o: {str(k): to_serializable_primitive(v) for k, v in o.items()},
        Path: str,
    }

def to_serializable_primitive(obj: Any) -> Any:
    """Convert numpy, pandas, dataclass, and path types to standard JSON/msgpack serializable types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    
    for types, handler in _TYPE_HANDLERS.items():
        if isinstance(obj, types):
            return handler(obj)
    try:
        if pd.isna(cast(Any,obj)):
            return None
    except ValueError:
        pass
    return obj


@dataclass
class ExecutionParams:
    machine: Machine
    n_threads: int
    freq_level: int
    boost: int
    place_wideness: int
    binding: int
    numa: int | None = None

    @classmethod
    def from_dict(cls, machine: Machine, data: dict[str, Any]) -> "ExecutionParams":
        n_threads = int(data.get("N_THREADS", data.get("THREADS", machine.max_threads)))
        freq_level = int(data.get("CLK", data.get("CLK_LEVEL", len(machine.freq) - 1)))
        place_wideness = int(data.get("PLACES", len(machine.places) - 1))
        binding = int(data.get("BINDING", len(machine.proc_bind) - 1))
        boost = int(data.get("BOOST", len(machine.turbo_boost) - 1))
        numa_raw = data.get("NUMA")
        numa = int(numa_raw) if numa_raw is not None and str(numa_raw).strip() != "" else None
        return cls(
            machine=machine,
            n_threads=n_threads,
            freq_level=freq_level,
            place_wideness=place_wideness,
            binding=binding,
            boost=boost,
            numa=numa,
        )

    @classmethod
    def from_args(cls, machine: Machine, args: Sequence[Any]) -> "ExecutionParams":
        def arg(i: int, default: int = 0) -> int:
            if len(args) > i and str(args[i]).strip() != "":
                try:
                    return int(args[i])
                except (ValueError, TypeError):
                    return default
            return default

        numa = None
        if len(args) > 5 and str(args[5]).strip() != "":
            try:
                numa = int(args[5])
            except (ValueError, TypeError):
                numa = None

        return cls(
            machine=machine,
            n_threads=arg(0, machine.max_threads),
            freq_level=arg(1, len(machine.freq) - 1),
            place_wideness=arg(2, len(machine.places) - 1),
            binding=arg(3, len(machine.proc_bind) - 1),
            boost=arg(4, len(machine.turbo_boost) - 1),
            numa=numa,
        )

class ExecuteWrapper:
    def __init__(self, function: Callable[[dict[str, Any]], Any]):
        self.function = dill.dumps(function)

    def start(self, previous: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not previous:
            return None
        rundir = previous.get("rundir")
        old_dir = os.getcwd() if rundir else None
        try:
            if rundir:
                os.chdir(rundir)
            dill.loads(self.function)(previous["run_info"]["params"])
        finally:
            if old_dir:
                os.chdir(old_dir)
        return previous

    def finished(self) -> bool:
        return True

    def finalise(self) -> None:
        pass

    def succeeded(self) -> bool:
        return True


class EnergyUQCampaign:
    """Encapsulates an EasyVVUQ Campaign along with EnergyUQ metadata:
    root directory, machine profile.
    """

    def __init__(
        self,
        campaign: Campaign,
        root_path: Path | str,
        machine: Machine,
        numa: bool = False,
        qoi: str = QOI,
        qois: list[str] = QOIS,
    ):
        self.campaign: Campaign = campaign
        self.root_path = Path(root_path)
        self.machine: Machine = machine
        self.numa = numa
        self.qoi = qoi
        self.qois = qois

    @property
    def sampler(self) -> SCSampler:
        return cast(SCSampler, self.campaign.get_active_sampler())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.campaign, name)

    def __repr__(self) -> str:
        camp_name = getattr(self.campaign, "campaign_name", None)
        return (
            f"EnergyUQCampaign(name={camp_name!r}, "
            f"root_path={self.root_path}, machine={self.machine.name}, "
            f"numa={self.numa}, qoi={self.qoi} "
        )

@dataclass
class EnergyReading:
    start: int
    end: int
    package: int | str
    sub_package: int


@dataclass
class limit:
    lower: int | float
    upper: int | float

@dataclass
class Result:
    df: DataFrame
    qois: list[str]

@dataclass
class EasyResult(Result):
    analysis: SCAnalysis
    campaign: EnergyUQCampaign
    sampler: SCSampler
    results: SCAnalysisResults
