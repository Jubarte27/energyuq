from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from easyvvuq.campaign import Campaign
from easyvvuq.sampling.stochastic_collocation import SCSampler
from pandas import DataFrame
from easyvvuq.analysis.sc_analysis import SCAnalysis, SCAnalysisResults

if TYPE_CHECKING:
    from ..machines.machine import Machine


def compute_edp(energy_uj: Any, time: Any) -> Any:
    """Calculate Energy-Delay Product from energy in μJ and time in seconds."""
    return (energy_uj * 1e-6) * time


def to_serializable_primitive(obj: Any) -> Any:
    """Convert numpy, pandas, dataclass, and path types to standard JSON/msgpack serializable types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return [to_serializable_primitive(x) for x in obj]
    if isinstance(obj, list):
        return [to_serializable_primitive(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable_primitive(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if pd.isna(obj):
        return None
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


@dataclass
class EnergyReading():
    start: int
    end: int
    package: int | str
    sub_package: int


@dataclass
class limit():
    lower: int # | float
    upper: int # | float

@dataclass
class Result():
    df: DataFrame
    qois: list[str]

@dataclass
class EasyResult(Result):
    analysis: SCAnalysis
    campaign: Campaign
    sampler: SCSampler
    results: SCAnalysisResults
