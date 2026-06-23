from dataclasses import dataclass

from easyvvuq.campaign import Campaign
from easyvvuq.sampling.stochastic_collocation import SCSampler
from pandas import DataFrame
from easyvvuq.analysis.sc_analysis import SCAnalysis, SCAnalysisResults
from ..machines.machine import Machine

@dataclass
class  ExecutionParams():
    machine: type[Machine]
    n_threads: int
    freq_level: int
    place_wideness: int = 0
    affinity_distance: int = 0


@dataclass
class EnergyReading():
    start: int
    end: int
    package: int
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