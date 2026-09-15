from dataclasses import dataclass

from easyvvuq.campaign import Campaign
from easyvvuq.sampling.stochastic_collocation import SCSampler
from pandas import DataFrame
from easyvvuq.analysis.sc_analysis import SCAnalysis, SCAnalysisResults
from ..machines.machine import Machine

@dataclass
class  ExecutionParams():
    machine: Machine
    n_threads: int
    freq_level: int
    boost: int
    place_wideness: int
    affinity_distance: int


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
