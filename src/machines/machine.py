from typing import ClassVar, Protocol
from dataclasses import dataclass

class Machine(Protocol):
    name: ClassVar[str]
    freq: ClassVar[list[int]]
    max_threads: ClassVar[int]
    package: ClassVar[list[int]] = [0]
    sub_package: ClassVar[list[int]] = [-1] # nome ruim, valores ruins
    places: ClassVar[list[str]] = ["threads", "cores"]
    proc_bind: ClassVar[list[str]] = ["true", "false"]

class NONE(Machine):
    name = "NONE"
    freq = [0]
    max_threads = 0

@dataclass
class MachineParams():
    machine: type[Machine] = NONE
    n_threads: int = 1
    freq_level: int = 0
