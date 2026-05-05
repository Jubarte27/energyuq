from typing import ClassVar, Protocol
from dataclasses import dataclass

class Machine(Protocol):
    name: ClassVar[str]
    freq: ClassVar[list[int]]
    physical_core_count: ClassVar[int]

class NONE(Machine):
    name = "NONE"
    freq = [0]
    physical_core_count = 0

@dataclass
class MachineParams():
    machine: type[Machine] = NONE
    n_threads: int = 1
    freq_level: int = 0
