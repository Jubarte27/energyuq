from typing import ClassVar, Protocol


class Machine(Protocol):
    name: ClassVar[str]
    freq: ClassVar[list[int]]
    physical_core_count: ClassVar[int]
