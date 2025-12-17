from typing import Protocol


class Machine(Protocol):
    @staticmethod
    def name() -> str: ...
    @staticmethod
    def freq() -> list[int]: ...

