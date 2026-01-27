from subprocess import CompletedProcess
from typing import ClassVar, Protocol


class Program(Protocol):

    name: ClassVar[str]

    @classmethod
    def run(cls, params: list) -> CompletedProcess[str]: ...
    @classmethod
    def report(cls, ): ...
