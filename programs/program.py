from subprocess import CompletedProcess
from typing import Protocol, ClassVar


class Program(Protocol):
    name: str
    @classmethod
    def run(cls, params: list) -> CompletedProcess[str]: ...
    @classmethod
    def report(cls): ...
