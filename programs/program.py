from subprocess import CompletedProcess
from typing import Protocol


class Program(Protocol):
    @staticmethod
    def name() -> str: ...
    @staticmethod
    def run(params: list) -> CompletedProcess[str]: ...
    @staticmethod
    def report(): ...
