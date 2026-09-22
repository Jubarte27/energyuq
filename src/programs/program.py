from collections.abc import Iterable
from subprocess import CompletedProcess
from typing import ClassVar, Protocol

from ..util.data import ExecutionParams


class Program(Protocol):

    name: ClassVar[str]

    @classmethod
    def run(cls, params: ExecutionParams, args: Iterable[str]) -> CompletedProcess[str]: ...
    @classmethod
    def report(cls, ): ...
