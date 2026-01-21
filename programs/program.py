from subprocess import CompletedProcess
from typing import NotRequired, Protocol, ClassVar,  TypedDict, Unpack
from collections.abc import Callable

class RunArgs(TypedDict):
    before: NotRequired[Callable[[], None]]
    after: NotRequired[Callable[[], None]]

class Preparation(Protocol):
    @classmethod
    def valid(cls) -> bool: return True

class Program(Protocol):
    name: ClassVar[str]
    @classmethod
    def run(cls, params: list[str]) -> CompletedProcess[str]: ...
    @classmethod
    def report(cls) -> dict[str, list] | None: ...


