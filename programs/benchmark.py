from subprocess import CompletedProcess, run
import os
from typing import ClassVar
from .program import Program

class ExecuteSH(Program):
    @classmethod
    def run(cls, params: list) -> CompletedProcess[str]:
        params = [int(x) for x in params]
        N_THREADS = params[0]
        execute = f"{os.path.dirname(__file__)}/../scripts/execute.sh"
        print(f"Running {cls.name} with {N_THREADS} threads")
        # print(f"Current working directory: {os.getcwd()}")
        return run(
            [execute, cls.name, str(N_THREADS)],
            capture_output=True,
            text=True,
        )
    @classmethod
    def report(cls):
        pass

class FFT(ExecuteSH):
    name: ClassVar[str] = "FFT"

class HPCG(ExecuteSH):
     name: ClassVar[str] = "HPCG"

class JA(ExecuteSH):
     name: ClassVar[str] = "JA"

class LULESH(ExecuteSH):
     name: ClassVar[str] = "LULESH"

class NONE(ExecuteSH):
     name: ClassVar[str] = "NONE"