from subprocess import CompletedProcess, run
import os
from typing import ClassVar, Iterable
from ..util.data import ExecutionParams
from .program import Program

class ExecuteSH(Program):
    @classmethod
    def run(cls, params: ExecutionParams, args: Iterable[str]) -> CompletedProcess[str]:
        execute = f"{os.path.dirname(__file__)}/../../scripts/execute-hpc-benchmarks.sh"
        print(f"Running {cls.name} with {params.n_threads} threads")
        proc_bind = params.machine.proc_bind[params.affinity_distance]
        places = params.machine.places[params.place_wideness]
        return run(
            [execute, cls.name, str(params.n_threads)],
            env={"OMP_PLACES": places, "OMP_PROC_BIND": proc_bind},
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