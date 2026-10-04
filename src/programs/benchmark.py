import os
from collections.abc import Iterable
from subprocess import CompletedProcess, run
from typing import ClassVar

from ..util.data import ExecutionParams
from .program import Program


class ExecuteSH(Program):
    @classmethod
    def run(cls, params: ExecutionParams, args: Iterable[str]) -> CompletedProcess[str]:
        execute = f"{os.path.dirname(__file__)}/../../scripts/run/execute-hpc-benchmarks.sh"
        print(f"Running {cls.name} with {params.n_threads} threads")
        proc_bind = params.machine.proc_bind[params.binding]
        places = params.machine.places[params.place_wideness]
        freq = params.machine.freq[params.freq_level]
        boost = params.machine.turbo_boost[params.boost]
        numa = params.machine.numactl[params.numa] if params.numa is not None else "false"
        return run(
            [execute, cls.name, str(params.n_threads), str(freq), str(boost), str(numa)],
            env=os.environ | {"OMP_PLACES": places, "OMP_PROC_BIND": proc_bind},
            capture_output=True,
            text=True,
            check=False
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

class PO(ExecuteSH):
     name: ClassVar[str] = "PO"

class ST(ExecuteSH):
     name: ClassVar[str] = "ST"

class LULESH(ExecuteSH):
     name: ClassVar[str] = "LULESH"

class MW(ExecuteSH):
     name: ClassVar[str] = "MW"

class NONE(ExecuteSH):
     name: ClassVar[str] = "NONE"

class FAKEWORK(ExecuteSH):
     name: ClassVar[str] = "FAKE_WORK"


class NAS_BT(ExecuteSH):
     name: ClassVar[str] = "NAS_BT"

class NAS_CG(ExecuteSH):
     name: ClassVar[str] = "NAS_CG"

class NAS_EP(ExecuteSH):
     name: ClassVar[str] = "NAS_EP"

class NAS_FT(ExecuteSH):
     name: ClassVar[str] = "NAS_FT"

class NAS_IS(ExecuteSH):
     name: ClassVar[str] = "NAS_IS"

class NAS_LU(ExecuteSH):
     name: ClassVar[str] = "NAS_LU"

class NAS_MG(ExecuteSH):
     name: ClassVar[str] = "NAS_MG"

class NAS_SP(ExecuteSH):
     name: ClassVar[str] = "NAS_SP"

class NAS_UA(ExecuteSH):
     name: ClassVar[str] = "NAS_UA"


class RODINIA_BFS(ExecuteSH):
     name: ClassVar[str] = "RODINIA_BFS"

class RODINIA_HOTSPOT(ExecuteSH):
     name: ClassVar[str] = "RODINIA_HOTSPOT"

class RODINIA_HOTSPOT3D(ExecuteSH):
     name: ClassVar[str] = "RODINIA_HOTSPOT3D"

class RODINIA_SRAD_V1(ExecuteSH):
     name: ClassVar[str] = "RODINIA_SRAD_V1"

class RODINIA_SRAD_V2(ExecuteSH):
     name: ClassVar[str] = "RODINIA_SRAD_V2"

class RODINIA_STREAMCLUSTER(ExecuteSH):
     name: ClassVar[str] = "RODINIA_STREAMCLUSTER"


class PARBOIL_SPMV(ExecuteSH):
     name: ClassVar[str] = "PARBOIL_SPMV"