from subprocess import CompletedProcess
import subprocess
import numpy as np
import os
from typing import Protocol, ClassVar, Unpack

from .program import Preparation, Program, RunArgs

class Benchmark(Program):
    short_name: str

    @classmethod
    def run(cls, params: list[str]) -> CompletedProcess[str]:
        N_THREADS = int(params[0])

        executable = f"{os.path.dirname(__file__)}../scripts/execute.sh"

        print(f"Running {cls.name} with {N_THREADS} threads")

        # for debugging
        print(f"Current working directory: {os.getcwd()}")

        # Run the executable and capture output
        result = subprocess.run(
            [
                executable,
                cls.short_name,
                str(N_THREADS),
            ],
            capture_output=True,
            text=True,
        )

        return result

    @classmethod
    def report(cls) -> None:
        return None
