from subprocess import CompletedProcess
import subprocess
from typing import ClassVar
import numpy as np
import os

from .program import Program

class Fletcher(Program):
    name: ClassVar[str] = "Fletcher"

    @classmethod
    def run(cls, params: list) -> CompletedProcess[str]:
        params = [int(x) for x in params]
        BSIZE_X = params[0]
        BSIZE_Y = params[1]

        fletcher_exe = f"{os.path.dirname(__file__)}/HIP-3D/ModelagemFletcher.exe"

        # Get the corresponding powers of two
        BSIZE_X = 2**BSIZE_X
        BSIZE_Y = 2**BSIZE_Y

        print(f"Running Fletcher with BSIZE_X={BSIZE_X}, BSIZE_Y={BSIZE_Y}")

        # Print current working directory for debugging
        print(f"Current working directory: {os.getcwd()}")

        # Run the executable and capture output
        return subprocess.run(
            [
                fletcher_exe,
                "TTI",
                "472",
                "472",
                "472",
                "16",
                "12.5",
                "12.5",
                "12.5",
                "0.001",
                "0.01",
                str(BSIZE_X),
                str(BSIZE_Y),
                "1",
            ],
            capture_output=True,
            text=True,
        )

    @classmethod
    def report(cls):
        msamples = np.array([0.0])
        # Check if Report.csv exists
        report_path = "Report.csv"
        if os.path.exists(report_path):
            print(f"Report.csv found at {report_path}")
            # Read Report.csv and extract MSamples
            with open(report_path, "r") as f:
                lines = f.readlines()
                # MSamples is in the second line
                if len(lines) >= 2:
                    # Parse the second line: walltime; 3.135677; MSamples; 408.282039; ...
                    parts = lines[1].split(";")
                    # Find MSamples value (it's after the MSamples key)
                    for i, part in enumerate(parts):
                        if "MSamples" in part and i + 1 < len(parts):
                            msamples_value = float(parts[i + 1].strip())
                            msamples = np.array([msamples_value])
                            print(f"Extracted MSamples: {msamples_value}")
                            break
                else:
                    print("Report.csv doesn't have enough lines")
        else:
            print(f"Report.csv NOT found at {report_path}")
            print(f"Files in current directory: {os.listdir('.')}")

        # output csv file
        header = "f"
        np.savetxt(
            "output.csv",
            np.atleast_2d(msamples).T,
            delimiter=",",
            comments="",
            header=header,
        )
