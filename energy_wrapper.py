#!/usr/bin/env python3
import numpy as np
import subprocess
from . import programs

POWER_CAP_POS = 2
FREQUECNY_POS = 3


def main(program: programs.Program):
    cpu_set()
    run(program)
    program.report()



def cpu_set():
    with open("input.csv", "r") as f:
        x = np.array(f.readline().split(",")).astype("int")

    power_cap = x[POWER_CAP_POS]
    power_cap *= 10**6

    frequency = x[FREQUECNY_POS]

    result = subprocess.run(
        ["sudo", "cpupower", "frequency-set", "--freq"],
        capture_output=True,
        text=True,
    )

    output_CompletedProcess("Frequency set", result)
    if result.returncode != 0: exit(result.returncode)

    # set_sysfs("/sys/class/powercap/intel-rapl:0/?????", power_cap, "Power cap")


def run(program: programs.Program):
    with open("input.csv", "r") as f:
        x = f.readline().split(",")

    output_CompletedProcess(program.name(), program.run(x))


def energy_report():
    result = subprocess.run(
        ["sudo", "cat", "/sys/class/powercap/intel-rapl:0/energy_uj"],
        capture_output=True,
        text=True,
    )


def set_sysfs(full_path, value, name=None):
    result = subprocess.run(
        ["sudo", "tee", full_path],
        input=str(value),
        capture_output=True,
        text=True,
    )

    output_CompletedProcess(full_path if name is None else name, result)

    if result.returncode != 0:
        exit(result.returncode)


def output_CompletedProcess(name: str, result: subprocess.CompletedProcess):
    print(f"--{name} set--")
    if result.returncode:
        print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f'Stdout: "{result.stdout}"')
    if result.stderr:
        print(f'Stderr: "{result.stderr}"')
    print(f"--{name} set--")


if __name__ == "__main__":
    program = programs.Fletcher
    main(program)
