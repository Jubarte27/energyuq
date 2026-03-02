#!/usr/bin/env python3
import numpy as np
import subprocess
from programs import *
from machines import *

from textwrap import indent

FREQUECNY_POS = 1
# POWER_CAP_POS = 2


def main(program: type[Program], machine: type[Machine]):
    cpu_set(machine)
    start, end = run(program)
    report(start, end)
    # program.report()


def cpu_set(machine: type[Machine]):
    with open("input.csv", "r") as f:
        x = np.array(f.readline().split(",")).astype("int")
    
    result = subprocess.run(
        ["cpupower", "frequency-set", "--governor", "userspace"],
        capture_output=True,
        text=True,
    )
    output_CompletedProcess("Governor set", result, True)
    if result.returncode != 0:
        exit(result.returncode)

    frequency = machine.freq[x[FREQUECNY_POS]]

    result = subprocess.run(
        ["cpupower", "frequency-set", "--freq", str(frequency)],
        capture_output=True,
        text=True,
    )
    output_CompletedProcess("Frequency set", result, True)
    if result.returncode != 0:
        exit(result.returncode)
    # power_cap = x[POWER_CAP_POS]
    # power_cap *= 10**6
    # set_sysfs("/sys/class/powercap/intel-rapl:0/?????", power_cap, "Power cap")


def run(program: type[Program]):
    with open("input.csv", "r") as f:
        x = f.readline().split(",")

    start = energy_uj()
    print()
    result = program.run(x)
    print()
    end = energy_uj()

    output_CompletedProcess(program.name, result)

    return start, end


def report(start_energy: int, end_energy: int):
    used_energy = end_energy - start_energy
    if used_energy < 0:
        max_energy = max_energy_range_uj()
        used_energy = (end_energy + max_energy) - start_energy

    with open("output.csv", "w") as f:
        f.write("energy_uj\n")
        f.write(f"{float(used_energy)}\n")


def prepare_report():
    return


def max_energy_range_uj() -> int:
    result = subprocess.run(
        ["cat", "/sys/class/powercap/intel-rapl:0/max_energy_range_uj"],
        capture_output=True,
        text=True,
    )
    output_CompletedProcess("max_energy_range_uj", result)
    if result.returncode != 0:
        exit(result.returncode)
    return int(result.stdout)

def energy_uj() -> int:
    result = subprocess.run(
        ["cat", "/sys/class/powercap/intel-rapl:0/energy_uj"],
        capture_output=True,
        text=True,
    )
    output_CompletedProcess("energy_uj", result)
    if result.returncode != 0:
        exit(result.returncode)
    return int(result.stdout)


def set_sysfs(full_path: str, value: object, name=None):
    result = subprocess.run(
        ["tee", full_path],
        input=str(value),
        capture_output=True,
        text=True,
    )

    output_CompletedProcess(full_path if name is None else name, result)

    if result.returncode != 0:
        exit(result.returncode)


def output_CompletedProcess(name: str, result: subprocess.CompletedProcess, quiet_success = False):
    success = result.returncode == 0
    if quiet_success and success:
        pretty_str = pretty_out(f"{name}", "Done")
    else:
        pretty_str = pretty_out(
            f"{name}",
            pretty_out("ExitStatus", result.returncode),
            pretty_out("Stdout", result.stdout),
            pretty_out("Stderr", result.stderr),
        )
    print(pretty_str)


def pretty_out(name: str, *args):
    # Removes falsey values, like "", 0, [], etc
    args = [arg for arg in args if arg]
    if len(args) < 1:
        return ""
    INDENTATION = "    "

    out = "\n".join(str(arg) for arg in args)
    out = out.rstrip("\n")
    if "\n" in out:
        out = f"{name}\n{indent(out, INDENTATION)}"
    else:
        out = f"{name}: {out}"
    return out


if __name__ == "__main__":
    program = NONE
    machine = Glados
    print("Running on Glados")
    main(program, machine)
