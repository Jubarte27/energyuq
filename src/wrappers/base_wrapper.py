import subprocess
from typing import Iterable, Union
from ..programs import *
from ..machines import *
from ..util.data import ExecutionParams
from time import perf_counter

from textwrap import indent

def prepare_and_exeute(machine: type[Machine], program: type[Program], params: ExecutionParams, args: Union[None, Iterable[str]]):
    if not args:
        args = []
    
    cpu_set(machine, params.freq_level)
    start, end, t = run(program, params, args)
    return report(start, end, t)
    

def cpu_set(machine: type[Machine], freq_level: int):
    frequency = machine.freq[freq_level]

    result = subprocess.run(
        ["cpupower", "frequency-set", "--governor", "userspace"],
        capture_output=True,
        text=True,
    )
    output_CompletedProcess("Governor set", result, True)
    if result.returncode != 0:
        raise Exception(f"Unable to set governor using cpupower, am i root? code: {result.returncode}")

    result = subprocess.run(
        ["cpupower", "frequency-set", "--freq", str(frequency)],
        capture_output=True,
        text=True,
    )
    output_CompletedProcess("Frequency set", result, True)
    if result.returncode != 0:
        raise Exception(f"Unable to set frequency using cpupower, am i root? code: {result.returncode}")
    # power_cap = x[POWER_CAP_POS]
    # power_cap *= 10**6
    # set_sysfs("/sys/class/powercap/intel-rapl:0/?????", power_cap, "Power cap")


def run(program: type[Program], params: ExecutionParams, parameter_list: Iterable[str]):

    start = energy_uj()
    t = perf_counter()

    print()
    result = program.run(params, parameter_list)
    print()

    t = perf_counter() - t
    end = energy_uj()

    output_CompletedProcess(program.name, result)

    return start, end, t


def report(start_energy: int, end_energy: int, elapsed: float):
    used_energy = end_energy - start_energy
    max_energy = max_energy_range_uj()

    if used_energy < 0:
        used_energy = (end_energy + max_energy) - start_energy
    
    energy_scaled = used_energy / max_energy

    return {
        "energy_uj": used_energy,
        "energy_scaled": energy_scaled,
        "time": elapsed
    }


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
