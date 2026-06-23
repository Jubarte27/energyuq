from dataclasses import dataclass
import subprocess
from typing import Iterable, Union
from ..programs import *
from ..machines import *
from ..util.data import ExecutionParams, EnergyReading
from time import perf_counter

from textwrap import indent

def prepare_and_exeute(machine: type[Machine], program: type[Program], params: ExecutionParams, args: Union[None, Iterable[str]]):
    if not args:
        args = []
    
    cpu_set(machine, params.freq_level)
    reading, t = run(machine, program, params, args)
    return report(reading, t)

def set_userspace():
    print("tried this")
    return subprocess.run(
        # which to use should be defined on the machine
    #     ["cpupower", "frequency-set", "--governor", "userspace"],
        ["cpufreq-set", "--governor", "userspace"],
        capture_output=True,
        text=True,
    )

def cpu_set(machine: type[Machine], freq_level: int):
    frequency = machine.freq[freq_level]

    result = set_userspace()

    output_CompletedProcess("Governor set", result, True)
    if result.returncode != 0:
        raise Exception(f"Unable to set governor using cpupower, am i root? code: {result.returncode}")

    print("also this")
    result = subprocess.run(
        # which to use should be defined on the machine
        # ["cpupower", "frequency-set", "--freq", str(frequency)],
        ["cpufreq-set", "--freq", str(frequency)],
        capture_output=True,
        text=True,
    )
    output_CompletedProcess("Frequency set", result, True)
    if result.returncode != 0:
        raise Exception(f"Unable to set frequency using cpupower, am i root? code: {result.returncode}")
    
    print("and it worked")
    # power_cap = x[POWER_CAP_POS]
    # power_cap *= 10**6
    # set_sysfs("/sys/class/powercap/intel-rapl:0/?????", power_cap, "Power cap")


def run(machine: type[Machine],program: type[Program], params: ExecutionParams, parameter_list: Iterable[str]):
    print("started")
    reading = all_energy_uj(machine)
    t = perf_counter()

    print()
    result = program.run(params, parameter_list)
    print()

    t = perf_counter() - t
    end = all_energy_uj(machine, reading)

    output_CompletedProcess(program.name, result)

    return reading, t


def report(readings: list[EnergyReading], elapsed: float):
    used_energy = 0
    energy_scaled = 0
    for reading in readings:
        used_energy = reading.end - reading.start
        
        # it can technically wrap around twice or more, so we shouldn't run it for longer that a whole day or something
        
        suf = f":{reading.sub_package}" if reading.sub_package >= 0 else ""
        max_energy = max_energy_range_uj(f"{reading.package}{suf}")
        if reading.end > reading.start:
            used_energy = (reading.end + max_energy) - reading.start
        
        energy_scaled = used_energy / max_energy

    return {
        "energy_uj": used_energy,
        "energy_scaled": energy_scaled,
        "time": elapsed
    }

def max_energy_range_uj(socket) -> int:
    result = subprocess.run(
        ["cat", f"/sys/class/powercap/intel-rapl:{socket}/max_energy_range_uj"],
        capture_output=True,
        text=True,
    )
    output_CompletedProcess(f"max_energy_range_uj:{socket}", result)
    if result.returncode != 0:
        exit(result.returncode)
    print("read once")
    return int(result.stdout)

def energy_uj(socket) -> int:
    result = subprocess.run(
        ["cat", f"/sys/class/powercap/intel-rapl:{socket}/energy_uj"],
        capture_output=True,
        text=True,
    )
    output_CompletedProcess(f"energy_uj:{socket}", result)
    if result.returncode != 0:
        exit(result.returncode)
    return int(result.stdout)


def all_energy_uj(machine: type[Machine], start: None | list[EnergyReading] = None) -> list[EnergyReading]:
    if start is not None:
        reading_requests = start
    else:
        reading_requests = [
            EnergyReading(-1, -1, package, sub_package)
            for package in machine.package
            for sub_package in machine.sub_package
        ]
    # feio
    return [
        EnergyReading(
            reading.start if reading.start < 0 else value_read,
            reading.end if reading.start >= 0 else value_read,
            reading.package,
            reading.sub_package
        )
        for reading in reading_requests
        if (suf := f":{reading.sub_package}" if reading.sub_package >= 0 else "") is not None
        if (value_read := energy_uj(f"{reading.package}{suf}")) >= 0
    ]
        

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
