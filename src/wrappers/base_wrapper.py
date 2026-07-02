from dataclasses import dataclass, replace
import subprocess
from sys import stderr
from typing import Iterable, Union
from ..programs import *
from ..machines import *
from ..util.data import ExecutionParams, EnergyReading
from time import perf_counter
from shutil import which

from textwrap import indent

def prepare_and_exeute(machine: type[Machine], program: type[Program], params: ExecutionParams, args: Union[None, Iterable[str]]):
    if not args:
        args = []
    
    cpu_set(machine, params.freq_level)
    reading, t = run(machine, program, params, args)
    return report(reading, t)

def try_exec(cmds: list[list[str]], err_msg: str = "") -> bool:
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if err_msg:
                print(err_msg, file=stderr)
            output_CompletedProcess(" ".join(cmd), result)
            return False
    return True

# which to use should be defined on the machine
def set_freq(machine: type[Machine], frequency):
    if which("cpufreq-set"):
        if try_exec([
            *(["cpufreq-set", f"--cpu {cpu}", "--governor", "userspace"] for cpu in range(machine.max_threads)),
            *(["cpufreq-set", f"--cpu {cpu}", "--freq", f"{frequency}"] for cpu in range(machine.max_threads))
        ]): return
        if try_exec([
            *(["sudo", "-n", "cpufreq-set", f"--cpu {cpu}", "--governor", "userspace"] for cpu in range(machine.max_threads)),
            *(["sudo", "-n", "cpufreq-set", f"--cpu {cpu}", "--freq", f"{frequency}"] for cpu in range(machine.max_threads))
        ]): return
    
    if which("cpupower"):
        if try_exec([
            ["cpupower", "frequency-set", "--governor", "userspace"],
            ["cpupower", "frequency-set", "--freq", f"{frequency}"]
        ]): return
        if try_exec([
            ["sudo", "-n", "cpupower", "frequency-set", "--governor", "userspace"],
            ["sudo", "-n", "cpupower", "frequency-set", "--freq", f"{frequency}"]
        ]): return
    
    raise Exception("Unable to use cpufreq-set or cpupower, do i have permission?")

def cpu_set(machine: type[Machine], freq_level: int):

    set_freq(machine, machine.freq[freq_level])
    
    # power_cap = x[POWER_CAP_POS]
    # power_cap *= 10**6
    # set_sysfs("/sys/class/powercap/intel-rapl:0/?????", power_cap, "Power cap")


def run(machine: type[Machine],program: type[Program], params: ExecutionParams, parameter_list: Iterable[str]):
    reading = all_energy_uj(machine)
    t = perf_counter()

    print()
    result = program.run(params, parameter_list)
    print()

    delta = perf_counter() - t
    end = all_energy_uj(machine, reading)

    output_CompletedProcess(program.name, result)

    return end, delta


def report(readings: list[EnergyReading], elapsed: float):
    used_energy = 0
    energy_scaled = 0
    for reading in readings:
        max_energy = max_energy_range_uj(f"{reading.package}{sub_package_sufix(reading.sub_package)}")

        # it can technically wrap around twice or more, so we shouldn't run it for longer than a whole day or something
        if reading.start > reading.end:
            used_energy += (reading.end + max_energy) - reading.start
        else:  
            used_energy += reading.end - reading.start
        
        energy_scaled += used_energy / max_energy
    
    print(f"Energy (μJ): {used_energy}")
    print(f"Time elapsed (s): {elapsed}")

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


def sub_package_sufix(sub_package: int):
    return f":{sub_package}" if sub_package >= 0 else ""

def get_energy(package: int = -1, sub_package: int= -1, reading: EnergyReading | None = None):
    if reading:
        package = reading.package
        sub_package = reading.sub_package
    return energy_uj(f"{package}{sub_package_sufix(sub_package)}")


def all_energy_uj(machine: type[Machine], start: None | list[EnergyReading] = None) -> list[EnergyReading]:
    if start is not None:
        return [replace(reading, end=get_energy(reading=reading)) for reading in start]
    return [
        EnergyReading(get_energy(package=package, sub_package=sub_package), -1, package, sub_package)
        for package in machine.package
        for sub_package in machine.sub_package
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
