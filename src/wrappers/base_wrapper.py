from dataclasses import replace
from pathlib import Path
import subprocess
from sys import stderr
from typing import Iterable, Union
from ..programs import *
from ..machines import *
from ..util.data import ExecutionParams, EnergyReading
from time import perf_counter
from abc import ABC, abstractmethod

from textwrap import indent


def prepare_and_execute(machine: Machine, program: type[Program], params: ExecutionParams, args: Union[None, Iterable[str]]):
    if not args:
        args = []

    cpu_set(machine, params.freq_level)
    set_boost(machine, params.boost)
    if params.numa is not None and machine.has_numa:
        set_numa(machine, params.numa)
    
    accum, t = run(machine, program, params, args)
    return report(accum, t)

def try_exec(cmds: list[list[str]], err_msg: str = "", input: str | None = None) -> bool:
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True, input=input)
        if result.returncode != 0:
            if err_msg:
                print(err_msg, file=stderr)
            output_CompletedProcess(" ".join(cmd), result)
            return False
    return True

def set_boost(machine: Machine, value: int):
    if machine.boost_setter == "cpufreq":
        boost = "0" if machine.turbo_boost[value] == "false" else "1"
        if try_exec([["tee", "/sys/devices/system/cpu/cpufreq/boost"]], input=boost):
            return machine.boost_setter

    raise Exception(f"Unable to use {machine.boost_setter} for setting turbo boost, do i have permission?")

def set_numa(machine: Machine, value: int):
    if machine.numa_setter == "sysctl":
        numa = "0" if machine.numactl[value] == "false" else "1"
        if try_exec([["sudo", "/sbin/sysctl", f"kernel.numa_balancing={numa}"]]):
            return machine.numa_setter

    raise Exception(f"Unable to use {machine.numa_setter} for setting numa balancing, do i have permission?")
    

def set_freq(machine: Machine, frequency):
    if machine.freq_setter == "cpufreq-set":
        if try_exec([
            *(["cpufreq-set", "--cpu", f"{cpu}", "--governor", "userspace"] for cpu in range(machine.max_threads)),
            *(["cpufreq-set", "--cpu", f"{cpu}", "--freq", f"{frequency}"] for cpu in range(machine.max_threads))
        ]):
            return machine.freq_setter

    if machine.freq_setter == "cpupower":
        if try_exec([
            ["cpupower", "frequency-set", "--governor", "userspace"],
            ["cpupower", "frequency-set", "--freq", f"{frequency}"]
        ]):
            return machine.freq_setter

    if machine.freq_getter == "slurm":
        return machine.freq_setter

    raise Exception(f"Unable to use {machine.freq_setter} for setting cpu frequency, do i have permission?")

def cpu_set(machine: Machine, freq_level: int):

    set_freq(machine, machine.freq[freq_level])
    
    # power_cap = x[POWER_CAP_POS]
    # power_cap *= 10**6
    # set_sysfs("/sys/class/powercap/intel-rapl:0/?????", power_cap, "Power cap")


def pick_reader(machine: Machine):
    if machine.energy_reader == "intel-rapl":
        reader: EnergyReader = intel_rapl(machine)
    elif machine.energy_reader == "cray":
        reader: EnergyReader = cray()
    else:
        raise RuntimeError("Couldn't find a way to read energy counters, do i have permission?")
    return reader

def run(machine: Machine,program: type[Program], params: ExecutionParams, parameter_list: Iterable[str]):
    reading = (reader := pick_reader(machine)).all_energy()
    t = perf_counter()

    print()
    result = program.run(params, parameter_list)
    print()

    delta = perf_counter() - t
    end = reader.all_energy(reading)

    output_CompletedProcess(program.name, result)
    result.check_returncode()

    return reader.accumulate(end), delta


def report(used_energy: int, elapsed: float):
    print(f"Energy (μJ): {used_energy}")
    print(f"Time elapsed (s): {elapsed}")

    return {
        "energy_uj": used_energy,
        "EDP": float((used_energy * 1e-6) * elapsed),
        "time": elapsed
    }
class EnergyReader(ABC):
    @abstractmethod
    def accumulate(self, readings: list[EnergyReading]) -> int: pass
    @abstractmethod
    def energy(self, counter) -> int: pass
    @abstractmethod
    def all_energy(self, start: None | list[EnergyReading] = None) -> list[EnergyReading]: pass

def set_sysfs(full_path: str, value: object, name=None):
    result = subprocess.run(
        ["tee", full_path],
        input=str(value),
        capture_output=True,
        text=True,
    )

    output_CompletedProcess(full_path if name is None else name, result)

    if result.returncode != 0:
        raise RuntimeError(f"\"tee {full_path}\" failed with exit code:{result.returncode}")

class intel_rapl(EnergyReader):
    def __init__(self, machine: Machine) -> None:
        super().__init__()
        self.packages = machine.package
        self.sub_packages = machine.sub_package
        
    def accumulate(self, readings: list[EnergyReading]):
        # If package-level readings (sub_package < 0) are present, accumulate only those
        # to avoid double-counting sub-domains (e.g. core, dram) which are already included in package energy.
        has_package_level = any(reading.sub_package < 0 for reading in readings)
        target_readings = [r for r in readings if r.sub_package < 0] if has_package_level else readings

        used_energy = 0
        for reading in target_readings:
            max_energy = self.max_energy_range_uj(f"{reading.package}{self.sub_package_sufix(reading.sub_package)}")
            # it can technically wrap around twice or more, so we shouldn't run it for longer than a whole day or something
            if reading.start > reading.end:
                used_energy += (reading.end + max_energy) - reading.start
            else:  
                used_energy += reading.end - reading.start
        return used_energy

    def max_energy_range_uj(self, socket) -> int:
        result = Path(f"/sys/class/powercap/intel-rapl:{socket}/max_energy_range_uj").read_text()
        return int(result)

    def energy(self, counter) -> int:
        result = Path(f"/sys/class/powercap/intel-rapl:{counter}/energy_uj").read_text()
        return int(result)
    
    def all_energy(self, start: None | list[EnergyReading] = None) -> list[EnergyReading]:
        if start is not None:
            return [replace(reading, end=self.get_energy(reading=reading)) for reading in start]
        return [
            EnergyReading(self.get_energy(package=package, sub_package=sub_package), -1, package, sub_package)
            for package in self.packages
            for sub_package in self.sub_packages
        ]
    
    def get_energy(self, package: int = -1, sub_package: int= -1, reading: EnergyReading | None = None):
        if reading:
            package = int(reading.package)
            sub_package = reading.sub_package
        return self.energy(f"{package}{self.sub_package_sufix(sub_package)}")
    
    @staticmethod
    def sub_package_sufix(sub_package: int):
        return f":{sub_package}" if sub_package >= 0 else ""


#https://cray-hpe.github.io/docs-csm/en-17/operations/power_management/user_access_to_compute_node_power_data/
class cray(EnergyReader):
    def accumulate(self, readings: list[EnergyReading]):
        return int(sum(reading.end - reading.start for reading in readings))

    def energy(self, counter) -> int:
        result = Path(f"/sys/cray/pm_counters/{counter}").read_text()
        
        return int(result.split()[0]) * 1_000_000
    
    def all_energy(self, start: None | list[EnergyReading] = None) -> list[EnergyReading]:
        if start is not None:
            return [replace(reading, end=self.energy(reading.package)) for reading in start]
        return [
            EnergyReading(self.energy(package), -1, package, -1)
            for package in ("cpu_energy", "memory_energy")
        ]

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
