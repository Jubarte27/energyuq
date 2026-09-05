import os
import socket
from dataclasses import dataclass, field, replace
from shutil import which
from pathlib import Path
import subprocess


@dataclass
class Machine:
    name: str
    freq: list[int]
    max_threads: int
    package: list[int] = field(default_factory=lambda: [0])
    sub_package: list[int] = field(default_factory=lambda: [-1]) # nome ruim, valores ruins
    places: list[str] = field(default_factory=lambda: ["threads", "cores", "sockets"])
    proc_bind: list[str] = field(default_factory=lambda: ["true", "close", "spread", "false"])
    freq_getter: str | None = None
    freq_setter: str | None = None
    energy_reader: str | None = None
    energy_accum: str | None = None


NONE = Machine(name="NONE", freq=[0], max_threads=0)


def _environment_list(name: str, parser):
    value = os.environ.get(name)
    if value is None:
        return None

    try:
        values = [parser(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        return None
    return values or None

def _read_ints_file(path: Path) -> list[int]:
    try:
        return [int(value) for value in path.read_text().split()]
    except (OSError, ValueError):
        return []

def _read_int_file(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _system_frequencies() -> list[int] | None:
    frequencies: set[int] = set()
    for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_available_frequencies"):
        frequencies.update(_read_ints_file(path))

    if frequencies:
        return sorted(frequencies)

    for path in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_max_freq"):
        if (frequency := _read_int_file(path)) is not None:
            frequencies.add(frequency)
    return sorted(frequencies) or None


def _system_rapl_domains() -> tuple[list[int], list[int]] | None:
    powercap = Path("/sys/class/powercap")
    packages: set[int] = set()

    try:
        paths = list(powercap.iterdir())
    except OSError:
        return None

    for path in paths:
        if not path.name.startswith("intel-rapl:"):
            continue
        parts = path.name.removeprefix("intel-rapl:").split(":")
        if len(parts) != 1:
            continue
        try:
            package = int(parts[0])
        except ValueError:
            continue

        # Check domain name: only consider CPU sockets (e.g. "package-0", "package-1"),
        # and ignore platform domains like "psys" to avoid double-counting.
        name_path = path / "name"
        try:
            if name_path.exists() and not name_path.read_text().strip().startswith("package-"):
                continue
        except OSError:
            pass

        packages.add(package)

    if not packages:
        return None

    # Sub-packages [-1] means the entire package/socket domain.
    # Individual sub-domains (e.g. core, uncore, dram) are subsets of the package
    # and should not be combined with -1 to avoid double-counting.
    return sorted(packages), [-1]


def _available_programs(machine: Machine, slurm: bool=True) -> Machine:
    freq_tool = None
    energy_tool = None
    if slurm:
        freq_tool="slurm"
    if which("cpufreq-set"):
        freq_tool = "cpufreq-set"
    elif which("cpupower"):
        freq_tool = "cpupower"

    if freq_tool is None:
        raise Exception("Unable to use cpufreq-set or cpupower, do i have permission?")

    if _check_rapl(machine):
        energy_tool = "intel-rapl"
    
    if _check_cray():
        energy_tool = "cray"

    if energy_tool is None:
        raise Exception("Couldn't find a way to read energy counters, do i have permission?")
        exit(42)
    return replace(machine,
        freq_getter=freq_tool,
        freq_setter=freq_tool,
        energy_reader=energy_tool,
        energy_accum=energy_tool,
    )

def try_exec(cmds: list[list[str]]) -> bool:
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False
    return True

def _check_rapl(machine: Machine) -> bool:
    commands = [
        [
            "cat",
            f"/sys/class/powercap/intel-rapl:{package}"
            f"{f':{sub_package}' if sub_package >= 0 else ''}"
            "/energy_uj",
        ]
        for package in machine.package
        for sub_package in machine.sub_package
    ]
    return len(commands) > 0 and try_exec(commands)


def _check_cray() -> bool:
    return try_exec([
        ["cat", f"/sys/cray/pm_counters/{package}"]
        for package in ("cpu_energy", "memory_energy")
    ])

def guess_machine() -> Machine:
    """Build a Machine from environment settings, system discovery, and safe defaults.

    Each ``ENERGYUQ_MACHINE_*`` variable takes precedence over discovery:
    ``NAME``, ``FREQ``, ``MAX_THREADS``, ``PACKAGE``, ``SUB_PACKAGE``,
    ``PLACES``, and ``PROC_BIND``. List values are comma-separated.
    """
    system_domains = _system_rapl_domains()
    slurm = os.environ.get("ENERGYUQ_SLURM") not in ("False", "false", "no")

    name = os.environ.get("ENERGYUQ_MACHINE_NAME") or socket.gethostname() or "unknown"
    frequencies = _environment_list("ENERGYUQ_MACHINE_FREQ", int) or _system_frequencies() or [0]

    env_max_threads = _environment_list("ENERGYUQ_MACHINE_MAX_THREADS", int)
    max_threads = env_max_threads[0] if env_max_threads else (os.cpu_count() or 1)
    if max_threads < 1:
        max_threads = 1

    packages = (
        _environment_list("ENERGYUQ_MACHINE_PACKAGE", int)
        or (system_domains[0] if system_domains else None)
        or [0]
    )
    sub_packages = (
        _environment_list("ENERGYUQ_MACHINE_SUB_PACKAGE", int)
        or (system_domains[1] if system_domains else None)
        or [-1]
    )
    places = _environment_list("ENERGYUQ_MACHINE_PLACES", str) or ["threads"]
    proc_bind = _environment_list("ENERGYUQ_MACHINE_PROC_BIND", str) or ["true"]

    return _available_programs(Machine(
        name=name,
        freq=frequencies,
        max_threads=max_threads,
        package=packages,
        sub_package=sub_packages,
        places=places,
        proc_bind=proc_bind,
    ), slurm)

@dataclass
class MachineParams():
    machine: Machine = field(default_factory=lambda: NONE)
    n_threads: int = 1
    freq_level: int = 0
