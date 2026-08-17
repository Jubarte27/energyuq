import os
import socket
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Machine:
    name: str
    freq: list[int]
    max_threads: int
    package: list[int] = field(default_factory=lambda: [0])
    sub_package: list[int] = field(default_factory=lambda: [-1]) # nome ruim, valores ruins
    places: list[str] = field(default_factory=lambda: ["threads", "cores", "sockets"])
    proc_bind: list[str] = field(default_factory=lambda: ["true", "close", "spread", "false"])


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
    sub_packages_by_package: dict[int, set[int]] = {}

    try:
        paths = list(powercap.iterdir())
    except OSError:
        return None

    for path in paths:
        if not path.name.startswith("intel-rapl:"):
            continue
        parts = path.name.removeprefix("intel-rapl:").split(":")
        try:
            package = int(parts[0])
        except ValueError:
            continue
        packages.add(package)
        if len(parts) == 2:
            try:
                sub_packages_by_package.setdefault(package, set()).add(int(parts[1]))
            except ValueError:
                continue

    if not packages:
        return None

    # We apply every sub-package to every package, so
    # only retain sub-packages that exist on all detected packages.
    # Not ideal, will cause problems in the future
    common_sub_packages = set.intersection(
        *(sub_packages_by_package.get(package, set()) for package in packages)
    )
    return sorted(packages), [-1, *sorted(common_sub_packages)]


def guess_machine() -> Machine:
    """Build a Machine from environment settings, system discovery, and safe defaults.

    Each ``ENERGYUQ_MACHINE_*`` variable takes precedence over discovery:
    ``NAME``, ``FREQ``, ``MAX_THREADS``, ``PACKAGE``, ``SUB_PACKAGE``,
    ``PLACES``, and ``PROC_BIND``. List values are comma-separated.
    """
    system_domains = _system_rapl_domains()

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

    return Machine(
        name=name,
        freq=frequencies,
        max_threads=max_threads,
        package=packages,
        sub_package=sub_packages,
        places=places,
        proc_bind=proc_bind,
    )

@dataclass
class MachineParams():
    machine: Machine = field(default_factory=lambda: NONE)
    n_threads: int = 1
    freq_level: int = 0
