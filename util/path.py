from pathlib import Path
from typing import Union

from collections.abc import Callable
import re


def compile_name_pattern(
    prefix: str = "",
    suffix="",
) -> re.Pattern:
    return re.compile(f"{re.escape(prefix)}([0-9]+){re.escape(suffix)}")


def latest_path(
    dir: Union[str, Path],
    prefix: str = "",
    suffix="",
    restriction: Callable[[Path], bool] = lambda _: True,
) -> Union[Path, None]:
    name_pattern = compile_name_pattern(prefix, suffix)

    parent = Path(dir)
    _, latest = max(
        {
            (int(m.group(1)), p)
            for p in parent.iterdir()
            if restriction(p)
            and (name := p.relative_to(parent).as_posix())
            and (m := name_pattern.fullmatch(name))
        },
        default=(-1, None),
    )
    return latest


def latest_dir(dir: Union[str, Path], base_name: str) -> Union[Path, None]:
    return latest_path(dir, prefix=base_name + "_", restriction=Path.is_dir)


def latest_file(dir: Union[str, Path], base_name: str) -> Union[Path, None]:
    return latest_path(dir, prefix=base_name + "_", restriction=Path.is_file)


def next_path(
    dir: Union[str, Path],
    prefix: str = "",
    suffix="",
    restriction: Callable[[Path], bool] = lambda _: True,
) -> Path:
    name_pattern = compile_name_pattern(prefix, suffix)
    parent = Path(dir)
    last = max(
        (
            int(m.group(1))
            for p in parent.iterdir()
            if restriction(p)
            and (name := p.relative_to(parent).as_posix())
            and (m := name_pattern.fullmatch(name))
            and m
        ),
        default=-1,
    )
    return Path(parent, f"{prefix}{last + 1}{suffix}")


def next_dir(dir: Union[str, Path], base_name: str) -> Path:
    return next_path(dir, prefix=base_name + "_", restriction=Path.is_dir)


def next_file(dir: Union[str, Path], base_name: str) -> Path:
    return next_path(dir, prefix=base_name + "_", restriction=Path.is_file)


def change_dir_permissions(dir: Path, mask: int):
    dir.chmod(mask)
    for sub in dir.iterdir():
        if sub.is_dir():
            change_dir_permissions(sub, mask)