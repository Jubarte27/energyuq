"""
Operating system interface utilities for EnergyUQ.
"""
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from shutil import make_archive


def pack_dir(run_dir: str | Path, out: str | Path | None = None) -> Path:
    """Archive a run directory into a single .tar.gz. Returns the archive path."""
    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise NotADirectoryError(run)
    target = Path(out) if out else run
    return Path(make_archive(str(target.with_suffix("")), "gztar", root_dir=run.parent, base_dir=run.name))

def try_exec(
    cmds: Sequence[Sequence[str]],
    err_msg: str = "",
    input: str | None = None,
) -> bool:
    """
    Execute a sequence of system commands in order.
    Returns True if all commands complete with returncode 0; False otherwise.
    Stops executing at the first failed command.
    """
    for cmd in cmds:
        cmd_list = list(cmd)
        print(f"Executing: {cmd_list}")
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            input=input, 
            check=False,
        )
        if result.returncode != 0:
            if err_msg:
                print(err_msg, file=sys.stderr)
            return False
    return True

def try_exec_shell(
    cmds: Sequence[str],
    err_msg: str = "",
    input: str | None = None,
):
    """
    Execute a sequence of shell commands in order.
    Returns True if all commands complete with returncode 0; False otherwise.
    Stops executing at the first failed command.
    """
    for cmd in cmds:
        print(f"Executing: {cmd}")
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            input=input, 
            check=False,
        )
        if result.returncode != 0:
            if err_msg:
                print(err_msg, file=sys.stderr)
            return False
    return True