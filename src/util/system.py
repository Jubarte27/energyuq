"""
Operating system interface utilities for EnergyUQ.
"""

from collections.abc import Sequence
import subprocess
import sys


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
        result = subprocess.run(cmd_list, capture_output=True, text=True, input=input)
        if result.returncode != 0:
            if err_msg:
                print(err_msg, file=sys.stderr)
            return False
    return True

