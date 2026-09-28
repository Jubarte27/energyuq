#!/usr/bin/env python3
from ..machines import Machine, guess_machine
from ..programs import NONE, Program
from ..util.data import ExecutionParams
from . import base_wrapper


def main(
        program: type[Program],
        machine: Machine,
        input_file: str = "input.csv",
        output_file: str = "output.csv"
    ):
    with open(input_file) as f:
        args = f.readline().split(",")
    params = ExecutionParams.from_args(machine, args)
    program_args = (
        args[6:] if has_arg(args, 5) and params.numa is not None
        else args[5:]
    )
    result = base_wrapper.prepare_and_execute(machine, program, params, program_args)

    ks, vs = zip(*result.items())
    header = ",".join(ks)
    content = ",".join(f'{v}' for v in vs)

    with open(output_file, "w") as f:
        f.write(f"{header}\n{content}\n")

def has_arg(args, n):
    return len(args) > n and str(args[n]).strip() != ""

if __name__ == "__main__":
    program = NONE
    machine = guess_machine()
    print(f"Running on {machine.name}")
    main(program, machine)
