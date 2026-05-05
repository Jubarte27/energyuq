#!/usr/bin/env python3
from util.data import ExecutionParams

from . import base_wrapper

from programs import *
from machines import *

def main(program: type[Program], machine: type[Machine], input_file: str = "input.csv", output_file: str = "output.csv"):
    with open(input_file, "r") as f:
        args = f.readline().split(",")
    params = ExecutionParams(n_threads=int(args[0]), freq_level=int(args[1]))

    result = base_wrapper.prepare_and_exeute(machine, program, params, args[2:])

    ks, vs = zip(*result.items())
    header = ",".join(ks)
    content = ",".join(f'{v}' for v in vs)

    with open(output_file, "w") as f:
        f.write(f"{header}\n{content}\n")

if __name__ == "__main__":
    program = NONE
    machine = Glados
    print(f"Running on {machine.name}")
    main(program, machine)
