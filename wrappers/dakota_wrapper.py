#!/usr/bin/env python3
from dakota import interfacing

import base_wrapper as base_wrapper
from programs import *
from machines import *
from util.data import ExecutionParams

FREQUECNY_POS = 1
# POWER_CAP_POS = 2


def main(program: type[Program], machine: type[Machine], parameters: interfacing.Parameters, results: interfacing.Results):
    params = ExecutionParams(freq_level=parameters["CLK"], n_threads=parameters["N_THREADS"])
    result = base_wrapper.prepare_and_exeute(machine, program, params, [])

    results["energy_uj"].function = result["energy_uj"]
    results.write()

if __name__ == "__main__":
    program = NONE
    machine = Glados
    print(f"Running on {machine.name}")
    parameters, results = interfacing.read_parameters_file()
    if isinstance(parameters, interfacing.BatchParameters) or isinstance(results, interfacing.BatchResults):
        print("I don't work with batches")
        exit(1)
    main(program, machine, parameters, results)
