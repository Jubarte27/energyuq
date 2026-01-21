#!/usr/bin/env python3
import numpy as np
import subprocess
import textwrap
from programs import *

N_THREADS_POS = 1
FREQUECNY_POS = 2


def main(program: type[Program]):
    cpu_set()
    run(program)
    program.report()


def cpu_set():
    with open("input.csv", "r") as f:
        x = np.array(f.readline().split(",")).astype("int")

    frequency = x[FREQUECNY_POS]

    result = subprocess.run(
        ["sudo", "cpupower", "frequency-set", "--freq", str(frequency)],
        capture_output=True,
        text=True,
    )

    output_required_CompletedProcess("Frequency set", result)


def run(program: type[Program]):
    with open("input.csv", "r") as f:
        x = f.readline().split(",")

    output_CompletedProcess(program.name, program.run(x))

def report():
    with open('output.csv', 'wt') as f:
        f.write('energy_uq\n')
        # f.write()

def energy_report():
    result = subprocess.run(
        ["sudo", "cat", "/sys/class/powercap/intel-rapl:0/energy_uj"],
        capture_output=True,
        text=True,
    )

def set_sysfs(full_path, value, name=None):
    result = subprocess.run(
        ["sudo", "tee", full_path],
        input=str(value),
        capture_output=True,
        text=True,
    )
    output_required_CompletedProcess(full_path if name is None else name, result)


def output_CompletedProcess(name: str, result: subprocess.CompletedProcess):
    print(
        h_string(
            f"{name} set",
            h_string("Return code", result.returncode),
            h_string("Stdout", result.stdout),
            h_string("Stderr", result.stderr)
        )
    )

def output_required_CompletedProcess(name, result):
    output_CompletedProcess(name, result)
    if result.returncode != 0:
        exit(result.returncode)

def hierarquical_string(*args):
    INDENT = "\t"
    if len(args) == 0:
        return ""
    if len(args) == 1:
        return args[0]

    name, children = args[0], args[1:]
    # ignore falsy values ("", None, 0, etc)
    children_text = [ str(child) for child in children if child ]

    if len(children_text) == 0:
        return name
    
    if len(children_text) == 1:
        child = children_text[0]
        return f'{name}: {child}'
    
    def indent(string: str):
        return textwrap.indent(string.rstrip().rstrip('\n'), INDENT)
    
    return name + '\n' + "\n".join([indent(s) for s in children_text])

h_string = hierarquical_string




if __name__ == "__main__":
    program = Fletcher
    main(program)
