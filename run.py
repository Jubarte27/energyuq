from dotenv import load_dotenv

load_dotenv()

from src import energyuq

from src import programs
from src.machines import *
from src.programs.benchmark import ExecuteSH

from socket import gethostname

#todo: more command line parameters

import sys

if len(sys.argv) > 1:
    requested_benchmark = sys.argv[1]
    try:
        benchmark = next(
            benchmark for benchmark in ExecuteSH.__subclasses__()
            if benchmark.__name__.lower() == requested_benchmark.lower()
        )
    except StopIteration:
        raise ValueError(f"Unknown benchmark {requested_benchmark}")
else:
    benchmark = programs.FAKEWORK

mach = guess_machine()

if mach is None:
    raise RuntimeError("I don't know where I am at")

campaign, analysis = energyuq.create(benchmark, mach)

energyuq.refine_and_analyse(campaign, analysis, max_number_of_refinements=100)

energyuq.save(campaign, analysis)
