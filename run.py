from src import energyuq

from src import programs
from src import machines

from socket import gethostname

import re
import sys

#todo: more command line parameters


# todo: extract details programatically. Remove this horror
POSSIBILITIES: list[tuple[re.Pattern, type[machines.Machine]]] = [
    (re.compile(r"glados", re.IGNORECASE), machines.Glados),
    (re.compile(r"hype\d", re.IGNORECASE), machines.Hype),
]

host = sys.argv[1] if len(sys.argv) > 1 else gethostname()

mach = None

for pat, m in POSSIBILITIES:
    if pat.match(host):
        mach = m
        break

if mach is None:
    raise RuntimeError("I don't know where I am at")

campaign, analysis = energyuq.create(programs.HPCG, mach)

energyuq.refine_and_analyse(campaign, analysis, max_number_of_refinements=100)

energyuq.save(campaign, analysis)
