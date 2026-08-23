from dotenv import load_dotenv

load_dotenv()

from src import energyuq

from src import programs
from src.machines import *

from socket import gethostname

import re
import sys

#todo: more command line parameters


# todo: extract details programatically. Remove this horror
#/sys/devices/system/cpu/cpu*/cpufreq/scaling_available_frequencies
POSSIBILITIES: list[tuple[re.Pattern, Machine]] = [
    (re.compile(r"glados", re.IGNORECASE), Glados),
    (re.compile(r"hype\d", re.IGNORECASE), Hype),
    (re.compile(r"machado", re.IGNORECASE), Machado),
]

host = sys.argv[1] if len(sys.argv) > 1 else gethostname()


mach = guess_machine()

for pat, m in POSSIBILITIES:
    if pat.match(host):
        mach = m
        break

if mach is None:
    raise RuntimeError("I don't know where I am at")

campaign, analysis = energyuq.create(programs.FAKEWORK, mach)

energyuq.refine_and_analyse(campaign, analysis, max_number_of_refinements=100)

energyuq.save(campaign, analysis)
