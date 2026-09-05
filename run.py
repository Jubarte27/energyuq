from dotenv import load_dotenv

load_dotenv()

from src import energyuq

from src import programs
from src.machines import *

from socket import gethostname

#todo: more command line parameters

mach = guess_machine()

if mach is None:
    raise RuntimeError("I don't know where I am at")

campaign, analysis = energyuq.create(programs.FAKEWORK, mach)

energyuq.refine_and_analyse(campaign, analysis, max_number_of_refinements=100)

energyuq.save(campaign, analysis)
