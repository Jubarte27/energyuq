from src import energyuq

from src import programs
from src import machines

import os

campaign, analysis = energyuq.create(programs.HPCG, machines.Hype)

energyuq.refine_and_analyse(campaign, analysis, 20)

energyuq.save(campaign, analysis)
