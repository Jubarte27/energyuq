from src import energyuq

from src import programs
from src import machines

campaign, analysis = energyuq.create(programs.HPCG, machines.Glados)

energyuq.refine_and_analyse(campaign, analysis, max_number_of_refinements=100)

energyuq.save(campaign, analysis)
