#!/usr/bin/env python3

import os
import numpy as np
import easyvvuq as uq
import chaospy as cp
import matplotlib.pyplot as plt
from easyvvuq.actions import CreateRunDirectory, Encode, Decode, ExecuteLocal, Actions


params = {}

params['BSIZE_X'] = {'type': 'integer', 'default': 256.0}  # block size parameter
params['BSIZE_Y'] = {'type': 'integer', 'default': 2.0}  # block size parameter

params['CLOCK'] = {'type': 'integer', 'default': 1500.0}  # clock speed in MHz
params['POWER_CAP'] = {'type': 'integer', 'default': 220.0}  # power cap in watts

# input file encoder
encoder = uq.encoders.GenericEncoder(template_fname='fletcher.template', delimiter='$', target_filename='input.csv')

# Quantity of Interest, also the column name of the output CSV file
QOI = 'f'
# CSV output file decoder
decoder = uq.decoders.SimpleCSV(target_filename='output.csv', output_columns=[QOI])

execute = ExecuteLocal(f'{os.path.dirname(__file__)}/fletcher.py')


# location where the run directories are stored
WORK_DIR = '/tmp'
# actions to be undertaken: make rundirs, encode input files, execute local model ensemble, decode output files
actions = Actions(CreateRunDirectory(root=WORK_DIR, flatten=True), Encode(encoder), execute, 
                  Decode(decoder))


campaign = uq.Campaign(name='fletcher', params=params, actions=actions, work_dir=WORK_DIR)

powers_of_two = np.array([2**i for i in range(1, 10)])  # 1, 2, 4, ..., 512



# CHAOSPY 

vary = {}
vary['BSIZE_X'] = cp.Uniform(1, len(powers_of_two) - 1)
vary['BSIZE_X'].interpret_as_integer = True

vary['BSIZE_Y'] = cp.Uniform(1, len(powers_of_two) - 1)
vary['BSIZE_Y'].interpret_as_integer = True

vary['CLOCK'] = cp.Uniform(500, 1700)
vary['CLOCK'].interpret_as_integer = True

vary['POWER_CAP'] = cp.Uniform(140, 540)  # Power cap between
vary['POWER_CAP'].interpret_as_integer = True


sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=1, quadrature_rule='C', sparse=True,
                                growth=True, dimension_adaptive=True)
campaign.set_sampler(sampler)


campaign.execute(sequential=True, mark_invalid=True).collate(progress_bar=True)


analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=[QOI])

campaign.apply_analysis(analysis)


number_of_adaptations = 10
for i in range(number_of_adaptations):
    # compute candidate refinements
    sampler.look_ahead(analysis.l_norm)
    # run ensemble (at new locations only)
    campaign.execute(sequential=True, mark_invalid=True).collate(progress_bar=True)
    # get dataframe
    data_frame = campaign.get_collation_result()
    # adapt the sampling plan
    analysis.adapt_dimension(QOI, data_frame)
    # we must apply the analysis to update its internal state
    campaign.apply_analysis(analysis)



# Save campaign
import pickle
with open('fletcher_campaign.pkl', 'wb') as f:
    pickle.dump(campaign, f)
