import easyvvuq as uq
import pandas as pd
from pathlib import Path
import pickle

def save(analysis: uq.analysis.SCAnalysis, sampler: uq.sampling.SCSampler, data_frame: pd.DataFrame, vary, qoi_cols, directory = "run_results/"):
    Path(directory).mkdir(exist_ok=True)

    analysis.save_state(f"{directory}/analisys")
    sampler.save_state(f"{directory}/sampler")
    data_frame.to_pickle(f"{directory}/data_frame")

    with open(f"{directory}/vary", "wb") as f:
        pickle.dump(vary.vary_dict, f)

    with open(f"{directory}/QOI", "wb") as f:
        pickle.dump(qoi_cols, f)

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load(directory = "run_results/"):
    vary = read_pickle(f"{directory}/vary")
    print(type(vary))

    sampler = uq.sampling.SCSampler(vary=vary)
    sampler.load_state(f"{directory}/sampler")

    qoi_cols = read_pickle(f"{directory}/QOI")

    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=qoi_cols)
    analysis.load_state(f"{directory}/analisys")

    data_frame = pd.read_pickle(f"{directory}/data_frame")
    results = analysis.analyse(data_frame)
    return analysis, sampler, data_frame, qoi_cols, results