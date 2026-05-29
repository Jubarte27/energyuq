#!/usr/bin/env python3
import sys

import numpy as np
import pandas as pd
from scipy.stats import qmc
from typing import Union

#why did i not use csv?????
def parse_input_file(text: str) -> tuple[dict[str, list[Union[int,float,str]]], list[dict[str,Union[int,float,str]]]]:
    lines = [parts for line in text.strip().split('\n') if (parts := line.split())]
    if len(lines) < 1:
        raise ValueError("len(lines) < 1")
    data: dict[str, list[Union[int,float,str]]] = {}
    if (size := len(lines[0]) - 1) < 2:
        raise ValueError("len(line) < 2")

    for line in lines:
        key = line[0]
        values = line[1:]

        if size != len(values):
            raise ValueError("not every line have same amount of elements")
        
        if key == 'descriptors':
            data[key] = [v.strip("'\"") for v in values]
        else:
            data[key] = [float(v) if '.' in v else int(v) for v in values]
    if 'descriptors' not in data:
        raise ValueError("missing descriptors")
    
    vars: list[dict[str,Union[int,float,str]]] = [{} for _ in range(size)]
    # todo: add more
    for key, values in data.items():
        for i, v in enumerate(values):
            vars[i][key] = v

            
    return data, vars

def generate_lhs_samples(parsed_data, n_samples=10, is_discrete=None):
    lower = np.array(parsed_data['lower_bounds'])
    upper = np.array(parsed_data['upper_bounds'])
    descriptors = parsed_data['descriptors']
    d = len(descriptors)
    
    if is_discrete is None:
        is_discrete = [isinstance(b, int) for b in parsed_data['upper_bounds']]
        
    sampler = qmc.LatinHypercube(d=d)
    lhs_samples = sampler.random(n=n_samples)
    
    scaled_samples = np.zeros_like(lhs_samples)
    for i in range(d):
        if is_discrete[i]:
            L, U = lower[i], upper[i]
            scaled_samples[:, i] = np.floor(lhs_samples[:, i] * (U - L + 1) + L)
        else:
            scaled_samples[:, i] = qmc.scale( lhs_samples[:, i:i+1], [lower[i]], [upper[i]]).flatten()
            
    df = pd.DataFrame(scaled_samples, columns=descriptors)
    for i, col in enumerate(descriptors):
        if is_discrete[i]:
            df[col] = df[col].astype(int)
            
    return df
if __name__ == "__main__":
    config = """
    initial_point        2           32
    lower_bounds         0            1
    upper_bounds         2           32
    descriptors      'CLK'  'N_THREADS'
    """

    samples=sys.argv[1] if len(sys.argv) > 1 else 20
    file = sys.argv[1] if len(sys.argv) > 2 else "samples.csv"


    parsed_data, _ = parse_input_file(config)
    print(parsed_data)

    samples_df = generate_lhs_samples(parsed_data, n_samples=5)
    
    print(samples_df)
    samples_df.to_csv(file)