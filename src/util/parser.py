from pandas import DataFrame, read_csv 
import numpy as np

def parse_dakota_file(file_path: str) -> DataFrame:
    with open(file_path, 'r') as f:
        headers = f.readline().replace('%', '').strip().split()
        
        df = read_csv(f, sep=r'\s+', names=headers, header=None)
    
    # For now, the first few samples taken by dakota ignore the fact that variables are discrete.
    # todo: figure out why that is being ignored (most likely bad configurarion) 
    # This is a temporary "fix" for visualization
    for col in df.columns:
        df[col] = np.floor(df[col])
    return df