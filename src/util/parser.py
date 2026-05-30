from pandas import DataFrame, read_csv 

def parse_dakota_file(file_path: str) -> DataFrame:
    with open(file_path, 'r') as f:
        headers = f.readline().replace('%', '').strip().split()
        
        df = read_csv(f, sep=r'\s+', names=headers, header=None)
    return df