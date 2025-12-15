#!/usr/bin/env python3

import numpy as np
# import pandas as pd
import subprocess
import os

POWER_CAP_POS = 2


def rocm_set():
    with open('input.csv', 'r') as f:
        x = np.array(f.readline().split(',')).astype('int')

    # clk = x[CLK_POS]
    power_cap = x[POWER_CAP_POS]
    power_cap *= 1000000

    # result = subprocess.run(
    #     ['sudo', 'rocm-smi', '--setpoweroverdrive', f'{power_cap}'], #'--setsclk', f'{clk}'],
    #     capture_output=True,
    #     text=True
    # )

    result = subprocess.run(
    ['sudo', 'tee', '/sys/class/hwmon/hwmon1/power1_cap'],
    input=str(power_cap),
    capture_output=True,
    text=True
    )

    print(f"--PowerCap set--")
    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")
    print(f"--PowerCap set--")

    if result.returncode != 0:
        exit(result.returncode)
        
    


def run():
    fletcher_exe = f'{os.path.dirname(__file__)}/HIP-3D/ModelagemFletcher.exe'

    fp = open('input.csv', 'r')
    x = np.array(fp.readline().split(',')).astype('int')
    fp.close()

    BSIZE_X = x[0]
    BSIZE_Y = x[1]

    # check if valid BSIZE_X and BSIZE_Y
    # if BSIZE_X + BSIZE_Y > 10:
    #     header = 'f'
    #     np.savetxt('output.csv', np.atleast_2d(np.array([np.nan])).T,
    #             delimiter=",", comments='',
    #             header=header)
    #     print(f"Invalid BSIZE_X and BSIZE_Y combination: {BSIZE_X}, {BSIZE_Y}")
    #     exit(0)


    # Get the corresponding powers of two
    BSIZE_X = 2 ** BSIZE_X
    BSIZE_Y = 2 ** BSIZE_Y

    print(f"Running Fletcher with BSIZE_X={BSIZE_X}, BSIZE_Y={BSIZE_Y}")

    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")

    # Run the executable and capture output
    result = subprocess.run(
        [fletcher_exe, 'TTI', '472', '472', '472', 
        '16', '12.5', '12.5', '12.5', '0.001', '0.01', str(BSIZE_X), str(BSIZE_Y), '1'],
        capture_output=True,
        text=True
    )

    # Check if execution was successful
    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")

def omnistat_report():
    # df = pd.read_csv("omnistat-rocm.gpu.csv", header=[0, 1, 2], index_col=0)

    # # Select a single metric
    # df["rocm_utilization_percentage"]

    # # Select a single metric and node
    # df["rocm_utilization_percentage"]["node01"]

    # # Select a single metric, node, and GPU
    # df["rocm_utilization_percentage"]["node01"]["0"]

    # # Select GPU Utilization and GPU Memory Utilization for GPU ID 0 in all nodes
    # df.loc[:, pd.IndexSlice[["rocm_utilization_percentage", "rocm_vram_used_percentage"], :, ["0"]]]
    pass


def report():
    msamples = np.array([0.0])
    # Check if Report.csv exists
    report_path = 'Report.csv'
    if os.path.exists(report_path):
        print(f"Report.csv found at {report_path}")
        # Read Report.csv and extract MSamples
        with open(report_path, 'r') as f:
            lines = f.readlines()
            # MSamples is in the second line
            if len(lines) >= 2:
                # Parse the second line: walltime; 3.135677; MSamples; 408.282039; ...
                parts = lines[1].split(';')
                # Find MSamples value (it's after the MSamples key)
                for i, part in enumerate(parts):
                    if 'MSamples' in part and i + 1 < len(parts):
                        msamples_value = float(parts[i + 1].strip())
                        msamples = np.array([msamples_value])
                        print(f"Extracted MSamples: {msamples_value}")
                        break
            else:
                print("Report.csv doesn't have enough lines")
    else:
        print(f"Report.csv NOT found at {report_path}")
        print(f"Files in current directory: {os.listdir('.')}")

    # output csv file
    header = 'f'
    np.savetxt('output.csv', np.atleast_2d(msamples).T,
            delimiter=",", comments='',
            header=header)

def main():
    rocm_set()
    run()
    report()

if __name__ == '__main__':
    main()