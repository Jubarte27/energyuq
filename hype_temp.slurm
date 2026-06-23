#!/bin/bash

#SBATCH --job-name=energyuq
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=3:00:00
#SBATCH --output=energyuq_%x_%j.out
#SBATCH --error=energyuq_%x_%j.err

set -e
cd "$SCRATCH"

# if [ -d "energyuq" ]; then
#     rm -fr energyuq
# fi
# git clone https://github.com/Jubarte27/energyuq/

cd energyuq
# git submodule update --init easy/EasyVVUQ hpc-benchmarks

# ./scripts/install-easy.sh

# ./scripts/compile-hpc-benchmarks.sh

source easy/.venv/bin/activate
python3 run.py

# ./scripts/execute-hpc-benchmarks.sh HPCG 40

# todo copy back
cp -r "$SCRATCH/energyuq/easy/run_results" "$HOME/run_results"
