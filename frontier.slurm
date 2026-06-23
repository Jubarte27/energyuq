#!/bin/bash

#SBATCH --job-name=energyuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=3:00:00
#SBATCH --output=energyuq_%x_%j.out
#SBATCH --error=energyuq_%x_%j.err

set -e

git clone https://github.com/Jubarte27/energyuq/
cd energyuq

./scripts/install-easy.sh

./scripts/compile-hpc-benchmarks.sh

python3 run.py

# todo copy back
# cp -r results .......
