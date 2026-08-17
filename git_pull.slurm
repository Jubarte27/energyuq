#!/bin/env bash

#SBATCH --cpus-per-task=40
#SBATCH --partition=hype

#SBATCH --job-name=energyuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --output=execlog/%x_%j.out
#SBATCH --error=execlog/%x_%j.err

# shellcheck disable=SC1090

set -e

report_end() {
    local exit_code=$? 
    echo "Something made me finish. Last exit code was: $exit_code"
}
trap report_end EXIT

cd "$SCRATCH/energyuq"
git pull --rebase
