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
    echo "Something made me finnish. Last exit code was: $exit_code"
}
trap report_end EXIT

if ! [ -z "$SCRATCH" ]; then cd "$SCRATCH/energyuq"; fi

NEWEST_DIR=$(find run_results -mindepth 1 -maxdepth 1 -type d -exec stat --printf="%Y\t%n\n" {} + | sort -n -r | head -1 | cut -f2-)

cp -r -t "$HOME/run_results/" "$NEWEST_DIR"