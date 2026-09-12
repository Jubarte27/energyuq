#!/bin/env bash

#SBATCH --job-name=energyuq-run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --output=execlog/%x_%j.out
#SBATCH --error=execlog/%x_%j.err

# shellcheck disable=SC1090

set -e

report_end() {
    local exit_code=$? 
    echo "Something made me finnish running. Last exit code was: $exit_code"
}
trap report_end EXIT

DESTINATION="$(pwd)"

if ! [ -z "$SCRATCH" ]; then cd "$SCRATCH/energyuq"; fi
SCRIPT_DIR=$(pwd)

git pull --rebase

VENV=.venv/bin/activate
ROOTV="$SCRIPT_DIR/$VENV"
EASYV="$SCRIPT_DIR/easy/$VENV"

if [ -f "$ROOTV" ]; then source "$ROOTV"
elif [ -f "$EASYV" ]; then source "$EASYV"
else echo I have not been properly initialized; exit 2; fi

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SCRIPT_DIR"

python3 "$SCRIPT_DIR/run.py" "$@"

cd "$DESTINATION" && "$SCRIPT_DIR/slurm_pcad/copy_back.slurm"