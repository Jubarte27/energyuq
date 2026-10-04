#!/bin/env bash

#SBATCH --cpus-per-task=40
#SBATCH --partition=hype

#SBATCH --job-name=energyuq-copyback
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH --output=execlog/%x_%j.out
#SBATCH --error=execlog/%x_%j.err

# shellcheck disable=SC1090

set -e

report_end() {
    local exit_code=$? 
    echo "Something made me finnish copying. Last exit code was: $exit_code"
}
trap report_end EXIT

DESTINATION="$(pwd)"

if ! [ -z "$SCRATCH" ]; then cd "$SCRATCH/energyuq"; fi

NEWEST_DIR=$(find run_results -mindepth 1 -maxdepth 1 -type d -exec stat --printf="%Y\t%n\n" {} + | sort -n -r | head -1 | cut -f2-)
NEWEST_DIR=${NEWEST_DIR%/}

UUID=$(python3 -c 'import uuid; print(uuid.uuid4().hex)')
TARGET_DIR="$DESTINATION/run_results/$(hostname)${1:+/$1}/$(basename "${NEWEST_DIR:?}")_$UUID"

if [ -d "$TARGET_DIR" ]; then echo "$TARGET_DIR already exists"; exit 42 ; fi

mkdir -p "$TARGET_DIR"

cp -r -t "$(dirname "$TARGET_DIR")" "$NEWEST_DIR"*