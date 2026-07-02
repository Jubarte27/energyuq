#!/bin/bash

#SBATCH --job-name=energyuq
#SBATCH --partition=hype
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=3:00:00
#SBATCH --output=execlog/%x_%j.out
#SBATCH --error=execlog/%x_%j.err

set -e

report_end() {
    local exit_code=$? 
    echo "Something made me finnish. Last exit code was: $exit_code"
}
trap report_end EXIT

cd "$SCRATCH"

if [ -d "energyuq" ]; then
    rm -fr energyuq
fi
git clone https://github.com/Jubarte27/energyuq/

cd energyuq
git submodule update --init easy/EasyVVUQ hpc-benchmarks

./scripts/install-easy.sh

./scripts/compile-hpc-benchmarks.sh HPCG

source easy/.venv/bin/activate
python3 run.py

echo "Done executing. Copying back."

# copy back

PARENT_DIR="$HOME"
COUNTER=1

while [ -d "$PARENT_DIR/run_results_$COUNTER" ]; do ((COUNTER++)); done
UNIQUE="$PARENT_DIR/run_results_$COUNTER"
mkdir -p "$UNIQUE"
cp -r "$SCRATCH/energyuq/easy/run_results" "$UNIQUE"

echo All finished
