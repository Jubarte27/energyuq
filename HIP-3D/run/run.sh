#!/bin/bash

script_path="$(readlink -f "$0")"
script_dir="$(dirname "$script_path")"

echo "running ../ModelagemFletcher.exe TTI 472 472 472 16 12.5 12.5 12.5 0.001 1.0 64 1 4 | tee log.txt"
time "$script_dir/../ModelagemFletcher.exe" TTI 472 472 472 16 12.5 12.5 12.5 0.001 1.0 64 1 4 | tee log.txt
