#!/bin/bash

for run_dir in final_runs/final/*/*; do
    for energy_dir in "$run_dir"/*; do
        if ! [ -d "$energy_dir" ] ; then
            continue
        fi
        cp -r -t "$run_dir" "$energy_dir"/*
    done
done