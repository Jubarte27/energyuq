#!/usr/bin/env bash

echo "=========================================================================="
printf "%-22s %-12s %-12s %-18s\n" "PARTITION" "STATE" "NODE COUNT" "CPUs (A/I/O/T)"
echo "=========================================================================="

# Query sinfo for partitions in 'idle' (100% free) or 'mix' (partially free) states
sinfo -h -o "%P %T %D %C" | awk '$2 ~ /idle|mix/' | while read -r partition state nodes cpus; do
    printf "%-22s %-12s %-12s %-18s\n" "$partition" "$state" "$nodes" "$cpus"
done