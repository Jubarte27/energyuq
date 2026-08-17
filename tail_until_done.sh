#!/bin/bash
has_writer() {
    local file="$1"
    if command -v lsof >/dev/null 2>&1; then
        # Check for Write (w) or Read/Write (u) locks via lsof
        lsof -F f "$file" 2>/dev/null | grep -q -i '[wu]'
    elif command -v fuser >/dev/null 2>&1; then
        # Fallback to fuser
        fuser "$file" >/dev/null 2>&1
    else
        # Fallback for Linux inspecting /proc
        local file_realpath
        file_realpath=$(readlink -f "$file" 2>/dev/null)
        for fd in /proc/[0-9]*/fd/*; do
            if [ "$(readlink -f "$fd" 2>/dev/null)" = "$file_realpath" ]; then
                return 0
            fi
        done 2>/dev/null
        return 1
    fi
}

tail_until_done() {
    local file="$1"

    if [ -z "$file" ] || [ ! -f "$file" ]; then
        echo "Usage: tail_until_done <file>" >&2
        return 1
    fi

    tail -f "$file" &
    local tail_pid=$!

    trap 'kill "$tail_pid" 2>/dev/null' INT TERM EXIT

    while has_writer "$file"; do
        sleep 1
    done

    sleep 0.5
    kill "$tail_pid" 2>/dev/null
    exit 0
}

get_latest_file() {
    local dir="${1:-.}"
    local pattern="${2:-*}"

    shopt -s nullglob
    local files=("$dir"/$pattern)
    shopt -u nullglob

    if [ ${#files[@]} -eq 0 ]; then return 1; fi

    ls -td "${files[@]}" 2>/dev/null | head -n 1
}


HERE=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")

file=$(get_latest_file "$HERE/execlog" "${1:-"*.out"}")
if [ -z "$file" ]; then echo "No file found in \"$HERE/execlog\" that matches \"${1:-"*.out"}"; exit 1; fi
echo "Tailing $file..."
tail_until_done "$file"