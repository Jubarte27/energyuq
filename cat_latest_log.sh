#!/bin/bash

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
echo "cat $file..."
cat "$file"