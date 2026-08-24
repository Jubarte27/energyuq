#!/bin/bash

get_latest_file() {
    local dir="${1:-.}"
    local pattern="${2:-*}"

    shopt -s nullglob
    local files=("$dir"/"$pattern")
    shopt -u nullglob

    if [ ${#files[@]} -eq 0 ]; then return 1; fi

    local latest=""
    for f in "${files[@]}"; do
        if [ -z "$latest" ] || [ "$f" -nt "$latest" ]; then
            latest="$f"
        fi
    done
    printf '%s' "$latest"
}


SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"


file=$(get_latest_file "$PROJECT_DIR/execlog" "${1:-"*.out"}")
if [ -z "$file" ]; then echo "No file found in \"$PROJECT_DIR/execlog\" that matches \"${1:-"*.out"}"; exit 1; fi
echo "cat $file..."
cat "$file"