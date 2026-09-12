#!/usr/bin/bash
set -e

main() {
    local temporary_run_path="$PROJECT_DIR/run_results"
    local relative
    relative=$(realpath --relative-to="$temporary_run_path" "$RUN_PATH")
    local base_dir="$PROJECT_DIR/runs/$RUN_TYPE"

    local nodes=()
    
    shopt -s nullglob
    for node in "$PROJECT_DIR/slurm_nodes/"*; do
        nodes+=("$(basename "$node")")
    done

    local all_outputs=( "$RUN_PATH"/output/* )
    shopt -u nullglob

    local first_output="${all_outputs[0]}"

    if [ -n "$first_output" ]; then
        local benchmark
        benchmark=$(basename "$first_output" | rev | cut -d'_' -f2- | rev)
        base_dir="$base_dir/$benchmark"
    fi

    local target_dir_name
    target_dir_name="$(basename "$relative")"

    local maybe_node
    maybe_node=$(echo "$relative" | cut -d'/' -f1)
    for node in "${nodes[@]}"; do
        if [ "$maybe_node" = "$node" ]; then
            clean_node="${node//[0-9]/}"
            target_dir_name=$clean_node
        fi
    done
    local target_dir="$base_dir/$target_dir_name"

    if [ -d "$target_dir" ]; then
        echo \""$target_dir"\" already exists
        exit 42
    fi

    cp -r "$RUN_PATH" "$target_dir"

}

_setConfigArgs() {
    while [ "${1:-}" != '' ]; do
        case "$1" in
            ## end of Options
            [!-]*)
                break
                ;;
            *)
                log "$WARN" "Unknown option \"$1\", ignoring" 0 
            ;;
        esac
    done
    RUN_PATH=$1
    RUN_TYPE=$2
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
main "$@"