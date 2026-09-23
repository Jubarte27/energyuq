#!/bin/bash
set -e

main() {

    EXTRA_ARGS=()
    if [[ -n "${SBATCH_OPTS:-}" ]]; then
        read -r -a EXTRA_ARGS <<< "$SBATCH_OPTS"
    fi

    local target
    local name
    if [ -n "$DOTENV" ]; then
        target=$(basename "$DOTENV")
        for env_file in "$PROJECT_DIR/slurm_nodes"/*; do
            [ -f "$env_file" ] || continue
            name=$(basename "$env_file")
            if [[ "$target" == "$name" || "$target" == "$name"[0-9]* || "$target" == "$name"\[* ]]; then
                set -a && source "$env_file" && set +a
                break
            fi
        done
    fi

    if [ -n "$DOTENV" ] && ! [ "$target" == "$name" ]; then
        EXTRA_ARGS+=(--nodelist="$target")
    fi

    if [ -n "$SLURM_CPUS_PER_TASK" ]; then
        EXTRA_ARGS+=(--cpus-per-task="$SLURM_CPUS_PER_TASK")
    fi

    echo sbatch "${EXTRA_ARGS[@]}" "$PROJECT_DIR/slurm_pcad/$SCRIPT.slurm" "${EXTRA_SCRIPT_ARGS[@]}"
}

_setConfigArgs() {
    while [ "${1:-}" != '' ]; do
        case "$1" in
            ## end of Options
            --env)
                DOTENV="$2"
                shift
                ;;
            [!-]*)
                break
                ;;
            *)
                log "$WARN" "Unknown option \"$1\", ignoring" 0 
            ;;
        esac
        shift
    done
	if [ "${1:-}" == '' ]; then
		log_error "First argument must be the name of a slurm script"
	fi
    SCRIPT="$1"
    shift
    EXTRA_SCRIPT_ARGS=("$@")
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
main "$@"