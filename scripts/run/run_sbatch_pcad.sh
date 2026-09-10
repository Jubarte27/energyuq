#!/bin/bash
set -e

main() {
    if ! [ -z "$DOTENV" ]; then
        set -a && source "$DOTENV" && set +a
    fi

    EXTRA_ARGS=()
    if [[ -n "${SBATCH_OPTS:-}" ]]; then
        read -r -a EXTRA_ARGS <<< "$SBATCH_OPTS"
    fi

    if ! [ -z "$SLURM_CPUS_PER_TASK" ]; then
        EXTRA_ARGS+=(--cpus-per-task="$SLURM_CPUS_PER_TASK")
    fi

    sbatch "${EXTRA_ARGS[@]}" "$PROJECT_DIR/slurm_pcad/$SCRIPT.slurm" "${EXTRA_SCRIPT_ARGS[@]}"
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