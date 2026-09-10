#!/usr/bin/bash
set -e

main() {

    if [[ ! -d "$CONFIG_DIR" ]]; then
        log_error "Configuration directory '$CONFIG_DIR' not found."
    fi

    shopt -s nullglob
    env_files=("$CONFIG_DIR"/*)
    shopt -u nullglob

    if [[ ${#env_files[@]} -eq 0 ]]; then
        log_error "No files found in '$CONFIG_DIR'."
    fi

    log_info "Found ${#env_files[@]} configurations in '$CONFIG_DIR'."


    if [[ -v ONLY ]]; then
        files=()
        for env_file in "${env_files[@]}"; do
            for node in "${ONLY[@]}"; do
                if [[ "$(basename "$env_file")" == "$node" ]]; then
                    files+=("$env_file")
                    break
                fi
            done

        done
    else
        files=("${env_files[@]}")
    fi
    

    for env_file in "${files[@]}"; do
        # Run each submission in a subshell so variables do not leak between jobs
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[$(basename "$env_file")] \"$PROJECT_DIR/scripts/run/run_sbatch_pcad.sh\" --env \"$env_file\" \"$SCRIPT\""
        else
            log_info "Submitting: $(basename "$env_file") -> $SCRIPT"
            "$PROJECT_DIR/scripts/run/run_sbatch_pcad.sh" --env "$env_file" "$SCRIPT"
        fi
    done
}

_setConfigArgs() {
    while [ "${1:-}" != '' ]; do
        case "$1" in
            --only)
                export ONLY=()
                IFS=, read -r -a ONLY <<< "$2"
                shift
                ;;
            --envs-dir)
                export CONFIG_DIR="$2"
                shift
                ;;
            ## end of Options
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


    export SCRIPT="$1"
    export CONFIG_DIR="${CONFIG_DIR:-"$PROJECT_DIR/slurm_nodes"}"
    export DRY_RUN="${DRY_RUN:-false}"
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
main "$@"