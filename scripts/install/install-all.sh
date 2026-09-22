#!/bin/bash
main() {
    set_log_depth 0
    ensure ensure_uv
    ensure create_venv
    ensure install_jupyter
    ensure fetch_repos
    ensure install_local_easyvvuq
}
_setConfigArgs() {
    while [ "${1:-}" != '' ]; do
        case "$1" in
            ## Options
            
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
    EasyVVUQ_DIR="$PROJECT_DIR/easy/EasyVVUQ"
    BENCHMARKS_DIR="$PROJECT_DIR/hpc-benchmarks"
}

fetch_repos() {
    enter_new_func "Fetching submodules"

    git submodule update --init "$EasyVVUQ_DIR" "$BENCHMARKS_DIR"
    (cd "$BENCHMARKS_DIR" && git submodule update --init MW)
}

create_venv() {
    enter_new_func "Creating python venv"
    
    if [ ! -f "$PROJECT_DIR/.venv/bin/activate" ]; then
        uv venv --python 3.12 "$PROJECT_DIR/.venv"
    fi
    
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv/bin/activate"

    uv pip install -e "$PROJECT_DIR"
}

install_jupyter() {
    enter_new_func "Installing jupyter"

    uv pip install -e "${PROJECT_DIR}[jupyter]"
}

install_local_easyvvuq() {
    enter_new_func "Installing easyvvuq"

    uv pip install -e "$EasyVVUQ_DIR"
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
main "$@"
