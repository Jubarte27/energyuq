#!/bin/bash
set -e
main() {
    set_log_depth 0
    ensure install_uv
    ensure create_venv
    ensure install_local_easyvvuq
    # ensure install_jupyter
}
_setConfigArgs() {
    EasyDIR="$PROJECT_DIR/easy"
    EasyVVUQ_DIR="$EasyDIR/EasyVVUQ"
    VENV_DIR=$EasyDIR/.venv
    while [ "${1:-}" != '' ]; do
        case "$1" in
            ## Options
            -g|--global)
                VENV_DIR="$PROJECT_DIR/.venv"
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

}

create_venv() {
    enter_new_func "Creating python venv"
    install_uv
    
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        uv venv "$VENV_DIR"
    fi
    
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    uv pip install -e "$PROJECT_DIR"
}

install_local_easyvvuq() {
    enter_new_func "Installing easyvvuq"

    cd "$EasyDIR" || exit 1
    if [ ! -f "$EasyVVUQ_DIR/requirements.txt" ]; then
        git submodule update --init "$EasyVVUQ_DIR" "$PROJECT_DIR/hpc-benchmarks"
    fi
    uv pip install setuptools wheel build
    uv pip install -e "$EasyVVUQ_DIR"
}

install_jupyter() {
    enter_new_func "Installing jupyter"

    uv pip install -e "$PROJECT_DIR[jupyter]"
}

python39() {
    enter_new_func "Installing python 3.9"

    eval "$(pyenv init - bash)"
    pyenv install --skip-existing 3.9
    cd "$EasyDIR" || exit 1
    pyenv local 3.9
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
main "$@"
