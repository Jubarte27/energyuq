#!/bin/bash
main() {
    set_log_depth 0
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
    
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        python3 -m venv "$VENV_DIR"
    fi
    
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    which pip
    pip install --upgrade pip
    pip install -r "$VENV_DIR/../requirements.txt"
}

install_local_easyvvuq() {
    enter_new_func "Installing easyvvuq"

    cd "$EasyDIR" || exit 1
    if [ ! -f "$EasyVVUQ_DIR/requirements.txt" ]; then
        git submodule init && git submodule update
    fi
    pip install setuptools wheel build
    pip install -e "$EasyVVUQ_DIR"
}

install_jupyter() {
    enter_new_func "Installing jupyter"

    pip install jupyter notebook
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
