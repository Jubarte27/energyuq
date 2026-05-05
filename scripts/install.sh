#!/bin/bash
main() {
    set_log_depth 0
    ensure create_venv
    ensure install_local_easyvvuq
    ensure install_jupyter
    if [ "$COMPILE_HPC" == "true" ] ; then
        ensure compile_hpc_benchmarks
    fi
}
_setConfigArgs() {
    while [ "${1:-}" != '' ]; do
        case "$1" in
            ## Options
            --hpc)
                COMPILE_HPC=true
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

    EasyVVUQ_DIR="$PROJECT_DIR/EasyVVUQ"
}

create_venv() {
    enter_new_func "Creating python venv"
    
    if [ ! -f "$PROJECT_DIR/.venv/bin/activate" ]; then
        python3 -m venv "$PROJECT_DIR/.venv"
    fi
    
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv/bin/activate"

    pip install --upgrade pip
    pip install -r "$PROJECT_DIR/requirements.txt"
}

install_local_easyvvuq() {
    enter_new_func "Installing easyvvuq"

    cd "$PROJECT_DIR" || exit 1
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
    cd "$PROJECT_DIR" || exit 1
    pyenv local 3.9
}

compile_hpc_benchmarks() {
    enter_new_func "Compiling HPC benchmarks"

    "${SCRIPT_DIR}/compile-hpc-benchmarks.sh"
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
main "$@"
