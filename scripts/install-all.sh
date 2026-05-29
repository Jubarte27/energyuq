#!/bin/bash
main() {
    set_log_depth 0
    ensure python_install
    ensure create_venv
    ensure install_jupyter
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
}

create_venv() {
    enter_new_func "Creating python venv"
    
    if [ ! -f "$$PROJECT_DIR/.venv/bin/activate" ]; then
        python3 -m venv "$PROJECT_DIR/.venv"
    fi
    
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv/bin/activate"

    pip install --upgrade pip
    pip install -r "$PROJECT_DIR/requirements.txt"
}

install_jupyter() {
    enter_new_func "Installing jupyter"

    pip install jupyter notebook
}

python_install() {
    enter_new_func "Installing python"
    eval "$(pyenv init - bash)"

    local ver
    ver="$(cat "$PROJECT_DIR/.python-version")"
    ver="${ver#"${ver%%[![:space:]]*}"}" # leading
    ver="${ver%"${ver##*[![:space:]]}"}" # trailing

    pyenv install --skip-existing "$ver"
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
main "$@"
