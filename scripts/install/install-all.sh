#!/bin/bash
main() {
    set_log_depth 0
    ensure python_install
    ensure install_uv
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
    install_uv
    
    if [ ! -f "$PROJECT_DIR/.venv/bin/activate" ]; then
        uv venv "$PROJECT_DIR/.venv"
    fi
    
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv/bin/activate"

    uv pip install -e "$PROJECT_DIR"
}

install_jupyter() {
    enter_new_func "Installing jupyter"

    uv pip install -e "$PROJECT_DIR[jupyter]"
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
