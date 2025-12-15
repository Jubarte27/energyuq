#!/bin/bash
main() {
    set_log_depth 0
    ensure create_venv
    ensure HIP_installation
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

HIP_installation() {
    enter_new_func "Making HIP"

    cd "$PROJECT_ROOT/HIP-3D" && make
}

create_venv() {
    enter_new_func "Creating python venv"
    
    if [ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
        python3 -m venv "$PROJECT_ROOT/.venv"
    fi
    
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"

    pip install --upgrade pip -r "$PROJECT_ROOT/requirements.txt"
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
main "$@"
