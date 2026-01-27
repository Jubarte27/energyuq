#!/bin/bash

if [ "$EUID" -ne 0 ]; then
    echo "I need super user privileges to do anything."
    echo "You should also read me before giving me your password"
    exec sudo -E --preserve-env=PATH "$0" "$@" || exit 1
fi

main() {
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv/bin/activate"

    cd "$PROJECT_DIR" && jupyter notebook --allow-root
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

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
main "$@"
