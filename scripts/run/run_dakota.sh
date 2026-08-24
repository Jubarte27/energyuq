#!/bin/env bash
set -o pipefail
main() {
    set_log_depth 0
    
    source "$SCRIPT_DIR/dakota-to-path.sh"
    mkdir -p "$RUN_DIR"
    ensure cd "$RUN_DIR"
    cat << EOF > dakota_driver.sh
#!/bin/env bash
source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$PROJECT_DIR"
python3 -m src.wrappers.dakota_wrapper "\$@"
EOF
    chmod +x dakota_driver.sh
    dakota "$IN_FILE"
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

    IN_FILE="$PROJECT_DIR/dakota/dakota_pce.in"

    BASE_DIR="$PROJECT_DIR/dakota/dev"
    RUN_DIR="${1:-"$BASE_DIR/run"}"
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
main "$@"