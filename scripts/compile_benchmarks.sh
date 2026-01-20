#!/bin/bash
# need to have pyenv installed
main() {
    set_log_depth 0
    pyenv_setup

    cd "$BENCHMARKS_DIR" || 1

    dirty_make FFT
    dirty_make JA
    dirty_make PO
    dirty_make ST
    dirty_make LULESH
    dirty_make HPCG
    dirty_make RODINIA/hotspot

    dirty_make NAS > /dev/null
    dirty_make NAS BT CLASS=B > /dev/null
    dirty_make NAS CG CLASS=B > /dev/null
    dirty_make NAS FT CLASS=B > /dev/null
    dirty_make NAS LU CLASS=B > /dev/null
    dirty_make NAS MG CLASS=B > /dev/null
    dirty_make NAS SP CLASS=B > /dev/null
    dirty_make NAS UA CLASS=B > /dev/null

    # cd "$BENCHMARKS_DIR/parboil" && ./parboil list
    cd "$BENCHMARKS_DIR/parboil" && ./parboil compile stencil omp_base
    # cd "$BENCHMARKS_DIR/parboil" && ./parboil run     stencil omp_base default
}

clean_make() {
    make --silent -C "$1" clean
}

dirty_make() {
    local DIR=$1
    shift 1
    make --silent -C "$DIR" "$@"
}

pyenv_setup() {
    pyenv install --skip-existing 2
    cd "$BENCHMARKS_DIR" || exit 1
    pyenv local 2
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

set_env() {
    BENCHMARKS_DIR="$PROJECT_DIR/hpc-benchmarks"
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
set_env
main "$@"
