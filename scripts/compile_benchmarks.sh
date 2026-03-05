#!/bin/bash
# need to have pyenv installed
main() {
    set_log_depth 0
    if ! [ -d "$BENCHMARKS_DIR" ]; then
        (cd "$PROJECT_DIR" && git clone https://github.com/wbmaas/hpc-benchmarks)
    fi

    cd "$BENCHMARKS_DIR" || exit 1

    if [ "$JUSTCLEAN" == "true" ]; then
        clean || exit 1
        cd "$BENCHMARKS_DIR" && rm -f .python-version
        git restore "$BENCHMARKS_DIR/**/*.pyc" ## python 2 é feio
        return
    fi

    pyenv_setup

    silent_make FFT
    silent_make JA
    silent_make PO
    silent_make ST
    silent_make LULESH
    silent_make HPCG

    silent_make RODINIA/hotspot
    silent_make RODINIA/lud
    silent_make RODINIA/streamcluster
    silent_make RODINIA/data/hotspot/inputGen

    {
        silent_make NAS
        silent_make NAS BT CLASS=B
        silent_make NAS CG CLASS=B
        silent_make NAS FT CLASS=B
        silent_make NAS LU CLASS=B
        silent_make NAS MG CLASS=B
        silent_make NAS SP CLASS=B
        silent_make NAS UA CLASS=B
    } > /dev/null

    # cd "$BENCHMARKS_DIR/parboil" && ./parboil list
    cd "$BENCHMARKS_DIR/parboil" && ./parboil compile stencil omp_base
    # cd "$BENCHMARKS_DIR/parboil" && ./parboil run     stencil omp_base default
}

clean() {
    clean_make FFT;
    clean_make HPCG;
    clean_make JA;
    clean_make LULESH;
    clean_make PO;
    clean_make ST;

    clean_make RODINIA/hotspot
    clean_make RODINIA/lud
    clean_make RODINIA/streamcluster
    clean_make RODINIA/data/hotspot/inputGen

    silent_make NAS veryclean

    cd "$BENCHMARKS_DIR/parboil" && ./parboil clean stencil omp_base
}

clean_make() {
    make --silent -C "$1" clean
}

silent_make() {
    make --silent -C "$@"
}

pyenv_setup() {
    eval "$(pyenv init - bash)"
    pyenv install --skip-existing 2
    cd "$BENCHMARKS_DIR" || exit 1
    pyenv local 2
}

_setConfigArgs() {
    while [ "${1:-}" != '' ]; do
        case "$1" in
            ## Options
            -c | --clean)
                JUSTCLEAN=true
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

set_env() {
    BENCHMARKS_DIR="$PROJECT_DIR/hpc-benchmarks"
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
set_env
main "$@"
