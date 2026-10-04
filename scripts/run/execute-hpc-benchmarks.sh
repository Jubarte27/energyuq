#!/bin/bash
set -e

# Runs the benchmarks built by hpc-benchmarks/compile-hpc-benchmarks.sh.
#
# Suites shipping a single benchmark are named after the suite itself
# (HPCG, JA, LULESH, MW, PO, ST); suites shipping several are named
# SUITE_BENCHMARK (NAS_BT, PARBOIL_SPMV, RODINIA_HOTSPOT). Run with -l to get
# the full list.

NAS_KERNELS=(BT CG EP FT IS LU MG SP UA)
RODINIA_APPS=(BFS HOTSPOT HOTSPOT3D SRAD_V1 SRAD_V2 STREAMCLUSTER)

main() {
	
    set_log_depth 0

    if [ "$LIST" == "true" ]; then
        list_benchmarks
        return 0
    fi

	if (( ${#EXEC[@]} == 0 )); then
        log_error "No benchmark selected"
    fi

	OUT_DIR=${OUT_DIR:-"$PROJECT_DIR/.benchmark-results"}
    if ! [ -d "$OUT_DIR" ]; then
        mkdir "$OUT_DIR"
    fi
	echo "Running $NAME with $NT threads"
    set -o pipefail
	OMP_RUN 2>&1 | tee "$OUT_DIR/$NAME.$NT.txt"
    set +o pipefail
}

OMP_RUN() {
    if [ -z "$OMP_PROC_BIND" ]; then
        export OMP_PROC_BIND=CLOSE
    fi
    if [ -z "$OMP_PLACES" ]; then
        export OMP_PLACES=CORES
    fi
    
    export OMP_NUM_THREADS=$NT

    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo "OMP_PROC_BIND=$OMP_PROC_BIND"
    echo "OMP_PLACES=$OMP_PLACES"

    "${EXEC[@]}"
}

execute() {
    if [ "$DRY_RUN" == "true" ]; then
        printf 'DRY RUN: %s\n' "$*"
        return 0
    fi
    if [ "$SRUN" == "true" ]; then
        if [ -z "$FREQHZ" ]; then echo ; exit 42; fi
        srun --cpu-freq="$FREQHZ-$FREQHZ:UserSpace" --cpus-per-task="$NT" --ntasks=1 --exclusive "$@"
    elif ! [ "$BUFF" == "true" ]; then
        not_buffered "$@"
    else
        "$@"
    fi
}

not_buffered() { stdbuf -oL "$@"; }

require() {
    if ! [ -x "$1" ]; then
        log_error "\"$1\" is missing. Build it with hpc-benchmarks/compile-hpc-benchmarks.sh ${2:-$NAME}"
    fi
}

# Launches a benchmark from a writable working directory so that the files it
# produces never end up inside its sources.
run_in() {
    local dir="$1"
    shift
    require "$1"
    mkdir -p "$dir"
    cd "$dir" || exit 1
    execute "$@"
}

find_exec() {
    local name="${1^^}"

    case "$name" in
        FAKE_WORK)
            EXEC=(fake_work)
            ;;
        FFT)
            EXEC=(fft)
            ;;
        HPCG)
            EXEC=(hpcg)
            ;;
        JA)
            EXEC=(ja)
            ;;
        LULESH)
            EXEC=(lulesh)
            ;;
        MW)
            EXEC=(mw)
            ;;
        PO)
            EXEC=(po)
            ;;
        ST)
            EXEC=(st)
            ;;
        RODINIA)
            EXEC=(rodinia_hotspot)
            ;;
        RODINIA_*)
            # substring after _
            case "${name#RODINIA_}" in
                BFS)
                    EXEC=(rodinia_bfs)
                    ;;
                HOTSPOT)
                    EXEC=(rodinia_hotspot)
                    ;;
                HOTSPOT3D)
                    EXEC=(rodinia_hotspot3d)
                    ;;
                SRAD_V1)
                    EXEC=(rodinia_srad_v1)
                    ;;
                SRAD_V2)
                    EXEC=(rodinia_srad_v2)
                    ;;
                STREAMCLUSTER)
                    EXEC=(rodinia_streamcluster)
                    ;;
                *)
                    log_error "Unknown benchmark \"$1\", expected one of ${RODINIA_APPS[*]/#/RODINIA_}"
                    ;;
            esac
            ;;
        NAS)
            log_error "\"NAS\" is a suite, name one of its benchmarks: ${NAS_KERNELS[*]/#/NAS_}"
            ;;
        NAS_*)
            # substring after _
            local kernel="${name#NAS_}"
            if ! contains "$kernel" "${NAS_KERNELS[@]}"; then
                log_error "Unknown benchmark \"$1\", expected one of ${NAS_KERNELS[*]/#/NAS_}"
            fi
            EXEC=(nas "$kernel")
            ;;
        PARBOIL)
            log_error "\"PARBOIL\" is a suite, name one of its benchmarks: $(parboil_suites)"
            ;;
        PARBOIL_*)
            # substring after _
            EXEC=(parboil_run "${name#PARBOIL_}" "$PARBOIL_DATASET")
            ;;
        TESTE_ERRO_OMP)
			EXEC=(teste_erro_omp)
			NAME="$1"
			;;
		NONE)
			EXEC=(echo "Running None")
			NAME="$1"
			;;
        *)
            log_error "Unknown benchmark \"$1\". Run with -l for the list of benchmarks"
            ;;
    esac
    NAME="$1"
}

contains() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        if [ "$item" == "$needle" ]; then
            return 0
        fi
    done
    return 1
}

list_benchmarks() {
    local kernel app

    echo "Benchmarks:"
    _list_benchmark "FFT" "not built by compile-hpc-benchmarks.sh"
    _list_benchmark "FAKE_WORK" "fake OpenMP work, for testing the harness"
    _list_benchmark "NONE" "no benchmark at all, for testing the harness"
    _list_benchmark "TESTE_ERRO_OMP" "the OpenMP error reproduction under teste/"
    _list_benchmark "HPCG" "hpc-benchmarks/HPCG"
    _list_benchmark "JA" "hpc-benchmarks/JA"
    _list_benchmark "LULESH" "hpc-benchmarks/LULESH"
    _list_benchmark "MW" "hpc-benchmarks/MW (MPI + OpenMP, single rank)"
    _list_benchmark "PO" "hpc-benchmarks/PO"
    _list_benchmark "ST" "hpc-benchmarks/ST"
    for kernel in "${NAS_KERNELS[@]}"; do
        _list_benchmark "NAS_$kernel" "hpc-benchmarks/NAS (NPB class C)"
    done
    for app in "${RODINIA_APPS[@]}"; do
        _list_benchmark "RODINIA_$app" "hpc-benchmarks/RODINIA"
    done
    for app in $(parboil_benchmarks); do
        _list_benchmark "PARBOIL_${app^^}" "hpc-benchmarks/PARBOIL ($PARBOIL_VERSION)"
    done
    echo
    echo "RODINIA is accepted as an alias of RODINIA_HOTSPOT."
}

_list_benchmark() {
    printf '  %-22s %s\n' "$1" "$2"
}

# PARBOIL ships many benchmarks, but only the ones with downloaded datasets can
# actually be run.
parboil_benchmarks() {
    local dir bench
    dir="$BENCHMARK_DIR/PARBOIL/datasets"
    if ! [ -d "$dir" ]; then
        return 0
    fi
    for bench in "$dir"/*; do
        if [ -d "$bench" ]; then
            printf '%s\n' "$(basename "$bench")"
        fi
    done | sort
}

parboil_suites() { printf 'PARBOIL_%s' "$(parboil_benchmarks | tr '[:lower:]' '[:upper:]' | paste -sd, -)"; }

# The PARBOIL driver resolves benchmarks and datasets relative to its own root,
# so it has to be started from there. "-C" skips the output check because the
# bundled compare-output tools are python2 scripts that cannot run here.
# Benchmarks are named in upper case here, the driver wants them lower case.
parboil_run() {
    local bench="${1,,}"
    local dataset="${2,,}"

    if ! [ -d "$BENCHMARK_DIR/PARBOIL/datasets/$bench/$dataset" ]; then
        log_error "PARBOIL benchmark \"$bench\" has no dataset \"$dataset\", available: $(ls "$BENCHMARK_DIR/PARBOIL/datasets/$bench" 2>/dev/null | paste -sd, -)"
    fi

    cd "$BENCHMARK_DIR/PARBOIL" || exit 1
    execute ./parboil run "$bench" "$PARBOIL_VERSION" "$dataset" -C
}

fft()    { run_in "$BENCHMARK_DIR/FFT/out" "$BENCHMARK_DIR/FFT/fft_omp"; }
hpcg()   { run_in "$BENCHMARK_DIR/HPCG/out" "$BENCHMARK_DIR/HPCG/HPCCG_BIN" 256 256 128; }
ja()     { run_in "$BENCHMARK_DIR/JA/out" "$BENCHMARK_DIR/JA/omp_ja"; }
lulesh() { run_in "$BENCHMARK_DIR/LULESH/out" "$BENCHMARK_DIR/LULESH/lulesh2.0" -i 5000 -s 50; }
# MPI + OpenMP build; one rank, so OMP_NUM_THREADS alone drives the scaling.
mw()     { run_in "$BENCHMARK_DIR/MW/c/build/out" "$BENCHMARK_DIR/MW/c/build/openmp"; }
po()     { run_in "$BENCHMARK_DIR/PO/out" "$BENCHMARK_DIR/PO/omp_po"; }
st()     { run_in "$BENCHMARK_DIR/ST/out" "$BENCHMARK_DIR/ST/stream"; }
nas()    { run_in "$BENCHMARK_DIR/NAS/out" "$BENCHMARK_DIR/NAS/bin/${1,,}.C.x"; }
teste_erro_omp() { cd "$PROJECT_DIR/teste" && execute ./execute.sh; }

rodinia_bfs() {
    run_in "$RODINIA_DIR/out/bfs" "$RODINIA_DIR/openmp/bfs/bfs" \
        "$NT" "$RODINIA_DIR/data/bfs/graph1MW_6.txt"
}

# <rows> <cols> <iterations> <nthreads> <temp file> <power file> <output file>
rodinia_hotspot() {
    run_in "$RODINIA_DIR/out/hotspot" "$RODINIA_DIR/openmp/hotspot/hotspot" \
        1024 1024 "$HOTSPOT_ITERS" "$NT" \
        "$RODINIA_DIR/data/hotspot/temp_1024" "$RODINIA_DIR/data/hotspot/power_1024" output.out
}

# <rows/cols> <layers> <iterations> <power file> <temp file> <output file>
rodinia_hotspot3d() {
    run_in "$RODINIA_DIR/out/hotspot3d" "$RODINIA_DIR/openmp/hotspot3D/3D" \
        512 8 "$HOTSPOT3D_ITERS" \
        "$RODINIA_DIR/data/hotspot3D/power_512x8" "$RODINIA_DIR/data/hotspot3D/temp_512x8" \
        output.out
}

# <k1> <k2> <dim> <chunksize> <clustersize> <n> <mode> <outfile> <nthreads>
rodinia_streamcluster() {
    run_in "$RODINIA_DIR/out/streamcluster" "$RODINIA_DIR/openmp/streamcluster/sc_omp" \
        10 20 256 65536 65536 "$STREAMCLUSTER_ITERS" none output.txt "$NT"
}

# <iterations> <saturation> <rows> <cols> <nthreads>.
# main.c hardcodes its input as "../../../data/srad/image.pgm", so the working
# directory has to sit exactly three levels below RODINIA. The nesting in
# run_in below is what satisfies that, and it keeps image_out.pgm out of the
# sources.
rodinia_srad_v1() {
    local run_dir="$RODINIA_DIR/out/srad/v1"

    mkdir -p "$run_dir"
    if ! [ -r "$run_dir/../../../data/srad/image.pgm" ]; then
        log_error "missing $run_dir/../../../data/srad/image.pgm"
    fi

    run_in "$run_dir" "$RODINIA_DIR/openmp/srad/srad_v1/srad" \
        "$SRAD_ITERS" 0.5 502 458 "$NT"
}

# <rows> <cols> <y1> <y2> <x1> <x2> <nthreads> <lambda> <iterations>
rodinia_srad_v2() {
    run_in "$RODINIA_DIR/out/srad_v2" "$RODINIA_DIR/openmp/srad/srad_v2/srad" \
        2048 2048 0 127 0 127 "$NT" 0.5 2
}

fake_work() {
    local iters=${1:-500}

    echo "Fake work: $iters"

    for ((thread_i = 0; thread_i < iters; thread_i++)); do
        (
            local acc=0
            local i
            for ((i = 0; i < iters; i++)); do
                acc=$(( (acc + i + thread_i) % 420 ))
            done
            printf '%d' "$acc"
        ) &
    done
    wait
}

_setConfigArgs() {
    while [ "${1:-}" != '' ]; do
        case "$1" in
            ## Options
            -b)
                BUFF=false
                ;;
            -l|--list)
                LIST=true
                ;;
            -n|--dry-run)
                DRY_RUN=true
                ;;
            -s|--srun)
                SRUN=true
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
	if [ "$LIST" == "true" ]; then
		return
	fi
	if [ "${1:-}" == '' ]; then
		log_error "First argument must be the name of a benchmark"
	fi
	if [ "${2:-}" == '' ]; then
		log_error "Second argument must be the number of threads"
	elif ! [[ "$2" =~ ^[+-]?[0-9]+$ ]]; then
		log_error "The number of threads must be an integer"
	fi

	find_exec "$1"
	NT=$2

    FREQHZ=$3
    BOOST=$4
    NUMA=$5
}


set_env() {
	BENCHMARK_DIR="$PROJECT_DIR/hpc-benchmarks"
    RODINIA_DIR="$BENCHMARK_DIR/RODINIA"

    # The miniWeather and PARBOIL binaries link against the toolchain and
    # libraries staged in .deps by hpc-benchmarks/initialize.sh.
    if [ -d "$BENCHMARK_DIR/.deps/bin" ]; then
        export PATH="$BENCHMARK_DIR/.deps/bin:$PATH"
    fi
    if [ -d "$BENCHMARK_DIR/.deps/lib" ]; then
        export LD_LIBRARY_PATH="$BENCHMARK_DIR/.deps/lib:${LD_LIBRARY_PATH:-}"
    fi

	EXEC=()
	NAME=""
    BUFF=true
    LIST=false
    DRY_RUN=false
    # Workload sizes, kept overridable so the runs can be tuned without editing
    # this script.
    HOTSPOT_ITERS=${HOTSPOT_ITERS:-100000}
    HOTSPOT3D_ITERS=${HOTSPOT3D_ITERS:-100}
    STREAMCLUSTER_ITERS=${STREAMCLUSTER_ITERS:-1000}
    SRAD_ITERS=${SRAD_ITERS:-100}
    PARBOIL_DATASET=${PARBOIL_DATASET:-large}
    PARBOIL_VERSION=${PARBOIL_VERSION:-omp_base}

    if ! [ -z "$ENERGYUQ_SLURM" ]; then
        SRUN=$ENERGYUQ_SLURM
    elif ! [ -z "$SLURM_JOB_ID" ]; then
        SRUN=true
    else
        SRUN=false
    fi
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
set_env
_setConfigArgs "$@"
main "$@"
