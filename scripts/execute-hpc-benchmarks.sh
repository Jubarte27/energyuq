#!/bin/bash

main() {
	
    set_log_depth 0

	if (( ${#EXEC[@]} == 0 )); then
        exit 1
    fi

	OUT_DIR=${OUT_DIR:-"$PROJECT_DIR/.benchmark-results"}
    if ! [ -d "$OUT_DIR" ]; then
        mkdir "$OUT_DIR"
    fi
	echo "Running $NAME with $NT threads"
	{
	    export OMP_NUM_THREADS=$NT
	    export OMP_PROC_BIND=CLOSE
	    export OMP_PLACES=CORES

		echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
		echo "OMP_PROC_BIND=$OMP_PROC_BIND"
		echo "OMP_PLACES=$OMP_PLACES"

		"${EXEC[@]}"
	} 2>&1 | tee "$OUT_DIR/$NAME.$NT.txt"
}

maybe_buffered() {
    if [ "$BUFF" == "true" ]; then
        "$@"
    else
        stdbuf -oL "$@"
    fi
}

find_exec() {
    case "$1" in
        FFT | fft)
            EXEC=(fft)
            ;;
        HPCG | hpcg)
            EXEC=(hpcg)
            ;;
        JA | ja)
            EXEC=(ja)
            ;;
        LULESH | lulesh)
            EXEC=(lulesh)
            ;;
        PO | po)
            EXEC=(po)
            ;;
        RODINIA | rodinia)
            EXEC=(rodinia_hotspot)
            ;;
        ST | st)
            EXEC=(st)
            ;;
        NAS_* | nas_*)
            # substring after _
            case ${1#*_} in
                BT | bt | \
                CG | cg | \
                FT | ft | \
                LU | lu | \
                MG | mg | \
                SP | sp | \
                UA | ua)
                    EXEC=(nas "${1#*_}")
                    ;;
                *)
                    log_error "Unknown benchmark \"$1\""
                    ;;
            esac
            ;;
        PARBOIL_* | parboil_*)
            # substring after _
            case ${1#*_} in
                BFS | bfs)
                    EXEC=(parboil_bfs)
                    ;;
                LBM | lbm)
                    EXEC=(parboil_lbm)
                    ;;
                *)
                    log_error "Unknown benchmark \"$1\""
                    ;;
            esac
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
            log_error "Unknown benchmark \"$1\""
            ;;
    esac
    NAME="$1"
}

fft()    { cd "$BENCHMARK_DIR/FFT"    && maybe_buffered ./fft_omp; }
hpcg()   { cd "$BENCHMARK_DIR/HPCG"   && maybe_buffered ./HPCCG_BIN 256 256 128; }
ja()     { cd "$BENCHMARK_DIR/JA"     && maybe_buffered ./omp_ja; }
lulesh() { cd "$BENCHMARK_DIR/LULESH" && maybe_buffered ./lulesh2.0 -i 5000 -s 50; }
po()     { cd "$BENCHMARK_DIR/po"     && maybe_buffered ./omp_po; }
st()     { cd "$BENCHMARK_DIR/ST"     && maybe_buffered ./stream ;}
nas()    { cd "$BENCHMARK_DIR/NAS"    && maybe_buffered "./$1.B.x"; }
teste_erro_omp() { cd "$PROJECT_DIR/teste" && maybe_buffered ./execute.sh; }

parboil_bfs() { cd "$BENCHMARK_DIR/parboil" && maybe_buffered ./parboil run bfs omp_base SF ;}
parboil_lbm() { cd "$BENCHMARK_DIR/parboil" && maybe_buffered ./parboil run lbm omp_cpu long ; }

rodinia_hotspot() { cd "$BENCHMARK_DIR/RODINIA/hotspot" && maybe_buffered ./hotspot 1024 1024 100000 "$N_THREADS" ../data/hotspot/temp_1024 ../data/hotspot/power_1024 output.out ;}

_setConfigArgs() {
    while [ "${1:-}" != '' ]; do
        case "$1" in
            ## Options
            -b)
                BUFF=false
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
}


set_env() {
	BENCHMARK_DIR="$PROJECT_DIR/hpc-benchmarks"

	EXEC=()
	NAME=""
    BUFF=true
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
set_env
_setConfigArgs "$@"
main "$@"
