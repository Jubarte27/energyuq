#!/bin/bash

main() {
	
    set_log_depth 0

	if [ -z "$EXEC" ]; then
        exit 1
    fi
	export OMP_PROC_BIND=CLOSE
	export OMP_PLACES=CORES
	export OMP_NUM_THREADS=$NT

	OUT_DIR=${OUT_DIR:-"$PROJECT_DIR/.benchmark-results"}
    if ! [ -d "$OUT_DIR" ]; then
        mkdir "$OUT_DIR"
    fi
	echo "Running $NAME with $NT threads"
	{
		echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
		echo "OMP_PROC_BIND=$OMP_PROC_BIND"
		echo "OMP_PLACES=$OMP_PLACES"
		$EXEC
	} | tee "$OUT_DIR/$NAME.$NT.txt"
}

find_exec() {
    case "$1" in
        FFT | fft)
            EXEC=fft
            ;;
        HPCG | hpcg)
            EXEC=hpcg
            ;;
        JA | ja)
            EXEC=ja
            ;;
        LULESH | lulesh)
            EXEC=lulesh
            ;;
        PO | po)
            EXEC=po
            ;;
        RODINIA | rodinia)
            EXEC=rodinia_hotspot
            ;;
        ST | st)
            EXEC=st
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
                    EXEC=nas ${1#*_}
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
                    EXEC=parboil_bfs
                    ;;
                LBM | lbm)
                    EXEC=parboil_lbm
                    ;;
                *)
                    log_error "Unknown benchmark \"$1\""
                    ;;
            esac
            ;;
		NONE)
			EXEC="echo \"Running None\""
			NAME="$1"
			;;
        *)
            log_error "Unknown benchmark \"$1\""
            ;;
    esac
}

fft()    { cd "$BENCHMARK_DIR/FFT"    && ./fft_omp; }
hpcg()   { cd "$BENCHMARK_DIR/HPCG"   && ./HPCCG_BIN 256 256 128; }
ja()     { cd "$BENCHMARK_DIR/JA"     && ./omp_ja; }
lulesh() { cd "$BENCHMARK_DIR/LULESH" && ./lulesh2.0 -i 5000 -s 50; }
po()     { cd "$BENCHMARK_DIR/po"     && ./omp_po; }
st()     { cd "$BENCHMARK_DIR/ST"     && ./stream ;}
nas()    { cd "$BENCHMARK_DIR/NAS"    && "./$1.B.x"; }

parboil_bfs() { cd "$BENCHMARK_DIR/parboil" && ./parboil run bfs omp_base SF ;}
parboil_lbm() { cd "$BENCHMARK_DIR/parboil" && ./parboil run lbm omp_cpu long ; }

rodinia_hotspot() { cd "$BENCHMARK_DIR/RODINIA/hotspot" && ./hotspot 1024 1024 100000 "$N_THREADS" ../data/hotspot/temp_1024 ../data/hotspot/power_1024 output.out ;}

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
	if [ "${1:-}" == '' ]; then
		log_error "First argument must be the name of a benchmark"
	fi
	if [ "${2:-}" == '' ]; then
		log_error "Second argument must be the number of threads"
	elif ! [[ "$2" =~ ^[+-]?[0-9]+$ ]]; then
		log_error "The number of threads must be an integer"
	fi

	find_exec "$1"
	case "$1" in
		FFT | HPCG | JA | LULESH)
			EXEC="${!1}"
			NAME="$1"
			;;
		NONE)
			EXEC="echo \"Running None\""
			NAME="$1"
			;;
		*)
			log_error "Unknown benchmark \"$1\""
	esac
	
	NT=$2


}


set_env() {
	BENCHMARK_DIR="$PROJECT_DIR/hpc-benchmarks"

	EXEC=""
	NAME=""
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
set_env
_setConfigArgs "$@"
main "$@"
