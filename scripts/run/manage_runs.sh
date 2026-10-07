#!/bin/bash
set -e

readonly JOB_NAME_PREFIX="energyuq"
readonly LEDGER_SUBDIR=".ledger"

declare -A L_STATUS L_JOBID L_SUBMITTED L_UPDATED L_NOTE
FAILED_JOBS=()
INTERRUPTS=0
SUBMITTING=true

main() {
    set_log_depth 0

    enter_new_func "Managing $TYPE runs of ${BENCHMARKS[*]:-none} on ${NODES[*]:-none}"

    if [ "$LIST" == "true" ]; then
        ledger_load
        print_matrix
        return 0
    fi

    ledger_load
    report_done_pairs

    if [ -n "$COPY_FROM" ]; then
        scan_copy_from "$COPY_FROM" true
    elif [ -n "$CHECK_COPY_FROM" ]; then
        scan_copy_from "$CHECK_COPY_FROM" false
        print_matrix
        return 0
    fi

    supervise
    finalize
}

report_done_pairs() {
    local bench node target

    for bench in "${BENCHMARKS[@]}"; do
        for node in "${NODES[@]}"; do
            if pair_done "$bench" "$node"; then
                target=$(runs_path "$bench" "$node")
                log_info "[$node] $bench already executed in $target, ignoring the pair"
            fi
        done
    done
}

supervise() {
    while true; do
        reconcile

        if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
            # Not log_error: the jobs still running have to be waited for.
            # Halt submissions only for the failed machine(s); others continue.
            log "$ERROR" "Job(s) failed: ${FAILED_JOBS[*]}. Submission halted for failed machine(s) only." 0
        fi

        if [ "$SUBMITTING" == "true" ]; then
            submit_ready
        fi

        ledger_save
        print_matrix

        if [ "$(in_flight)" -eq 0 ]; then
            if [ "$SUBMITTING" == "false" ] || [ -z "$(pending_pairs)" ]; then
                return 0
            fi
        fi

        nap
    done
}

submit_ready() {
    local partition pair bench node

    for partition in "${PARTITIONS[@]}"; do
        if [ "$(in_flight "$partition")" -lt 1 ]; then
            if ! pair=$(next_pair "$partition"); then
                continue
            fi
            IFS='/' read -r bench node <<< "$pair"
            submit_pair "$bench" "$node"
        fi
    done
}

submit_pair() {
    local bench="$1"
    local node="$2"
    local pair="$bench/$node"
    local job_name="$JOB_NAME_PREFIX-$bench"
    local env_file
    env_file=$(node_env "$node")

    EXTRA_ARGS=()

    if ! pair_done "$bench" "$node"; then
        EXTRA_ARGS=(--dest "$(runs_path "$bench" "$node")")
    fi
    
    local output
    if ! output=$(SBATCH_OPTS="--job-name=$job_name" "$PROJECT_DIR/scripts/run/run_sbatch_pcad.sh" --env "$env_file" run "${EXTRA_ARGS[@]}" "${RUN_ARGS[@]}" "$bench" 2>&1); then
        ledger_set "$pair" FAILED "" "submission failed: $output"
        FAILED_JOBS+=("$pair (not submitted: $output)")
        return 0
    fi

    local jobid
    jobid=$(printf '%s\n' "$output" | grep -o 'Submitted batch job [0-9]*' | awk '{print $4}' | head -1)
    if [ -z "$jobid" ]; then
        ledger_set "$pair" FAILED "" "no job id in sbatch output: $output"
        FAILED_JOBS+=("$pair (no job id)")
        return 0
    fi

    ledger_set "$pair" PENDING "$jobid" ""
    log_info "[$node] $bench submitted as job $jobid"
}

reconcile() {
    local pair jobid state note

    FAILED_JOBS=()

    for pair in "${!L_STATUS[@]}"; do
        case "${L_STATUS[$pair]}" in
            PD|PENDING | R|RUNNING | HALTED) ;;
            *) continue ;;
        esac
        jobid="${L_JOBID[$pair]}"
        state=$(slurm_state "$jobid")
        note=""
        case "$state" in
            PD|PENDING) ledger_set "$pair" PENDING "$jobid" "" ;;
            R|RUNNING) ledger_set "$pair" RUNNING "$jobid" "" ;;
            CD|COMPLETED|DONE) ledger_set "$pair" DONE "$jobid" "" ;;
            F|FAILED)
                ledger_set "$pair" FAILED "$jobid" "$state"
                FAILED_JOBS+=("$pair (job $jobid: $state)")
                report_job_tail "$pair" "$jobid"
                ;;
            CA|CANCELLED)
                ledger_set "$pair" FAILED "$jobid" "job $jobid cancelled"
                FAILED_JOBS+=("$pair (job $jobid cancelled)")
                ;;
            TIMEOUT)
                ledger_set "$pair" FAILED "$jobid" "job $jobid timeout"
                FAILED_JOBS+=("$pair (job $jobid timeout)")
                ;;
            *) note="unexpected state \"$state\"" ;;
        esac
        if [ -n "$note" ]; then
            log "$WARN" "$pair: $note" 1
        fi
    done

    return 0
}

in_flight() {
    local partition="${1:-}"
    if [ -n "$partition" ]; then partition=$(node_partition "$partition"); fi
    local pair count=0
    for pair in "${!L_STATUS[@]}"; do
        if [ -n "$partition" ] && [ "$(node_partition "${pair#*/}")" != "$partition" ]; then
            continue
        fi
        case "${L_STATUS[$pair]}" in
            PD|PENDING|R|RUNNING|HALTED) count=$((count + 1)) ;;
        esac
    done
    printf '%s' "$count"
}

node_blocked() {
    local node="$1"
    local pair
    for pair in "${!L_STATUS[@]}"; do
        if [ "${pair#*/}" == "$node" ] && [ "${L_STATUS[$pair]}" == "FAILED" ]; then
            return 0
        fi
    done
    return 1
}

next_pair() {
    local partition="$1"
    local bench node pair

    for bench in "${BENCHMARKS[@]}"; do
        for node in "${NODES[@]}"; do
            if node_blocked "$node"; then
                continue
            fi
            if [ "$(node_partition "$node")" != "$partition" ]; then
                continue
            fi
            pair="$bench/$node"
            if pair_done "$bench" "$node"; then
                continue
            fi
            if [ -n "${L_STATUS[$pair]+set}" ]; then
                continue
            fi
            printf '%s' "$pair"
            return 0
        done
    done
    return 1
}

pending_pairs() {
    local partition pair
    for partition in "${PARTITIONS[@]}"; do
        if pair=$(next_pair "$partition"); then
            printf '%s\n' "$pair"
        fi
    done
}

pair_done() {
    local dir
    dir="$(runs_path "$1" "$2")"
    [ -d "$dir" ] && [ -n "$(ls -A "$dir")" ]
}

runs_path() {
    printf '%s/%s/%s/%s' "$RUNS_DIR" "$TYPE" "$1" "$2"
}

node_partition() {
    local node="$1"
    node="${node%%[*}"
    printf '%s' "${node%%[0-9]*}"
}

node_env() {
    printf '%s/slurm_nodes/%s/%s' "$PROJECT_DIR" "$(node_partition "$1")" "$1"
}

slurm_state() {
    local jobid="$1"
    local state

    state=$(squeue -h -j "$jobid" -o '%i %t' | awk 'NR == 1 {print $2}')
    if [ -z "$state" ]; then
        # scontrol prints JobState on a line of its own, not on the first one.
        state=$(scontrol show job "$jobid" |
            grep -m1 -o 'JobState=[A-Z_]*' | cut -d= -f2)
    fi
    if [ -z "$state" ]; then
        state=$(sacct -X -j "$jobid" --format=State --noheader |
            awk 'NR == 1 {print $1}')
        state="${state%%[.+]*}"
    fi

    case "$state" in
        COMPLETED) printf 'DONE' ;;
        FAILED | TIMEOUT | NODE_FAIL | OUT_OF_MEMORY | CANCELLED | PREEMPTED | BOOT_FAIL | DEADLINE)
            printf '%s' "$state"
            ;;
        PENDING | CONFIGURING | REQUEUED | REQUEUE_HOLD | RESIZING) printf 'PENDING' ;;
        RUNNING | COMPLETING | STAGE_OUT | SIGNALING | SUSPENDED | STOPPED | REVOKED) printf 'RUNNING' ;;
        "") printf 'UNKNOWN' ;;
        *) printf '%s' "$state" ;;
    esac
}

report_job_tail() {
    local pair="$1"
    local jobid="$2"
    local log="$PROJECT_DIR/execlog/$JOB_NAME_PREFIX-${pair%%/*}-${pair#*/}_$jobid.out"

    log_warn "$pair: tail of $log"
    if [ -f "$log" ]; then
        tail -n 5 "$log" || true
    else
        log_warn "$log not found"
    fi
}

scan_copy_from() {
    local dir="$1"
    local copying="$2"
    local checkpoint run relative host node bench note
    local -a report=()
    local managed

    if [ ! -d "$dir" ]; then
        log_error "Copy from directory '$dir' not found"
    fi

    mapfile -t checkpoints < <(find "$dir" -maxdepth 4 -name checkpoint.json -type f | sort)

    if [ ${#checkpoints[@]} -eq 0 ]; then
        log_info "No runs found in '$dir'"
        return 0
    fi

    log_info "Runs found in '$dir':"
    printf '  %-22s %-12s %-24s %-20s %-10s %s\n' BENCHMARK NODE RUN DATE COMPLETED CONVERGED

    for checkpoint in "${checkpoints[@]}"; do
        run=$(dirname "$checkpoint")
        relative=${run#"$dir"/}
        host=${relative%%/*}
        node=$(node_partition "$host")
        if ! bench=$(run_benchmark "$run"); then
            log_warn "$relative: unknown benchmark, ignoring"
            continue
        fi

        managed=true
        if ! contains "$bench" "${BENCHMARKS[@]}" || ! contains "$node" "${NODES[@]}"; then
            managed=false
            note=" (not in the requested list)"
        else
            note=""
        fi

        # Collected so that the copying does not get in the middle of the table.
        report+=("$(printf '  %-22s %-12s %-24s %-20s %-10s %s%s' \
            "$bench" "$node" "$relative" "$(run_date "$run")" \
            "$(run_completed "$run")" "$(run_converged "$run")" "$note")")

        if [ "$copying" == "true" ] && [ "$managed" == "true" ]; then
            promote "$run" "$bench" "$node"
        fi
    done

    printf '%s\n' "${report[@]}"
}

run_benchmark() {
    local run="$1"
    local first bench

    shopt -s nullglob
    local outputs=("$run"/output/*)
    shopt -u nullglob

    if [ ${#outputs[@]} -eq 0 ]; then
        return 1
    fi
    first=$(basename "${outputs[0]}")

    for bench in "${BENCHMARKS[@]}"; do
        if [ "$first" == "$bench" ] || [[ "$first" == "$bench"_* ]]; then
            printf '%s' "$bench"
            return 0
        fi
    done
    if [[ "$first" == *_* ]]; then
        printf '%s' "${first%_*}"
        return 0
    fi
    return 1
}

run_date() {
    local run="$1"
    local stamp
    stamp=$(json_field "$run/checkpoint.json" timestamp)
    if [ -z "$stamp" ]; then
        stamp=$(date -r "$run" '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || true)
    fi
    printf '%s' "${stamp%%.*}"
}

run_completed() {
    local status
    status=$(json_field "$1/checkpoint.json" status)
    case "$status" in
        completed | converged) printf 'yes' ;;
        *) printf 'no (%s)' "${status:-unknown}" ;;
    esac
}

run_converged() {
    printf '%s' "$(json_field "$1/checkpoint.json" converged)"
}

json_field() {
    local file="$1"
    local key="$2"
    if [ ! -f "$file" ]; then
        return 0
    fi
    sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\"\?\([^\",}]*\)\"\?.*/\1/p" "$file" | head -1
}

promote() {
    local run="$1"
    local bench="$2"
    local node="$3"
    local target
    target=$(runs_path "$bench" "$node")

    if pair_done "$bench" "$node"; then
        log_info "$target already exists, nothing to copy"
        return 0
    fi

    mkdir -p "$(dirname "$target")"
    cp -r "$run" "$target"
    log_info "$run -> $target"
    ledger_set "$bench/$node" COPIED "" "copied from $run"
}

ledger_set() {
    local pair="$1"
    local status="$2"
    local jobid="${3:-}"
    local note="${4:-}"

    if [ -z "${L_SUBMITTED[$pair]+set}" ] && [ -n "$jobid" ]; then
        L_SUBMITTED[$pair]=$(now)
    fi
    L_STATUS[$pair]=$status
    L_JOBID[$pair]=$jobid
    L_UPDATED[$pair]=$(now)
    if [ -n "$note" ]; then
        L_NOTE[$pair]=$note
    fi
}

ledger_load() {
    local bench node status jobid submitted updated note

    if [ ! -f "$LEDGER" ]; then
        log_info "No ledger at '$LEDGER', starting one"
        return 0
    fi

    while IFS=$'\t' read -r bench node status jobid submitted updated note; do
        if [ "$bench" == "bench" ]; then
            continue
        fi
        local pair="$bench/$node"
        L_STATUS[$pair]=$status
        L_JOBID[$pair]=$jobid
        L_SUBMITTED[$pair]=$submitted
        L_UPDATED[$pair]=$updated
        L_NOTE[$pair]=$note
    done < "$LEDGER"

    log_info "Loaded ${#L_STATUS[@]} pair(s) from '$LEDGER'"
}

ledger_save() {
    local pair
    local tmp

    mkdir -p "$(dirname "$LEDGER")"
    tmp=$(mktemp "$LEDGER.XXXXXX")

    {
        printf 'bench\tnode\tstatus\tjobid\tsubmitted_at\tupdated_at\tnote\n'
        for pair in "${!L_STATUS[@]}"; do
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${pair%%/*}" "${pair#*/}" "${L_STATUS[$pair]}" "${L_JOBID[$pair]}" \
                "${L_SUBMITTED[$pair]}" "${L_UPDATED[$pair]}" "${L_NOTE[$pair]}"
        done | sort
    } > "$tmp"

    mv "$tmp" "$LEDGER"
}

print_matrix() {
    local bench node pair target

    if [ ${#BENCHMARKS[@]} -eq 0 ] || [ ${#NODES[@]} -eq 0 ]; then
        log_info "Nothing to report: no benchmark and node pair was given"
        return 0
    fi

    log_info "Results of $TYPE, jobs in flight: $(in_flight)"
    printf '%-22s' 'BENCHMARK'
    for node in "${NODES[@]}"; do
        printf '%-14s' "$node"
    done
    printf '\n'
    for bench in "${BENCHMARKS[@]}"; do
        printf '%-22s' "$bench"
        for node in "${NODES[@]}"; do
            pair="$bench/$node"
            if pair_done "$bench" "$node"; then
                target=$(runs_path "$bench" "$node")
                printf '%-14s' "done($(du -sh "$target" 2>/dev/null | cut -f1))"
            else
                case "${L_STATUS[$pair]:-}" in
                    PENDING) printf '%-14s' "pending" ;;
                    RUNNING) printf '%-14s' "running" ;;
                    HALTED) printf '%-14s' "halted" ;;
                    FAILED) printf '%-14s' "failed" ;;
                    COPIED) printf '%-14s' "copied" ;;
                    DONE) printf '%-14s' "finished" ;;
                    *) printf '%-14s' '-' ;;
                esac
            fi
        done
        printf '\n'
    done
}

# First Ctrl-C stops submitting and waits for the running jobs, second one leaves them going.
on_interrupt() {
    INTERRUPTS=$((INTERRUPTS + 1))
    ledger_save

    if [ "$INTERRUPTS" -eq 1 ]; then
        SUBMITTING=false
        log_warn "Submission halted. Waiting for $(in_flight) job(s); press Ctrl-C again to leave them running."
    else
        log_warn "Leaving now. $(in_flight) job(s) keep running, their ids are in '$LEDGER'."
        finalize
        exit 0
    fi
}

finalize() {
    local pair

    for pair in "${!L_STATUS[@]}"; do
        case "${L_STATUS[$pair]}" in
            PD|PENDING | R|RUNNING) L_STATUS[$pair]=HALTED ;;
        esac
    done
    ledger_save

    if [ "$(in_flight)" -gt 0 ]; then
        log_warn "$(in_flight) job(s) still running, recorded in '$LEDGER'. Run this command again to follow them."
    fi
    if [ ${#FAILED_JOBS[@]} -gt 0 ]; then
        log_warn "Failed pairs: ${FAILED_JOBS[*]}. Remove their lines from '$LEDGER' to try them again."
    fi
}

nap() {
    sleep "$POLL" &
    wait "$!" || true
}

now() {
    date '+%Y-%m-%dT%H:%M:%S'
}

# Comma separated list, @file with one name per line, or, for nodes.
expand_list() {
    local arg="$1"
    local file line

    if [ -z "$arg" ]; then
        return 0
    fi

    printf '%s\n' "${arg//,/$'\n'}"
}

abs_path() {
    local path="$1"
    if [ -z "$path" ]; then
        return 0
    fi
    if [ "${path:0:1}" != "/" ]; then
        path="$PROJECT_DIR/$path"
    fi
    printf '%s' "$path"
}

set_env() {
    RUNS_DIR="$PROJECT_DIR/runs"
    COPY_FROM=""
    CHECK_COPY_FROM=""
    BENCH_RAW=""
    NODE_RAW="all"
    POLL=15
    LIST=false
    RUN_ARGS=()
    BENCHMARKS=()
    NODES=()
    PARTITIONS=()
    FAILED_JOBS=()
}

_setConfigArgs() {
    while [ "${1:-}" != '' ]; do
        case "$1" in
            --type)
                TYPE="$2"
                shift
                ;;
            --benchmarks)
                BENCH_RAW="$2"
                shift
                ;;
            --nodes)
                NODE_RAW="$2"
                shift
                ;;
            --runs-dir)
                RUNS_DIR="$2"
                shift
                ;;
            --copy-from)
                COPY_FROM="$2"
                shift
                ;;
            --check-copy-from)
                CHECK_COPY_FROM="$2"
                shift
                ;;
            --poll)
                POLL="$2"
                shift
                ;;
            --list|-l)
                LIST=true
                ;;
            --)
                shift
                RUN_ARGS=("$@")
                break
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

    if [ -z "$TYPE" ]; then
        log_error "First argument must be the run type, use --type <TYPE>"
    fi

    if ! [[ "$POLL" =~ ^[0-9]+$ ]] || [ "$POLL" -lt 1 ]; then
        log_error "The polling interval must be a positive integer of seconds"
    fi

    RUNS_DIR=$(abs_path "$RUNS_DIR")
    COPY_FROM=$(abs_path "$COPY_FROM")
    CHECK_COPY_FROM=$(abs_path "$CHECK_COPY_FROM")

    mkdir -p "$RUNS_DIR"
    LEDGER="$RUNS_DIR/$LEDGER_SUBDIR/$TYPE.tsv"

    local line
    while read -r line; do
        BENCHMARKS+=("$line")
    done < <(expand_list "$BENCH_RAW")
    while read -r line; do
        NODES+=("$line")
    done < <(expand_list "$NODE_RAW")

    if [ ${#BENCHMARKS[@]} -eq 0 ] && [ -z "$COPY_FROM" ] && [ -z "$CHECK_COPY_FROM" ]; then
        log_error "No benchmark selected, use --benchmarks <list>"
    fi
    if [ ${#NODES[@]} -eq 0 ]; then
        log_error "No node selected, use --nodes <list>"
    fi

    local node partition
    for node in "${NODES[@]}"; do
        partition=$(node_partition "$node")
        if ! contains "$partition" "${PARTITIONS[@]}"; then
            PARTITIONS+=("$partition")
        fi
        if [ ! -e "$PROJECT_DIR/slurm_nodes/$partition" ]; then
            log_warn "No configuration for partition '$partition' in slurm_nodes"
        fi
    done
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

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
trap on_interrupt INT
set_env
_setConfigArgs "$@"
main "$@"