#!/bin/env bash

### Log ###
### base styling for log messages NAME;COLORCODE
WARN="WARN;33"
INFO="INFO;32"
ERROR="ERROR;31"
log() {
    local level="$1"
    local message="$2"
    local log_depth=${3:-$LOG_DEPTH}
    TABS=""
    for _ in $(seq 1 "$log_depth"); do
        TABS="$TABS\t"
    done

    IFS=";" read -r LEVEL_NAME LEVEL_COLOR <<< "$level"

    echo -e "\e[${LEVEL_COLOR}m$(date '+%Y-%m-%d %H:%M:%S') - $LEVEL_NAME:\e[0m$TABS $message"
}

log_warn() {
    log $WARN "$@" 
}

log_info() {
    log $INFO "$@" 
}

log_error() {
    log $ERROR "$@" 
    exit 1
}

set_log_depth() {
    LOG_DEPTH=${1:-$LOG_DEPTH}
    NEXT_LOG_DEPTH=$((LOG_DEPTH + 1))
}

enter_new_func() {
    set_log_depth "$2"
    log $INFO "$1"
}

ensure() {
    local NEXT_DEPTH=$NEXT_LOG_DEPTH
    local DEPTH=$LOG_DEPTH
    local ERR_MSG;

    if ! [[ -t 0 || -p /dev/stdin ]]; then
        # stdin is redirected from a file or here-document/string
        ERR_MSG="$(cat)"
    fi

    set_log_depth $NEXT_LOG_DEPTH

    if "$@"; then
        :;
    else
        ERR_MSG=${ERR_MSG:-"command \"$*\" failed with exit status $?"}
        log_error "$ERR_MSG" "$NEXT_DEPTH"
    fi
    set_log_depth "$DEPTH"
}

set_log_depth 0
