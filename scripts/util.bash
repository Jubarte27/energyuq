#!/bin/env bash
# shellcheck disable=SC2317
join_by() {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

not() {
    if [ "$1" == true ]; then
        echo false
    else
        echo true
    fi
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")
# shellcheck disable=SC2034
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../" && pwd)"
source "$SCRIPT_DIR/log.bash"