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
PROJECT_DIR="$(cd "$SCRIPT_DIR/../" && pwd)"
export PATH="$PROJECT_DIR/.bin:$PATH"
source "$SCRIPT_DIR/log.bash"

install_uv() {
    enter_new_func "Installing uv"

    local uv_dir="$PROJECT_DIR/.bin"
    if [ ! -x "$uv_dir/uv" ]; then
        mkdir -p "$uv_dir"
        if command -v uv >/dev/null 2>&1; then
            cp "$(command -v uv)" "$uv_dir/uv"
            if command -v uvx >/dev/null 2>&1; then
                cp "$(command -v uvx)" "$uv_dir/uvx"
            fi
        elif command -v curl >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$uv_dir" UV_NO_MODIFY_PATH=1 sh
        elif command -v wget >/dev/null 2>&1; then
            wget -qO- https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$uv_dir" UV_NO_MODIFY_PATH=1 sh
        else
            log_error "Neither curl nor wget is available to install uv"
            return 1
        fi
    fi
    export PATH="$uv_dir:$PATH"
}

ensure_uv() {
    if command -v uv > /dev/null; then
        return
    fi
    install_uv
}