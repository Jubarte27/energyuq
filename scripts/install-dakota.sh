#!/bin/env bash
set -o pipefail
main() {
    find "$BASE_DIR" -iname "*.log" -type f -delete
    set_log_depth 0
    ensure install_deps
    ensure install_python
    ensure create_venv
    ensure install_dakota

    ensure cleanup
}

cleanup() {
    enter_new_func "Cleaning up"
    if ! [ "$KEEP_AFTER_INSTALL" = "true" ]; then
        rm -rf "$DAK_BUILD" "$DAK_VENV"
        if [ "$RELEASE" = "true" ]; then
            rm -rf "$DAK_SRC"
        fi
    fi
}

install_dakota() {
    enter_new_func "Installing Dakota"

    if [ "$RELEASE" == "true" ]; then
        ensure get_release
    else
        ensure get_dev
    fi
    ensure configure_dakota
    ensure build_dakota
    # ensure test_dakota
}

configure_dakota() {
    enter_new_func "Configuring Dakota"

    ensure cd "$BASE_DIR"

    local extras=()
    if ! [ "$RELEASE" == "true" ]; then
      extras+=("-DENABLE_SPEC_MAINT=ON")
    fi

    mkdir -p "$DAK_BUILD"
    ensure cd "$DAK_BUILD"
    # ensure cmake -C "$BASE_DIR/BuildDakotaTemplate.cmake" -DCMAKE_INSTALL_PREFIX="$DAK_INSTALL" $DAK_SRC
    
    #   -DTrilinos_DIR="/usr/lib/x86_64-linux-gnu/cmake/Trilinos" \
    make_me > >(tee "$BASE_DIR/configure.log") 2> >(tee "$BASE_DIR/configureerr.log" >&2)
}

make_me() {
    if ! cmake --fresh -DCMAKE_INSTALL_PREFIX="$DAK_INSTALL" \
      -DCMAKE_Fortran_COMPILER="gfortran" \
      -DCMAKE_C_COMPILER="gcc" \
      -DCMAKE_C_COMPILER_LAUNCHER="ccache" \
      -DCMAKE_CXX_COMPILER="g++" \
      -DCMAKE_CXX_COMPILER_LAUNCHER="ccache" \
      -DMPI_CXX_COMPILER="mpicxx" \
      -DCMAKE_Fortran_FLAGS="-fallow-argument-mismatch" \
      -DDAKOTA_HAVE_MPI=ON \
      -DDAKOTA_HAVE_HDF5=ON \
      -DHAVE_MUQ=ON \
      -DDAKOTA_JAVA_SURROGATES=ON \
      -DDAKOTA_PYTHON=ON \
      -DDAKOTA_PYTHON_DIRECT_INTERFACE=ON \
      -DDAKOTA_PYTHON_DIRECT_INTERFACE_NUMPY=ON \
      -DDAKOTA_PYTHON_SURROGATES=ON \
      -DDAKOTA_PYTHON_WRAPPER=ON \
      -DDAKOTA_HAVE_GSL=ON \
      -DHAVE_QUESO=ON \
      "${extras[@]}" \
      "$DAK_SRC"; then
        log "$ERROR" "Configuration failed, check $BASE_DIR/configureerr.log for details"
        exit 1
    fi
}

test_dakota() {
    enter_new_func "Testing Dakota"
    ensure cd "$BASE_DIR"

    ensure cd "$DAK_BUILD"
    ctest -j "$(nproc --ignore=2)"
}

build_dakota() {
    enter_new_func "Building Dakota"
    ensure cd "$BASE_DIR"
    
    ensure cd "$DAK_BUILD"
    a > >(tee "$BASE_DIR/build.log") 2> >(tee "$BASE_DIR/builderr.log" >&2)
    b > >(tee "$BASE_DIR/install.log") 2> >(tee "$BASE_DIR/installerr.log" >&2)
}

b() {
    if ! make -j "$(nproc --ignore=2)" --no-keep-going install; then
        log "$ERROR" "Installation failed, check $BASE_DIR/installerr.log for details"
        exit 1
    fi
}

a() {
    if ! make -j "$(nproc --ignore=2)" --no-keep-going ; then
        log "$ERROR" "Build failed, check $BASE_DIR/builderr.log for details"
        exit 1
    fi
}

get_release() {
    enter_new_func "Downloading release source files"
    ensure cd "$BASE_DIR"

    wget https://github.com/snl-dakota/dakota/releases/download/v6.24.0/$DAK_VERSION.tar.gz -O "$BASE_DIR/dak.tar.gz"
    tar -xzf "$BASE_DIR/dak.tar.gz" -C "$BASE_DIR"
    rm "$BASE_DIR/dak.tar.gz"
}

get_dev() {
    enter_new_func "Downloading git source files"
    ensure cd "$BASE_DIR"

    ensure cd "$PROJECT_DIR"
    git submodule update --remote "$DAK_SRC"
    ensure cd "$DAK_SRC"
    # git submodule update packages/external packages/pecos packages/surfpack
    # a feiura abaixo funciona, enquanto a belezura acima não
    # sem acesso aos submodules, copiar da release
    ensure get_release
    cp -r --update=all $BASE_DIR/$DAK_VERSION/packages/** $DAK_SRC/packages/
    rm -fr $BASE_DIR/$DAK_VERSION
}

create_venv() {
    enter_new_func "Creating python venv"
    
    if [ ! -f "$DAK_VENV/bin/activate" ]; then
        ensure cd "$BASE_DIR"
        python3 -m venv "$DAK_VENV"
    fi
    
    # shellcheck disable=SC1091
    source "$DAK_VENV/bin/activate"
    pip install --upgrade pip
    pip install sphinx myst-parser sphinx-rtd-theme sphinxcontrib-bibtex h5py scipy numpy
}

install_python() {
    ensure cd "$BASE_DIR"
    enter_new_func "Installing python"
    if ! command -v pyenv &> /dev/null; then
        if ! command -v python3 &> /dev/null; then
            # never tested?
            apt install python3
        fi
    else
        pyenv install --skip-existing 3.14
        pyenv local 3.14
    fi

    # do i need this?
    if ! python3 -m pip &> /dev/null; then
        python3 -m ensurepip --upgrade
    fi
}

install_deps() {
    enter_new_func "Installing dependencies"

    log "$INFO" "Hoping they are here already"
    # apt install gcc g++ gfortran cmake ccache libboost-all-dev libblas-dev liblapack-dev libopenmpi-dev openmpi-bin gsl-bin libgsl-dev perl libhdf5-dev doxygen texlive-latex-base openjdk-11-jre-headless libatlas-base-dev
}

_setConfigArgs() {
    while [ "${1:-}" != '' ]; do
        case "$1" in
            ## Options
            -k|--keep-after-install)
                KEEP_AFTER_INSTALL=true
                ;;
            
            -r|--release)
                RELEASE=true
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

    BASE_DIR="${1:-"$PROJECT_DIR/dakota"}"
    DEV_DIR="$BASE_DIR/dev"
    DAK_VERSION=dakota-6.24.0-public-src-cli

    if [ "$RELEASE" == "true" ]; then
        DAK_SRC="$BASE_DIR/$DAK_VERSION"
    else
        DAK_SRC="$BASE_DIR/snl-dakota"
        BASE_DIR="$DEV_DIR"
    fi

    DAK_INSTALL="$BASE_DIR/install"
    DAK_BUILD="$BASE_DIR/build"
    DAK_VENV="$BASE_DIR/.venv"


}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
main "$@"