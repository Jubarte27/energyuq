#!/bin/env bash

main() {
    set_log_depth 0
    ensure install_deps
    ensure install_python
    ensure create_venv
    ensure install_dakota

    ensure cleanup
}

cleanup() {
    enter_new_func "Cleaning up"
    if ! [ "$KEEP_AFTER_INSTALL" = true ]; then
        rm -rf "$DAK_SRC" "$DAK_BUILD" "$BASE_DIR/.dakota-venv"
    fi
}

install_dakota() {
    enter_new_func "Installing Dakota"

    ensure get_source
    ensure configure_dakota
    ensure build_dakota
    # ensure test_dakota
}

configure_dakota() {
    enter_new_func "Configuring Dakota"

    #   -DMPI_CXX_COMPILER="$MPICXX" \
    #   -DBLAS_LIBS="$BLAS_LIB" \
    #   -DLAPACK_LIBS="$LAPACK_LIB" \
    mkdir "$DAK_BUILD"
    cd "$DAK_BUILD" || exit
    cmake -DCMAKE_INSTALL_PREFIX="$DAK_INSTALL" \
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
      "$DAK_SRC"
}

test_dakota() {
    enter_new_func "Testing Dakota"

    cd "$DAK_BUILD" || exit
    ctest -j "$(nproc --ignore=2)"
}

build_dakota() {
    enter_new_func "Building Dakota"
    
    cd "$DAK_BUILD" || exit
    make -j "$(nproc --ignore=2)"
    make -j "$(nproc --ignore=2)" install
}

get_source() {
    enter_new_func "Downloading source files"

    wget https://github.com/snl-dakota/dakota/releases/download/v6.23.0/$DAK_VERSION.tar.gz -O dak.tar.gz
    tar -xzvf dak.tar.gz
    rm dak.tar.gz
}

create_venv() {
    enter_new_func "Creating python venv"
    
    if [ ! -f "$BASE_DIR/.dakota-venv/bin/activate" ]; then
        python3 -m venv "$BASE_DIR/.dakota-venv"
    fi
    
    # shellcheck disable=SC1091
    source "$BASE_DIR/.dakota-venv/bin/activate"

    pip install --upgrade pip
    pip install sphinx myst-parser sphinx-rtd-theme sphinxcontrib-bibtex h5py
}

install_python() {
    enter_new_func "Installing python"

    if ! command -v python3 &> /dev/null; then
        # do i need this?
        if ! command -v pyenv &> /dev/null; then
            apt install python3
        else
            pyenv install 3.10
            pyenv local 3.10
        fi
    fi

    # do i need this?
    if ! python3 -m pip &> /dev/null; then
        python3 -m ensurepip --upgrade
    fi
}

install_deps() {
    enter_new_func "Installing dependencies"

    log "$INFO" "Hoping they are here already"
    # apt install gcc g++ gfortran cmake libboost-all-dev libblas-dev liblapack-dev libopenmpi-dev openmpi-bin gsl-bin libgsl-dev perl libhdf5-dev doxygen texlive-latex-base openjdk-11-jre-headless libatlas-base-dev
}

_setConfigArgs() {
    while [ "${1:-}" != '' ]; do
        case "$1" in
            ## Options
            -k|--keep-after-install)
                KEEP_AFTER_INSTALL=true
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

    BASE_DIR="${1:-$PROJECT_DIR}"

    DAK_VERSION=dakota-6.23.0-public-src-cli
    DAK_SRC="$BASE_DIR/$DAK_VERSION"
    DAK_INSTALL="$BASE_DIR/dakota"
    DAK_BUILD="$BASE_DIR/dak-build"


    # MPICXX=$(find /usr/ -iname mpicxx)
    # BLAS_LIB=$(find /usr/ -iwholename "*atlas/libblas.so")
    # LAPACK_LIB=$(find /usr/ -iwholename "*atlas/liblapack.so")
}

SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
_setConfigArgs "$@"
main "$@"