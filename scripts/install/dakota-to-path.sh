#!/bin/bash
# todo: melhorar isso aqui, tá ruim de usar, não é a prova de idiota

HERE=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")

__add_dakota_to_path() {
    local INSTALL_DIR=$1
    export PATH=$INSTALL_DIR/bin:$INSTALL_DIR/share/dakota/test:$INSTALL_DIR/bin:$PATH
    export PYTHONPATH=$PYTHONPATH:$INSTALL_DIR/share/dakota/Python
}

__dakota_path() {
    local HERE
    HERE=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")
    
    readlink -e "$HERE/../dakota/dev/install"
}

__add_dakota_to_path "$(__dakota_path)"
unset -f __add_dakota_to_path
unset -f __dakota_to_path
