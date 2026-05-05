#!/bin/bash
__add_dakota_to_path() {
    local INSTALL_DIR="/home/mmachado/HPC/energyuq/dakota"
    export PATH=$INSTALL_DIR/bin:$INSTALL_DIR/share/dakota/test:$INSTALL_DIR/gui:$PATH
    export PYTHONPATH=$PYTHONPATH:$INSTALL_DIR/share/dakota/Python
}
__add_dakota_to_path
unset -f __add_dakota_to_path
