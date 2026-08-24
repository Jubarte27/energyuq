#!/bin/bash
SCRIPT_DIR=$(dirname "$(readlink -e "${BASH_SOURCE[0]}")") && source "$SCRIPT_DIR/util.bash"
cat <<EOF > .env
PYTHONPATH=$PROJECT_DIR:\$PYTHONPATH
EOF