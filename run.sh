#!/usr/bin/env bash

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$DIR/venv/bin/python"

VENDOR_DIRS="$DIR/venv/lib/python3.11/site-packages/nvidia/cublas/lib:$DIR/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
if [ -e "$DIR/venv/lib/python3.11/site-packages/nvidia/cublas/lib/libcublas.so.12" ]; then
    export LD_LIBRARY_PATH="${VENDOR_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [ -d /usr/local/lib/ollama/cuda_v12 ]; then
    export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$PY" "$DIR/main.py" "$@"
