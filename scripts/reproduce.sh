#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root="$PWD"
python="${RADIOHEAD_PYTHON:-$root/.venv/bin/python}"
cd upstream/radiohead/code/lab

case "${1:-test}" in
  test) "$python" tests.py ;;
  recall|smoke)
    extra=()
    if [[ "$1" == smoke ]]; then extra+=(--smoke); fi
    for model in KM SM; do
      RADIOHEAD_RUNS="$root/runs" "$python" -c '
import os
import train

train.RES = os.environ["RADIOHEAD_RUNS"]
train.main()
' --task mqar --model "$model" \
        --d_model 64 --heads 1 --layers 2 --lr 1e-3 --epochs 80 --seed 0 ${extra[@]+"${extra[@]}"}
    done
    ;;
  *) echo "Usage: scripts/reproduce.sh [test|smoke|recall]" >&2; exit 2 ;;
esac
