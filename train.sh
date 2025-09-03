#!/usr/bin/env bash

set -euo pipefail

# Resolve this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional: initialize pyenv if available; otherwise skip silently
if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init -)"
  if command -v pyenv-virtualenv-init >/dev/null 2>&1; then
    eval "$(pyenv virtualenv-init -)" || true
  fi
fi

# Dataset tag (override by exporting TAG in environment if desired)
TAG="${TAG:-aou_2021_2022}"

# Prefer train split if present; otherwise fall back to root processed dir
BASE_DIR="${SCRIPT_DIR}/processed_data_${TAG}"
if [[ -f "${BASE_DIR}/train/tokenized_timelines.pkl" ]]; then
  DATA_DIR="${BASE_DIR}/train"
elif [[ -f "${BASE_DIR}/tokenized_timelines.pkl" ]]; then
  DATA_DIR="${BASE_DIR}"
else
  echo "❌ Processed data not found for tag '${TAG}'." >&2
  echo "Expected at: '${BASE_DIR}/train/tokenized_timelines.pkl' or '${BASE_DIR}/tokenized_timelines.pkl'" >&2
  echo "Run data processing first, e.g.:" >&2
  echo "  python ${SCRIPT_DIR}/data_processor.py --data_path /path/to/omop_data --tag ${TAG} --split_by_person --train_frac 0.7 --split_seed 42" >&2
  exit 1
fi

python "${SCRIPT_DIR}/train.py" \
  --tag "${TAG}" \
  --data_dir "${DATA_DIR}"
