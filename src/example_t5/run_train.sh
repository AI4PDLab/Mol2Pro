#!/usr/bin/env bash
set -euo pipefail

# Tokenize (Hub-first by default) then train. Intended for Git Bash / WSL / Linux.
#
# Required env:
#   TOKENIZER_AA, TOKENIZER_MOL, MODEL_NAME_OR_PATH
#
# Optional env:
#   DATASET             Hub dataset id (default: contributor-anonymous/Mol2Pro-Binder-Dataset)
#   LOCAL_DATASET_PATH  If set, passed as --dataset instead of DATASET (local JSON file or split dir)
#   TOKENIZED_DIR       Output of tokenize_dataset.py / input layout for train.py (default: ./example_t5/tokenized)
#   FORCE_TOKENIZE      If 1, always rerun tokenize_dataset.py
#   HF_REVISION         Passed as --revision to tokenize_dataset.py when loading from the Hub
#   TRUST_REMOTE_CODE   If 1, adds --trust_remote_code for Hub datasets
#   OUTPUT_DIR          Training checkpoints (default: ./example_t5/training_output)
#   TRAIN_EXTRA_ARGS    Extra args appended to train.py (quoted string)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET="${DATASET:-contributor-anonymous/Mol2Pro-Binder-Dataset}"
TOKENIZED_DIR="${TOKENIZED_DIR:-${SCRIPT_DIR}/tokenized}"
FORCE_TOKENIZE="${FORCE_TOKENIZE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/training_output}"

: "${TOKENIZER_AA:?Set TOKENIZER_AA to the amino-acid tokenizer directory}"
: "${TOKENIZER_MOL:?Set TOKENIZER_MOL to the molecule tokenizer directory}"
: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH (base checkpoint, e.g. google/t5-small)}"

if [[ -n "${LOCAL_DATASET_PATH:-}" ]]; then
  DATASET_ARG="${LOCAL_DATASET_PATH}"
else
  DATASET_ARG="${DATASET}"
fi

RUN_TOKENIZE=0
if [[ "${FORCE_TOKENIZE}" == "1" ]]; then
  RUN_TOKENIZE=1
elif [[ ! -f "${TOKENIZED_DIR}/dataset_dict.json" ]]; then
  RUN_TOKENIZE=1
fi

if [[ "${RUN_TOKENIZE}" == "1" ]]; then
  mkdir -p "${TOKENIZED_DIR}"
  TOKENIZE_CMD=(
    python "${SCRIPT_DIR}/tokenize_dataset.py"
    --dataset "${DATASET_ARG}"
    --tokenizer_aa "${TOKENIZER_AA}"
    --tokenizer_mol "${TOKENIZER_MOL}"
    --output_dir "${TOKENIZED_DIR}"
  )
  if [[ -n "${HF_REVISION:-}" ]]; then
    TOKENIZE_CMD+=(--revision "${HF_REVISION}")
  fi
  if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
    TOKENIZE_CMD+=(--trust_remote_code)
  fi
  "${TOKENIZE_CMD[@]}"
else
  echo "[run_train] Skipping tokenize; found ${TOKENIZED_DIR}/dataset_dict.json (set FORCE_TOKENIZE=1 to rebuild)."
fi

TRAIN_CMD=(
  python "${SCRIPT_DIR}/train.py"
  --tokenized_dir "${TOKENIZED_DIR}"
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --output_dir "${OUTPUT_DIR}"
)

if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  TRAIN_CMD+=(${TRAIN_EXTRA_ARGS})
fi

"${TRAIN_CMD[@]}"
