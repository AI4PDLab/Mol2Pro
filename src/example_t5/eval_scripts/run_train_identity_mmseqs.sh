#!/usr/bin/env bash
set -euo pipefail

# Minimal local equivalent of novelty_mmseqs_array_MN5.sh.
# Usage:
#   bash run_train_identity_mmseqs.sh <inference_folder> <train_dataset_json>

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INFERENCE_FOLDER="${1:?Usage: $0 <inference_folder> <train_dataset_json>}"
TRAIN_DATASET_JSON="${2:?Usage: $0 <inference_folder> <train_dataset_json>}"

MMSEQS_BIN_DIR="${MMSEQS_BIN_DIR:-}"
TMP_ROOT="${TMP_ROOT:-${INFERENCE_FOLDER}/tmp}"
TRAIN_FASTA="${TRAIN_DATASET_JSON%.json}.fasta"
QUERY_FASTA="${INFERENCE_FOLDER}/mmseqs/collated.fasta"
INPUT_METADATA_FILE="${INFERENCE_FOLDER}/molecule_gt-input_metadata.json"

mkdir -p "${TMP_ROOT}"
TMP_FOLDER="$(mktemp -d -p "${TMP_ROOT}" "mmseqs_train_identity_XXXX")"
trap 'rm -rf "${TMP_FOLDER}"' EXIT

python "${ROOT_DIR}/eval/train_identity/mmseq_file_preparation.py" \
  --fasta_folder "${INFERENCE_FOLDER}" \
  --json_file "${TRAIN_DATASET_JSON}"

MMSEQS_BIN="${MMSEQS_BIN_DIR:+${MMSEQS_BIN_DIR%/}/}mmseqs"
"${MMSEQS_BIN}" easy-search \
  "${QUERY_FASTA}" \
  "${TRAIN_FASTA}" \
  "${INFERENCE_FOLDER}/mmseqs/mmseqs_output" \
  "${TMP_FOLDER}" \
  --cov-mode 0 -c 0.3 --format-output query,target,fident,alnlen,qlen,tlen

python "${ROOT_DIR}/eval/train_identity/mmseq_interpretation.py" \
  --mmseqs_path "${INFERENCE_FOLDER}/mmseqs" \
  --query_fasta "${QUERY_FASTA}" \
  --training_correspondence "${TRAIN_FASTA%.fasta}_sequence_correspondence.json" \
  --input_metadata_file "${INPUT_METADATA_FILE}"
