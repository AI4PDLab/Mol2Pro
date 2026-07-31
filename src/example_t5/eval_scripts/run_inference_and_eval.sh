#!/usr/bin/env bash
set -euo pipefail

# Minimal local equivalent of full_eval_MN5_base_EXTENDED_array_SAIR_outside.sh.
# It runs inference, GT MMseqs, train-identity MMseqs, the GT/train join, and
# optional pLDDT folding. Paths can be supplied as environment variables.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${MODEL_PATH:?Set MODEL_PATH to the trained model/checkpoint directory}"
: "${TOKENIZER_AA:?Set TOKENIZER_AA to the amino-acid tokenizer directory}"
: "${TOKENIZER_MOL:?Set TOKENIZER_MOL to the molecule tokenizer directory}"
: "${INFERENCE_INPUT:?Set INFERENCE_INPUT to the test JSON/JSONL/TXT input file}"
: "${TEST_DATASET:?Set TEST_DATASET to the test dataset JSON used for GT evaluation}"
: "${TRAIN_DATASET:?Set TRAIN_DATASET to the train dataset JSON used for novelty/train identity}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/example_t5/outputs/$(basename "${INFERENCE_INPUT%.*}")}"
GT_FASTA_DIR="${GT_FASTA_DIR:-${OUTPUT_DIR}/gt_fasta}"
MMSEQS_BIN_DIR="${MMSEQS_BIN_DIR:-}"
TMP_ROOT="${TMP_ROOT:-${OUTPUT_DIR}/tmp}"
RUN_PLDDT="${RUN_PLDDT:-0}"
ESMFOLD_PATH="${ESMFOLD_PATH:-}"
MAX_FOLD_LEN="${MAX_FOLD_LEN:-840}"

mkdir -p "${OUTPUT_DIR}" "${TMP_ROOT}"

echo "[1/5] Running inference into ${OUTPUT_DIR}"
python "${ROOT_DIR}/example_t5/inference_t5.py" \
  --input_file "${INFERENCE_INPUT}" \
  --model_path "${MODEL_PATH}" \
  --output_folder "${OUTPUT_DIR}" \
  --tokenizer_mol "${TOKENIZER_MOL}" \
  --tokenizer_aa "${TOKENIZER_AA}"

echo "[2/5] Running GT MMseqs evaluation"
MMSEQS_ARGS=()
if [[ -n "${MMSEQS_BIN_DIR}" ]]; then
  MMSEQS_ARGS+=(--mmseq_path "${MMSEQS_BIN_DIR}")
fi

python "${ROOT_DIR}/eval/gt_acc/gt_inference_mmseq.py" \
  --inference_folder "${OUTPUT_DIR}" \
  --dataset_json "${TEST_DATASET}" \
  --output_folder_gt "${GT_FASTA_DIR}" \
  --cache_path "${TMP_ROOT}" \
  "${MMSEQS_ARGS[@]}"

python "${ROOT_DIR}/eval/gt_acc/gt_mmseq_interpretation_balanced.py" \
  --mmseqs_path "${OUTPUT_DIR}/gt_mmseqs" \
  --inference_folder "${OUTPUT_DIR}" \
  --gt_fasta_folder "${GT_FASTA_DIR}"

echo "[3/5] Running train-identity MMseqs evaluation"
bash "${ROOT_DIR}/example_t5/eval_scripts/run_train_identity_mmseqs.sh" \
  "${OUTPUT_DIR}" \
  "${TRAIN_DATASET}"

echo "[4/5] Joining GT and train-identity results"
bash "${ROOT_DIR}/example_t5/eval_scripts/join_mmseqs_results.sh" \
  "${OUTPUT_DIR}" \
  "${TRAIN_DATASET}"

if [[ "${RUN_PLDDT}" == "1" ]]; then
  : "${ESMFOLD_PATH:?Set ESMFOLD_PATH or disable RUN_PLDDT}"
  echo "[5/5] Running pLDDT folding"
  python "${ROOT_DIR}/eval/plddt/run-esm.py" \
    --fasta_path "${OUTPUT_DIR}" \
    --output_path "${OUTPUT_DIR}/foldings_${MAX_FOLD_LEN}" \
    --max_len "${MAX_FOLD_LEN}" \
    --esmfold_path "${ESMFOLD_PATH}"

  python "${ROOT_DIR}/eval/plddt/plot_plddt.py" \
    --results_folder "${OUTPUT_DIR}/foldings_${MAX_FOLD_LEN}"
else
  echo "[5/5] Skipping pLDDT folding. Set RUN_PLDDT=1 to enable it."
fi

echo "Done. Results are under ${OUTPUT_DIR}"
