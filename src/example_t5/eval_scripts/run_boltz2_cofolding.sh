#!/usr/bin/env bash
set -euo pipefail

# Minimal local equivalent of boltz2_cofold_inference_best.sh.
# It prepares Boltz input YAMLs from an inference run, executes Boltz for each
# target in the manifest, then collects cofolding results.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

RUN_DIR="${1:?Usage: $0 <inference_run_dir>}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"

: "${BOLTZ_BIN:=boltz}"
: "${BOLTZ_CACHE:?Set BOLTZ_CACHE to the Boltz model cache directory}"

FOLDINGS_SUBDIR="${FOLDINGS_SUBDIR:-foldings_840}"
COFOLDING_SUBDIR="${COFOLDING_SUBDIR:-cofolding_results}"
CF_ROOT="${RUN_DIR}/${COFOLDING_SUBDIR}"
MANIFEST="${CF_ROOT}/manifest.json"
TOP_K_OVERRIDE="${TOP_K_OVERRIDE:-}"
TARGET_INDICES="${TARGET_INDICES:-}"
EXTRA_BOLTZ_ARGS="${EXTRA_BOLTZ_ARGS:-}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  PREP_ARGS=(--run_dir "${RUN_DIR}" --out_dir "${CF_ROOT}" --foldings_subdir "${FOLDINGS_SUBDIR}")
  if [[ -n "${FOLDINGS_DIR:-}" ]]; then
    PREP_ARGS+=(--foldings_dir "${FOLDINGS_DIR}")
  fi
  if [[ -n "${TOP_K_OVERRIDE}" ]]; then
    PREP_ARGS+=(--top_k "${TOP_K_OVERRIDE}")
  fi
  if [[ -n "${TARGET_INDICES}" ]]; then
    PREP_ARGS+=(--indices "${TARGET_INDICES}")
  fi
  python "${ROOT_DIR}/eval/cofolding/boltz_inference_cofold_prepare.py" "${PREP_ARGS[@]}"
else
  echo "[info] SKIP_PREPARE=1; using existing ${MANIFEST}"
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 1
fi

mapfile -t STEMS < <(python -c "import json,sys; m=json.load(open(sys.argv[1])); [print(t['stem']) for t in m.get('targets', [])]" "${MANIFEST}")
mapfile -t YAMLS < <(python -c "import json,sys; m=json.load(open(sys.argv[1])); [print(t['yaml_path']) for t in m.get('targets', [])]" "${MANIFEST}")

FAILURES=0
echo "Targets to process: ${#STEMS[@]}"

for ((i=0; i<${#STEMS[@]}; i++)); do
  stem="${STEMS[$i]}"
  yaml_path="${YAMLS[$i]}"
  [[ -z "${stem}" ]] && continue

  out_run="${CF_ROOT}/runs/${stem}"
  boltz_out="${out_run}/boltz_output"

  if compgen -G "${boltz_out}/predictions/*/confidence_*.json" >/dev/null 2>&1 \
    || compgen -G "${boltz_out}/boltz_results_*/predictions/*/confidence_*.json" >/dev/null 2>&1; then
    echo "[skip] ${stem} already has Boltz confidence output."
    continue
  fi

  mkdir -p "${out_run}"
  echo "Running Boltz for ${stem}"
  if ! "${BOLTZ_BIN}" predict "${yaml_path}" \
      --cache "${BOLTZ_CACHE}" \
      --out_dir "${boltz_out}" \
      --output_format pdb \
      --use_potentials \
      --override ${EXTRA_BOLTZ_ARGS}; then
    echo "[error] Boltz failed for ${stem}; continuing."
    ((FAILURES++)) || true
  fi
done

echo "Boltz loop done. Failures: ${FAILURES}/${#STEMS[@]}"

if [[ "${SKIP_COLLECT}" != "1" ]]; then
  python "${ROOT_DIR}/eval/cofolding/collect_boltz_cofold_inference_results.py" \
    --run_dir "${RUN_DIR}" \
    --cofolding_subdir "${COFOLDING_SUBDIR}"
else
  echo "[info] SKIP_COLLECT=1; skipping result collection."
fi

echo "Done. Results under ${CF_ROOT}"
