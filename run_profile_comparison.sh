#!/usr/bin/env bash
# Run FSDP then DDP; full logs under profile_runs/<timestamp>/console.log
#
#   export NPROC=4
#   export MODEL=gpt2
#   export EXTRA_ARGS="--seq-len 512 --steps 30"
#   ./run_profile_comparison.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PROFILE="${ROOT}/profile_llm_distributed.py"
[[ -f "$PROFILE" ]] || { echo "error: missing $PROFILE" >&2; exit 1; }

NPROC="${NPROC:-1}"
MODEL="${MODEL:-gpt2}"
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
OUT="${ROOT}/profile_runs/${STAMP}"
mkdir -p "$OUT"
LOG="${OUT}/console.log"
: >"$LOG"

echo "[run_profile_comparison] OUT_DIR=$OUT NPROC=$NPROC MODEL=$MODEL" | tee -a "$LOG"
echo "[run_profile_comparison] EXTRA_ARGS=${EXTRA_ARGS:-}" | tee -a "$LOG"

run_one() {
  local strategy="$1"
  echo "" | tee -a "$LOG"
  echo "========== strategy=${strategy} ==========" | tee -a "$LOG"
  # shellcheck disable=SC2086
  torchrun --nproc_per_node="${NPROC}" "$PROFILE" \
    --strategy "${strategy}" \
    --model "${MODEL}" \
    --bf16 \
    --mode both \
    ${EXTRA_ARGS:-} \
    2>&1 | tee -a "$LOG"
}

run_one fsdp
run_one ddp
echo "" | tee -a "$LOG"
echo "[run_profile_comparison] done: $LOG" | tee -a "$LOG"
