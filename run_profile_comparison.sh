#!/usr/bin/env bash
# Run FSDP and DDP Wikitext-2 profiles and write plot-friendly artifacts under profile_runs/<timestamp>/.
#
# Usage:
#   export NPROC=4                    # GPUs per node (1, 2, 4, …)
#   export MODEL=gpt2               # optional HF model id
#   export EXTRA_ARGS="--seq-len 512 --steps 30"   # optional, passed to the profiler
#   ./run_profile_comparison.sh
#
# Outputs (in the printed directory):
#   metrics.jsonl   — one JSON object per strategy (append if you reuse --plot-jsonl path)
#   metrics.csv     — same rows flattened for Excel / R / ggplot
#   console.log     — full stdout/stderr from both runs
#
# Plotting (Python):
#   import pandas as pd
#   df = pd.read_json("profile_runs/<stamp>/metrics.jsonl", lines=True)
#   df.plot.bar(x="strategy", y=["train_tokens_per_s", "infer_tokens_per_s"])

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PROFILE="${ROOT}/profile_llm_distributed.py"
if [[ ! -f "$PROFILE" ]]; then
  echo "error: missing $PROFILE" >&2
  exit 1
fi

NPROC="${NPROC:-1}"
MODEL="${MODEL:-gpt2}"
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
OUT="${ROOT}/profile_runs/${STAMP}"
mkdir -p "$OUT"

METRICS_JSONL="${OUT}/metrics.jsonl"
METRICS_CSV="${OUT}/metrics.csv"
CONSOLE_LOG="${OUT}/console.log"

: >"$CONSOLE_LOG"
: >"$METRICS_JSONL"

echo "[run_profile_comparison] OUT_DIR=$OUT NPROC=$NPROC MODEL=$MODEL" | tee -a "$CONSOLE_LOG"
echo "[run_profile_comparison] EXTRA_ARGS=${EXTRA_ARGS:-}" | tee -a "$CONSOLE_LOG"

run_one() {
  local strategy="$1"
  echo "" | tee -a "$CONSOLE_LOG"
  echo "========== strategy=${strategy} ==========" | tee -a "$CONSOLE_LOG"
  # shellcheck disable=SC2086
  torchrun --nproc_per_node="${NPROC}" "$PROFILE" \
    --strategy "${strategy}" \
    --model "${MODEL}" \
    --bf16 \
    --mode both \
    --plot-jsonl "$METRICS_JSONL" \
    ${EXTRA_ARGS:-} \
    2>&1 | tee -a "$CONSOLE_LOG"
}

run_one fsdp
run_one ddp

python3 - "$METRICS_JSONL" "$METRICS_CSV" <<'PY'
import csv, json, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
lines = [ln.strip() for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
if not lines:
    print("warning: no metrics rows; skipping CSV", file=sys.stderr)
    sys.exit(0)
rows = [json.loads(ln) for ln in lines]
keys = sorted(set().union(*(r.keys() for r in rows)))
with dst.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in keys})
print(f"[run_profile_comparison] wrote {dst} ({len(rows)} rows)")
PY

echo "" | tee -a "$CONSOLE_LOG"
echo "[run_profile_comparison] done. Plot from:" | tee -a "$CONSOLE_LOG"
echo "  $METRICS_JSONL" | tee -a "$CONSOLE_LOG"
echo "  $METRICS_CSV" | tee -a "$CONSOLE_LOG"
