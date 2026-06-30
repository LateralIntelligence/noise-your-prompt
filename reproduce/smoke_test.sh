#!/usr/bin/env bash
#
# smoke_test.sh — fast end-to-end sanity check of the train -> eval pipeline for
# BOTH main experiments, using tiny step counts and tiny datasets so the whole
# thing finishes in ~1-2 minutes on a single GPU. It does NOT reproduce paper
# numbers; it only proves the code path is intact (imports, data generation,
# training loop, checkpointing, and sampling/eval all run and write results).
#
# It drives the two real sweep scripts with tiny env overrides (every knob they
# expose is reused here), then asserts a non-empty results.json landed for each.
#
# Usage:
#   bash reproduce/smoke_test.sh
#
# wandb is forced offline so no account / API key is required.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE=offline

SUDOKU_ROOT="outputs/smoke-sudoku"
NQUEENS_ROOT="outputs/smoke-nqueens"

echo "==================== smoke: Sudoku infill ===================="
MAX_STEPS=20 VAL_CHECK_INTERVAL=10 CKPT_EVERY=10 \
  SUBSET_SIZES=100 NUM_TRAIN=100 NUM_VALID=20 DIFFICULTY=easy \
  GLOBAL_BATCH=16 SAMPLING_STEPS=8 CONDITIONING_PROBS_CLEAN=1.0 \
  SWEEP_ROOT="${SUDOKU_ROOT}" \
  ./reproduce/sweep_infill_sudoku_gen.sh

echo "==================== smoke: N-Queens infill ===================="
MAX_STEPS=20 VAL_CHECK_INTERVAL=10 CKPT_EVERY=10 \
  NUM_TRAIN=200 NUM_VALID=50 GLOBAL_BATCH=32 SAMPLING_STEPS=8 \
  NUM_PUZZLES=10 NUM_SAMPLES=4 CONDITIONING_PROBS_CLEAN=1.0 \
  SWEEP_ROOT="${NQUEENS_ROOT}" \
  ./reproduce/sweep_infill_nqueens.sh

echo "==================== verifying results ===================="
sudoku_hits=$(find "${SUDOKU_ROOT}" -path '*/sudoku_eval/*/results.json' 2>/dev/null | wc -l)
nqueens_hits=$(find "${NQUEENS_ROOT}" -path '*/nqueens_eval/*/results.json' 2>/dev/null | wc -l)
echo "sudoku results.json: ${sudoku_hits}"
echo "nqueens results.json: ${nqueens_hits}"

if [[ "${sudoku_hits}" -lt 1 || "${nqueens_hits}" -lt 1 ]]; then
  echo "SMOKE TEST FAILED: missing results.json (see logs above)"
  exit 1
fi
echo "== smoke OK =="
