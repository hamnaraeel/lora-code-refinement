#!/usr/bin/env bash
# Full evaluation: base vs fine-tuned on the benchmark and (optionally) test,
# plus LLM-as-judge and the catastrophic-forgetting probes, then the report.
#
# Usage:
#   bash scripts/run_evaluation.sh artifacts/runs/<run>/adapter [BASE_MODEL]
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:?usage: run_evaluation.sh <adapter-dir> [base-model]}"
BASE_MODEL="${2:-mistralai/Mistral-7B-Instruct-v0.3}"
FOURBIT="${FOURBIT:---load-in-4bit}"

echo "== 1/6  Base model on the benchmark =="
coderefine evaluate --split benchmark --base-model "$BASE_MODEL" --tag base $FOURBIT

echo "== 2/6  Fine-tuned model on the benchmark =="
coderefine evaluate --split benchmark --base-model "$BASE_MODEL" --adapter "$ADAPTER" --tag tuned $FOURBIT

echo "== 3/6  Head-to-head comparison =="
coderefine compare \
  artifacts/eval/base__benchmark.predictions.jsonl \
  artifacts/eval/tuned__benchmark.predictions.jsonl

echo "== 4/6  LLM-as-judge (skipped without an API key) =="
if [ -n "${ANTHROPIC_API_KEY:-}${OPENAI_API_KEY:-}" ]; then
  PROVIDER=$([ -n "${ANTHROPIC_API_KEY:-}" ] && echo anthropic || echo openai)
  coderefine judge \
    artifacts/eval/base__benchmark.predictions.jsonl \
    artifacts/eval/tuned__benchmark.predictions.jsonl \
    --provider "$PROVIDER"
else
  echo "   no ANTHROPIC_API_KEY or OPENAI_API_KEY set; skipping."
fi

echo "== 5/6  Catastrophic forgetting probes =="
coderefine forgetting --base-model "$BASE_MODEL" $FOURBIT
coderefine forgetting --base-model "$BASE_MODEL" --adapter "$ADAPTER" $FOURBIT

echo "== 6/6  Experiment report =="
coderefine report

echo
echo "Done. reports/EXPERIMENT_REPORT.md"
echo
echo "The test split is still untouched. When the configuration is frozen, score it once:"
echo "  coderefine evaluate --split test --final --tag base  --base-model $BASE_MODEL $FOURBIT"
echo "  coderefine evaluate --split test --final --tag tuned --adapter $ADAPTER --base-model $BASE_MODEL $FOURBIT"
