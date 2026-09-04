#!/usr/bin/env bash
# Run the full hyperparameter sweep and print a comparison table.
#
# Every arm is selected on VALIDATION loss. The test split is not touched here
# and must not be — see the note at the end of this script.
#
# Usage:
#   bash scripts/run_sweep.sh                       # all arms
#   bash scripts/run_sweep.sh configs/sweep/s0*.yaml # a subset
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(configs/sweep/*.yaml)
fi

echo "Sweep: ${#CONFIGS[@]} arm(s)"
echo

FAILED=()
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  echo "=============================================================="
  echo "  $name"
  echo "=============================================================="
  if ! coderefine train "$cfg"; then
    echo "!! arm failed: $name"
    FAILED+=("$name")
  fi
  echo
done

echo "=============================================================="
echo "  Sweep summary (validation loss)"
echo "=============================================================="
python - <<'PY'
import json, pathlib
rows = []
for path in sorted(pathlib.Path("artifacts/runs").glob("*/train_summary.json")):
    s = json.loads(path.read_text())
    cfg = s.get("config", {})
    rows.append((
        s.get("best_eval_loss") if s.get("best_eval_loss") is not None else float("inf"),
        s.get("run_name", "?"),
        cfg.get("lora", {}).get("r"),
        cfg.get("train", {}).get("learning_rate"),
        cfg.get("train", {}).get("num_epochs"),
        ",".join(cfg.get("lora", {}).get("target_modules", [])),
        s.get("best_checkpoint_step"),
        s.get("train_runtime_s", 0) / 60,
        s.get("gpu_peak_memory_gb"),
        s.get("adapter_size_mb"),
    ))
rows.sort()
hdr = f"{'run':30s} {'r':>3} {'lr':>8} {'ep':>3} {'targets':14s} {'eval_loss':>10} {'step':>6} {'min':>6} {'GB':>5} {'MB':>7}"
print(hdr); print("-" * len(hdr))
for loss, name, r, lr, ep, tgt, step, mins, gb, mb in rows:
    ls = f"{loss:10.4f}" if loss != float("inf") else "         -"
    print(f"{name:30s} {r if r else '-':>3} {lr if lr else '-':>8} {ep if ep else '-':>3} "
          f"{tgt[:14]:14s} {ls} {str(step):>6} {mins:6.1f} {str(gb):>5} {str(mb):>7}")
if rows:
    print()
    print(f"Best on validation loss: {rows[0][1]}")
    print("Freeze this configuration before touching the test split.")
PY

if [ ${#FAILED[@]} -gt 0 ]; then
  echo
  echo "Arms that failed: ${FAILED[*]}"
  exit 1
fi
