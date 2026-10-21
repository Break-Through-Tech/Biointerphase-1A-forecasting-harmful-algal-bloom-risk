#!/usr/bin/env bash
# Rebuild the weekly table and run every model at 1, 2 and 4 weeks ahead, then the
# bootstrap against persistence and the comparison table. Each step is timed:
# results/run_times.csv has one row per step, and results/run_times.md is a readable
# summary with the machine and package versions the times were measured on.
# Usage, from the models/ folder: bash run_all.sh
set -e
cd "$(dirname "$0")"
mkdir -p results
TIMES=results/run_times.csv
echo "horizon_weeks,step,seconds" > "$TIMES"

# Run a command and append how long it took to the times file.
timed() {
  local step=$1; shift
  local start=$(date +%s)
  "$@"
  echo "$h,$step,$(( $(date +%s) - start ))" >> "$TIMES"
}

for h in 1 2 4; do
  export HAB_HORIZON=$h
  timed weekly_table python -c "from common.weekly import build_weekly; build_weekly(rebuild=True)"
  for m in persistence seasonal_climatology logistic_regression random_forest xgboost hybrid_escalation; do
    timed $m python $m/train.py
  done
  timed bootstrap python experiments/bootstrap_vs_persistence.py
done
python compare.py
python write_run_times.py
