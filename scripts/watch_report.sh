#!/usr/bin/env bash
# Wait until every listed run has produced results/<run>/seed_0.jsonl, then print
# the sliced Hit@1 report. Used as a background babysitter so results are collected
# and surfaced automatically when training+inference finish.
#
# Usage: watch_report.sh <label> <run1> <run2> ...
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec
LABEL=$1; shift; RUNS="$@"
echo "[watch:$LABEL] waiting for COMPLETION of: $RUNS  ($(date))"
# IMPORTANT: infer.py creates seed_0.jsonl at the START and fills it incrementally,
# so file existence != complete. Wait until every file's line count is STABLE across
# two consecutive polls (and non-zero) -> inference has finished writing.
declare -A prev
while true; do
  done_all=1
  for r in $RUNS; do
    f="results/$r/seed_0.jsonl"
    if [ ! -f "$f" ]; then done_all=0; prev[$r]=-1; continue; fi
    c=$(wc -l < "$f")
    if [ "$c" -le 0 ] || [ "${prev[$r]:-(-1)}" != "$c" ]; then done_all=0; fi
    prev[$r]=$c
  done
  [ "$done_all" -eq 1 ] && break
  sleep 120
done
echo "[watch:$LABEL] ALL COMPLETE $(date)"
echo "======================= $LABEL RESULTS ======================="
python -m clam_rec.eval.report --runs_dir results --variants $RUNS --ks 1 --fuzzy 0.90
echo "[watch:$LABEL] done $(date)"
