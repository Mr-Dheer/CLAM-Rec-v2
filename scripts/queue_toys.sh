#!/usr/bin/env bash
# Autonomously finish the Toys ladder. Toys clip_inject is already running (GPU 2);
# this waits for Prime Pantry to finish (frees GPUs 0/1/3), then launches the
# remaining Toys models: text (GPU 0) + clip_align (GPU 3). Self-healing driver
# (OOM -> bs4) + namespaced outputs. Runs detached; safe to leave overnight.
cd "$(dirname "$0")/.."
echo "[queue_toys] waiting for Prime Pantry to finish (frees GPUs 0/1/3) $(date)"
while true; do
  n=0
  for f in r_pp_text r_pp_align r_pp_inject; do
    grep -q "DONE" "logs/$f.log" 2>/dev/null && n=$((n+1))
  done
  [ "$n" -ge 3 ] && break
  sleep 300
done
echo "[queue_toys] Prime Pantry complete -> launching Toys text (GPU0) + clip_align (GPU3) $(date)"
setsid nohup bash scripts/run_train_infer.sh configs/toys_and_games_sub.yaml text       -               0 0 > logs/r_toys_text.log  2>&1 &
setsid nohup bash scripts/run_train_infer.sh configs/toys_and_games_sub.yaml clip_align vitl14_zeroshot 3 0 > logs/r_toys_align.log 2>&1 &
echo "[queue_toys] launched Toys text + clip_align $(date)"
