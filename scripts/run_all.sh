#!/usr/bin/env bash
# Full experiment matrix for the "When Does Vision Help?" paper.
# Run on the A6000 (>=24GB) for OPT-6.7B. Uses the ALLM-Rec conda env.
#
# Design:
#   - Mechanism contrast: variants = text (baseline), clip_align, clip_inject   (fusion=concat)
#   - RQ2 fusion ablation: variant=clip_inject, fusion in {concat, mean, gating}
#   - 10 seeds each for inference (training is seed-agnostic; Stage1/2 trained once per variant/fusion)
set -e
cd "$(dirname "$0")/.."
PY="conda run -n ALLM-Rec python"
CFG=configs/luxury_beauty.yaml
SEEDS="0 1 2 3 4 5 6 7 8 9"

run_variant () {   # $1=variant $2=fusion
  local V=$1 F=$2
  echo "=== TRAIN $V/$F ==="
  $PY scripts/train.py --config $CFG --variant $V --fusion $F --stage 1
  $PY scripts/train.py --config $CFG --variant $V --fusion $F --stage 2
  echo "=== INFER $V/$F (10 seeds) ==="
  for s in $SEEDS; do
    $PY scripts/infer.py --config $CFG --variant $V --fusion $F --seed $s
  done
}

# Mechanism contrast (fusion fixed = concat)
run_variant text        concat
run_variant clip_align  concat
run_variant clip_inject concat

# RQ2 fusion ablation (best mechanism = clip_inject)
run_variant clip_inject mean
run_variant clip_inject gating

echo "=== REPORT ==="
$PY -m clam_rec.eval.report --runs_dir results --baseline text \
  --variants text clip_align clip_inject --ks 1 5 10 --fuzzy 0.90
echo "=== FUSION REPORT ==="
$PY -m clam_rec.eval.report --runs_dir results --baseline clip_inject_concat \
  --variants clip_inject_concat clip_inject_mean clip_inject_gating --ks 1 --fuzzy 0.90 || true
