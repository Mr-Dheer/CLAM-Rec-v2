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

run_variant () {   # $1=variant $2=fusion $3=clip_variant(optional, default bigG)
  local V=$1 F=$2 CV=${3:-bigG}
  local EXTRA=""
  [ "$CV" != "bigG" ] && EXTRA="--clip_variant $CV"
  echo "=== TRAIN $V/$F/$CV ==="
  $PY scripts/train.py --config $CFG --variant $V --fusion $F $EXTRA --stage 1
  $PY scripts/train.py --config $CFG --variant $V --fusion $F $EXTRA --stage 2
  echo "=== INFER $V/$F/$CV (10 seeds) ==="
  for s in $SEEDS; do
    $PY scripts/infer.py --config $CFG --variant $V --fusion $F $EXTRA --seed $s
  done
}

# --- RQ1 + mechanism contrast (fusion fixed = concat, bigG CLIP) ---
run_variant text        concat
run_variant clip_align  concat
run_variant clip_inject concat

# --- RQ2 fusion ablation (best mechanism = clip_inject, bigG) ---
run_variant clip_inject mean
run_variant clip_inject gating

# --- RQ3 domain fine-tuning (ViT-L/14 zero-shot vs fine-tuned, clip_inject/concat) ---
# Run dirs: results/clip_inject_concat_vitl14_zeroshot and ..._vitl14_ft
# (drivers append the non-bigG clip_variant to the run name -> no collision with
# the bigG clip_inject_concat run).
run_variant clip_inject concat vitl14_zeroshot
run_variant clip_inject concat vitl14_ft

echo "=== REPORT: RQ1 + mechanism (bigG concat) ==="
$PY -m clam_rec.eval.report --runs_dir results --baseline text_concat \
  --variants text_concat clip_align_concat clip_inject_concat --ks 1 5 10 --fuzzy 0.90
echo "=== REPORT: RQ2 fusion (clip_inject, bigG) ==="
$PY -m clam_rec.eval.report --runs_dir results --baseline clip_inject_concat \
  --variants clip_inject_concat clip_inject_mean clip_inject_gating --ks 1 --fuzzy 0.90 || true
echo "=== REPORT: RQ3 fine-tuning (ViT-L zero-shot vs fine-tuned) ==="
$PY -m clam_rec.eval.report --runs_dir results --baseline clip_inject_concat_vitl14_zeroshot \
  --variants clip_inject_concat_vitl14_zeroshot clip_inject_concat_vitl14_ft --ks 1 5 10 --fuzzy 0.90 || true
