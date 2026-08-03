#!/usr/bin/env bash
# RQ3 resume: vitl14_ft already trained (checkpoints exist) -> just infer it.
# Then train+infer vitl14_zeroshot. Uses the separate RQ3 config (batch_infer=16).
# Skips any stage/seed whose output already exists (idempotent, safe to re-run).
set -e
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/luxury_beauty_rq3.yaml
SEEDS="0"

train_if_needed () {  # $1=clip_variant
  local CV=$1 CK=results/checkpoints/clip_inject_concat_$1
  if [ -f "$CK/mm_proj.pt" ]; then
    echo "## $CV: checkpoints exist, SKIP training $(date)"
  else
    echo "## $CV: STAGE1 $(date)"; python -u scripts/train.py --config $CFG --variant clip_inject --fusion concat --clip_variant $CV --stage 1
    echo "## $CV: STAGE2 $(date)"; python -u scripts/train.py --config $CFG --variant clip_inject --fusion concat --clip_variant $CV --stage 2
  fi
}

infer_all () {  # $1=clip_variant
  local CV=$1 OUT=results/clip_inject_concat_$1
  for s in $SEEDS; do
    if [ -f "$OUT/seed_$s.jsonl" ]; then echo "## $CV seed $s exists, skip"; continue; fi
    echo "## $CV INFER seed $s $(date)"
    python -u scripts/infer.py --config $CFG --variant clip_inject --fusion concat --clip_variant $CV --seed $s
  done
}

train_if_needed vitl14_ft
infer_all       vitl14_ft
train_if_needed vitl14_zeroshot
infer_all       vitl14_zeroshot

echo "## RQ3 REPORT $(date)"
python -u -m clam_rec.eval.report --runs_dir results --baseline clip_inject_concat_vitl14_zeroshot \
  --variants clip_inject_concat_vitl14_zeroshot clip_inject_concat_vitl14_ft --ks 1 5 10 --fuzzy 0.90
echo "## RQ3 DONE $(date)"
