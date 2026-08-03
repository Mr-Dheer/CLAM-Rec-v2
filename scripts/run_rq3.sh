#!/usr/bin/env bash
# RQ3 only: clip_inject with vitl14_ft AND vitl14_zeroshot (fair same-model pair).
# Pinned to GPU 2. Run detached: setsid bash scripts/run_rq3.sh &> logs/rq3.log
set -e
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/luxury_beauty.yaml
SEEDS="0 1 2 3 4 5 6 7 8 9"

run () {  # $1 = clip_variant
  local CV=$1
  echo "############ RQ3 $CV : STAGE1 ############ $(date)"
  python -u scripts/train.py --config $CFG --variant clip_inject --fusion concat --clip_variant $CV --stage 1
  echo "############ RQ3 $CV : STAGE2 ############ $(date)"
  python -u scripts/train.py --config $CFG --variant clip_inject --fusion concat --clip_variant $CV --stage 2
  echo "############ RQ3 $CV : INFER 10 seeds ############ $(date)"
  for s in $SEEDS; do
    echo "---- $CV seed $s ---- $(date)"
    python -u scripts/infer.py --config $CFG --variant clip_inject --fusion concat --clip_variant $CV --seed $s
  done
}

run vitl14_ft
run vitl14_zeroshot

echo "############ RQ3 REPORT ############ $(date)"
python -m clam_rec.eval.report --runs_dir results --baseline clip_inject_concat_vitl14_zeroshot \
  --variants clip_inject_concat_vitl14_zeroshot clip_inject_concat_vitl14_ft --ks 1 5 10 --fuzzy 0.90
echo "############ RQ3 DONE ############ $(date)"
