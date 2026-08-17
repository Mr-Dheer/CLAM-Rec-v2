#!/usr/bin/env bash
# clip_align + ViT-L/14 zero-shot (the HEADLINE mechanism middle: the "bottleneck"
# variant on the chosen single backbone). GPU 2, 1 seed, sliced cold/warm.
# Completes the ViT-L mechanism triple: text -> clip_align -> clip_inject.
set -e
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/luxury_beauty_rq3.yaml   # batch_size2=6, batch_size_infer=16
CV=vitl14_zeroshot
echo "## clip_align ViT-L STAGE1 $(date)"; python -u scripts/train.py --config $CFG --variant clip_align --fusion concat --clip_variant $CV --stage 1
echo "## clip_align ViT-L STAGE2 $(date)"; python -u scripts/train.py --config $CFG --variant clip_align --fusion concat --clip_variant $CV --stage 2
echo "## clip_align ViT-L INFER seed 0 $(date)"; python -u scripts/infer.py --config $CFG --variant clip_align --fusion concat --clip_variant $CV --seed 0
echo "## clip_align ViT-L REPORT (mechanism triple) $(date)"; python -u -m clam_rec.eval.report --runs_dir results --baseline text_concat \
  --variants text_concat clip_align_concat_vitl14_zeroshot clip_inject_concat_vitl14_zeroshot --ks 1 5 10 --fuzzy 0.90
echo "## clip_align ViT-L DONE $(date)"
