#!/usr/bin/env bash
# clip_align + bigG (the RQ1 "bottleneck" variant: CLIP is a Stage-1 alignment
# target only; vision does NOT reach the LLM at inference). GPU 2, 1 seed,
# sliced cold/warm. This is the missing MIDDLE of the mechanism contrast
# (text -> clip_align -> clip_inject); see PROJECT.md §4.4.
set -e
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/luxury_beauty_rq3.yaml   # batch_size2=4, batch_size_infer=16 (safe)
# NOTE: no --clip_variant => defaults to bigG (clip_{text,image}_Luxury_Beauty.npy, 1280-dim).
echo "## clip_align STAGE1 $(date)"; python -u scripts/train.py --config $CFG --variant clip_align --fusion concat --stage 1
echo "## clip_align STAGE2 $(date)"; python -u scripts/train.py --config $CFG --variant clip_align --fusion concat --stage 2
echo "## clip_align INFER seed 0 $(date)"; python -u scripts/infer.py --config $CFG --variant clip_align --fusion concat --seed 0
echo "## clip_align REPORT $(date)"; python -u -m clam_rec.eval.report --runs_dir results --baseline text_concat \
  --variants text_concat clip_align_concat clip_inject_concat --ks 1 5 10 --fuzzy 0.90
echo "## clip_align DONE $(date)"
