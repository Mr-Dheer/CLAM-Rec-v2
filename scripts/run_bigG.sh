#!/usr/bin/env bash
# clip_inject + bigG (fresh zero-shot LAION bigG), GPU 2, 1 seed, sliced cold/warm.
set -e
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/luxury_beauty_rq3.yaml   # batch_size2=4, batch_size_infer=16 (safe)
# NOTE: no --clip_variant => defaults to bigG (uses clip_{text,image}_Luxury_Beauty.npy, 1280-dim)
echo "## bigG STAGE1 $(date)"; python -u scripts/train.py --config $CFG --variant clip_inject --fusion concat --stage 1
echo "## bigG STAGE2 $(date)"; python -u scripts/train.py --config $CFG --variant clip_inject --fusion concat --stage 2
echo "## bigG INFER seed 0 $(date)"; python -u scripts/infer.py --config $CFG --variant clip_inject --fusion concat --seed 0
echo "## bigG REPORT $(date)"; python -u -m clam_rec.eval.report --runs_dir results --variants clip_inject_concat --ks 1 5 10 --fuzzy 0.90
echo "## bigG DONE $(date)"
