#!/usr/bin/env bash
# A-LLMRec TEXT baseline (SBERT, no vision) on GPU 0. 1 seed. Sliced cold/warm.
set -e
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=configs/luxury_beauty_rq3.yaml   # has safe batch_size2=4, batch_size_infer=16

echo "## TEXT baseline STAGE1 $(date)"
python -u scripts/train.py --config $CFG --variant text --fusion concat --stage 1
echo "## TEXT baseline STAGE2 $(date)"
python -u scripts/train.py --config $CFG --variant text --fusion concat --stage 2
echo "## TEXT baseline INFER seed 0 $(date)"
python -u scripts/infer.py --config $CFG --variant text --fusion concat --seed 0
echo "## TEXT baseline REPORT $(date)"
python -u -m clam_rec.eval.report --runs_dir results --variants text_concat --ks 1 5 10 --fuzzy 0.90
echo "## TEXT baseline DONE $(date)"
