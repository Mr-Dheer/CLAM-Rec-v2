#!/usr/bin/env bash
# CLIP-text-only ablation. Train clip_inject with fusion=text_only (inject the
# CLIP-TEXT embedding only, NO image) + RANK-infer. Decomposes the CLIP injection:
#   text (SBERT)  vs  clip_inject text_only (CLIP-text)  vs  clip_inject concat (CLIP-text+image)
# -> isolates whether any effect is the image or just CLIP-text. Run name:
#   {dataset}_clip_inject_text_only_vitl14_zeroshot
set -e
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec
CFG=$1; GPU=$2
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V=clip_inject; CV=vitl14_zeroshot; F=text_only
echo "## $CFG $F STAGE1 $(date)"
python -u scripts/train.py --config $CFG --variant $V --fusion $F --clip_variant $CV --stage 1
echo "## $CFG $F STAGE2 $(date)"
set +e
python -u scripts/train.py --config $CFG --variant $V --fusion $F --clip_variant $CV --stage 2
RC=$?
set -e
[ $RC -ne 0 ] && python -u scripts/train.py --config $CFG --variant $V --fusion $F --clip_variant $CV --stage 2 --batch_size2 4
echo "## $CFG $F RANK-INFER $(date)"
python -u scripts/infer.py --config $CFG --variant $V --fusion $F --clip_variant $CV --seed 0 --rank --rank_chunk 20
echo "## $CFG $F DONE $(date)"
