#!/usr/bin/env bash
# Generic driver: train Stage 1 + Stage 2, then infer one seed, for one
# variant on one dataset/GPU. Reusable across datasets to avoid script sprawl.
#
# Usage: run_train_infer.sh <config> <variant> <clip_variant|-> <gpu> [seed]
#   e.g. run_train_infer.sh configs/all_beauty.yaml text          - 0 0
#        run_train_infer.sh configs/all_beauty.yaml clip_inject vitl14_zeroshot 1 0
set -e
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec
CFG=$1; V=$2; CV=$3; GPU=$4; SEED=${5:-0}; FUSION=${6:-concat}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXTRA=""; { [ "$CV" != "-" ] && [ -n "$CV" ]; } && EXTRA="--clip_variant $CV"
echo "## $V/${CV} on $CFG (GPU $GPU) STAGE1 $(date)"
python -u scripts/train.py --config $CFG --variant $V --fusion $FUSION $EXTRA --stage 1
echo "## $V/${CV} STAGE2 $(date)"
set +e
python -u scripts/train.py --config $CFG --variant $V --fusion $FUSION $EXTRA --stage 2
RC=$?
set -e
if [ $RC -ne 0 ]; then
  echo "## $V/${CV} STAGE2 FAILED (rc=$RC, likely OOM) -> RETRY with batch_size2=4 $(date)"
  python -u scripts/train.py --config $CFG --variant $V --fusion $FUSION $EXTRA --stage 2 --batch_size2 4
fi
echo "## $V/${CV} INFER seed $SEED $(date)"
python -u scripts/infer.py --config $CFG --variant $V --fusion $FUSION $EXTRA --seed $SEED
echo "## $V/${CV} DONE $(date)"
