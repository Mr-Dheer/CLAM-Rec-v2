#!/usr/bin/env bash
# Shared-candidate re-inference for CF+LLM fusion. For one dataset on one GPU:
#   1) SASRec ranks the shared pool (fast)   2) text LLM ranks the same pool (slow)
# Both log per-candidate scores -> results/{ds}_{sasrec,text_concat}_shared/seed_0.jsonl
# Usage: bash scripts/run_shared_reinfer.sh <config> <gpu>
set -e
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec
CFG=$1; GPU=$2
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DS=$(python -c "from clam_rec.config import load_config; print(load_config('$CFG', stage='inference').dataset)")
CF=data/candidates/${DS}_seed0.json
echo "## $DS shared re-infer on GPU $GPU $(date)"
[ -f "$CF" ] || python scripts/make_candidates.py --config "$CFG" --seed 0
# SASRec (skip if already done)
if [ ! -s "results/${DS}_sasrec_shared/seed_0.jsonl" ]; then
  echo "## $DS SASRec-shared $(date)"
  python scripts/eval_sasrec.py --config "$CFG" --seed 0 --candidates_file "$CF"
fi
echo "## $DS text-LLM-shared $(date)"
python scripts/infer.py --config "$CFG" --variant text --fusion concat --seed 0 \
  --rank --rank_chunk 20 --candidates_file "$CF"
echo "## $DS DONE $(date)"
