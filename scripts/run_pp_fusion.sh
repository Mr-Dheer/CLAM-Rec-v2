#!/usr/bin/env bash
# Prime Pantry fusion data: SASRec (fast) + text LLM sharded across 4 GPUs, then merge.
set +u   # override any inherited nounset; conda activate references unbound vars
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CF=data/candidates/Prime_Pantry_seed0.json
echo "## PP SASRec-shared $(date)"
CUDA_VISIBLE_DEVICES=0 python scripts/eval_sasrec.py --config configs/prime_pantry.yaml --seed 0 --candidates_file "$CF"
echo "## PP text shards x4 $(date)"
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python scripts/infer.py --config configs/prime_pantry.yaml --variant text --fusion concat \
    --seed 0 --rank --rank_chunk 20 --candidates_file "$CF" --nshards 4 --shard $g > logs/pp_shard$g.log 2>&1 &
done
wait
echo "## merge shards $(date)"
OUT=results/Prime_Pantry_text_concat_shared
cat $OUT/seed_0.part0of4.jsonl $OUT/seed_0.part1of4.jsonl $OUT/seed_0.part2of4.jsonl $OUT/seed_0.part3of4.jsonl > $OUT/seed_0.jsonl
echo "## PP FUSION DONE $(date) — merged $(wc -l < $OUT/seed_0.jsonl) records"
