#!/usr/bin/env bash
# Autonomous ranking-eval queue. Once ALL current training finishes (Luxury text/
# clip_inject retrains + Toys text/clip_align), re-run inference in RANKING mode
# (--rank) for every dataset/variant -> Hit@1/5, NDCG@5. Generation results are
# backed up first (results/_backup_gen/). 4 GPU lanes run in parallel; each lane
# processes its jobs sequentially (no GPU contention). rank_chunk=20 (free GPUs).
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[rankq] waiting for training to finish $(date)"
while true; do
  n=0
  for f in r_lux_text_retrain r_lux_inject_retrain r_toys_text r_toys_align; do
    grep -q "DONE" "logs/$f.log" 2>/dev/null && n=$((n+1))
  done
  [ "$n" -ge 4 ] && break
  sleep 300
done

echo "[rankq] training done -> backup generation results $(date)"
mkdir -p results/_backup_gen
for d in results/*/; do
  b=$(basename "$d")
  { [ "$b" = "_backup_gen" ] || [ "$b" = "_backup" ]; } && continue
  [ -f "$d/seed_0.jsonl" ] && cp "$d/seed_0.jsonl" "results/_backup_gen/${b}_seed0.jsonl"
done

echo "[rankq] SASRec ranking (fast) $(date)"
for CFG in luxury_beauty_rq3 all_beauty amazon_fashion prime_pantry toys_and_games_sub; do
  CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_sasrec.py --config configs/$CFG.yaml --seed 0 2>&1 | grep -i wrote
done

# LLM rank-inference jobs: "config|variant|clip_variant"  (- = none/bigG/text)
JOBS=(
  "luxury_beauty_rq3|text|-"
  "luxury_beauty_rq3|clip_align|vitl14_zeroshot"
  "luxury_beauty_rq3|clip_inject|vitl14_zeroshot"
  "luxury_beauty_rq3|clip_inject|vitl14_ft"
  "luxury_beauty_rq3|clip_inject|-"
  "all_beauty|text|-"
  "all_beauty|clip_align|vitl14_zeroshot"
  "all_beauty|clip_inject|vitl14_zeroshot"
  "amazon_fashion|text|-"
  "amazon_fashion|clip_align|vitl14_zeroshot"
  "amazon_fashion|clip_inject|vitl14_zeroshot"
  "prime_pantry|text|-"
  "prime_pantry|clip_align|vitl14_zeroshot"
  "prime_pantry|clip_inject|vitl14_zeroshot"
  "toys_and_games_sub|text|-"
  "toys_and_games_sub|clip_align|vitl14_zeroshot"
  "toys_and_games_sub|clip_inject|vitl14_zeroshot"
)

run_lane() {
  local GPU=$1; shift
  for J in "$@"; do
    IFS='|' read -r CFG V CV <<< "$J"
    local EXTRA=""; [ "$CV" != "-" ] && EXTRA="--clip_variant $CV"
    echo "[rankq] GPU$GPU: $CFG/$V/$CV $(date)"
    CUDA_VISIBLE_DEVICES=$GPU python -u scripts/infer.py --config configs/$CFG.yaml \
      --variant $V --fusion concat $EXTRA --seed 0 --rank --rank_chunk 20 \
      > "logs/rank_${CFG}_${V}_${CV}.log" 2>&1
  done
}

# split jobs round-robin into 4 GPU lanes
lane0=(); lane1=(); lane2=(); lane3=(); i=0
for J in "${JOBS[@]}"; do
  case $((i % 4)) in
    0) lane0+=("$J");; 1) lane1+=("$J");; 2) lane2+=("$J");; 3) lane3+=("$J");;
  esac
  i=$((i + 1))
done
echo "[rankq] ${#JOBS[@]} LLM rank-inferences across 4 GPU lanes $(date)"
run_lane 0 "${lane0[@]}" &
run_lane 1 "${lane1[@]}" &
run_lane 2 "${lane2[@]}" &
run_lane 3 "${lane3[@]}" &
wait

echo "[rankq] ALL DONE -> report (Hit@1/5, NDCG@5) $(date)"
python -u -m clam_rec.eval.report --runs_dir results --ks 1 5 --fuzzy 0.90 --variants \
  Luxury_Beauty_sasrec Luxury_Beauty_text_concat Luxury_Beauty_clip_align_concat_vitl14_zeroshot Luxury_Beauty_clip_inject_concat_vitl14_zeroshot Luxury_Beauty_clip_inject_concat_vitl14_ft Luxury_Beauty_clip_inject_concat \
  All_Beauty_sasrec All_Beauty_text_concat All_Beauty_clip_align_concat_vitl14_zeroshot All_Beauty_clip_inject_concat_vitl14_zeroshot \
  AMAZON_FASHION_sasrec AMAZON_FASHION_text_concat AMAZON_FASHION_clip_align_concat_vitl14_zeroshot AMAZON_FASHION_clip_inject_concat_vitl14_zeroshot \
  Prime_Pantry_sasrec Prime_Pantry_text_concat Prime_Pantry_clip_align_concat_vitl14_zeroshot Prime_Pantry_clip_inject_concat_vitl14_zeroshot \
  Toys_and_Games_sub_sasrec Toys_and_Games_sub_text_concat Toys_and_Games_sub_clip_align_concat_vitl14_zeroshot Toys_and_Games_sub_clip_inject_concat_vitl14_zeroshot
echo "[rankq] QUEUE COMPLETE $(date)"
