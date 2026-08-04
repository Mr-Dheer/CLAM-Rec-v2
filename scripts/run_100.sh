#!/usr/bin/env bash
# 100-candidate ranking pass for the paper. Runs an assigned list of {dataset:variant}
# jobs sequentially on ONE GPU, on a SHARED 100-candidate pool (logs per-candidate
# scores -> enables both the main ranking table AND fusion, all at 100 candidates).
# Idempotent via .done markers. Eval user cap via env MAXU (0 = all users).
# Usage: MAXU=2000 bash scripts/run_100.sh <gpu> Luxury_Beauty:text All_Beauty:sasrec ...
set -u
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec
GPU=$1; shift
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAXU=${MAXU:-0}
declare -A CFG=( [Luxury_Beauty]=configs/luxury_beauty_rq3.yaml [All_Beauty]=configs/all_beauty.yaml \
                 [AMAZON_FASHION]=configs/amazon_fashion.yaml [Toys_and_Games_sub]=configs/toys_and_games_sub.yaml )
MU=""; [ "$MAXU" -gt 0 ] && MU="--max_users $MAXU"

for job in "$@"; do
  ds=${job%%:*}; var=${job##*:}; cfg=${CFG[$ds]}
  CF=data/candidates/${ds}_seed0_n100.json
  case $var in
    sasrec)      run=${ds}_sasrec_n100;      mark=results/$run/.done ;;
    text)        run=${ds}_text_concat_n100; mark=results/$run/.done ;;
    clip_align)  run=${ds}_clip_align_concat_vitl14_zeroshot_n100; mark=results/$run/.done ;;
    clip_inject) run=${ds}_clip_inject_concat_vitl14_zeroshot_n100; mark=results/$run/.done ;;
  esac
  if [ -f "$mark" ]; then echo "## SKIP $job (done) $(date)"; continue; fi
  echo "## START $job on GPU $GPU $(date)"
  if [ "$var" = sasrec ]; then
    python scripts/eval_sasrec.py --config "$cfg" --seed 0 --candidates_file "$CF" --out_tag n100 $MU
  elif [ "$var" = text ]; then
    python scripts/infer.py --config "$cfg" --variant text --fusion concat --seed 0 \
      --rank --rank_chunk 50 --candidates_file "$CF" --out_tag n100 $MU
  else
    python scripts/infer.py --config "$cfg" --variant "$var" --fusion concat --clip_variant vitl14_zeroshot \
      --seed 0 --rank --rank_chunk 50 --candidates_file "$CF" --out_tag n100 $MU
  fi
  [ $? -eq 0 ] && touch "$mark" && echo "## DONE $job $(date)"
done
echo "## GPU $GPU QUEUE COMPLETE $(date)"
