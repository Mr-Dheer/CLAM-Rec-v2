"""
Crossover analysis (ranking eval): Hit@1 vs target item's TRAIN interaction count,
per dataset, for SASRec / text / clip_inject. Shows the CF->LLM handoff: the LLM
(text) beats CF (SASRec) on cold/rare items and loses on warm/popular items, and
vision (clip_inject) tracks text. Uses the ranked seed_0.jsonl (generated=ranked[0]).

Usage: python scripts/analysis_crossover.py
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.eval.metrics import record_hit_at_1

FUZZY = 0.90
DATASETS = ["Luxury_Beauty", "All_Beauty", "AMAZON_FASHION", "Prime_Pantry", "Toys_and_Games_sub"]
BINS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]
LABELS = ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51+"]


def load(ds, v):
    p = Path(f"results/{ds}_{v}/seed_0.jsonl")
    if not p.exists():
        return None
    return {json.loads(l)["user"]: json.loads(l) for l in open(p) if l.strip()}


for ds in DATASETS:
    sas = load(ds, "sasrec")
    txt = load(ds, "text_concat")
    inj = load(ds, "clip_inject_concat_vitl14_zeroshot")
    if not (sas and txt and inj):
        print(f"[{ds}] missing runs"); continue
    print(f"\n=== {ds} ===")
    print(f"{'train_cnt':9s} {'n':>7s} {'SASRec':>7s} {'text':>7s} {'inject':>7s} "
          f"{'text-CF':>8s} {'inj-text':>9s}")
    xover_prev = None
    for (lo, hi), lab in zip(BINS, LABELS):
        us = [u for u in txt if txt[u].get("train_count") is not None
              and lo <= txt[u]["train_count"] <= hi and u in sas and u in inj]
        if len(us) < 20:
            continue
        s = np.mean([record_hit_at_1(sas[u], FUZZY) for u in us])
        t = np.mean([record_hit_at_1(txt[u], FUZZY) for u in us])
        j = np.mean([record_hit_at_1(inj[u], FUZZY) for u in us])
        flag = ""
        if xover_prev is not None and (xover_prev >= 0) != ((t - s) >= 0):
            flag = "  <-- CF/LLM crossover"
        xover_prev = t - s
        print(f"{lab:9s} {len(us):7d} {s:7.3f} {t:7.3f} {j:7.3f} {t-s:+8.3f} {j-t:+9.3f}{flag}")
