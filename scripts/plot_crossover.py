"""
Publication figure: the CF<->LLM crossover. Left panel: Hit@1 advantage of the LLM
(text) over CF (SASRec) vs the target item's TRAIN interaction count, one line per
dataset -> monotonic decay crossing zero (LLM wins cold, CF wins warm). Right panel:
vision effect (clip_inject - text) -> flat near zero (vision null). Ranking eval.

Usage: python scripts/plot_crossover.py  ->  figures/crossover.{pdf,png}
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.eval.metrics import record_hit_at_1

FUZZY = 0.90
DATASETS = [("Luxury_Beauty", "Luxury Beauty"), ("All_Beauty", "All Beauty"),
            ("AMAZON_FASHION", "Fashion"), ("Prime_Pantry", "Prime Pantry"),
            ("Toys_and_Games_sub", "Toys")]
BINS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]
XLAB = ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51+"]


def load(ds, v):
    p = Path(f"results/{ds}_{v}/seed_0.jsonl")
    return {json.loads(l)["user"]: json.loads(l) for l in open(p)} if p.exists() else None


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for ds, name in DATASETS:
    sas, txt, inj = load(ds, "sasrec"), load(ds, "text_concat"), load(ds, "clip_inject_concat_vitl14_zeroshot")
    if not (sas and txt and inj):
        continue
    xs, d_tc, d_it = [], [], []
    for k, ((lo, hi), lab) in enumerate(zip(BINS, XLAB)):
        us = [u for u in txt if txt[u].get("train_count") is not None
              and lo <= txt[u]["train_count"] <= hi and u in sas and u in inj]
        if len(us) < 20:
            continue
        s = np.mean([record_hit_at_1(sas[u], FUZZY) for u in us])
        t = np.mean([record_hit_at_1(txt[u], FUZZY) for u in us])
        j = np.mean([record_hit_at_1(inj[u], FUZZY) for u in us])
        xs.append(k); d_tc.append(t - s); d_it.append(j - t)
    ax1.plot(xs, d_tc, marker="o", label=name)
    ax2.plot(xs, d_it, marker="o", label=name)

for ax, title, ylab in [(ax1, "LLM vs CF (text $-$ SASRec)", "$\\Delta$ Hit@1  (LLM $-$ CF)"),
                        (ax2, "Vision effect (clip_inject $-$ text)", "$\\Delta$ Hit@1  (vision)")]:
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(XLAB))); ax.set_xticklabels(XLAB, rotation=0)
    ax.set_xlabel("target item train interaction count"); ax.set_ylabel(ylab)
    ax.set_title(title); ax.grid(alpha=0.3)
ax1.legend(fontsize=8, loc="upper right")
ax1.annotate("LLM wins (cold)", (0.05, 0.92), xycoords="axes fraction", fontsize=8, color="green")
ax1.annotate("CF wins (warm)", (0.6, 0.08), xycoords="axes fraction", fontsize=8, color="firebrick")
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/crossover.pdf"); fig.savefig("figures/crossover.png", dpi=150)
print("saved -> figures/crossover.{pdf,png}")
