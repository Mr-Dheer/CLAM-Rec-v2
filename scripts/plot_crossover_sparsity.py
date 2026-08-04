"""
Figure + analysis: the CF->LLM crossover and its dependence on dataset density.

Left panel : (text - SASRec) Hit@1 vs target item TRAIN interaction count, 4 datasets.
             The zero-crossing = the CF<->LLM crossover point.
Right panel: crossover point vs dataset mean interactions/item, with y=x identity line.
             Shows the crossover is *predicted* by density (crossover ~ mean pop/item).

Also prints a DEPLOYABLE router (route by user history length -> which model to trust),
computable from existing logs (no re-inference, no shared candidate set needed).

Usage: python scripts/plot_crossover_sparsity.py
Outputs: figures/crossover_sparsity.{pdf,png}
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.eval.metrics import record_hit_at_1, record_rank, hit_at_k, ndcg_at_k

FUZZY = 0.90
ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["Luxury_Beauty", "AMAZON_FASHION", "Toys_and_Games_sub", "Prime_Pantry"]  # AB dropped, PP added 2026-08-04
NICE = {"Luxury_Beauty": "Luxury Beauty", "Prime_Pantry": "Prime Pantry",
        "AMAZON_FASHION": "Fashion", "Toys_and_Games_sub": "Toys (sub)"}
COLORS = {"Luxury_Beauty": "#1f77b4", "Prime_Pantry": "#2ca02c",
          "AMAZON_FASHION": "#d62728", "Toys_and_Games_sub": "#9467bd"}
FINE = [(0,0),(1,1),(2,2),(3,4),(5,7),(8,12),(13,20),(21,35),(36,60),(61,10**9)]
CENTERS = [0.5,1,2,3.5,6,10,16,28,48,80]  # x positions (0->0.5 for log axis)


def load(ds, v):
    p = ROOT / f"results/{ds}_{v}/seed_0.jsonl"
    return {json.loads(l)["user"]: json.loads(l) for l in open(p) if l.strip()} if p.exists() else None


def sparsity(ds):
    ic, uc, n = {}, {}, 0
    for line in open(ROOT / f"assets/{ds}.txt"):
        p = line.split()
        if len(p) < 2:
            continue
        uc[p[0]] = uc.get(p[0], 0) + 1
        ic[p[1]] = ic.get(p[1], 0) + 1
        n += 1
    return dict(inter_per_item=n/len(ic), inter_per_user=n/len(uc), items=len(ic))


def metrics(ranks):
    h1 = np.mean([1.0 if r == 1 else 0.0 for r in ranks])
    h5 = np.mean([hit_at_k(r, 5) if r else 0.0 for r in ranks])
    n5 = np.mean([ndcg_at_k(r, 5) if r else 0.0 for r in ranks])
    return h1, h5, n5


curves, xovers, sp_item, sp_user = {}, {}, {}, {}
router_rows = []
for ds in DATASETS:
    sas, txt = load(ds, "sasrec"), load(ds, "text_concat")
    if not (sas and txt):
        print(f"[{ds}] missing"); continue
    users = [u for u in txt if u in sas and txt[u].get("train_count") is not None]

    xs, ys = [], []
    for (lo, hi), c in zip(FINE, CENTERS):
        us = [u for u in users if lo <= txt[u]["train_count"] <= hi]
        if len(us) < 20:
            continue
        d = np.mean([record_hit_at_1(txt[u], FUZZY) for u in us]) - \
            np.mean([record_hit_at_1(sas[u], FUZZY) for u in us])
        xs.append(c); ys.append(d)
    curves[ds] = (xs, ys)

    xo = None
    for i in range(1, len(ys)):
        if ys[i-1] >= 0 > ys[i]:
            xo = xs[i-1] + (xs[i]-xs[i-1]) * (ys[i-1] / (ys[i-1]-ys[i])); break
    xovers[ds] = xo
    s = sparsity(ds); sp_item[ds] = s["inter_per_item"]; sp_user[ds] = s["inter_per_user"]

    # DEPLOYABLE router: route by USER history length (known at inference; no shared cands).
    # short history -> trust LLM (text); long -> trust CF (SASRec). Sweep threshold, best Hit@1.
    r_sas = {u: record_rank(sas[u], FUZZY) for u in users}
    r_txt = {u: record_rank(txt[u], FUZZY) for u in users}
    hlen = {u: (txt[u].get("hist_len") or txt[u].get("seq_len")) for u in users}
    if all(hlen[u] is None for u in users):
        hlen = None  # history length not logged; skip deployable router for this ds
    ms, mt = metrics([r_sas[u] for u in users]), metrics([r_txt[u] for u in users])
    row = dict(ds=ds, sas=ms, txt=mt, best_pure=max(ms[0], mt[0]))
    if hlen is not None:
        best = None
        for thr in sorted(set(hlen.values())):
            ranks = [r_txt[u] if hlen[u] <= thr else r_sas[u] for u in users]
            h = metrics(ranks)[0]
            if best is None or h > best[1]:
                best = (thr, h, ranks)
        row["router_hist"] = (best[0], metrics(best[2]))
    router_rows.append(row)

# ---------------- figure ----------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

for ds in DATASETS:
    if ds not in curves:
        continue
    xs, ys = curves[ds]
    axL.plot(xs, ys, "-o", ms=4, color=COLORS[ds], label=NICE[ds])
axL.axhline(0, color="k", lw=0.8, ls="--")
axL.set_xscale("log")
axL.set_xlabel("target item train interaction count (log)")
axL.set_ylabel("Hit@1(text) − Hit@1(SASRec)")
axL.set_title("(a) CF→LLM crossover: LLM wins cold, CF wins warm")
axL.legend(fontsize=8, frameon=False)
axL.text(0.6, 0.02, "LLM better ↑", fontsize=8, color="gray")
axL.text(0.6, -0.04, "CF better ↓", fontsize=8, color="gray")

xo = np.array([xovers[d] for d in DATASETS if xovers.get(d) is not None])
xi = np.array([sp_item[d] for d in DATASETS if xovers.get(d) is not None])
xu = np.array([sp_user[d] for d in DATASETS if xovers.get(d) is not None])
names = [d for d in DATASETS if xovers.get(d) is not None]
lim = max(xo.max(), xi.max()) * 1.15
axR.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")
for d in names:
    axR.scatter(sp_item[d], xovers[d], s=70, color=COLORS[d], zorder=3)
    axR.annotate(NICE[d], (sp_item[d], xovers[d]), textcoords="offset points",
                 xytext=(6, 4), fontsize=8)
r_item = np.corrcoef(xi, xo)[0, 1]
r_user = np.corrcoef(xu, xo)[0, 1]
axR.set_xlim(0, lim); axR.set_ylim(0, lim)
axR.set_xlabel("dataset mean interactions / item")
axR.set_ylabel("crossover point (train_count)")
axR.set_title("(b) crossover point ≈ dataset density")
axR.legend(fontsize=8, frameon=False, loc="upper left")
axR.text(0.05, 0.80, f"Pearson r = {r_item:+.2f}\n(vs hist. length r = {r_user:+.2f})",
         transform=axR.transAxes, fontsize=9,
         bbox=dict(boxstyle="round", fc="white", ec="0.7"))

plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(ROOT / f"figures/crossover_sparsity.{ext}", dpi=160, bbox_inches="tight")
print("wrote figures/crossover_sparsity.{pdf,png}")
print(f"\ncrossover points: " + ", ".join(f"{NICE[d]}={xovers[d]:.1f}" for d in names))
print(f"mean inter/item : " + ", ".join(f"{NICE[d]}={sp_item[d]:.1f}" for d in names))
print(f"Pearson r(crossover, inter/item)={r_item:+.3f}   r(crossover, hist.len)={r_user:+.3f}")

print("\n=== DEPLOYABLE router (route by user history length; no re-inference) ===")
for row in router_rows:
    tag = ""
    if "router_hist" in row:
        thr, m = row["router_hist"]
        tag = (f"router(hist<= {thr:2d}) Hit@1={m[0]:.4f} "
               f"(Δ best-pure {m[0]-row['best_pure']:+.4f}) Hit@5={m[1]:.4f} NDCG@5={m[2]:.4f}")
    else:
        tag = "history length NOT logged in records -> deployable router needs it (see note)"
    print(f"  {NICE[row['ds']]:14s} pure: CF {row['sas'][0]:.3f} / text {row['txt'][0]:.3f}  |  {tag}")
