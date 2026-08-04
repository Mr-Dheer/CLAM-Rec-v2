"""
Honest CF+LLM fusion on a SHARED candidate set (per-candidate scores logged by the
_shared runs). Compares, per dataset: pure SASRec, pure text, and fusions:
  - RRF (rank fusion)                         : deployable, no popularity
  - z-score fusion (equal weight)             : deployable, no popularity
  - popularity-gated z-fusion                 : deployable, uses candidate popularity
        weight w(pop)=pop/(pop+pivot); trust CF for popular candidates, LLM for rare
  - ORACLE popularity router (by target pop)  : upper bound reference
Metrics: Hit@1, Hit@5, NDCG@5 (overall + cold/warm). Title-independent (uses target_id).

Usage: python scripts/analysis_ensemble.py
"""
import sys, json
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.config import load_config
from clam_rec.data.partition import data_partition

ROOT = Path(__file__).resolve().parents[1]
DS = {"Luxury_Beauty": "luxury_beauty_rq3", "AMAZON_FASHION": "amazon_fashion",
      "Toys_and_Games_sub": "toys_and_games_sub", "Prime_Pantry": "prime_pantry"}  # AB dropped, PP added 2026-08-04


def load(run):
    p = ROOT / f"results/{run}/seed_0.jsonl"
    return {json.loads(l)["user"]: json.loads(l) for l in open(p) if l.strip()} if p.exists() else None


def item_train_pop(cfg):
    ut, _, _, _, _ = data_partition(cfg.interactions)
    c = Counter()
    for u, items in ut.items():
        for it in items:
            c[int(it)] += 1
    return c


def rank_of(target, ids, sc):                     # 1-based rank of target by score desc
    order = np.argsort(-np.asarray(sc))
    ids = np.asarray(ids)[order]
    pos = np.where(ids == target)[0]
    return int(pos[0]) + 1 if len(pos) else None


def z(a):
    a = np.asarray(a, float); s = a.std()
    return (a - a.mean()) / (s + 1e-9)


def agg(ranks, mask=None):
    r = np.array([x if x else 999 for x in ranks], float)
    if mask is not None:
        r = r[mask]
    h1 = np.mean(r == 1); h5 = np.mean(r <= 5)
    n5 = np.mean([(1/np.log2(x+1)) if x <= 5 else 0.0 for x in r])
    return h1, h5, n5


for ds, cfgname in DS.items():
    sas, txt = load(f"{ds}_sasrec_shared"), load(f"{ds}_text_concat_shared")
    if not (sas and txt):
        print(f"[{ds}] shared runs missing (sas={bool(sas)} txt={bool(txt)})"); continue
    cfg = load_config(f"configs/{cfgname}.yaml", stage="inference")
    pop = item_train_pop(cfg)
    inter_per_item = sum(pop.values()) / max(len(pop), 1)
    pivot = inter_per_item                          # principled 50/50 pivot (no fitting)

    users = [u for u in txt if u in sas]
    methods = ["CF", "text", "RRF", "z-fuse", "pop-gate", "ORACLE"]
    R = {m: [] for m in methods}
    cold = []
    for u in users:
        a, b = sas[u], txt[u]
        tid = a["target_id"]
        # align both models' scores by candidate id
        cs = {int(c): s for c, s in zip(a["candidate_ids"], a["scores"])}
        ct = {int(c): s for c, s in zip(b["candidate_ids"], b["scores"])}
        ids = [c for c in a["candidate_ids"] if c in ct]
        s_cf = np.array([cs[c] for c in ids]); s_tx = np.array([ct[c] for c in ids])
        zc, zt = z(s_cf), z(s_tx)
        R["CF"].append(rank_of(tid, ids, s_cf))
        R["text"].append(rank_of(tid, ids, s_tx))
        # RRF
        rc = {c: i for i, c in enumerate(np.array(ids)[np.argsort(-s_cf)])}
        rt = {c: i for i, c in enumerate(np.array(ids)[np.argsort(-s_tx)])}
        rrf = [1/(60+rc[c]+1) + 1/(60+rt[c]+1) for c in ids]
        R["RRF"].append(rank_of(tid, ids, rrf))
        R["z-fuse"].append(rank_of(tid, ids, zc + zt))
        # popularity-gated: per-candidate weight on CF by that candidate's popularity
        w = np.array([pop.get(c, 0) / (pop.get(c, 0) + pivot) for c in ids])
        R["pop-gate"].append(rank_of(tid, ids, w * zc + (1 - w) * zt))
        # oracle: route whole decision by TARGET popularity (tau = pivot)
        R["ORACLE"].append(rank_of(tid, ids, s_cf if pop.get(tid, 0) > pivot else s_tx))
        cold.append(bool(a.get("cold")))
    cold = np.array(cold)

    print(f"\n=== {ds}  (n={len(users)}, pivot pop={pivot:.1f}) ===")
    print(f"{'method':9s} | {'Hit@1':>6s} {'Hit@5':>6s} {'NDCG5':>6s} | "
          f"{'cold H@1':>8s} {'warm H@1':>8s}")
    base = max(agg(R['CF'])[0], agg(R['text'])[0])
    for m in methods:
        o = agg(R[m]); c = agg(R[m], cold); w = agg(R[m], ~cold)
        d = "" if m in ("CF", "text") else f"  (Δbest {o[0]-base:+.3f})"
        star = " *" if m == "ORACLE" else ""
        print(f"{m:9s} | {o[0]:6.4f} {o[1]:6.4f} {o[2]:6.4f} | {c[0]:8.4f} {w[0]:8.4f}{d}{star}")
    print("  * ORACLE routes by the true target popularity = upper bound, not deployable")
