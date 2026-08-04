"""Full fusion table: Hit@1/5/10 + NDCG@5/10 (overall + cold/warm Hit@1) for every method,
all datasets. Post-hoc on the shared-candidate logged scores. Emits markdown."""
import sys, json, math
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.config import load_config
from clam_rec.data.partition import data_partition

ROOT = Path(__file__).resolve().parents[1]
DS = [("Luxury_Beauty", "luxury_beauty_rq3", "Luxury Beauty"),
      ("AMAZON_FASHION", "amazon_fashion", "AMAZON_FASHION"),
      ("Toys_and_Games_sub", "toys_and_games_sub", "Toys & Games (sub)"),
      ("Prime_Pantry", "prime_pantry", "Prime Pantry")]   # All Beauty dropped 2026-08-04 (small); Prime Pantry added
METHODS = ["CF (SASRec)", "text (LLM)", "RRF", "z-fuse", "pop-gate", "ORACLE"]


def load(run):
    p = ROOT / f"results/{run}/seed_0.jsonl"
    return {json.loads(l)["user"]: json.loads(l) for l in open(p) if l.strip()} if p.exists() else None


def item_pop(cfg):
    ut, _, _, _, _ = data_partition(cfg.interactions)
    c = Counter()
    for u, items in ut.items():
        for it in items:
            c[int(it)] += 1
    return c


def z(a):
    a = np.asarray(a, float); return (a - a.mean()) / (a.std() + 1e-9)


def rank_of(scores, tgt_local):                 # 1-based rank of the target index
    order = np.argsort(-np.asarray(scores))
    return int(np.where(order == tgt_local)[0][0]) + 1


def hk(r, k): return 1.0 if r <= k else 0.0
def nk(r, k): return (1/math.log2(r+1)) if r <= k else 0.0


for dsid, cfgn, nice in DS:
    sas, txt = load(f"{dsid}_sasrec_shared"), load(f"{dsid}_text_concat_shared")
    if not (sas and txt):
        print(f"[{nice}] missing"); continue
    cfg = load_config(f"configs/{cfgn}.yaml", stage="inference")
    pop = item_pop(cfg); pivot = sum(pop.values()) / max(len(pop), 1)
    users = [u for u in txt if u in sas]

    # per method: lists of (rank, cold)
    R = {m: [] for m in METHODS}
    for u in users:
        a, b = sas[u], txt[u]
        cs = {int(c): s for c, s in zip(a["candidate_ids"], a["scores"])}
        ct = {int(c): s for c, s in zip(b["candidate_ids"], b["scores"])}
        ids = [c for c in a["candidate_ids"] if c in ct]
        if a["target_id"] not in ids:
            continue
        ti = ids.index(a["target_id"]); cold = bool(a.get("cold"))
        zc, zt = z([cs[c] for c in ids]), z([ct[c] for c in ids])
        pops = np.array([pop.get(c, 0) for c in ids]); w = pops / (pops + pivot)
        rc = {c: i for i, c in enumerate(np.array(ids)[np.argsort(-zc)])}
        rt = {c: i for i, c in enumerate(np.array(ids)[np.argsort(-zt)])}
        rrf = [1/(60+rc[c]+1) + 1/(60+rt[c]+1) for c in ids]
        scoremap = {
            "CF (SASRec)": zc, "text (LLM)": zt, "RRF": rrf,
            "z-fuse": zc + zt, "pop-gate": w * zc + (1 - w) * zt,
            "ORACLE": zc if pop.get(a["target_id"], 0) > pivot else zt,
        }
        for m, sc in scoremap.items():
            R[m].append((rank_of(sc, ti), cold))

    print(f"\n#### {nice} — n={len(R['CF (SASRec)'])} "
          f"({sum(c for _,c in R['CF (SASRec)'])} cold / {sum(1-c for _,c in R['CF (SASRec)'])} warm)")
    print()
    print("| Method | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | cold H@1 | warm H@1 |")
    print("|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    for m in METHODS:
        rk = [r for r, _ in R[m]]
        cold = [r for r, c in R[m] if c]; warm = [r for r, c in R[m] if not c]
        h1 = np.mean([hk(r, 1) for r in rk]); h5 = np.mean([hk(r, 5) for r in rk]); h10 = np.mean([hk(r, 10) for r in rk])
        n5 = np.mean([nk(r, 5) for r in rk]); n10 = np.mean([nk(r, 10) for r in rk])
        ch1 = np.mean([hk(r, 1) for r in cold]); wh1 = np.mean([hk(r, 1) for r in warm])
        star = " *" if m == "ORACLE" else ""
        print(f"| {m}{star} | {h1:.3f} | {h5:.3f} | {h10:.3f} | {n5:.3f} | {n10:.3f} | {ch1:.3f} | {wh1:.3f} |")
    print("\n_* ORACLE routes by the true target popularity — upper bound, not deployable._")
