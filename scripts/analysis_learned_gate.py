"""
Learned-gate ablation (post-hoc, no GPU, no retraining). On the SHARED-candidate logged
scores, compare popularity-gating variants:
  - fixed  : w = pop/(pop+pivot)          (the hand-picked formula in analysis_ensemble)
  - learned: w = sigmoid(a*log(pop+1)+b)  (2 params fit to rank the target first)
Fit the learned gate on a TRAIN split of users, evaluate all methods on the held-out TEST
split (honest — no overfitting). Reports Hit@1/5, NDCG@5. Confirms whether learning the gate
beats the fixed formula (expectation: it matches it, near the oracle ceiling).

Usage: python scripts/analysis_learned_gate.py
"""
import sys, json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.config import load_config
from clam_rec.data.partition import data_partition

ROOT = Path(__file__).resolve().parents[1]
DS = {"Luxury_Beauty": "luxury_beauty_rq3", "AMAZON_FASHION": "amazon_fashion",
      "Toys_and_Games_sub": "toys_and_games_sub", "Prime_Pantry": "prime_pantry"}  # AB dropped, PP added 2026-08-04


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


def zrows(a):                          # per-row (per-user) z-normalize
    m = a.mean(1, keepdims=True); s = a.std(1, keepdims=True)
    return (a - m) / (s + 1e-9)


def metrics(fused, tgt_idx):           # fused: (N,20), tgt_idx: (N,)
    order = np.argsort(-fused, 1)
    rank = (order == tgt_idx[:, None]).argmax(1) + 1   # 1-based rank of target
    h1 = np.mean(rank == 1); h5 = np.mean(rank <= 5)
    n5 = np.mean(np.where(rank <= 5, 1/np.log2(rank+1), 0.0))
    return h1, h5, n5


for ds, cfgname in DS.items():
    sas, txt = load(f"{ds}_sasrec_shared"), load(f"{ds}_text_concat_shared")
    if not (sas and txt):
        print(f"[{ds}] shared runs missing"); continue
    cfg = load_config(f"configs/{cfgname}.yaml", stage="inference")
    pop = item_pop(cfg)
    pivot = sum(pop.values()) / max(len(pop), 1)

    users = [u for u in txt if u in sas]
    ZCF, ZTX, LP, TGT = [], [], [], []
    for u in users:
        a, b = sas[u], txt[u]
        cs = {int(c): s for c, s in zip(a["candidate_ids"], a["scores"])}
        ct = {int(c): s for c, s in zip(b["candidate_ids"], b["scores"])}
        ids = [c for c in a["candidate_ids"] if c in ct]
        if a["target_id"] not in ids or len(ids) != 20:
            continue
        ZCF.append([cs[c] for c in ids]); ZTX.append([ct[c] for c in ids])
        LP.append([np.log(pop.get(c, 0) + 1) for c in ids])
        TGT.append(ids.index(a["target_id"]))
    ZCF = zrows(np.array(ZCF, float)); ZTX = zrows(np.array(ZTX, float))
    LP = np.array(LP, float); TGT = np.array(TGT)
    N = len(TGT)

    rng = np.random.default_rng(0)
    tr = rng.random(N) < 0.5; te = ~tr                 # 50/50 user split

    def fuse(w):                                       # w: (N,20)
        return w * ZCF + (1 - w) * ZTX

    # learned gate: w = sigmoid(a*logpop + b), fit (a,b) on TRAIN via softmax CE
    def loss(theta):
        a, b = theta
        w = 1 / (1 + np.exp(-(a * LP[tr] + b)))
        f = fuse(w)[np.arange(N)[tr]] if False else (w * ZCF[tr] + (1 - w) * ZTX[tr])
        f = f - f.max(1, keepdims=True)
        logp = f - np.log(np.exp(f).sum(1, keepdims=True))
        return -logp[np.arange(tr.sum()), TGT[tr]].mean()
    res = minimize(loss, x0=[0.0, 0.0], method="Nelder-Mead")
    a_, b_ = res.x

    # evaluate everything on TEST split
    def ev(name, fused):
        h1, h5, n5 = metrics(fused[te], TGT[te])
        return f"{name:16s} Hit@1 {h1:.4f}  Hit@5 {h5:.4f}  NDCG@5 {n5:.4f}"

    pivot_w = LP  # placeholder
    w_fixed = np.exp(LP) - 1; w_fixed = w_fixed / (w_fixed + pivot)   # pop/(pop+pivot)
    w_learn = 1 / (1 + np.exp(-(a_ * LP + b_)))
    # oracle: route whole decision by target popularity (test)
    tgt_pop = np.exp(LP[np.arange(N), TGT]) - 1
    orc = np.where((tgt_pop > pivot)[:, None], ZCF, ZTX)

    print(f"\n=== {ds}  (N={N}, test={te.sum()}, pivot pop={pivot:.1f}, learned a={a_:.2f} b={b_:.2f}) ===")
    print("  " + ev("pure CF", ZCF))
    print("  " + ev("pure text", ZTX))
    print("  " + ev("fixed pop-gate", fuse(w_fixed)))
    print("  " + ev("LEARNED gate", fuse(w_learn)))
    print("  " + ev("ORACLE (upper)", orc))
