"""
Mechanism analyses (no training; uses existing ViT-L embeddings + result JSONLs):

  1. Image-discriminativeness vs vision benefit. Per dataset compute how
     item-discriminative the images are (mean image<->text cosine = per-item
     consistency; mean image nearest-neighbour cosine = how look-alike items are)
     and line it up against the measured cold-slice vision benefit (inject - text).
  2. Coldness curve. Vision benefit (inject - text Hit@1) binned by the target
     item's TRAIN interaction count -> does vision help more as items get colder?
  3. Image-present vs image-missing. Slice the vision benefit by whether the
     TARGET item has an image -> near-causal check that the IMAGE drives the gain.

Usage: python scripts/analysis_mechanism.py
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.config import load_config
from clam_rec.data.partition import data_partition
from clam_rec.eval.metrics import title_match

FUZZY = 0.90
DS = [  # (name, config, measured cold-slice vision Δ from the ladder)
    ("Luxury_Beauty",  "configs/luxury_beauty.yaml", +0.076),
    ("All_Beauty",     "configs/all_beauty.yaml",    +0.006),
    ("AMAZON_FASHION", "configs/amazon_fashion.yaml", -0.024),
    ("Prime_Pantry",   "configs/prime_pantry.yaml",   None),   # rec pending
]

def load_recs(ds, variant):
    p = Path(f"results/{ds}_{variant}/seed_0.jsonl")
    if not p.exists(): return None
    return {json.loads(l)["user"]: json.loads(l) for l in open(p) if l.strip()}

def hit(r): return 1.0 if title_match(r["answer"], r["generated"], FUZZY) else 0.0

# ---------- 1. image discriminativeness ----------
def discriminativeness(ds):
    ti = np.load(f"data/clip/clip_text_{ds}_vitl14_zeroshot.npy")
    im = np.load(f"data/clip/clip_image_{ds}_vitl14_zeroshot.npy")
    has = np.linalg.norm(im, axis=1) > 1e-6
    idx = np.where(has)[0]
    imh = im[idx]                                   # normalized image emb (items w/ image)
    # per-item image<->text consistency
    it_cos = float((imh * ti[idx]).sum(1).mean())
    # nearest-neighbour image cosine (sample queries for speed)
    rng = np.random.default_rng(0)
    q = idx if len(idx) <= 2500 else rng.choice(idx, 2500, replace=False)
    S = im[q] @ imh.T                               # (q, N_img)
    # mask self
    self_pos = {v: k for k, v in enumerate(idx)}
    for r, gi in enumerate(q):
        S[r, self_pos[gi]] = -1
    nn = float(S.max(1).mean())                     # avg similarity to closest OTHER item
    return it_cos, nn, has

def exp1():
    print("\n" + "="*78)
    print("EXP 1: image discriminativeness vs cold-slice vision benefit")
    print("="*78)
    print(f"{'dataset':16s} {'img<->text cos':>14s} {'img NN cos':>11s} {'Δ vision (cold)':>16s}")
    print("  (higher img-text = image matches item; higher NN cos = more look-alikes = LESS discriminative)")
    rows=[]
    for ds, cfg, dc in DS:
        it, nn, _ = discriminativeness(ds)
        rows.append((ds, it, nn, dc))
        print(f"{ds:16s} {it:14.3f} {nn:11.3f} {('%+.3f'%dc) if dc is not None else 'pending':>16s}")
    # correlation over datasets with a measured Δ
    m=[(it,nn,dc) for _,it,nn,dc in rows if dc is not None]
    it_a=np.array([x[0] for x in m]); nn_a=np.array([x[1] for x in m]); d_a=np.array([x[2] for x in m])
    if len(m)>=3:
        print(f"\n  corr(img<->text, Δvision) = {np.corrcoef(it_a,d_a)[0,1]:+.2f}   "
              f"corr(img NN cos, Δvision) = {np.corrcoef(nn_a,d_a)[0,1]:+.2f}")
        print("  expect: img<->text +corr (consistent images help), img NN cos -corr (look-alikes hurt)")

# ---------- 2. coldness curve ----------
def exp2():
    print("\n" + "="*78)
    print("EXP 2: vision benefit (inject - text Hit@1) by target train-count bin")
    print("="*78)
    bins=[(0,0),(1,2),(3,5),(6,10),(11,20),(21,10**9)]
    labels=["0","1-2","3-5","6-10","11-20","21+"]
    for ds,_,_ in DS:
        t=load_recs(ds,"text_concat"); inj=load_recs(ds,"clip_inject_concat_vitl14_zeroshot")
        if not t or not inj: continue
        print(f"\n[{ds}]  (train_count bin: n | text | inject | Δ)")
        for (lo,hi),lab in zip(bins,labels):
            us=[u for u in t if inj.get(u) and (t[u]["train_count"] is not None) and lo<=t[u]["train_count"]<=hi]
            if len(us)<10: continue
            th=np.mean([hit(t[u]) for u in us]); ih=np.mean([hit(inj[u]) for u in us])
            print(f"   {lab:6s}: n={len(us):5d}  {th:.3f}  {ih:.3f}  {ih-th:+.3f}")

# ---------- 3. image-present vs missing ----------
def exp3():
    print("\n" + "="*78)
    print("EXP 3: vision benefit by whether the TARGET item has an image")
    print("="*78)
    for ds,cfg,_ in DS:
        t=load_recs(ds,"text_concat"); inj=load_recs(ds,"clip_inject_concat_vitl14_zeroshot")
        if not t or not inj: continue
        c=load_config(cfg, stage="inference")
        _,_,user_test,_,_=data_partition(c.interactions)
        im=np.load(f"data/clip/clip_image_{ds}_vitl14_zeroshot.npy")
        has=np.linalg.norm(im,axis=1)>1e-6
        def slice_delta(present):
            us=[u for u in t if inj.get(u) and user_test.get(u) and bool(has[user_test[u][0]])==present]
            if len(us)<10: return None
            return len(us), np.mean([hit(t[u]) for u in us]), np.mean([hit(inj[u]) for u in us])
        print(f"\n[{ds}]  (target: n | text | inject | Δ)")
        for present,lab in [(True,"image-PRESENT"),(False,"image-MISSING")]:
            r=slice_delta(present)
            if r: print(f"   {lab:14s}: n={r[0]:5d}  {r[1]:.3f}  {r[2]:.3f}  {r[2]-r[1]:+.3f}")

if __name__=="__main__":
    exp1(); exp2(); exp3()
