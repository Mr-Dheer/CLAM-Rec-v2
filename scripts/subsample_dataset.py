"""
Create a principled random-user subsample of a (preprocessed) dataset so a big
category becomes tractable for OPT-6.7B. Method (report this in the paper):
  1. take a random subset of users (fixed seed),
  2. iterative 5-core: drop users with <k items and items with <k users until stable,
  3. reindex users/items to 1..N; keep item->asin (from the full itemmap) so images
     and CLIP rows still align.

Outputs (into out_dir): <name>_sub.txt, <name>_sub_itemmap.pkl (id->asin), and prints
the resulting size / image-URL coverage / cold-warm split so we can decide BEFORE training.

Usage:
  python scripts/subsample_dataset.py --name Toys_and_Games \
    --full_txt data/raw/analysis/Toys_and_Games/Toys_and_Games.txt \
    --full_itemmap data/raw/analysis/Toys_and_Games/Toys_and_Games_itemmap.pkl \
    --meta data/raw/meta_Toys_and_Games.json --frac 0.10 --seed 42 --out_dir data/raw/sub
"""
import argparse, json, pickle, random
from collections import defaultdict
from pathlib import Path


def kcore(user_items, k=5):
    """Iterative k-core on a {user:set(items)} dict until stable."""
    ui = {u: set(v) for u, v in user_items.items()}
    while True:
        item_users = defaultdict(set)
        for u, its in ui.items():
            for i in its: item_users[i].add(u)
        drop_items = {i for i, us in item_users.items() if len(us) < k}
        if drop_items:
            for u in list(ui): ui[u] -= drop_items
        ui = {u: its for u, its in ui.items() if len(its) >= k}
        # recompute item support after user drops
        item_users = defaultdict(set)
        for u, its in ui.items():
            for i in its: item_users[i].add(u)
        if not any(len(us) < k for us in item_users.values()) and \
           all(len(its) >= k for its in ui.values()):
            break
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--full_txt", required=True)
    ap.add_argument("--full_itemmap", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--frac", type=float, default=0.10, help="fraction of users to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kcore", type=int, default=5)
    ap.add_argument("--cold_threshold", type=int, default=5)
    ap.add_argument("--out_dir", default="data/raw/sub")
    args = ap.parse_args()

    # load full interactions (chronological order preserved per user)
    user_seq = defaultdict(list)
    with open(args.full_txt) as f:
        for line in f:
            u, i = line.split(); user_seq[int(u)].append(int(i))
    users = list(user_seq)
    rng = random.Random(args.seed); rng.shuffle(users)
    keep = set(users[:int(len(users) * args.frac)])
    sub = {u: user_seq[u] for u in keep}
    core = kcore({u: set(v) for u, v in sub.items()}, args.kcore)   # stable id sets
    # keep chronological order, only surviving items
    sub = {u: [i for i in user_seq[u] if i in core[u]] for u in core}

    # reindex users (order) + items (first-appearance)
    old_id_to_asin = pickle.load(open(args.full_itemmap, "rb"))["id_to_asin"]
    umap, imap = {}, {}
    lines = []
    for u in sorted(sub):
        if len(sub[u]) < args.kcore: continue
        umap[u] = len(umap) + 1
        for i in sub[u]:
            if i not in imap: imap[i] = len(imap) + 1
            lines.append((umap[u], imap[i]))
    out = Path(args.out_dir) / args.name; out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.name}_sub.txt", "w") as f:
        for u, i in lines: f.write(f"{u} {i}\n")
    new_id_to_asin = {new: old_id_to_asin[old] for old, new in imap.items()}
    pickle.dump({"id_to_asin": new_id_to_asin,
                 "asin_to_id": {a: i for i, a in new_id_to_asin.items()}},
                open(out / f"{args.name}_sub_itemmap.pkl", "wb"))

    usernum, itemnum, inter = len(umap), len(imap), len(lines)
    # cold/warm on train portion (leave-one-out: last=test, 2nd-last=valid)
    from collections import Counter
    train_cnt = Counter()
    per_user = defaultdict(list)
    for u, i in lines: per_user[u].append(i)
    cold = warm = 0
    for u, seq in per_user.items():
        if len(seq) < 3: continue
        for i in seq[:-2]: train_cnt[i] += 1
    for u, seq in per_user.items():
        if len(seq) < 3: continue
        tgt = seq[-1]
        (cold := cold + 1) if train_cnt.get(tgt, 0) <= args.cold_threshold else (warm := warm + 1)
    # image URL coverage of the subsample items
    asins = set(new_id_to_asin.values()); have = 0
    with open(args.meta) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: o = json.loads(line)
            except: continue
            if o["asin"] in asins and (o.get("imageURLHighRes") or o.get("imageURL")): have += 1

    n_test = cold + warm
    print(f"\n=== SUBSAMPLE {args.name} (frac={args.frac}, seed={args.seed}) ===")
    print(f"users={usernum:,} items={itemnum:,} interactions={inter:,} avg_seq={inter/usernum:.2f}")
    print(f"image URL coverage: {have:,}/{itemnum:,} = {100*have/itemnum:.1f}%")
    print(f"cold/warm (thr {args.cold_threshold}): cold={cold:,} ({100*cold/max(n_test,1):.1f}%) warm={warm:,}")
    print(f"wrote -> {out}/{args.name}_sub.txt")


if __name__ == "__main__":
    main()
