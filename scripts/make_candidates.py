"""
Generate a SHARED candidate set per user (1 positive + N-1 negatives), independent
of any model, so SASRec and the LLM rank the *identical* candidates -> enables honest
score/rank fusion (the earlier RRF was bogus because each model sampled its own set).

Output: data/candidates/{dataset}_seed{seed}.json  =  {user: {"target": id, "cands": [ids]}}
Negatives are sampled from items NOT in the user's train history and != target
(mirrors the in-model / eval_sasrec exclusion). Deterministic given --seed.

Usage: python scripts/make_candidates.py --config configs/luxury_beauty_rq3.yaml --seed 0
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.config import load_config
from clam_rec.data.partition import data_partition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num", type=int, default=0, help="override candidate count (default = cfg.candidate_num)")
    args = ap.parse_args()
    cfg = load_config(args.config, stage="inference")
    num = args.num or cfg.candidate_num
    rng = np.random.default_rng(args.seed)

    user_train, user_valid, user_test, usernum, itemnum = data_partition(cfg.interactions)
    users = [u for u in range(1, usernum + 1)
             if len(user_train.get(u, [])) >= 1 and len(user_test.get(u, [])) >= 1]

    out = {}
    for u in users:                                   # sorted -> deterministic
        target = int(user_test[u][0])
        hist = set(int(x) for x in user_train.get(u, [])) | {target}
        neg = set()
        while len(neg) < num - 1:
            t = int(rng.integers(1, itemnum + 1))
            if t not in hist:
                neg.add(t)
        cands = np.array([target] + list(neg))
        cands = cands[rng.permutation(len(cands))].tolist()
        out[str(u)] = {"target": target, "cands": cands}

    out_dir = Path("data/candidates"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.dataset}_seed{args.seed}_n{num}.json"
    json.dump(out, open(out_path, "w"))
    print(f"wrote {len(out)} users x {num} candidates -> {out_path}")


if __name__ == "__main__":
    main()
