"""
SASRec-only baseline, evaluated the SAME way as our LLM models so numbers are
directly comparable: same leave-one-out split, same 20-candidate set (1 positive +
19 negatives, built identically to the model's _make_candidates), same cold/warm
tagging. Hit@1 = SASRec ranks the ground-truth item #1 among the 20 candidates.

This is the collaborative-filtering FLOOR — how well pure CF does, sliced cold/warm.
Fast (no LLM, no training): scores via frozen SASRec item embeddings.

Usage:
  conda run -n ALLM-Rec python scripts/eval_sasrec.py --config configs/luxury_beauty_rq3.yaml --seed 0
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.config import load_config
from clam_rec.model.sasrec import RecSys
from clam_rec.data.partition import data_partition, tag_cold_warm
from clam_rec.eval.metrics import evaluate_sliced


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_candidates(interact_ids, num, target_id, item_num):
    """Replicate clam_rec._make_candidates EXACTLY (same RNG calls -> same negatives
    under the same seed as infer.py)."""
    neg = []
    while len(neg) < 50:
        t = np.random.randint(1, item_num + 1)
        if t not in interact_ids and t not in neg:
            neg.append(t)
    random.shuffle(neg)
    cand_ids = [target_id]
    for c in neg[:num - 1]:
        cand_ids.append(c)
    perm = np.random.permutation(len(cand_ids))
    return np.array(cand_ids)[perm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    cfg = load_config(args.config, device=args.device)
    set_seed(args.seed)

    recsys = RecSys(cfg.sasrec_checkpoint, cfg.device)
    sas = recsys.model
    sas.eval()
    item_num = recsys.item_num
    maxlen = cfg.maxlen

    user_train, user_valid, user_test, usernum, itemnum = data_partition(cfg.interactions)
    tags, _ = tag_cold_warm(user_test, user_train, cold_threshold=cfg.cold_threshold)

    users = [u for u in range(1, usernum + 1)
             if len(user_train.get(u, [])) >= 1 and len(user_test.get(u, [])) >= 1]

    records = []
    with torch.no_grad():
        for u in users:
            # build eval sequence exactly like SeqDatasetInference: train + valid item
            seq = np.zeros([maxlen], dtype=np.int32)
            idx = maxlen - 1
            if user_valid[u]:
                seq[idx] = user_valid[u][0]; idx -= 1
            for i in reversed(user_train[u]):
                if idx < 0:
                    break
                seq[idx] = i; idx -= 1

            target = user_test[u][0]
            interact_ids = user_train[u]  # for negative exclusion (matches model: seq[seq>0])
            cand_ids = make_candidates(np.array(interact_ids), cfg.candidate_num, target, item_num)

            # score candidates with SASRec: log_feat (last position) · item_emb
            log_feat = sas.log2feats(seq[np.newaxis, :])[:, -1, :]           # (1, d)
            cand = torch.LongTensor(cand_ids).to(cfg.device)
            cand_emb = sas.item_emb(cand)                                    # (20, d)
            scores = (cand_emb @ log_feat.squeeze(0)).cpu().numpy()          # (20,)

            # Hit@1 = ground-truth item has the top score
            pred_id = int(cand_ids[int(scores.argmax())])
            hit = int(pred_id == target)

            tag = tags.get(u, {"cold": None, "train_count": None})
            # store as answer/generated STRINGS so we can reuse evaluate_sliced:
            # exact-match on the item id string == Hit@1.
            records.append({"user": u, "seed": args.seed,
                            "cold": tag["cold"], "train_count": tag["train_count"],
                            "answer": str(target), "generated": str(pred_id)})

    # write jsonl (comparable format) + report
    out_dir = Path("results") / "sasrec_only"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{args.seed}.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    res = evaluate_sliced(records, ks=(1,))  # exact id match = Hit@1
    print(f"\n=== SASRec-only (20-candidate Hit@1, seed {args.seed}) ===")
    for sl in ["overall", "cold", "warm"]:
        m = res[sl]
        print(f"  {sl:8s}: Hit@1={m['Hit@1']:.4f}  (n={m['count']})")
    print(f"\nwrote {len(records)} records -> {out_path}")


if __name__ == "__main__":
    main()
