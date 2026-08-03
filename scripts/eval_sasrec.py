"""
SASRec-only baseline eval (CF, no LLM).

Ranks the same 20-candidate set (1 positive + 19 negatives) by SASRec's
dot-product score and takes the top-1 as the prediction. Writes JSONL records in
the SAME schema as scripts/infer.py so the identical sliced report / metrics
pipeline scores it (Hit@1, cold/warm) alongside the LLM variants.

This anchors "how much does the LLM (+vision) add over plain collaborative
filtering." SASRec is frozen/pretrained -> no training, inference only.

Usage:
  conda run -n ALLM-Rec python scripts/eval_sasrec.py --config configs/luxury_beauty.yaml --seed 0
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.config import load_config
from clam_rec.model.sasrec import RecSys
from clam_rec.data.partition import data_partition, tag_cold_warm
from clam_rec.data.seq_dataset import SeqDatasetInference


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_candidates(history_ids, target_id, item_num, num):
    """1 positive + (num-1) negatives not in the user's history (mirrors the
    in-model candidate protocol used by the LLM variants)."""
    neg = []
    hist = set(history_ids.tolist()) if hasattr(history_ids, "tolist") else set(history_ids)
    while len(neg) < num - 1:
        t = np.random.randint(1, item_num + 1)
        if t not in hist and t != target_id and t not in neg:
            neg.append(t)
    cands = np.array([target_id] + neg)
    return cands[np.random.permutation(len(cands))]


def title_of(text_name_dict, item):
    return text_name_dict["title"].get(item, "No Title")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max_users", type=int, default=0, help="limit users (smoke test only)")
    args = ap.parse_args()

    cfg = load_config(args.config, stage="inference")
    set_seed(args.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    with open(cfg.text_name_dict, "rb") as f:
        text_name_dict = pickle.load(f)
    recsys = RecSys(cfg.sasrec_checkpoint, device)
    item_emb = recsys.model.item_emb  # (item_num+1, hidden), frozen

    user_train, user_valid, user_test, usernum, itemnum = data_partition(cfg.interactions)
    tags, _ = tag_cold_warm(user_test, user_train, cold_threshold=cfg.cold_threshold)
    users = [u for u in range(1, usernum + 1)
             if len(user_train.get(u, [])) >= 1 and len(user_test.get(u, [])) >= 1]
    if args.max_users:
        users = users[:args.max_users]

    ds = SeqDatasetInference(user_train, user_valid, user_test, users, itemnum, cfg.maxlen)
    loader = DataLoader(ds, batch_size=cfg.batch_size_infer, pin_memory=True)

    out_dir = Path("results") / f"{cfg.dataset}_sasrec"   # namespaced per dataset (uniform {dataset}_... scheme)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{args.seed}.jsonl"

    n = 0
    with torch.no_grad(), open(out_path, "w", encoding="utf-8") as f:
        for data in loader:
            u, seq, pos, neg = (x.numpy() for x in data)
            log_emb = recsys.model(u, seq, pos, neg, mode="log_only")  # (B, hidden)
            for j in range(len(u)):
                uid = int(u[j])
                target = int(pos[j])
                cands = sample_candidates(seq[j][seq[j] > 0], target, itemnum, cfg.candidate_num)
                cand_t = item_emb(torch.LongTensor(cands).to(device))       # (C, hidden)
                scores = (log_emb[j].unsqueeze(0) * cand_t).sum(-1)          # (C,)
                order = torch.argsort(scores, descending=True).cpu().numpy()  # rank candidates
                ranked = [title_of(text_name_dict, int(cands[o])) for o in order]
                tag = tags.get(uid, {"cold": None, "train_count": None})
                rec = {"user": uid, "seed": args.seed,
                       "cold": tag["cold"], "train_count": tag["train_count"],
                       "answer": title_of(text_name_dict, target),
                       "generated": ranked[0], "ranked": ranked}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} records -> {out_path}")


if __name__ == "__main__":
    main()
