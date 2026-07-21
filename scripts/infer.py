"""
Run inference for a trained CLAM-Rec variant over one seed, writing a clean
JSONL log (one record per test case) with cold/warm metadata for sliced eval.

Fixes the old pipeline's append-mode bug: each seed writes its own file in
WRITE mode.

Usage:
  conda run -n ALLM-Rec python scripts/infer.py --config configs/luxury_beauty.yaml \
    --variant clip_inject --fusion concat --seed 0
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.config import load_config
from clam_rec.model.clam_rec import ClamRec
from clam_rec.data.partition import data_partition, tag_cold_warm
from clam_rec.data.seq_dataset import SeqDatasetInference


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--fusion", default=None)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    over = {"stage": "inference"}
    if args.variant:
        over["variant"] = args.variant
    if args.fusion:
        over["fusion"] = args.fusion
    cfg = load_config(args.config, **over)
    set_seed(args.seed)

    model = ClamRec(cfg).to(cfg.device)
    prefix = f"results/checkpoints/{cfg.variant}_{cfg.fusion}/"
    model.load_stage1(prefix, freeze=True)
    model.load_stage2(prefix)
    model.eval()

    user_train, user_valid, user_test, usernum, itemnum = data_partition(cfg.interactions)
    tags, _ = tag_cold_warm(user_test, user_train, cold_threshold=cfg.cold_threshold)

    users = [u for u in range(1, usernum + 1)
             if len(user_train.get(u, [])) >= 1 and len(user_test.get(u, [])) >= 1]
    ds = SeqDatasetInference(user_train, user_valid, user_test, users, itemnum, cfg.maxlen)
    loader = DataLoader(ds, batch_size=cfg.batch_size_infer, pin_memory=True)

    out_dir = Path("results") / f"{cfg.variant}_{cfg.fusion}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"seed_{args.seed}.jsonl"

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for data in loader:
            u, seq, pos, neg = (x.numpy() for x in data)
            gen, ans = model.generate([u, seq, pos, neg])
            for j in range(len(u)):
                uid = int(u[j])
                tag = tags.get(uid, {"cold": None, "train_count": None})
                rec = {"user": uid, "seed": args.seed,
                       "cold": tag["cold"], "train_count": tag["train_count"],
                       "answer": ans[j], "generated": gen[j]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} records -> {out_path}")


if __name__ == "__main__":
    main()
