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
    ap.add_argument("--clip_variant", default=None, help="bigG | vitl14_zeroshot | vitl14_ft")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max_users", type=int, default=0, help="limit users (smoke test only)")
    ap.add_argument("--rank", action="store_true",
                    help="likelihood-ranking eval: rank the candidate set -> logs 'ranked' for Hit@1/5, NDCG@5")
    ap.add_argument("--rank_chunk", type=int, default=0, help="candidates scored per forward (0=default 8; use 20 on a free GPU)")
    ap.add_argument("--candidates_file", default=None, help="shared candidate pool json (for fusion); logs target_id/candidate_ids/scores")
    ap.add_argument("--out_tag", default=None, help="override output dir suffix (e.g. smoke100); avoids clobbering _shared runs")
    ap.add_argument("--nshards", type=int, default=1, help="split users across N GPUs (each shard a separate process)")
    ap.add_argument("--shard", type=int, default=0, help="which shard this process handles (0..nshards-1)")
    args = ap.parse_args()

    over = {"stage": "inference"}
    if args.rank_chunk:
        over["rank_chunk"] = args.rank_chunk
    if args.variant:
        over["variant"] = args.variant
    if args.fusion:
        over["fusion"] = args.fusion
    if args.clip_variant:
        over["clip_variant"] = args.clip_variant
    cfg = load_config(args.config, **over)
    set_seed(args.seed)

    run = f"{cfg.dataset}_{cfg.variant}_{cfg.fusion}"   # dataset prefix: avoid cross-dataset collisions
    if getattr(cfg, "clip_variant", "bigG") != "bigG":
        run += f"_{cfg.clip_variant}"

    model = ClamRec(cfg).to(cfg.device)
    prefix = f"results/checkpoints/{run}/"
    model.load_stage1(prefix, freeze=True)
    model.load_stage2(prefix)
    if args.candidates_file:
        model.load_shared_candidates(args.candidates_file)
    model.eval()

    user_train, user_valid, user_test, usernum, itemnum = data_partition(cfg.interactions)
    tags, _ = tag_cold_warm(user_test, user_train, cold_threshold=cfg.cold_threshold)

    users = [u for u in range(1, usernum + 1)
             if len(user_train.get(u, [])) >= 1 and len(user_test.get(u, [])) >= 1]
    if args.max_users:                      # for quick smoke tests only
        users = users[:args.max_users]
    if args.nshards > 1:                     # multi-GPU: strided user split (balanced)
        users = users[args.shard::args.nshards]
    ds = SeqDatasetInference(user_train, user_valid, user_test, users, itemnum, cfg.maxlen)
    loader = DataLoader(ds, batch_size=cfg.batch_size_infer, pin_memory=True)

    if args.out_tag:
        out_run = f"{run}_{args.out_tag}"
    else:
        out_run = run + "_shared" if args.candidates_file else run   # keep fusion runs separate
    out_dir = Path("results") / out_run
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.nshards > 1:
        out_path = out_dir / f"seed_{args.seed}.part{args.shard}of{args.nshards}.jsonl"
    else:
        out_path = out_dir / f"seed_{args.seed}.jsonl"

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for data in loader:
            u, seq, pos, neg = (x.numpy() for x in data)
            extras = None
            if args.rank:
                gen, ans, ranked, extras = model.rank_candidates([u, seq, pos, neg])
            else:
                gen, ans = model.generate([u, seq, pos, neg])
            for j in range(len(u)):
                uid = int(u[j])
                tag = tags.get(uid, {"cold": None, "train_count": None})
                rec = {"user": uid, "seed": args.seed,
                       "cold": tag["cold"], "train_count": tag["train_count"],
                       "answer": ans[j], "generated": gen[j]}
                if args.rank:
                    rec["ranked"] = ranked[j]
                    if args.candidates_file:
                        rec.update(extras[j])   # target_id, candidate_ids, scores
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} records -> {out_path}")


if __name__ == "__main__":
    main()
