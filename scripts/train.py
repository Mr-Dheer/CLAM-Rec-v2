"""
Train a CLAM-Rec variant (Stage 1 then Stage 2). Run on a GPU with >=24GB for
the OPT-6.7B Stage 2 (A6000). Stage 1 alone fits on smaller GPUs.

Usage:
  conda run -n ALLM-Rec python scripts/train.py --config configs/luxury_beauty.yaml \
    --variant clip_inject --fusion concat --stage 1
  conda run -n ALLM-Rec python scripts/train.py --config configs/luxury_beauty.yaml \
    --variant clip_inject --fusion concat --stage 2
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.config import load_config
from clam_rec.model.clam_rec import ClamRec
from clam_rec.data.partition import data_partition
from clam_rec.data.seq_dataset import SeqDataset


def run_name(cfg):
    """Unique run id incorporating variant, fusion, and (if not bigG) the CLIP set."""
    name = f"{cfg.variant}_{cfg.fusion}"
    if getattr(cfg, "clip_variant", "bigG") != "bigG":
        name += f"_{cfg.clip_variant}"
    return name


def ckpt_prefix(cfg, stage):
    d = Path("results/checkpoints") / run_name(cfg)
    d.mkdir(parents=True, exist_ok=True)
    return str(d) + "/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--fusion", default=None)
    ap.add_argument("--clip_variant", default=None, help="bigG | vitl14_zeroshot | vitl14_ft")
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2])
    args = ap.parse_args()

    over = {"stage": args.stage}
    if args.variant:
        over["variant"] = args.variant
    if args.fusion:
        over["fusion"] = args.fusion
    if args.clip_variant:
        over["clip_variant"] = args.clip_variant
    cfg = load_config(args.config, **over)

    model = ClamRec(cfg).to(cfg.device)
    prefix = ckpt_prefix(cfg, args.stage)

    part = data_partition(cfg.interactions)
    user_train, _, _, usernum, itemnum = part
    print(f"users={usernum} items={itemnum} variant={cfg.variant} fusion={cfg.fusion} stage={args.stage}")

    bs = cfg.batch_size1 if args.stage == 1 else cfg.batch_size2
    ds = SeqDataset(user_train, usernum, itemnum, cfg.maxlen)
    loader = DataLoader(ds, batch_size=bs, shuffle=(args.stage == 2), pin_memory=True)
    num_batch = len(user_train) // bs

    if args.stage == 2:
        model.load_stage1(prefix, freeze=True)

    lr = cfg.stage1_lr if args.stage == 1 else cfg.stage2_lr
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr, betas=(0.9, 0.98))
    model.train()
    t0 = time.time()
    for epoch in range(1, cfg.num_epochs + 1):
        for step, data in enumerate(loader):
            u, seq, pos, neg = (x.numpy() for x in data)
            bi = [epoch, cfg.num_epochs, step, num_batch]
            if args.stage == 1:
                model.stage1_step([u, seq, pos, neg], opt, bi)
            else:
                model.stage2_step([u, seq, pos, neg], opt, bi)
        if args.stage == 1:
            model.save_stage1(prefix)
        else:
            model.save_stage2(prefix)
    print(f"stage{args.stage} done in {time.time()-t0:.1f}s -> {prefix}")


if __name__ == "__main__":
    main()
