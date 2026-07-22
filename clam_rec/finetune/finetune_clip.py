"""
Domain image-text contrastive fine-tuning of CLIP ViT-L/14 with LoRA.

Fits on a 16GB GPU. Trains LoRA adapters on both the vision and text towers with a
symmetric InfoNCE loss over item (image, text) pairs. Splits BY ITEM so validation
items are unseen. Early-stops on validation image->text retrieval Recall@1.

Usage:
  conda run -n ALLM-Rec python -m clam_rec.finetune.finetune_clip \
    --itemmap data/processed/Luxury_Beauty_itemmap.pkl \
    --metadata /home/kavach/Dev/Extension-Paper/A-LLMRec/data/amazon/meta_Luxury_Beauty.json \
    --images_dir data/images/Luxury_Beauty \
    --out_dir results/finetune_vitl14 \
    --model ViT-L-14 --pretrained laion2b_s32b_b82k \
    --epochs 20 --batch_size 128 --lr 5e-5 --lora_r 8 --val_frac 0.15
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import open_clip
from peft import LoraConfig, get_peft_model

from clam_rec.finetune.pairs_dataset import (
    build_pair_index, split_by_item, ClipPairDataset, collate)


def find_lora_targets(model):
    """Target the attention in/out projections in both towers."""
    names = set()
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            leaf = n.split(".")[-1]
            if leaf in ("out_proj", "c_fc", "c_proj"):
                names.add(leaf)
    # attention qkv is packed as in_proj_weight (not Linear); we adapt out_proj +
    # the MLP (c_fc/c_proj), which is a common, stable LoRA target set for CLIP.
    return sorted(names)


@torch.no_grad()
def encode_val(model, loader, device):
    model.eval()
    imgs, txts, ids = [], [], []
    for image, text, item_id in loader:
        image, text = image.to(device), text.to(device)
        with torch.cuda.amp.autocast():
            ie = F.normalize(model.encode_image(image).float(), dim=-1)
            te = F.normalize(model.encode_text(text).float(), dim=-1)
        imgs.append(ie.cpu()); txts.append(te.cpu()); ids.append(item_id)
    return torch.cat(imgs), torch.cat(txts), torch.cat(ids)


def retrieval_recall(img_emb, txt_emb, ks=(1, 5, 10)):
    """image->text retrieval: for each image, rank all val texts; correct = same index."""
    sims = img_emb @ txt_emb.t()                     # (N, N)
    ranks = sims.argsort(dim=1, descending=True)
    n = sims.size(0)
    gt = torch.arange(n).unsqueeze(1)
    pos = (ranks == gt).nonzero()[:, 1]              # rank position of the true text
    out = {}
    for k in ks:
        out[f"R@{k}"] = float((pos < k).float().mean())
    out["median_rank"] = float(pos.float().median() + 1)
    return out, pos


def contrastive_loss(image_emb, text_emb, logit_scale):
    logits = logit_scale * image_emb @ text_emb.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--itemmap", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b82k")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad_checkpointing", action="store_true", default=True)
    ap.add_argument("--no_grad_checkpointing", dest="grad_checkpointing", action="store_false")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data -----------------------------------------------------------
    pairs = build_pair_index(args.itemmap, args.metadata, args.images_dir)
    train_pairs, val_pairs = split_by_item(pairs, args.val_frac, args.seed)
    print(f"pairs: total={len(pairs)} train={len(train_pairs)} val={len(val_pairs)}")

    # ---- model ----------------------------------------------------------
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device)
    # gradient checkpointing: recompute activations in backward -> big memory save,
    # lets us fine-tune ViT-L/14 (both towers) with a useful batch size on 16GB.
    if args.grad_checkpointing:
        try:
            model.set_grad_checkpointing(True)
            print("gradient checkpointing: ON")
        except Exception as e:
            print(f"grad checkpointing unavailable: {e}")

    targets = find_lora_targets(model)
    print("LoRA target modules:", targets)
    lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                          target_modules=targets, lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    logit_scale = model.logit_scale.exp().detach()   # keep CLIP temperature fixed

    train_ds = ClipPairDataset(train_pairs, preprocess, tokenizer)
    val_ds = ClipPairDataset(val_pairs, preprocess, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    # ---- zero-shot baseline on val -------------------------------------
    ie, te, _ = encode_val(model, val_loader, device)
    base_metrics, _ = retrieval_recall(ie, te)
    print(f"[val zero-shot] {base_metrics}")

    best = base_metrics["R@1"]; best_ep = 0; bad = 0
    history = [{"epoch": 0, "split": "val_zeroshot", **base_metrics}]

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0; nb = 0
        for image, text, _ in train_loader:
            image, text = image.to(device), text.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                ie = F.normalize(model.encode_image(image).float(), dim=-1)
                te = F.normalize(model.encode_text(text).float(), dim=-1)
                loss = contrastive_loss(ie, te, logit_scale)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1

        ie, te, _ = encode_val(model, val_loader, device)
        vm, _ = retrieval_recall(ie, te)
        print(f"[ep {ep}] train_loss={tot/nb:.4f}  val {vm}")
        history.append({"epoch": ep, "split": "val", "train_loss": tot/nb, **vm})

        if vm["R@1"] > best:
            best = vm["R@1"]; best_ep = ep; bad = 0
            model.save_pretrained(str(out_dir / "lora_best"))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at ep {ep} (best R@1={best:.4f} @ ep {best_ep})")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), "history": history,
                   "best_R@1": best, "best_epoch": best_ep,
                   "zeroshot_R@1": base_metrics["R@1"]}, f, indent=2)
    print(f"\nDONE. zero-shot R@1={base_metrics['R@1']:.4f} -> best R@1={best:.4f} "
          f"(ep {best_ep}). adapters: {out_dir/'lora_best'}")


if __name__ == "__main__":
    main()
