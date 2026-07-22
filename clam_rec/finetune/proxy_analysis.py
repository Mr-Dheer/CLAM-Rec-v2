"""
Cold/warm-aware proxy analysis of CLIP domain fine-tuning (RQ1 x RQ3 preview).

We cannot measure recommendation impact without the LLM (needs the A6000), but we
CAN measure whether fine-tuning improves the *visual representation* more for cold
items than warm ones, using image->text retrieval on the held-out validation items
(the same by-item split used in fine-tuning, so these items were never trained on).

For each held-out item we ask: does its image retrieve its own text among all
held-out texts? We compute Recall@1/5/10 and median rank, sliced by whether the
item is cold (train interaction count <= threshold) or warm, for zero-shot vs
fine-tuned ViT-L/14. A larger fine-tuning gain on the cold slice is the proxy
signal that vision fine-tuning helps most where CF is weak.

Usage:
  conda run -n ALLM-Rec python -m clam_rec.finetune.proxy_analysis \
    --lora results/finetune_vitl14/lora_best --val_frac 0.15 --seed 0 --cold_threshold 5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import open_clip
from peft import PeftModel

from clam_rec.finetune.pairs_dataset import (
    build_pair_index, split_by_item, ClipPairDataset, collate)
from clam_rec.data.partition import data_partition, item_train_counts

DEF_ITEMMAP = "data/processed/Luxury_Beauty_itemmap.pkl"
DEF_META = "/home/kavach/Dev/Extension-Paper/A-LLMRec/data/amazon/meta_Luxury_Beauty.json"
DEF_IMAGES = "data/images/Luxury_Beauty"
DEF_INTERACT = "data/processed/Luxury_Beauty.txt"


@torch.no_grad()
def encode(model, loader, device):
    model.eval()
    imgs, txts, ids = [], [], []
    for image, text, item_id in loader:
        with torch.cuda.amp.autocast():
            ie = F.normalize(model.encode_image(image.to(device)).float(), dim=-1)
            te = F.normalize(model.encode_text(text.to(device)).float(), dim=-1)
        imgs.append(ie.cpu()); txts.append(te.cpu()); ids.append(item_id)
    return torch.cat(imgs), torch.cat(txts), torch.cat(ids)


def ranks_of(img_emb, txt_emb):
    sims = img_emb @ txt_emb.t()
    order = sims.argsort(dim=1, descending=True)
    gt = torch.arange(sims.size(0)).unsqueeze(1)
    return (order == gt).nonzero()[:, 1]     # 0-based rank of the true text


def metrics_for(pos, ks=(1, 5, 10)):
    out = {f"R@{k}": float((pos < k).float().mean()) for k in ks}
    out["median_rank"] = float(pos.float().median() + 1)
    out["n"] = int(pos.numel())
    return out


def load_model(model_name, pretrained, lora, device):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    if lora:
        model = PeftModel.from_pretrained(model, lora)
        model = model.merge_and_unload()
    return model.to(device).eval(), preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="laion2b_s32b_b82k")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cold_threshold", type=int, default=5)
    ap.add_argument("--out", default="results/finetune_vitl14/proxy_coldwarm.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = open_clip.get_tokenizer(args.model)

    # same held-out val split as fine-tuning
    pairs = build_pair_index(DEF_ITEMMAP, DEF_META, DEF_IMAGES)
    _, val_pairs = split_by_item(pairs, args.val_frac, args.seed)

    # cold/warm tag per item by TRAIN interaction count
    user_train, _, _, _, _ = data_partition(DEF_INTERACT)
    counts = item_train_counts(user_train)

    def encode_split(lora):
        model, preprocess = load_model(args.model, args.pretrained, lora, device)
        ds = ClipPairDataset(val_pairs, preprocess, tokenizer)
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate)
        ie, te, ids = encode(model, loader, device)
        del model; torch.cuda.empty_cache()
        return ie, te, ids

    zs_i, zs_t, ids = encode_split(None)
    ft_i, ft_t, _ = encode_split(args.lora)

    item_ids = ids.numpy()
    is_cold = np.array([counts.get(int(i), 0) <= args.cold_threshold for i in item_ids])

    def slice_metrics(img, txt, mask, shared_pool=False):
        # shared_pool=False: retrieve within the slice only (pool size = slice size).
        # shared_pool=True: query items in the slice, but retrieve against the FULL
        #   val text pool -> removes the pool-size confound so cold vs warm is a fair
        #   comparison (both face the same 829-way retrieval difficulty).
        idx = np.where(mask)[0]
        if len(idx) < 2:
            return None
        if shared_pool:
            sims = img[idx] @ txt.t()               # queries x FULL pool
            order = sims.argsort(dim=1, descending=True)
            gt = torch.tensor(idx).unsqueeze(1)
            pos = (order == gt).nonzero()[:, 1]
        else:
            pos = ranks_of(img[idx], txt[idx])
        return metrics_for(pos)

    result = {"cold_threshold": args.cold_threshold,
              "n_val": len(val_pairs),
              "n_cold": int(is_cold.sum()), "n_warm": int((~is_cold).sum())}
    for name, mask in [("overall", np.ones(len(item_ids), bool)),
                       ("cold", is_cold), ("warm", ~is_cold)]:
        # shared_pool=True for cold/warm so the comparison is not confounded by
        # differing pool sizes (598 cold vs 231 warm). overall uses full pool anyway.
        sp = name != "overall"
        zs = slice_metrics(zs_i, zs_t, mask, shared_pool=sp)
        ft = slice_metrics(ft_i, ft_t, mask, shared_pool=sp)
        result[name] = {"zeroshot": zs, "finetuned": ft, "shared_pool": sp,
                        "delta_R@1": (ft["R@1"] - zs["R@1"]) if (zs and ft) else None}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== PROXY: img->text retrieval, zero-shot vs fine-tuned (within-slice pool) ===")
    print(f"val items: {result['n_val']}  cold: {result['n_cold']}  warm: {result['n_warm']}")
    for name in ["overall", "cold", "warm"]:
        r = result[name]
        if r["zeroshot"] and r["finetuned"]:
            print(f"\n[{name}]  (n={r['zeroshot']['n']})")
            print(f"  zero-shot  R@1={r['zeroshot']['R@1']:.4f} R@5={r['zeroshot']['R@5']:.4f} R@10={r['zeroshot']['R@10']:.4f}")
            print(f"  fine-tuned R@1={r['finetuned']['R@1']:.4f} R@5={r['finetuned']['R@5']:.4f} R@10={r['finetuned']['R@10']:.4f}")
            print(f"  Δ R@1 = {r['delta_R@1']:+.4f}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
