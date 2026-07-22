"""
Extract aligned CLIP embeddings with ViT-L/14, either zero-shot or with the
domain fine-tuned LoRA adapters. Produces the fair same-model pair for RQ3.

Reuses the alignment + fusion logic from clam_rec.clip.extract so rows line up
with SASRec item ids identically to the bigG pipeline.

Usage:
  # zero-shot
  conda run -n ALLM-Rec python -m clam_rec.finetune.extract_vitl14 \
    --tag vitl14_zeroshot --out_dir data/clip
  # fine-tuned (LoRA)
  conda run -n ALLM-Rec python -m clam_rec.finetune.extract_vitl14 \
    --tag vitl14_ft --lora results/finetune_vitl14/lora_best --out_dir data/clip
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

import open_clip
from peft import PeftModel

from clam_rec.clip.extract import (
    build_item_text, load_metadata, encode_texts, encode_images,
    align, fuse_concat)

DEFAULTS = dict(
    itemmap="data/processed/Luxury_Beauty_itemmap.pkl",
    metadata="/home/kavach/Dev/Extension-Paper/A-LLMRec/data/amazon/meta_Luxury_Beauty.json",
    images_dir="data/images/Luxury_Beauty",
    dataset="Luxury_Beauty",
    model="ViT-L-14",
    pretrained="laion2b_s32b_b82k",
    prefix="a product photo of ",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="e.g. vitl14_zeroshot or vitl14_ft")
    ap.add_argument("--lora", default=None, help="path to LoRA adapter dir (fine-tuned)")
    ap.add_argument("--out_dir", default="data/clip")
    ap.add_argument("--itemmap", default=DEFAULTS["itemmap"])
    ap.add_argument("--metadata", default=DEFAULTS["metadata"])
    ap.add_argument("--images_dir", default=DEFAULTS["images_dir"])
    ap.add_argument("--dataset", default=DEFAULTS["dataset"])
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--pretrained", default=DEFAULTS["pretrained"])
    ap.add_argument("--prefix", default=DEFAULTS["prefix"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.itemmap, "rb") as f:
        itemmap = pickle.load(f)
    id_to_asin = itemmap["id_to_asin"]
    asins = [id_to_asin[i] for i in sorted(id_to_asin)]

    meta = load_metadata(args.metadata)
    texts = [build_item_text(meta.get(a, {}), args.prefix) for a in asins]

    print(f"loading {args.model} / {args.pretrained}"
          + (f" + LoRA {args.lora}" if args.lora else " (zero-shot)"))
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained)
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
        model = model.merge_and_unload()   # bake LoRA into weights for clean inference
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    with torch.no_grad():
        dim = model.encode_text(tokenizer(["x"]).to(device)).shape[-1]
    print(f"embedding dim: {dim}")

    text_emb = encode_texts(model, tokenizer, texts, device, args.batch_size, args.fp16)
    text_map = {a: e for a, e in zip(asins, text_emb)}
    image_map = encode_images(model, preprocess, asins, args.images_dir,
                              device, args.batch_size, args.fp16)

    text_aligned, image_aligned, n_txt, n_img = align(id_to_asin, dim, text_map, image_map)
    fused = fuse_concat(text_aligned, image_aligned)
    ntot = len(asins)
    print(f"text cov {n_txt}/{ntot}  image cov {n_img}/{ntot}  fused {fused.shape}")

    suffix = f"{args.dataset}_{args.tag}"
    np.save(out_dir / f"clip_text_{suffix}.npy", text_aligned)
    np.save(out_dir / f"clip_image_{suffix}.npy", image_aligned)
    np.save(out_dir / f"clip_fused_{suffix}.npy", fused)
    with open(out_dir / f"clip_meta_{suffix}.json", "w") as f:
        json.dump({"tag": args.tag, "model": args.model, "pretrained": args.pretrained,
                   "lora": args.lora, "dim": int(dim), "n_items": ntot,
                   "text_coverage": n_txt, "image_coverage": n_img,
                   "fused_shape": list(fused.shape)}, f, indent=2)
    print(f"saved -> {out_dir}/clip_fused_{suffix}.npy")


if __name__ == "__main__":
    main()
