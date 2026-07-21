"""
Clean CLIP extraction + alignment + fusion, rewritten from scratch.

Produces, for a dataset with a VERIFIED itemmap:
  - clip_text_{dataset}.npy   (itemnum+1, D)  L2-normalized text embeddings
  - clip_image_{dataset}.npy  (itemnum+1, D)  L2-normalized image embeddings
  - clip_fused_{dataset}.npy  (itemnum+1, 2D) [text || image], each half L2-normed
Row 0 is padding (zeros). Items with no image get a ZERO image half (paper's
zero-vector fallback). Rows are aligned to SASRec item ids via itemid_to_asin.

Text per item (paper Eq. 1): prefix + title (+ brand) (+ description), HTML-unescaped,
whitespace-collapsed. open_clip's tokenizer truncates to the model's context (77).

Usage:
  conda run -n ALLM-Rec python -m clam_rec.clip.extract \
    --dataset Luxury_Beauty \
    --itemmap data/processed/Luxury_Beauty_itemmap.pkl \
    --metadata /home/kavach/Dev/Extension-Paper/A-LLMRec/data/amazon/meta_Luxury_Beauty.json \
    --images_dir data/images/Luxury_Beauty \
    --out_dir data/clip \
    --model_id "hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
    --batch_size 128 --fp16
"""

import argparse
import html
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip


# --------------------------------------------------------------------------- #
# text construction
# --------------------------------------------------------------------------- #
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(str(s))
    s = re.sub(r"<[^>]+>", " ", s)          # strip stray HTML tags
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_item_text(meta: dict, prefix: str) -> str:
    title = clean_text(meta.get("title", ""))
    brand = clean_text(meta.get("brand", ""))
    desc = meta.get("description", "")
    if isinstance(desc, list):
        desc = desc[0] if desc else ""
    desc = clean_text(desc)
    parts = [p for p in [title, brand, desc] if p]
    body = ", ".join(parts) if parts else "product"
    return f"{prefix}{body}"


def load_metadata(path: str) -> dict:
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta[obj["asin"]] = obj
    return meta


# --------------------------------------------------------------------------- #
# encoders
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size, fp16):
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="text"):
        toks = tokenizer(texts[i:i + batch_size]).to(device)
        if fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                feats = model.encode_text(toks)
        else:
            feats = model.encode_text(toks)
        feats = F.normalize(feats.float(), p=2, dim=-1)
        out.append(feats.cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def encode_images(model, preprocess, asins, images_dir, device, batch_size, fp16):
    """Encode images for the asins that have a valid file; return dict asin->emb."""
    images_dir = Path(images_dir)
    valid = [(a, images_dir / f"{a}.jpg") for a in asins
             if (images_dir / f"{a}.jpg").exists()]
    emb_map = {}
    for i in tqdm(range(0, len(valid), batch_size), desc="image"):
        chunk = valid[i:i + batch_size]
        tensors, keep = [], []
        for asin, fp in chunk:
            try:
                img = Image.open(fp).convert("RGB")
                tensors.append(preprocess(img))
                keep.append(asin)
            except Exception:
                continue
        if not tensors:
            continue
        batch = torch.stack(tensors).to(device)
        if fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=torch.float16):
                feats = model.encode_image(batch)
        else:
            feats = model.encode_image(batch)
        feats = F.normalize(feats.float(), p=2, dim=-1)
        feats = feats.cpu().numpy()
        for asin, e in zip(keep, feats):
            emb_map[asin] = e
    return emb_map


# --------------------------------------------------------------------------- #
# alignment + fusion
# --------------------------------------------------------------------------- #
def align(itemid_to_asin, dim, text_map, image_map):
    """Build (itemnum+1, dim) text & image arrays; row 0 = pad zeros."""
    n = max(itemid_to_asin) + 1
    text = np.zeros((n, dim), dtype=np.float32)
    image = np.zeros((n, dim), dtype=np.float32)
    n_txt = n_img = 0
    for item_id, asin in itemid_to_asin.items():
        if asin in text_map:
            text[item_id] = text_map[asin]
            n_txt += 1
        if asin in image_map:
            image[item_id] = image_map[asin]
            n_img += 1
    return text, image, n_txt, n_img


def fuse_concat(text, image):
    """[text || image]; each half already L2-normed. Missing image half stays 0."""
    return np.concatenate([text, image], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--itemmap", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", default="hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    ap.add_argument("--prefix", default="a product photo of ")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.itemmap, "rb") as f:
        itemmap = pickle.load(f)
    itemid_to_asin = itemmap["id_to_asin"]
    asins = [itemid_to_asin[i] for i in sorted(itemid_to_asin)]
    print(f"items: {len(asins)}")

    meta = load_metadata(args.metadata)
    texts = [build_item_text(meta.get(a, {}), args.prefix) for a in asins]
    print("sample text:", texts[0][:120])

    print(f"loading CLIP: {args.model_id}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_id)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model_id)
    with torch.no_grad():
        dim = model.encode_text(tokenizer(["x"]).to(device)).shape[-1]
    print(f"embedding dim: {dim}")

    text_emb = encode_texts(model, tokenizer, texts, device, args.batch_size, args.fp16)
    text_map = {a: e for a, e in zip(asins, text_emb)}
    image_map = encode_images(model, preprocess, asins, args.images_dir,
                              device, args.batch_size, args.fp16)

    text_aligned, image_aligned, n_txt, n_img = align(itemid_to_asin, dim, text_map, image_map)
    fused = fuse_concat(text_aligned, image_aligned)

    ntot = len(asins)
    print(f"text coverage:  {n_txt}/{ntot} ({100*n_txt/ntot:.1f}%)")
    print(f"image coverage: {n_img}/{ntot} ({100*n_img/ntot:.1f}%)")
    print(f"fused shape: {fused.shape}  (expect ({ntot+1}, {2*dim}))")

    np.save(out_dir / f"clip_text_{args.dataset}.npy", text_aligned)
    np.save(out_dir / f"clip_image_{args.dataset}.npy", image_aligned)
    fused_path = out_dir / f"clip_fused_{args.dataset}.npy"
    np.save(fused_path, fused)
    print(f"saved: {fused_path}")

    meta_out = {
        "dataset": args.dataset, "model_id": args.model_id, "dim": int(dim),
        "prefix": args.prefix, "n_items": ntot,
        "text_coverage": n_txt, "image_coverage": n_img,
        "fused_shape": list(fused.shape),
    }
    with open(out_dir / f"clip_meta_{args.dataset}.json", "w") as f:
        json.dump(meta_out, f, indent=2)


if __name__ == "__main__":
    main()
