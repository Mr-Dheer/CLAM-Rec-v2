"""
Item image-text pair dataset for domain fine-tuning of CLIP.

Each sample = (preprocessed image tensor, tokenized text) for one item that has
BOTH an image on disk and non-empty text. Split is BY ITEM (not by pair) so the
validation items are never seen in training -> we measure generalization, not
memorization (critical given only ~5.5k items).

Text is built the same way as the extraction pipeline (title + brand + desc,
HTML-cleaned) so fine-tuning aligns the same content the recommender will use.
"""

import json
import pickle
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from clam_rec.clip.extract import build_item_text, load_metadata


def build_pair_index(itemmap_path, metadata_path, images_dir, prefix="a product photo of "):
    """Return list of dicts: {item_id, asin, text, image_path} for items with both."""
    with open(itemmap_path, "rb") as f:
        itemmap = pickle.load(f)
    id_to_asin = itemmap["id_to_asin"]
    meta = load_metadata(metadata_path)
    images_dir = Path(images_dir)

    pairs = []
    for item_id in sorted(id_to_asin):
        asin = id_to_asin[item_id]
        img_path = images_dir / f"{asin}.jpg"
        if not img_path.exists():
            continue
        text = build_item_text(meta.get(asin, {}), prefix)
        if not text or text.strip() == prefix.strip():
            continue
        pairs.append({"item_id": item_id, "asin": asin,
                      "text": text, "image_path": str(img_path)})
    return pairs


def split_by_item(pairs, val_frac=0.15, seed=0):
    """Split the pair list into train/val by item (disjoint item sets)."""
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val_ids = set(idx[:n_val])
    train = [pairs[i] for i in idx if i not in val_ids]
    val = [pairs[i] for i in idx if i in val_ids]
    return train, val


class ClipPairDataset(Dataset):
    def __init__(self, pairs, preprocess, tokenizer):
        self.pairs = pairs
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        img = Image.open(p["image_path"]).convert("RGB")
        image = self.preprocess(img)
        text = self.tokenizer([p["text"]])[0]  # (context_len,)
        return image, text, p["item_id"]


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    texts = torch.stack([b[1] for b in batch])
    item_ids = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return images, texts, item_ids
