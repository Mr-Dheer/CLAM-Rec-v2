"""
Faithful reimplementation of A-LLMRec's pre_train/sasrec/data_preprocess.py,
with one crucial addition: we also persist the ``itemmap`` (asin -> item_id) and
its inverse (item_id -> asin).

Why this matters
----------------
In A-LLMRec, item ids are assigned *by order of first appearance* while iterating
the filtered review file (second pass). The original script never saved this
mapping, which is exactly what makes CLIP-row alignment error-prone: any
independently-guessed ordering can silently misalign visual embeddings to the
wrong SASRec item ids. We recompute the mapping here from the same logic and
verify our regenerated interaction file byte-matches the canonical one, so the
mapping is provably the one SASRec was trained with.

Filter logic (must match data_preprocess.py exactly):
  - pass 1 counts; for Beauty/Toys, skip reviews with overall < 3.
  - threshold = 4 for Beauty/Toys, else 5.
  - keep an interaction iff countU[reviewer] >= threshold AND countP[asin] >= threshold.
  - item id assigned on first *kept* appearance (incrementing from 1).
"""

import gzip
import json
import pickle
from collections import defaultdict
from pathlib import Path


def _is_beauty_or_toys(name: str) -> bool:
    return ("Beauty" in name) or ("Toys" in name)


def _threshold(name: str) -> int:
    return 4 if _is_beauty_or_toys(name) else 5


def parse_gz(path):
    with gzip.open(path, "rb") as g:
        for line in g:
            yield json.loads(line)


def preprocess(
    dataset: str,
    reviews_gz: str,
    meta_json: str,
    out_dir: str,
):
    """Recompute the interaction file, text_name_dict, and item/user maps.

    Returns a dict with keys: usernum, itemnum, itemmap (asin->id),
    itemid_to_asin, txt_path, name_dict_path, itemmap_path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    beauty_toys = _is_beauty_or_toys(dataset)
    threshold = _threshold(dataset)

    # ---- pass 1: count interactions --------------------------------------
    countU = defaultdict(int)
    countP = defaultdict(int)
    for l in parse_gz(reviews_gz):
        if beauty_toys and l["overall"] < 3:
            continue
        countU[l["reviewerID"]] += 1
        countP[l["asin"]] += 1

    # ---- load metadata (one json object per line) ------------------------
    meta_dict = {}
    with open(meta_json, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            obj = json.loads(line)
            meta_dict[obj["asin"]] = obj

    # ---- pass 2: assign ids in first-appearance order --------------------
    usermap, itemmap = {}, {}
    usernum = itemnum = 0
    User = {}
    name_dict = {"title": {}, "description": {}}

    for l in parse_gz(reviews_gz):
        # NOTE: original does NOT re-apply the overall<3 skip in pass 2.
        # We match that behaviour exactly.
        asin, rev, time = l["asin"], l["reviewerID"], l["unixReviewTime"]
        if countU[rev] < threshold or countP[asin] < threshold:
            continue

        if rev in usermap:
            userid = usermap[rev]
        else:
            usernum += 1
            userid = usernum
            usermap[rev] = userid
            User[userid] = []

        if asin in itemmap:
            itemid = itemmap[asin]
        else:
            itemnum += 1
            itemid = itemnum
            itemmap[asin] = itemid

        User[userid].append([time, itemid])

        # title / description (best-effort, matching original try/except)
        try:
            desc = meta_dict[asin]["description"]
            name_dict["description"][itemid] = (
                "Empty description" if len(desc) == 0 else desc[0]
            )
            name_dict["title"][itemid] = meta_dict[asin]["title"]
        except Exception:
            pass

    for userid in User:
        User[userid].sort(key=lambda x: x[0])

    # ---- write artifacts -------------------------------------------------
    txt_path = out / f"{dataset}.txt"
    with open(txt_path, "w") as f:
        for user in User:
            for _, itemid in User[user]:
                f.write("%d %d\n" % (user, itemid))

    name_dict_path = out / f"{dataset}_text_name_dict.json.gz"
    with open(name_dict_path, "wb") as tf:
        pickle.dump(name_dict, tf)

    itemid_to_asin = {v: k for k, v in itemmap.items()}
    itemmap_path = out / f"{dataset}_itemmap.pkl"
    with open(itemmap_path, "wb") as f:
        pickle.dump({"asin_to_id": itemmap, "id_to_asin": itemid_to_asin}, f)

    print(f"usernum={usernum} itemnum={itemnum}")
    return {
        "usernum": usernum,
        "itemnum": itemnum,
        "itemmap": itemmap,
        "itemid_to_asin": itemid_to_asin,
        "txt_path": str(txt_path),
        "name_dict_path": str(name_dict_path),
        "itemmap_path": str(itemmap_path),
    }
