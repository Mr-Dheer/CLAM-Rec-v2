"""
Analyze a candidate Amazon dataset for the "when does vision help" paper:
runs the A-LLMRec-style k-core preprocess, then reports the things that decide
whether the dataset is USEFUL for us:
  - filtered size (users / items / interactions)
  - image coverage (fraction of filtered items with an image URL in metadata)
  - cold/warm split at the target-item train-count threshold (need a real cold pop)
  - optional: does it match an existing SASRec checkpoint (reusable vs must-retrain)

Usage:
  python scripts/analyze_dataset.py --dataset AMAZON_FASHION \
    --reviews data/raw/AMAZON_FASHION.json.gz --meta data/raw/meta_AMAZON_FASHION.json \
    [--sasrec assets/sasrec_AMAZON_FASHION.pth] [--cold_threshold 5]
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.data.preprocess import preprocess
from clam_rec.data.partition import data_partition, tag_cold_warm, cold_warm_summary


def image_coverage(meta_json, itemid_to_asin):
    """Fraction of filtered items whose metadata has a non-empty image URL."""
    have = set()
    with open(meta_json, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            img = o.get("imageURLHighRes") or o.get("imageURL") or o.get("image")
            if img:
                have.add(o["asin"])
    asins = set(itemid_to_asin.values())
    n_img = sum(1 for a in asins if a in have)
    return n_img, len(asins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--sasrec", default=None)
    ap.add_argument("--cold_threshold", type=int, default=5)
    ap.add_argument("--out_dir", default="data/raw/analysis")
    args = ap.parse_args()

    out = Path(args.out_dir) / args.dataset
    r = preprocess(args.dataset, args.reviews, args.meta, str(out))
    txt = r["txt_path"]

    ut, uv, ute, usernum, itemnum = data_partition(txt)
    interactions = sum(len(v) for v in {**ut}.values()) + \
                   sum(len(v) for v in uv.values()) + sum(len(v) for v in ute.values())
    tags, _ = tag_cold_warm(ute, ut, cold_threshold=args.cold_threshold)
    cw = cold_warm_summary(tags)
    n_img, n_items = image_coverage(args.meta, r["itemid_to_asin"])
    # title coverage (Fashion-bug criterion: eval is title-generation based)
    import pickle as _pk
    _d = _pk.load(open(r["name_dict_path"], "rb"))
    _titles = [str(v).strip() for v in _d["title"].values() if v and str(v).strip()]
    n_title = len(_titles); n_title_uniq = len(set(_titles))

    print("\n" + "=" * 64)
    print(f"DATASET ANALYSIS: {args.dataset}")
    print("=" * 64)
    print(f"users               : {usernum:,}")
    print(f"items               : {itemnum:,}")
    print(f"interactions        : {interactions:,}")
    print(f"avg seq len         : {interactions/usernum:.2f}")
    print(f"image coverage      : {n_img:,}/{n_items:,} = {100*n_img/max(n_items,1):.1f}%")
    print(f"title coverage      : {n_title:,}/{itemnum:,} = {100*n_title/max(itemnum,1):.1f}%  "
          f"(unique {100*n_title_uniq/max(n_title,1):.1f}%)")
    print(f"test users (w/ test): {cw['n_test_users']:,}")
    print(f"  cold (<= {args.cold_threshold})       : {cw['cold']:,} ({100*cw['cold_frac']:.1f}%)")
    print(f"  warm             : {cw['warm']:,} ({100*(1-cw['cold_frac']):.1f}%)")

    if args.sasrec and Path(args.sasrec).exists():
        import torch
        kw, _ = torch.load(args.sasrec, map_location="cpu", weights_only=False)
        match = (kw["user_num"] == usernum and kw["item_num"] == itemnum)
        print(f"SASRec ckpt         : users={kw['user_num']:,} items={kw['item_num']:,} "
              f"-> {'MATCH (reusable)' if match else 'MISMATCH (retrain SASRec)'}")

    # crude usefulness verdict
    verdict = []
    if itemnum < 3000: verdict.append("items<3k (small, cold slice may be thin)")
    if cw["cold_frac"] < 0.20: verdict.append("cold<20% (warm-skewed like All Beauty)")
    if n_items and n_img/n_items < 0.60: verdict.append("image cov<60% (vision under-supported)")
    if itemnum and n_title/itemnum < 0.80: verdict.append("title cov<80% (title-eval degenerate, cf. Fashion)")
    if usernum > 20000: verdict.append("users>20k (Stage-2 runtime heavy)")
    print("FLAGS               :", "; ".join(verdict) if verdict else "none — looks usable")
    print("=" * 64)


if __name__ == "__main__":
    main()
