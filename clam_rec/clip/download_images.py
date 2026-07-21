"""
Download one product image per item for a dataset.

Based on the Smol-Rec download script (trusted, user-provided). Adapted to:
  - reuse our VERIFIED itemmap (data/processed/{dataset}_itemmap.pkl) as the source
    of truth for the valid-ASIN set, and to cross-check it against the replayed
    two-pass filter (a second alignment guard);
  - key images by ASIN ({asin}.jpg), with a coverage CSV + PIL validation + resume.

Run:
  conda run -n ALLM-Rec python -m clam_rec.clip.download_images \
    --dataset Luxury_Beauty \
    --reviews /home/kavach/Dev/Extension-Paper/A-LLMRec/data/amazon/Luxury_Beauty.json.gz \
    --metadata /home/kavach/Dev/Extension-Paper/A-LLMRec/data/amazon/meta_Luxury_Beauty.json \
    --itemmap data/processed/Luxury_Beauty_itemmap.pkl \
    --output_dir data/images/Luxury_Beauty --workers 8
"""

import argparse
import csv
import gzip
import json
import pickle
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


def derive_valid_asins(reviews_path: Path, dataset: str) -> set:
    """Replay data_preprocess filter to derive the valid-ASIN set."""
    beauty_toys = ("Beauty" in dataset) or ("Toys" in dataset)
    threshold = 4 if beauty_toys else 5
    countU, countP = defaultdict(int), defaultdict(int)
    with gzip.open(reviews_path, "rb") as f:
        for raw in tqdm(f, desc="  pass1 count", unit="line"):
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if beauty_toys and rec.get("overall", 5) < 3:
                continue
            countU[rec["reviewerID"]] += 1
            countP[rec["asin"]] += 1
    valid = set()
    with gzip.open(reviews_path, "rb") as f:
        for raw in tqdm(f, desc="  pass2 filter", unit="line"):
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if beauty_toys and rec.get("overall", 5) < 3:
                continue
            if countU[rec["reviewerID"]] >= threshold and countP[rec["asin"]] >= threshold:
                valid.add(rec["asin"])
    return valid


def load_image_urls(metadata_path: Path, valid_asins: set) -> dict:
    url_map = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="  scan meta", unit="line"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = obj.get("asin", "")
            if asin not in valid_asins:
                continue
            hi = obj.get("imageURLHighRes") or []
            lo = obj.get("imageURL") or []
            url_map[asin] = (hi[0] if hi and hi[0] else (lo[0] if lo and lo[0] else ""))
    return url_map


def download_one(asin, url, output_dir, timeout=10):
    dest = output_dir / f"{asin}.jpg"
    if not url:
        return asin, "no_url_in_metadata", "", ""
    if dest.exists():
        try:
            with Image.open(dest) as img:
                img.verify()
            return asin, "success", url, str(dest)
        except Exception:
            dest.unlink(missing_ok=True)
    time.sleep(random.uniform(0.05, 0.2))
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or len(resp.content) <= 500:
            return asin, "failed_download", url, ""
        dest.write_bytes(resp.content)
    except Exception:
        return asin, "failed_download", url, ""
    try:
        with Image.open(dest) as img:
            img.verify()
        return asin, "success", url, str(dest)
    except Exception:
        dest.unlink(missing_ok=True)
        return asin, "failed_validation", url, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reviews", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--itemmap", required=True, help="verified itemmap.pkl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--report_dir", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=10)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir) if args.report_dir else output_dir.parent / "download_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # source of truth: verified itemmap
    with open(args.itemmap, "rb") as f:
        itemmap = pickle.load(f)
    map_asins = set(itemmap["asin_to_id"].keys())

    print("[Step 1] deriving valid ASINs (replay filter) ...")
    replay_asins = derive_valid_asins(Path(args.reviews), args.dataset)

    # cross-check the two agree (alignment guard)
    only_map = map_asins - replay_asins
    only_replay = replay_asins - map_asins
    print(f"  itemmap ASINs={len(map_asins)}  replay ASINs={len(replay_asins)}")
    print(f"  only in itemmap={len(only_map)}  only in replay={len(only_replay)}")
    if only_map or only_replay:
        print("  WARNING: ASIN set mismatch between itemmap and replay filter!")
    else:
        print("  ✅ ASIN sets identical — alignment guard passed")

    # download for the itemmap ASINs (the true item set)
    print("[Step 2] resolving image URLs from metadata ...")
    url_map = load_image_urls(Path(args.metadata), map_asins)
    matched = sum(1 for v in url_map.values() if v)
    print(f"  matched {matched}/{len(map_asins)} ASINs to image URLs")

    print(f"[Step 3] downloading with {args.workers} workers ...")
    asins_list = sorted(map_asins)
    rows, success, failed = [], 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_one, a, url_map.get(a, ""), output_dir, args.timeout): a
                for a in asins_list}
        with tqdm(total=len(asins_list), unit="img", desc="  downloading") as bar:
            for fut in as_completed(futs):
                asin, status, url_used, fp = fut.result()
                rows.append((asin, status, url_used, fp))
                success += status == "success"
                failed += status != "success"
                bar.update(1)

    rows.sort(key=lambda r: r[0])
    report_path = report_dir / f"{args.dataset}_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asin", "item_id", "status", "url_used", "file_path"])
        for asin, status, url_used, fp in rows:
            w.writerow([asin, itemmap["asin_to_id"].get(asin, ""), status, url_used, fp])

    total = len(asins_list)
    print("=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Total items:        {total}")
    print(f"Successfully saved: {success:>5} ({100*success/total:.1f}%)")
    print(f"Failed/missing:     {failed:>5} ({100*failed/total:.1f}%)")
    print(f"Report: {report_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
