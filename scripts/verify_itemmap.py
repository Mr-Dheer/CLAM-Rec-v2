"""
Regenerate the Luxury Beauty item mapping and VERIFY it matches the canonical
SASRec data. This anchors CLIP-row alignment to the exact SASRec item ids.

Checks:
  1. Regenerated {dataset}.txt is byte-identical to the canonical Luxury_Beauty.txt
     (=> identical user ids, item ids, and ordering).
  2. Our regenerated id->asin map agrees with the existing itemid_to_asin pkl
     (if present) on every overlapping id.

Run:
  conda run -n ALLM-Rec python scripts/verify_itemmap.py
"""

import hashlib
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.data.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[1]
OLD = Path("/home/kavach/Dev/Extension-Paper")

DATASET = "Luxury_Beauty"
REVIEWS = OLD / "A-LLMRec/data/amazon/Luxury_Beauty.json.gz"
META = OLD / "A-LLMRec/data/amazon/meta_Luxury_Beauty.json"
CANON_TXT = OLD / "A-LLMRec/data/amazon/Luxury_Beauty.txt"
OLD_ID2ASIN = OLD / "Clip/allm_itemmap/itemid_to_asin_Lux_beauty.pkl"
OUT = ROOT / "data" / "processed"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("Regenerating item map + interaction file ...")
    res = preprocess(DATASET, str(REVIEWS), str(META), str(OUT))

    # ---- check 1: byte-identical interaction file ------------------------
    regen_txt = Path(res["txt_path"])
    h_new, h_old = sha256(regen_txt), sha256(CANON_TXT)
    same_bytes = h_new == h_old
    print("\n[Check 1] Interaction file byte-match vs canonical Luxury_Beauty.txt")
    print(f"  regenerated: {regen_txt}  sha256={h_new[:16]}...")
    print(f"  canonical  : {CANON_TXT}  sha256={h_old[:16]}...")
    print(f"  -> {'IDENTICAL ✅' if same_bytes else 'DIFFERENT ❌'}")

    if not same_bytes:
        # Line-level diff summary to diagnose.
        a = regen_txt.read_text().splitlines()
        b = CANON_TXT.read_text().splitlines()
        print(f"  line counts: regen={len(a)} canonical={len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"  first diff at line {i}: regen='{x}' canonical='{y}'")
                break

    # ---- check 2: id->asin map agreement ---------------------------------
    print("\n[Check 2] id->asin map vs existing pkl")
    if OLD_ID2ASIN.exists():
        with open(OLD_ID2ASIN, "rb") as f:
            old_map = pickle.load(f)
        new_map = res["itemid_to_asin"]
        common = set(new_map) & set(old_map)
        agree = sum(1 for i in common if new_map[i] == old_map[i])
        print(f"  existing pkl entries: {len(old_map)}, ours: {len(new_map)}")
        print(f"  overlapping ids: {len(common)}, agree: {agree} "
              f"({100*agree/max(1,len(common)):.1f}%)")
        if agree != len(common):
            mism = [i for i in common if new_map[i] != old_map[i]][:5]
            for i in mism:
                print(f"  MISMATCH id={i}: ours={new_map[i]} old={old_map[i]}")
    else:
        print(f"  (no existing pkl at {OLD_ID2ASIN}, skipping)")

    print("\nSummary:")
    print(f"  itemnum={res['itemnum']} usernum={res['usernum']}")
    print(f"  map saved to {res['itemmap_path']}")
    return 0 if same_bytes else 1


if __name__ == "__main__":
    raise SystemExit(main())
