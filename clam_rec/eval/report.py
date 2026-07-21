"""
Turn per-seed inference JSONL logs into the paper's result tables.

Given a directory of runs like:
  results/<variant>/seed_<s>.jsonl
produce, per variant:
  - overall / cold / warm  Hit@K, NDCG@K  (mean ± std over seeds)
and, given a baseline variant, paired-t significance vs baseline per slice.

Usage:
  conda run -n ALLM-Rec python -m clam_rec.eval.report \
    --runs_dir results --baseline text \
    --variants text clip_align clip_inject \
    --ks 1 5 10 --fuzzy 0.90
"""

import argparse
import glob
import os
import re

from .metrics import load_jsonl, aggregate_seeds, paired_ttest


def load_variant(runs_dir, variant):
    """Return {seed: [records]} for a variant."""
    s2r = {}
    for fp in sorted(glob.glob(os.path.join(runs_dir, variant, "seed_*.jsonl"))):
        m = re.search(r"seed_(\d+)\.jsonl", fp)
        if not m:
            continue
        s2r[int(m.group(1))] = load_jsonl(fp)
    return s2r


def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="results")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--fuzzy", type=float, default=None)
    args = ap.parse_args()

    loaded = {v: load_variant(args.runs_dir, v) for v in args.variants}
    loaded = {v: s2r for v, s2r in loaded.items() if s2r}
    if not loaded:
        print("No run files found.")
        return

    ks = tuple(args.ks)
    print("\n" + "=" * 72)
    print("RESULTS (mean ± std over seeds)   fuzzy =", args.fuzzy)
    print("=" * 72)
    for v, s2r in loaded.items():
        agg, _ = aggregate_seeds(s2r, ks, args.fuzzy)
        n_seeds = len(s2r)
        print(f"\n[{v}]  ({n_seeds} seeds)")
        for sl in ["overall", "cold", "warm"]:
            metrics = agg.get(sl, {})
            cells = []
            for m in sorted(metrics):
                cells.append(f"{m}={fmt(metrics[m]['mean'], metrics[m]['std'])}")
            print(f"  {sl:8s}: " + "  ".join(cells))

    if args.baseline and args.baseline in loaded:
        print("\n" + "=" * 72)
        print(f"SIGNIFICANCE vs baseline '{args.baseline}' (paired t-test, Hit@1)")
        print("=" * 72)
        base = loaded[args.baseline]
        for v, s2r in loaded.items():
            if v == args.baseline:
                continue
            print(f"\n[{v}] vs [{args.baseline}]")
            for sl in ["overall", "cold", "warm"]:
                res = paired_ttest(s2r, base, sl, "Hit@1", args.fuzzy)
                if res is None:
                    print(f"  {sl:8s}: (insufficient shared seeds)")
                    continue
                delta = res["mean_a"] - res["mean_b"]
                sig = "***" if res["p"] < 0.001 else ("**" if res["p"] < 0.01
                       else ("*" if res["p"] < 0.05 else "n.s."))
                print(f"  {sl:8s}: {v}={res['mean_a']:.4f} vs base={res['mean_b']:.4f} "
                      f"Δ={delta:+.4f}  t={res['t']:+.2f} p={res['p']:.2e} {sig}")


if __name__ == "__main__":
    main()
