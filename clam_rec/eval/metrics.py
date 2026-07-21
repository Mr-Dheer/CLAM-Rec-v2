"""
Sliced evaluation: Hit@K / NDCG@K overall and by cold/warm slice, with
paired-t significance across seeds.

Design decisions
----------------
- The LLM generates the *title* of the predicted next item. Following A-LLMRec we
  score by matching the generated title against the ground-truth title. This gives
  a top-1 decision (Hit@1). For Hit@K / NDCG@K we additionally require the model to
  emit a *ranked list*; when only a single generated title is available we report
  Hit@1 (== NDCG@1) and, if a ranked candidate list is logged, the ranked metrics.
- Each evaluation record carries metadata (user_id, cold flag, target train_count)
  so metrics can be sliced. Records are written as JSONL by the inference step
  (one dict per test case) — NOT free-form text — to avoid the append/parse bugs
  in the old pipeline.

Record schema (one JSON object per line):
  {"user": int, "seed": int, "cold": bool, "train_count": int,
   "answer": str, "generated": str,
   "ranked": [str, ...]  # optional: model's ranked candidate titles, best first
  }
"""

import html
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np


# --------------------------------------------------------------------------- #
# title normalization + matching
# --------------------------------------------------------------------------- #
def normalize_title(s: str) -> str:
    if s is None:
        return ""
    s = html.unescape(s).strip()
    while len(s) >= 2 and (
        (s[0] == s[-1] and s[0] in "\"'") or (s[0], s[-1]) in [("“", "”"), ("‘", "’")]
    ):
        s = s[1:-1].strip()
    s = s.strip("\"'“”‘’").lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*/\s*", "/", s)
    return s.strip()


def title_match(a: str, b: str, fuzzy_threshold=None) -> bool:
    a, b = normalize_title(a), normalize_title(b)
    if a == b:
        return True
    if fuzzy_threshold is not None:
        return SequenceMatcher(None, a, b).ratio() >= fuzzy_threshold
    return False


# --------------------------------------------------------------------------- #
# per-record hit / ndcg
# --------------------------------------------------------------------------- #
def record_hit_at_1(rec, fuzzy_threshold=None) -> float:
    return 1.0 if title_match(rec["answer"], rec["generated"], fuzzy_threshold) else 0.0


def record_rank(rec, fuzzy_threshold=None):
    """Rank (1-based) of the ground-truth answer in the model's ranked list.
    Returns None if no ranked list logged or answer not found."""
    ranked = rec.get("ranked")
    if not ranked:
        return None
    for i, cand in enumerate(ranked):
        if title_match(rec["answer"], cand, fuzzy_threshold):
            return i + 1
    return None


def hit_at_k(rank, k):
    return 1.0 if (rank is not None and rank <= k) else 0.0


def ndcg_at_k(rank, k):
    if rank is None or rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1)


# --------------------------------------------------------------------------- #
# aggregate over a set of records (one seed)
# --------------------------------------------------------------------------- #
def evaluate_records(records, ks=(1, 5, 10), fuzzy_threshold=None):
    """Return dict of metric -> value for one seed's records (overall)."""
    n = len(records)
    if n == 0:
        return {}
    out = {}
    hits1 = [record_hit_at_1(r, fuzzy_threshold) for r in records]
    out["Hit@1"] = float(np.mean(hits1))
    # ranked metrics only if any record logs a ranked list
    have_ranked = any(r.get("ranked") for r in records)
    if have_ranked:
        ranks = [record_rank(r, fuzzy_threshold) for r in records]
        for k in ks:
            out[f"Hit@{k}"] = float(np.mean([hit_at_k(r, k) for r in ranks]))
            out[f"NDCG@{k}"] = float(np.mean([ndcg_at_k(r, k) for r in ranks]))
    out["count"] = n
    return out


def evaluate_sliced(records, ks=(1, 5, 10), fuzzy_threshold=None):
    """Overall + cold + warm metrics for one seed."""
    cold = [r for r in records if r.get("cold")]
    warm = [r for r in records if not r.get("cold")]
    return {
        "overall": evaluate_records(records, ks, fuzzy_threshold),
        "cold": evaluate_records(cold, ks, fuzzy_threshold),
        "warm": evaluate_records(warm, ks, fuzzy_threshold),
    }


# --------------------------------------------------------------------------- #
# multi-seed aggregation + significance
# --------------------------------------------------------------------------- #
def load_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def aggregate_seeds(seed_to_records, ks=(1, 5, 10), fuzzy_threshold=None):
    """seed_to_records: {seed: [records]}. Returns mean±std per slice/metric."""
    per_seed = {seed: evaluate_sliced(recs, ks, fuzzy_threshold)
                for seed, recs in seed_to_records.items()}
    slices = ["overall", "cold", "warm"]
    agg = {}
    for sl in slices:
        metric_vals = defaultdict(list)
        for seed, res in per_seed.items():
            for m, v in res[sl].items():
                if m == "count":
                    continue
                metric_vals[m].append(v)
        agg[sl] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v)), "seeds": len(v)}
                   for m, v in metric_vals.items()}
    return agg, per_seed


def paired_ttest(seed_to_records_a, seed_to_records_b, slice_name="overall",
                 metric="Hit@1", fuzzy_threshold=None):
    """Paired t-test of metric between two models across shared seeds.
    Returns (t, p, mean_a, mean_b, n_seeds)."""
    from scipy import stats

    def per_seed_metric(s2r):
        vals = {}
        for seed, recs in s2r.items():
            sub = recs if slice_name == "overall" else \
                [r for r in recs if (r.get("cold") == (slice_name == "cold"))]
            res = evaluate_records(sub, fuzzy_threshold=fuzzy_threshold)
            if metric in res:
                vals[seed] = res[metric]
        return vals

    a = per_seed_metric(seed_to_records_a)
    b = per_seed_metric(seed_to_records_b)
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return None
    av = np.array([a[s] for s in shared])
    bv = np.array([b[s] for s in shared])
    t, p = stats.ttest_rel(av, bv)
    return {"t": float(t), "p": float(p), "mean_a": float(av.mean()),
            "mean_b": float(bv.mean()), "n_seeds": len(shared)}
