# CLAM-Rec v2 — Results Log

> A single, growing record of **every experiment run**: which model/variant, which
> dataset, which CLIP embeddings (zero-shot vs fine-tuned, which backbone), the
> config used, and the metrics. Append new results here as they land. This is the
> factual results ledger — for the *why/positioning*, see `PROJECT.md`.

Metric convention: **Hit@1** with fuzzy title matching at threshold **0.90** (generated
next-item title vs ground-truth title). Evaluation is leave-one-out, 1 ground-truth +
19 random negatives = 20 candidates. Metrics are sliced by **cold** (target item's
training interaction count ≤ 5) vs **warm** (> 5).

---

## Fixed experimental setup (unless noted)

| Item | Value |
|------|-------|
| Dataset | Amazon **Luxury Beauty** |
| Users / Items / Interactions | 9,930 / 6,141 / 63,953 |
| Test users evaluated | 9,912 (3,763 cold / 6,149 warm @ threshold 5) |
| Backbone LLM | OPT-6.7B (frozen, 8-bit) |
| Sequential recommender | SASRec (frozen), hidden 50, maxlen 50 |
| Base framework | A-LLMRec (2-stage: alignment + LLM integration) |
| Stage-1 / Stage-2 epochs | 10 / 10 |
| Stage-1 / Stage-2 LR | 1e-4 / 1e-4 |
| Candidates | 20 (1 pos + 19 neg) |
| Cold threshold | ≤ 5 training interactions |
| Hardware | NVIDIA RTX A6000 (GPU id 2), remote `dreal_gpu` |

**Model variants** (the `variant` switch):
- `text` — SBERT alignment target, no vision (A-LLMRec baseline).
- `clip_align` — fused CLIP as Stage-1 alignment target only (vision does NOT reach the LLM at inference; the "bottleneck").
- `clip_inject` — CLIP alignment target PLUS a per-item `[MMEmb]` CLIP soft token fed to the LLM at inference (bottleneck fixed).

**CLIP embedding sets** (the `clip_variant` switch):
- `bigG` — CLIP-ViT-bigG-14 (laion2b), 1280-dim/modality, 2560 fused. Zero-shot.
- `vitl14_zeroshot` — CLIP-ViT-L/14 (laion2b), 768-dim/modality, 1536 fused. Zero-shot.
- `vitl14_ft` — same ViT-L/14, LoRA domain-fine-tuned on Luxury Beauty image↔text.

---

## Results table (Hit@1, fuzzy@0.90)

| # | Variant | CLIP set | Seeds | Overall | Cold | Warm | Date |
|---|---------|----------|:-----:|:-------:|:----:|:----:|------|
| 0 | **SASRec only** (CF floor) | — | 1 (seed 0) | 0.5270 | **0.1929** | **0.7315** | 2026-07-29 |
| 1 | clip_inject | vitl14_zeroshot | 1 (seed 0) | **0.5976** | 0.4845 | 0.6668 | 2026-07-27 |
| 2 | clip_inject | vitl14_ft | 1 (seed 0) | **0.5674** | 0.4494 | 0.6396 | 2026-07-27 |

Run config for #1–#2: `configs/luxury_beauty_rq3.yaml` (batch_size2=4, batch_size_infer=16).
Output: `results/clip_inject_concat_vitl14_{zeroshot,ft}/seed_0.jsonl` (9,912 records each).
Row #0 (SASRec-only): `scripts/eval_sasrec.py` — same leave-one-out split, same 20-candidate
set (1 pos + 19 neg, seed 0), same cold/warm tags; Hit@1 = SASRec scores true item #1.
Ran on local CPU (tiny). Output: `results/sasrec_only/seed_0.jsonl`.

### RQ1 headline: SASRec (CF) vs vision-augmented, sliced cold/warm

| Model | Overall | Cold | Warm |
|-------|--------:|-----:|-----:|
| SASRec only (CF) | 0.5270 | **0.1929** | 0.7315 |
| clip_inject + ViT-L (vision) | 0.5976 | **0.4845** | 0.6668 |
| Δ (vision − CF) | +0.071 | **+0.292** | −0.065 |

**Finding:** Pure collaborative filtering (SASRec) is strong on warm items (0.73) but
**collapses on cold items (0.19)** — no interaction signal. Adding vision + LLM
**dramatically recovers cold-item recommendation (0.19 → 0.48, +29 pts)** at a small
cost to warm items (−6.5 pts), and wins overall. This is the core "when does vision
help" result: **vision rescues cold-start, where CF fails.** (1 seed; confirm with more.)

### RQ3 comparison (fine-tuned vs zero-shot CLIP, ViT-L/14, clip_inject)

| Slice | Zero-shot | Fine-tuned | Δ (ft − zs) |
|-------|----------:|-----------:|------------:|
| Overall | 0.5976 | 0.5674 | **−0.030** |
| Cold | 0.4845 | 0.4494 | **−0.035** |
| Warm | 0.6668 | 0.6396 | **−0.027** |

**Finding:** Domain fine-tuning of CLIP **degrades recommendation** by ~3 points across
all slices — despite *improving* image↔text retrieval (proxy R@1 0.53 → 0.63). Zero-shot
CLIP's general semantic space is more useful for recommendation than domain-specialized
embeddings. (Interpretation: fine-tuning makes each item's image/text more self-consistent
but collapses cross-item semantic structure the recommender relies on.)

**Caveat:** 1 seed only → the −3pt is suggestive, not yet statistically confirmed
(paired t-test needs ≥ 2 shared seeds). Cold ≪ warm everywhere, as expected.

---

## Proxy result (not recommendation — image↔text retrieval, held-out items)

CLIP ViT-L/14 LoRA domain fine-tuning, image→text retrieval R@1 on unseen items:

| Split | Zero-shot R@1 | Fine-tuned R@1 | Δ |
|-------|--------------:|---------------:|----:|
| seed 0 | 0.530 | 0.633 | +0.104 |
| seed 1 | 0.577 | 0.677 | +0.100 |

Fine-tuning reliably improves *retrieval* (~+0.10 R@1, robust across splits) — but this
does NOT translate to recommendation (see RQ3 above). Retrieval quality ≠ rec quality.

---

## Still to run (not yet done)

- [ ] **RQ1 / mechanism (bigG):** `text` vs `clip_align` vs `clip_inject` — the paper's
  headline (does vision help? does injection beat the bottleneck?).
- [ ] **RQ2 fusion (bigG):** `clip_inject` with concat vs mean vs gating.
- [ ] **Multi-seed** (≥10) for any result we want significance on (currently 1 seed each).
- [ ] Same-CLIP baselines for a clean RQ3 (bigG zero-shot vs a bigG fine-tune, if desired).

_Last updated: 2026-07-29._
