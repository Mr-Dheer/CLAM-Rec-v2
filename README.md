# CLAM-Rec v2 — *When Does Vision Help LLM-based Sequential Recommendation?*

An **analysis** of when multimodal (visual) signal actually improves LLM-based
sequential recommendation, rather than a new architecture. We use a CLIP-augmented
A-LLMRec (OPT-6.7B backbone) as the instrument and characterize gains along
item- and design-level conditions.

## Research questions
- **RQ1 — Item coldness.** Does visual signal help more for *cold* items (few
  interactions) than *warm* ones? Global averages hide this; we slice metrics by
  item interaction count.
- **RQ2 — Fusion strategy.** Does *how* text+image CLIP embeddings are fused
  (concat / mean / learned gating) matter?
- **Mechanism contrast.** Original CLAM-Rec uses CLIP only as a Stage-1 *alignment
  target* (visual signal never reaches the LLM at inference). We compare this
  against a variant that *injects* the CLIP embedding into the LLM at inference.

## Relationship to Smol-Rec
This is a **distinct** paper from the authors' *Smol-Rec* (native VLM visual pathway,
image-budget study). Here the backbone is a **text LLM with pre-computed CLIP
embeddings** (no VLM inference), and the contribution is an **empirical analysis of
conditions** (coldness, fusion), not a new visual mechanism. Different title,
different RQ. Do not reuse Smol-Rec's title/framing.

## Dataset
Amazon **Luxury Beauty** (9930 users, 6141 items, 63953 interactions). Leave-one-out.

## Layout
```
clam_rec/
  data/     preprocessing (item-id/asin alignment), partition, cold/warm tagging
  clip/     CLIP extraction + alignment + fusion (rewritten from scratch)
  model/    SASRec (frozen CF), Stage-1 alignment, Stage-2 OPT integration, variants
  fusion/   concat / mean / gating fusion modules
  eval/     Hit@K / NDCG@K, sliced (cold vs warm), significance
configs/    one yaml per experiment
scripts/    end-to-end run scripts
assets/     symlinked reusable inputs (SASRec ckpt, fused CLIP npy, interactions, text dict)
```

## Provenance / trust notes
- A-LLMRec base code is reused (trusted KDD 2024 base).
- **All CLIP code is rewritten from scratch** — the old CLIP pipeline had
  contradictory approaches and unverifiable alignment.
- Item→ASIN alignment is **re-derived and cross-checked** so CLIP rows provably
  line up with SASRec item IDs.
