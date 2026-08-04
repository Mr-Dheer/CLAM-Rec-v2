# CLAM-Rec v2 — *When Do LLMs Help Sequential Recommendation?*

An **empirical analysis** of *when* LLMs (and visual signal) help LLM-based
sequential recommendation, rather than a new architecture. We use a CLIP-augmented
A-LLMRec (OPT-6.7B backbone) as the instrument and characterize gains by item
popularity across four Amazon datasets.

## Headline finding — a CF↔LLM crossover by item popularity
- On **cold / rare items** the LLM (`text`) beats collaborative filtering (`SASRec`)
  in every dataset — CF has no signal for rarely-seen items.
- On **warm / popular items** CF beats the LLM in every dataset.
- `text − SASRec` **decays monotonically and crosses zero** at a dataset-dependent
  interaction count. LLMs add value *exactly* where CF is weak.
- **Adding visual (CLIP) signal does not shift this crossover** — the vision angle is
  a null result under the ranking protocol (see below).

## Research questions
- **RQ1 — Where does the LLM help?** Slice metrics by item interaction count to expose
  the CF↔LLM crossover (global averages hide it).
- **Mechanism contrast (vision).** Original CLAM-Rec uses CLIP only as a Stage-1
  *alignment target* (visual signal never reaches the LLM at inference; `clip_align`).
  We compare against *injecting* the CLIP embedding into the LLM (`clip_inject`).
  Neither shifts the crossover → vision is a ruled-out confound.
- **Fusion (note, not an RQ).** We tried concat / mean / learned gating for combining
  CLIP text+image; **plain concatenation was best**, so we fix fusion = concat
  throughout and do not report fusion as a separate study.

## Relationship to Smol-Rec
This is a **distinct** paper from the authors' *Smol-Rec* (native VLM visual pathway,
image-budget study). Here the backbone is a **text LLM with pre-computed CLIP
embeddings** (no VLM inference), and the contribution is an **empirical analysis of
CF/LLM complementarity along the popularity axis** (plus the vision null), not a new
visual mechanism. Different question, different framing.

## Datasets
Four Amazon Review 2018 (McAuley v2) datasets: **Luxury Beauty** (9930u/6141i),
**All Beauty** (2169u/1854i), **AMAZON_FASHION** (3679u/7310i), and **Toys & Games**
(a 9513u/7253i random-user subsample). Leave-one-out, ranked over 20 candidates.
(Prime Pantry was prepared but excluded — results not favorable.)

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
