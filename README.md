# CLAM-Rec v2 — *When Do LLMs Help Sequential Recommendation?*

An **empirical analysis** of *when* LLMs help LLM-based sequential recommendation and how they
relate to collaborative filtering (CF) — not a new architecture. We use **A-LLMRec** (a frozen
SASRec CF model aligned with a frozen OPT-6.7B LLM) as the instrument and characterize behaviour by
item popularity across four Amazon datasets.

> **New sessions: read [`CLAUDE.md`](./CLAUDE.md) / [`AGENTS.md`](./AGENTS.md) first** — full project
> context and reading order. The detailed story is in `FINDINGS.md` + `FINDING_1/2/4.md`; numbers in
> `RESULTS.md`.

## The three findings
1. **A CF↔LLM crossover by item popularity.** On **cold / rare** items the LLM (`text`) beats CF
   (`SASRec`) in every dataset (CF has no signal for rarely-seen items — at 0 interactions SASRec
   scores exactly 0.000); on **warm / popular** items CF beats the LLM. `text − SASRec` decays
   monotonically and crosses zero. LLMs add value *exactly where CF is weak*.
2. **The crossover point is predicted by dataset density.** It lands near the dataset's mean
   interactions/item (Pearson **r = 0.98**) — predictive, not just descriptive.
3. **A popularity-gated fusion beats both.** Weighting each candidate's CF vs LLM score by that
   candidate's popularity beats *both* pure models on **Hit@1 and NDCG** across all 4 datasets
   (≈ an oracle upper bound); naive fusion (RRF, equal-weight) does not.

## Setup
- **Variants:** `SASRec` (CF only) and `text` (A-LLMRec). Likelihood ranking over **20 candidates**
  (1 ground-truth + 19 negatives); metrics Hit@1/5/10 & NDCG@5/10, sliced cold (target train
  count ≤ 5) / warm.
- **Datasets (4):** **Luxury Beauty** (9930u/6141i), **AMAZON_FASHION** (3679u/7310i),
  **Toys & Games** (a 9513u/7253i random-user subsample), **Prime Pantry** (15611u/7841i).
  Leave-one-out. (All Beauty was used earlier but dropped — small / density outlier.)

## Relationship to the sibling papers
- **Vision is *not* in this paper.** The repo also contains a CLIP-augmented variant and a
  vision-null result; that is a **separate future paper** — see `FINDING_3.md`. Ignore the
  `clam_rec/clip|fusion|finetune` code for this paper.
- Distinct from the authors' *Smol-Rec* (a VLM visual-budget study): different question (CF/LLM
  complementarity along popularity), different backbone (text LLM), no vision.

## Layout
```
clam_rec/
  data/     preprocessing (item-id/asin alignment), partition, cold/warm tagging
  model/    SASRec (frozen CF), A-LLMRec Stage-1/2, generate + rank_candidates
  eval/     Hit@K / NDCG@K, sliced (cold vs warm)
  clip/, fusion/, finetune/   ← vision code (separate paper; FINDING_3.md)
scripts/    eval_sasrec.py, infer.py (ranking), analysis_*.py (the results), plot_*.py (figures)
configs/    one yaml per dataset      results/  per-seed .jsonl + checkpoints
```

## Docs
`CLAUDE.md`/`AGENTS.md` (entry) · `FINDINGS.md` (master index) · `FINDING_1/2/4.md` (deep-dives) ·
`RESULTS.md` (numbers) · `DATASETS.md` · `EVAL_PROTOCOL.md` (why 20 candidates) ·
`FINDING_3.md` (vision — separate paper) · `PROJECT.md`/`STORY.md` (design history, superseded).
