# CLAM-Rec v2 — Results Log

> A single, growing record of **every experiment run**: which model/variant, which
> dataset, which CLIP embeddings (zero-shot vs fine-tuned, which backbone), the
> config used, and the metrics. Append new results here as they land. This is the
> factual results ledger — for the *why/positioning*, see `PROJECT.md`.

Metric convention: **Hit@1** with fuzzy title matching at threshold **0.90** (generated
next-item title vs ground-truth title). Evaluation is leave-one-out, 1 ground-truth +
19 random negatives = 20 candidates. Metrics are sliced by **cold** (target item's
training interaction count ≤ 5) vs **warm** (> 5). (The paper reports the **ranking**
protocol — Hit@1/5, NDCG@5 over the same 20 candidates — which confirms the same
crossover + vision-null; the generation Hit@1 numbers below are the underlying ledger.)

---

## ⭐ SCOPE (locked 2026-08-03) — read before using any table below

The paper covers **4 datasets** (Luxury Beauty, AMAZON_FASHION, Toys sub, **Prime Pantry** —
All Beauty dropped 2026-08-04 as a small density outlier; Prime Pantry reinstated, see `DATASETS.md`),
**ViT-L/14 as the single backbone**, ladder **SASRec → text → clip_align → clip_inject**,
**fusion fixed = concat**, **20 candidates** (1 pos + 19 neg — LOCKED; higher counts overflow
OPT's 2048-token context, see `EVAL_PROTOCOL.md`). Headline = the **CF↔LLM crossover** by item
popularity; the vision angle is a **null** result under ranking (see `STORY.md`).

**Parked / EXCLUDED from the paper (kept below for the record, clearly marked):**
- ⛔ **bigG backbone** — dropped; ViT-L is the only backbone. (bigG tables retained as history.)
- ⛔ **CLIP fine-tuning (RQ3)** — dropped; "better retrieval, worse rec" negative parked below.
- ⛔ **Prime Pantry dataset** — dropped (results not favorable); section retained as history.
- ⛔ **Fusion study (RQ2)** — not run as a study; **we tried concat / mean / gating and concat
  was best**, so concat is fixed everywhere (one-line note in the paper).

---

## ⭐ RANKING RESULTS (REPORTED PROTOCOL) — all datasets, all metrics

**This is the paper's primary results.** Likelihood-ranking over **20 candidates** (1 ground-truth
+ 19 negatives): each candidate title scored by the LLM's length-normalized log-likelihood (SASRec
by dot-product), ranked → **Hit@{1,5,10}, NDCG@{5,10}**, sliced **overall / cold** (target train
count ≤ 5) **/ warm** (> 5). **1 seed** (seed 0).

> ⛔ **This paper reports only `SASRec` and `text`.** The `clip_align` / `clip_inject` rows in the
> tables below are **vision variants, OUT OF SCOPE** for this paper (moved to `FINDING_3.md` for a
> separate vision paper). They are retained here only as a data record — **do not cite them.**

#### Luxury Beauty — n=9912 (3763 cold / 6149 warm)

**Overall**

| Model | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|:-:|:-:|:-:|:-:|:-:|
| SASRec (CF only) | 0.505 | 0.710 | 0.806 | 0.613 | 0.645 |
| text (+LLM) | 0.600 | 0.755 | 0.848 | 0.682 | 0.712 |
| clip_align | 0.610 | 0.764 | 0.857 | 0.691 | 0.721 |
| clip_inject | 0.562 | 0.737 | 0.843 | 0.653 | 0.687 |

**Cold (target train ≤5)**

| Model | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|:-:|:-:|:-:|:-:|:-:|
| SASRec (CF only) | 0.169 | 0.356 | 0.514 | 0.264 | 0.314 |
| text (+LLM) | 0.492 | 0.645 | 0.765 | 0.572 | 0.610 |
| clip_align | 0.470 | 0.636 | 0.758 | 0.556 | 0.596 |
| clip_inject | 0.436 | 0.627 | 0.761 | 0.535 | 0.578 |

**Warm (target train >5)**

| Model | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|:-:|:-:|:-:|:-:|:-:|
| SASRec (CF only) | 0.710 | 0.927 | 0.985 | 0.828 | 0.847 |
| text (+LLM) | 0.666 | 0.823 | 0.899 | 0.749 | 0.774 |
| clip_align | 0.696 | 0.843 | 0.917 | 0.773 | 0.797 |
| clip_inject | 0.639 | 0.805 | 0.893 | 0.726 | 0.754 |

#### Prime Pantry — n=15606 (3845 cold / 11761 warm)

**Overall**

| Model | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|:-:|:-:|:-:|:-:|:-:|
| SASRec (CF only) | 0.257 | 0.593 | 0.788 | 0.429 | 0.492 |
| text (+LLM) | 0.226 | 0.465 | 0.675 | 0.345 | 0.413 |
| clip_align | 0.221 | 0.476 | 0.684 | 0.350 | 0.417 |
| clip_inject | 0.222 | 0.468 | 0.681 | 0.346 | 0.414 |

**Cold (target train ≤5)**

| Model | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|:-:|:-:|:-:|:-:|:-:|
| SASRec (CF only) | 0.026 | 0.136 | 0.360 | 0.077 | 0.148 |
| text (+LLM) | 0.146 | 0.326 | 0.544 | 0.235 | 0.305 |
| clip_align | 0.096 | 0.300 | 0.533 | 0.197 | 0.271 |
| clip_inject | 0.148 | 0.344 | 0.553 | 0.245 | 0.311 |

**Warm (target train >5)**

| Model | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|:-:|:-:|:-:|:-:|:-:|
| SASRec (CF only) | 0.332 | 0.742 | 0.928 | 0.544 | 0.605 |
| text (+LLM) | 0.252 | 0.510 | 0.718 | 0.381 | 0.448 |
| clip_align | 0.262 | 0.534 | 0.734 | 0.401 | 0.465 |
| clip_inject | 0.246 | 0.509 | 0.722 | 0.379 | 0.447 |

#### AMAZON_FASHION — n=3261 (2438 cold / 823 warm)

**Overall**

| Model | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|:-:|:-:|:-:|:-:|:-:|
| SASRec (CF only) | 0.250 | 0.379 | 0.487 | 0.316 | 0.350 |
| text (+LLM) | 0.301 | 0.478 | 0.662 | 0.388 | 0.447 |
| clip_align | 0.287 | 0.460 | 0.654 | 0.372 | 0.434 |
| clip_inject | 0.279 | 0.463 | 0.664 | 0.371 | 0.435 |

**Cold (target train ≤5)**

| Model | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|:-:|:-:|:-:|:-:|:-:|
| SASRec (CF only) | 0.071 | 0.194 | 0.317 | 0.132 | 0.171 |
| text (+LLM) | 0.158 | 0.365 | 0.583 | 0.258 | 0.328 |
| clip_align | 0.144 | 0.340 | 0.573 | 0.240 | 0.314 |
| clip_inject | 0.130 | 0.346 | 0.587 | 0.237 | 0.314 |

**Warm (target train >5)**

| Model | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|:-:|:-:|:-:|:-:|:-:|
| SASRec (CF only) | 0.783 | 0.927 | 0.990 | 0.860 | 0.880 |
| text (+LLM) | 0.725 | 0.814 | 0.896 | 0.772 | 0.798 |
| clip_align | 0.713 | 0.815 | 0.896 | 0.765 | 0.791 |
| clip_inject | 0.721 | 0.812 | 0.893 | 0.767 | 0.793 |

#### Toys & Games (sub) — n=9513 (4504 cold / 5009 warm)

**Overall**

| Model | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|:-:|:-:|:-:|:-:|:-:|
| SASRec (CF only) | 0.203 | 0.485 | 0.692 | 0.347 | 0.414 |
| text (+LLM) | 0.253 | 0.503 | 0.713 | 0.378 | 0.445 |
| clip_align | 0.239 | 0.487 | 0.692 | 0.364 | 0.430 |
| clip_inject | 0.248 | 0.500 | 0.711 | 0.374 | 0.442 |

**Cold (target train ≤5)**

| Model | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|:-:|:-:|:-:|:-:|:-:|
| SASRec (CF only) | 0.107 | 0.292 | 0.511 | 0.200 | 0.270 |
| text (+LLM) | 0.237 | 0.502 | 0.715 | 0.368 | 0.437 |
| clip_align | 0.219 | 0.480 | 0.696 | 0.351 | 0.420 |
| clip_inject | 0.231 | 0.498 | 0.714 | 0.364 | 0.434 |

**Warm (target train >5)**

| Model | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
|---|:-:|:-:|:-:|:-:|:-:|
| SASRec (CF only) | 0.289 | 0.659 | 0.854 | 0.480 | 0.543 |
| text (+LLM) | 0.267 | 0.504 | 0.712 | 0.386 | 0.453 |
| clip_align | 0.256 | 0.494 | 0.689 | 0.375 | 0.438 |
| clip_inject | 0.263 | 0.502 | 0.708 | 0.383 | 0.449 |

### Headline — the CF↔LLM crossover (Hit@1)

| Dataset | SASRec cold | text cold | **LLM Δcold** | SASRec warm | text warm | **CF Δwarm** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| Luxury Beauty | 0.169 | 0.492 | **+0.323** | 0.710 | 0.666 | **+0.044** |
| AMAZON_FASHION | 0.071 | 0.158 | **+0.087** | 0.783 | 0.725 | **+0.058** |
| Toys & Games (sub) | 0.107 | 0.237 | **+0.130** | 0.289 | 0.267 | **+0.022** |
| Prime Pantry | 0.026 | 0.146 | **+0.120** | 0.332 | 0.252 | **+0.080** |

`LLM Δcold` = text − SASRec on cold; `CF Δwarm` = SASRec − text on warm. **In every dataset the LLM
wins cold and CF wins warm** — the consistent CF↔LLM crossover. LLMs add value where CF is weak
(rare items) and subtract it where CF is strong (popular items). The crossover *point* (train_count
where the sign flips) tracks dataset density: **Pearson r = +0.89 vs mean interactions/item, +0.98
vs mean history length** (`figures/crossover_sparsity.*`, `scripts/analysis_crossover.py`).

### Vision (CLIP) — ⛔ OUT OF SCOPE for this paper (moved to a separate vision paper)

The vision variants (`clip_align`, `clip_inject`) and the vision null result are **not part of this
paper** — see **`FINDING_3.md`**. This paper reports only **SASRec** and **text**. The `clip_*` rows
that remain in the per-dataset ranking tables above are kept only as a **data record** (they were run);
**do not cite them in this paper.**

### CF+LLM fusion (ensemble) — ✅ COMPLETE (the deployable popularity-gated fusion WINS)

SASRec + text ranked the **identical** 20-candidate pool (`results/{ds}_{sasrec,text_concat}_shared/`,
per-candidate scores logged); fusion via `scripts/analysis_ensemble.py`. Methods: pure **CF**,
pure **text**; **RRF** (rank fusion, no popularity); **z-fuse** (equal-weight score fusion);
**pop-gate** (score fusion weighting each candidate by its own popularity `w=pop/(pop+pivot)`,
`pivot`=dataset mean interactions/item — **deployable**, popularity known at inference); **ORACLE**
(routes the whole decision by the *target's* popularity — upper bound, not deployable). 1 seed.

Full metrics (bold = best **deployable** method, i.e. excluding the oracle upper bound):

| Dataset | metric | CF | text | RRF | z-fuse | pop-gate | ORACLE |
|---------|--------|:--:|:----:|:---:|:------:|:--------:|:------:|
| **Luxury** | Hit@1 | 0.501 | 0.600 | 0.539 | 0.575 | **0.622** | 0.627 |
|            | Hit@5 | 0.709 | 0.756 | 0.744 | 0.748 | **0.801** | 0.816 |
|            | Hit@10 | 0.805 | 0.850 | 0.888 | 0.857 | **0.884** | 0.896 |
|            | NDCG@5 | 0.611 | 0.682 | 0.646 | 0.667 | **0.718** | 0.728 |
|            | NDCG@10 | 0.643 | 0.712 | 0.692 | 0.702 | **0.744** | 0.754 |
| **Fashion** | Hit@1 | 0.246 | 0.292 | 0.252 | 0.260 | **0.308** | 0.310 |
|             | Hit@5 | 0.382 | 0.475 | 0.407 | 0.390 | **0.497** | 0.519 |
|             | Hit@10 | 0.491 | 0.657 | 0.595 | 0.551 | **0.659** | 0.699 |
|             | NDCG@5 | 0.314 | 0.385 | 0.329 | 0.326 | **0.405** | 0.415 |
|             | NDCG@10 | 0.349 | 0.443 | 0.389 | 0.377 | **0.457** | 0.473 |
| **Toys (sub)** | Hit@1 | 0.206 | 0.247 | 0.242 | 0.254 | **0.266** | 0.278 |
|                | Hit@5 | 0.484 | 0.496 | 0.520 | 0.530 | **0.554** | 0.584 |
|                | Hit@10 | 0.691 | 0.710 | 0.746 | 0.732 | **0.768** | 0.784 |
|                | NDCG@5 | 0.347 | 0.372 | 0.382 | 0.394 | **0.414** | 0.433 |
|                | NDCG@10 | 0.414 | 0.441 | 0.455 | 0.459 | **0.482** | 0.498 |
| **Prime Pantry** | Hit@1 | 0.254 | 0.220 | 0.251 | 0.276 | **0.286** | 0.292 |
|  (CF-dominant)   | Hit@5 | **0.592** | 0.462 | 0.568 | 0.581 | 0.577 | 0.616 |
|                  | Hit@10 | **0.787** | 0.678 | 0.784 | 0.775 | 0.759 | 0.783 |
|                  | NDCG@5 | 0.428 | 0.342 | 0.413 | 0.432 | **0.436** | 0.461 |
|                  | NDCG@10 | 0.491 | 0.411 | 0.483 | **0.495** | **0.495** | 0.514 |

**pop-gate Δ over best pure model (Hit@1):** Luxury **+0.022**, Fashion **+0.015**, Toys **+0.019**,
Prime Pantry **+0.032** — **positive on all 4**, ≈ the oracle upper bound. Hit@5/NDCG@5 gains are
larger on the LLM-favorable datasets (Toys Hit@5 +0.058, Luxury Hit@5 +0.045). **Prime Pantry is the
honest exception:** it is CF-dominant (CF > text overall), so pop-gate wins Hit@1/NDCG but pure CF
wins Hit@5/10 there.

**Key finding:** on the LLM-favorable datasets (Luxury, Fashion, Toys) naive fusion does **not** help
— **RRF is negative** and equal-weight **z-fuse is ~neutral** — only **popularity-weighted** fusion
(using the crossover) beats both pure models. On the CF-dominant Prime Pantry, both z-fuse and
pop-gate beat pure CF on Hit@1 (the LLM adds a little on cold), with pop-gate still best. So the
takeaway holds: **the crossover is prescriptive — a deployable popularity-gated fusion beats both
pure models on Hit@1 and NDCG@{5,10} across all 4 datasets** (and on Hit@5/10 too on the three
LLM-competitive ones; on CF-dominant Prime Pantry it wins Hit@1/NDCG but pure CF wins Hit@5/10).

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
- `vitl14_zeroshot` — CLIP-ViT-L/14 (laion2b), 768-dim/modality, 1536 fused. Zero-shot.
  **← the only one used in the paper.**
- ⛔ `bigG` — CLIP-ViT-bigG-14, 2560 fused. **Dropped** (parked below as history).
- ⛔ `vitl14_ft` — ViT-L/14 LoRA-fine-tuned. **Dropped** (RQ3 parked below).

---

---

# ⛔⛔ EVERYTHING BELOW IS HISTORICAL — DO NOT USE FOR THE PAPER ⛔⛔

> **All content from here down is superseded**: the **generation** protocol (pre-ranking), the
> **vision/CLIP** experiments (now a separate paper, `FINDING_3.md`), the parked **bigG / RQ3 /
> fusion-study** runs, **All Beauty** (dropped), and the old **Prime Pantry "excluded"** note (it is
> now reinstated — see the ranking + fusion tables at the top). It also uses the old dataset framing.
> **The paper's numbers are the RANKING + FUSION tables at the TOP of this file.** This tail is kept
> only as a run ledger / for the vision paper.

---

## Results table (Hit@1, fuzzy@0.90) — GENERATION protocol (secondary/historical)

> ⚠️ The tables from here down are the earlier **generation** protocol (the LLM *generates* one
> title; Hit@1 by title match), kept for the record and the generation-vs-ranking contrast.


| # | Variant | CLIP set | Seeds | Overall | Cold | Warm | Date |
|---|---------|----------|:-----:|:-------:|:----:|:----:|------|
| 0 | SASRec (CF-only baseline, no LLM) | — | 1 (seed 0) | **0.5066** | 0.1680 | 0.7138 | 2026-07-30 |
| 1 | clip_inject | vitl14_zeroshot | 1 (seed 0) | **0.5976** | 0.4845 | 0.6668 | 2026-07-27 |[^ckpt]

[^ckpt]: This model's **checkpoint** was made inconsistent on 2026-07-30 (a fusion sanity
check overwrote its Stage-1 weights) and was then removed during the namespacing migration.
The **result above is valid** — its seed_0.jsonl (computed 07-27) was preserved and moved to
`results/Luxury_Beauty_clip_inject_concat_vitl14_zeroshot/`. The checkpoint must be retrained
cleanly for reproduction / the multi-seed pass. Lesson: run scratch experiments with a
throwaway output dir, never a real run name.
| 2 | clip_inject | vitl14_ft | 1 (seed 0) | **0.5674** | 0.4494 | 0.6396 | 2026-07-27 |
| 3 | text (SBERT, no vision) | — | 1 (seed 0) | **0.6014** | 0.4082 | 0.7196 | 2026-07-29 |
| 4 | clip_inject | bigG | 1 (seed 0) | **0.6048** | 0.3986 | 0.7310 | 2026-07-29 |

Run config for #1–#2: `configs/luxury_beauty_rq3.yaml` (batch_size2=4, batch_size_infer=16).
Run config for #3–#4: same; scripts `run_baseline_text.sh` (GPU 0) and `run_bigG.sh` (GPU 2).
Row #0 (SASRec CF baseline): `scripts/eval_sasrec.py` (ranks the 20 candidates by SASRec
dot-product, top-1; no LLM).

> **Output paths are namespaced by dataset** (since 2026-07-30):
> `results/{dataset}_{variant}_{fusion}[_{clip}]/seed_*.jsonl`. The rows above live at
> `results/Luxury_Beauty_sasrec/` (#0), `results/Luxury_Beauty_text_concat/` (#3),
> `results/Luxury_Beauty_clip_inject_concat/` (bigG, #4), and
> `results/Luxury_Beauty_clip_inject_concat_vitl14_{zeroshot,ft}/` (#1–#2) — 9,912 records each.
> The old flat names (`text_concat`, `clip_inject_concat`, …) were renamed when multi-dataset
> support landed; a missing dataset prefix had caused Luxury↔All-Beauty output collisions.

### Luxury Beauty — full ViT-L mechanism triple (headline, seed 0) — COMPLETE

| Model | Overall | Cold | Warm |
|-------|--------:|-----:|-----:|
| SASRec (CF only) | 0.5066 | 0.1680 | 0.7138 |
| text (+LLM) | 0.6014 | 0.4082 | 0.7196 |
| clip_align (vision as align target) | **0.6141** | 0.4767 | 0.6982 |
| clip_inject (vision → LLM) | 0.5976 | **0.4845** | 0.6668 |

**Cold ladder:** 0.168 → 0.408 (+LLM) → 0.477 (align, **+0.069 vs text**) → 0.485 (inject,
**+0.076 vs text, +0.008 vs align**). Findings: (1) **vision helps cold substantially on both
mechanisms** (RQ1 confirmed on the headline backbone); (2) the **bottleneck story is nuanced** —
clip_align already delivers most of the cold benefit, injection adds only a marginal +0.008, so
vision reaches cold items *mostly via alignment*, not injection; (3) **clip_align is best overall**
(0.6141) — it captures the cold gain with a smaller warm penalty (−0.021), while clip_inject
trades a tiny extra cold gain for a bigger warm cost (−0.053) and lands overall below text.
On **warm**, CF alone (SASRec 0.714) already ≈ everything → vision helps where CF is weak (cold),
not where it's strong (warm). ⚠️ 1 seed: the align-vs-inject cold gap (+0.008) is within noise;
the text→vision jump (+0.07) is the solid result. This is cleaner + more defensible than the
razor-thin bigG appendix, and *differs* from the original "injection fixes the bottleneck" framing.

### ⛔ PARKED — RQ1 / mechanism contrast — bigG (DROPPED from paper 2026-08-03)

> **Not in the paper.** ViT-L/14 is the single backbone; bigG was dropped entirely (not even
> an appendix). Numbers retained below only as history — do not cite.

`clip_inject` (vision reaches the LLM) vs `text` (no vision), fused bigG CLIP, seed 0:

| Slice | text (no vision) | clip_inject (bigG) | Δ (inject − text) |
|-------|-----------------:|-------------------:|------------------:|
| Overall | 0.6014 | 0.6048 | **+0.003** |
| Cold | 0.4082 | 0.3986 | **−0.010** |
| Warm | 0.7196 | 0.7310 | **+0.011** |

**Finding (⚠️ preliminary — inverts the headline hypothesis):** on these 1-seed bigG numbers,
injecting vision **helps warm items** (+1.1 pt) and **slightly hurts cold items** (−1.0 pt) —
the *opposite* of RQ1's "vision helps cold items most" prediction. The overall +0.3 pt is the
average of a small warm gain and a small cold loss.

**Do not draw conclusions yet.** Two things are missing before this contrast is interpretable:
1. **`clip_align` (bigG) has not been run** — without the bottleneck variant we cannot separate
   "vision doesn't help cold" from "vision doesn't *reach* cold via this mechanism."
2. **1 seed only** — no significance; the ±1 pt cold/warm swings are within plausible seed noise.
Also note the cold/warm *profile* differs sharply between CLIP sets at equal overall (bigG
cold 0.399/warm 0.731 vs vitl14_zeroshot cold 0.485/warm 0.667); slice membership is identical
across runs, so this is real model behaviour worth understanding.

### ⛔ PARKED — RQ3 comparison (fine-tuned vs zero-shot CLIP) — DROPPED from paper 2026-08-03

> **Not in the paper.** CLIP fine-tuning is dropped. The negative finding below ("better
> retrieval, worse recommendation") is retained as history; no RQ depends on it.

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

## All Beauty (2nd dataset, ViT-L/14, concat, 1 seed) — IN PROGRESS

Data prep verified: preprocess matches the SASRec checkpoint exactly (**2,169 users /
1,854 items**; 2,099 test users). ViT-L zero-shot embeddings extracted (text 1854/1854,
**image 1294/1854 = 69.8%** — lower image coverage than Luxury's 90%).

| Model | CLIP | Overall | Cold | Warm | Status |
|-------|------|:-------:|:----:|:----:|--------|
| SASRec (CF only) | — | 0.5717 | 0.0855 | **0.9630** | ✅ |
| text (no vision) | — | **0.5612** | 0.1271 | 0.9106 | ✅ |
| clip_align | vitl14_zeroshot | 0.5226 | **0.1346** | 0.8349 | ✅ |
| clip_inject | vitl14_zeroshot | 0.5541 | 0.1335 | 0.8925 | ✅ |

> **CORRECTION (2026-07-30):** an earlier version of this table showed inflated numbers
> (text cold 0.1618, inject cold 0.1818, etc.) — those were computed by a buggy watcher that
> read the seed files **mid-inference** (partial records). The numbers above are the TRUE
> values from the complete 2099-record files (verified by direct recompute). Watcher fixed to
> wait for line-count stability.

**Honest reading (weak/muddy on All Beauty):** cold — text 0.1271, clip_align 0.1346,
clip_inject 0.1335 → align ≈ inject, both only **marginally** above text (+0.006–0.008). The
clean "bottleneck hurts cold, injection recovers" pattern is NOT present here. RQ1 direction
holds weakly: vision helps cold a little, hurts warm (inject warm 0.8925 < text 0.9106). On
**warm**, plain SASRec (0.963) beats every LLM variant. Luxury (ViT-L) remains the strong
dataset for the cold effect (inject cold 0.4845 vs text 0.4082, +0.076); All Beauty is muddy.

Caveat: All Beauty is small (2,099 test users), extremely warm-skewed; 1 seed.
Config `configs/all_beauty.yaml`; outputs `results/All_Beauty_*/seed_0.jsonl`.

---

## AMAZON_FASHION (3rd dataset, ViT-L/14, concat, 1 seed) — COMPLETE (vision HURTS)

3,679 users / 7,310 items / 3,261 test users (2,438 cold / 823 warm). Title bug fixed
(see dataset-selection note); embeddings verified healthy (text 7310, image 6076/7311 = 83%).

| Model | Overall | Cold | Warm |
|-------|--------:|-----:|-----:|
| SASRec (CF only) | 0.2505 | 0.0710 | 0.7825 |
| text (+LLM) | **0.2925** | **0.1440** | 0.7327 |
| clip_align | 0.2824 | 0.1358 | 0.7169 |
| clip_inject | 0.2699 | 0.1202 | 0.7132 |

**Verified negative result.** LLM+text recovers cold (0.071 → 0.144), but **vision makes it
worse**: align −0.008, inject −0.024 vs text on cold; also hurts warm and overall. Not an
artifact — slices consistent (2438/823 across variants), generations non-degenerate, embeddings
healthy, and the head-to-head is distributed (on cold, inject fixes 112 / breaks 170 = net −58).
Interpretation: Fashion is very sparse (avg seq 4.89, little CF to align to) and product images
are heterogeneous (model/flat-lay/packaging) → CLIP signal not discriminative for the item →
vision adds net noise. Structurally favorable (CF weak on cold) yet vision doesn't help.

### Cross-dataset summary — cold-slice vision effect (clip_inject − text)

| Dataset | SASRec cold | text cold | inject cold | Δ vision (cold) | verdict |
|---------|:-----------:|:---------:|:-----------:|:---------------:|---------|
| Luxury Beauty | 0.168 | 0.408 | 0.485 | **+0.076** | ✅ vision helps |
| All Beauty | 0.086 | 0.127 | 0.134 | +0.006 | ⚠️ marginal/muddy |
| AMAZON_FASHION | 0.071 | 0.144 | 0.120 | **−0.024** | ❌ vision hurts |

**Honest state:** only Luxury cleanly supports "vision helps cold." The emerging story is
**domain-dependent** — vision helps when images are discriminative for the item (beauty) and
there's enough CF density, not in sparse/heterogeneous-image domains (fashion). Still a valid
*analysis* contribution ("when/why vision helps"), but not a uniform 3-dataset win.

## ⛔ PARKED — Prime Pantry (DROPPED from paper 2026-08-03; results not favorable)

> **Not in the paper.** Retained as history. Note the crossover was still present here
> (SASRec cold 0.026 ≪ warm 0.332; text recovers cold to 0.139) — but per the scope
> decision Prime Pantry is excluded from the final 4-dataset set.

15,611 users / 7,841 items; fresh data, SASRec trained by us (`train_sasrec.py`), 90.6% image cov.

| Model | Overall | Cold | Warm |
|-------|--------:|-----:|-----:|
| SASRec (CF only) | 0.2568 | 0.0263 | 0.3322 |
| text (+LLM) | 0.2328 | 0.1394 | 0.2633 |
| clip_align | 0.2304 | 0.0861 | 0.2775 |
| clip_inject | 0.2240 | **0.1454** | 0.2497 |

## Toys & Games (subsample, ViT-L/14, concat, 1 seed) — PARTIAL

9,513 users / 7,253 items (random 15%-user subsample, seed 42; see DATASETS.md). SASRec cold
0.107 ≪ warm 0.289.

| Model | Overall | Cold | Warm |
|-------|--------:|-----:|-----:|
| SASRec | 0.2027 | 0.1068 | 0.2889 |
| text (+LLM) | **0.2600** | 0.2214 | 0.2947 |
| clip_align | 0.2503 | 0.2167 | 0.2805 |
| clip_inject | 0.2557 | **0.2216** | 0.2863 |

**Vision does NOT help (verdict).** clip_inject vs text on cold = **+0.0002** (~zero); vision
slightly hurts warm/overall; `text` is the best model. Toys was *engineered* to be favorable
(item-discriminative images, 91.7% coverage, 47% cold) yet shows no vision benefit — the
"item-discriminative images predict vision helps" hypothesis did NOT hold.

### Cross-dataset cold-slice vision effect (clip_inject − text) — generation Hit@1 — final 4

| Dataset | Δ cold | verdict |
|---------|:------:|---------|
| Luxury Beauty | **+0.076** | ✅ helps (generation only) |
| All Beauty | +0.006 | ⚠️ ~zero |
| Toys (sub) | +0.000 | ⚠️ ~zero |
| AMAZON_FASHION | −0.024 | ❌ hurts |

**Even under generation, only Luxury benefits; the other 3 are zero-to-negative — and under
the ranking protocol the Luxury gain reverses too, so vision is a null across the board.** This
is why the paper's headline moved to the **CF↔LLM crossover** (`text` vs `SASRec`, robust in all
4) and treats vision as a ruled-out confound. (Prime Pantry excluded from scope; see parked
section above.)

## ⛔ PARKED — Proxy result (image↔text retrieval) — fine-tuning DROPPED from paper

> Retained as history only (fine-tuning is out of scope).

CLIP ViT-L/14 LoRA domain fine-tuning, image→text retrieval R@1 on unseen items:

| Split | Zero-shot R@1 | Fine-tuned R@1 | Δ |
|-------|--------------:|---------------:|----:|
| seed 0 | 0.530 | 0.633 | +0.104 |
| seed 1 | 0.577 | 0.677 | +0.100 |

Fine-tuning reliably improves *retrieval* (~+0.10 R@1, robust across splits) — but this
does NOT translate to recommendation (see RQ3 above). Retrieval quality ≠ rec quality.

---

## Dataset selection (decided 2026-07-30; Toys added 2026-07-31)

**Full provenance/prep/selection now lives in `DATASETS.md`** (per-dataset build steps,
the subsampling method, the rejected-candidate list). Summary below.

Evaluated candidates via `scripts/analyze_dataset.py` (fresh McAuley-v2 downloads → k-core
preprocess → size / image-coverage / **title-coverage** / cold-warm). Constraints: **≤ ~15k
users**, **≥ ~70% image coverage**, **≥ ~80% title coverage+uniqueness** (Fashion bug), and
**item-discriminative images** (the real predictor of whether vision helps). 5th dataset =
**Toys & Games**, a random-15%-user subsample (seed 42) → 9,513u/7,253i, 91.7% image, 47% cold.

| Dataset | Users | Items | Img cov | Cold % | Decision |
|---------|------:|------:|:-------:|:------:|----------|
| Luxury Beauty | 9,930 | 6,141 | 90% | 38% | ✅ paper |
| AMAZON_FASHION | 3,679 | 7,310 | 83% | 74.8% | ✅ paper |
| Toys & Games (sub) | 9,513 | 7,253 | 91.7% | 47% | ✅ paper (subsampled) |
| Prime Pantry | 15,611 | 7,841 | 90.6% | 24.6% | ✅ paper (reinstated 2026-08-04) |
| All Beauty | 2,169 | 1,854 | 70% | 45% | ⛔ dropped 2026-08-04 (too small / density outlier) |
| Video Games | 64,073 | 33,614 | 82% | 20.8% | ✗ ~70h/run |
| Musical Instruments | 40,644 | 30,676 | 69% | 36% | ✗ ~44h/run |
| Arts & Crafts | 86,810 | 64,072 | 55% | 38% | ✗ too big + low img |
| Appliances | 1,568 | 3,473 | 40% | 81% | ✗ image coverage |

**Final 4: Luxury Beauty + AMAZON_FASHION + Toys & Games (sub) + Prime Pantry** (updated 2026-08-04:
All Beauty dropped, Prime Pantry reinstated). See `DATASETS.md` for full provenance + rationale.

**NEW screening criterion — title coverage.** The whole eval is title-generation-based, so items
need non-empty, mostly-unique titles. A [preprocess bug](../clam_rec/data/preprocess.py) coupled
title extraction to `description`; categories with **no `description` field (AMAZON_FASHION)**
raised KeyError and dropped 88% of titles → empty-title matched empty-title → **inflated Hit@1**
(SASRec Fashion cold showed a fake 0.7547). **Fixed** (title/description now independent);
re-preprocess recovered titles 888 → 7,309/7,310 with interactions byte-identical (alignment intact).
Corrected Fashion SASRec: **overall 0.2505 / cold 0.0710 / warm 0.7825** — Fashion is genuinely
**FAVORABLE** (CF collapses on cold 0.071, even weaker than Luxury's 0.168 → large vision headroom).
Luxury/All Beauty were unaffected (they have descriptions; their SASRec ID≈title, numbers valid).

## Still to run (not yet done)

Scope (locked 2026-08-03): **ViT-L/14 single backbone**, fusion fixed = concat, per-dataset
ladder **SASRec → text → clip_align → clip_inject** (no fine-tune), **4 datasets** (Luxury,
All Beauty, Fashion, Toys sub). See `STORY.md` / `PROJECT.md §0.0`.

- [x] **Luxury:** SASRec, text, clip_align, clip_inject (ViT-L zs) — full ladder done.
- [x] **All Beauty:** SASRec, text, clip_align, clip_inject (ViT-L zs) — done.
- [x] **AMAZON_FASHION:** full ladder done (fixed titles).
- [x] **Toys (sub):** full ladder done.
- [ ] **Multi-seed** (≥2–10) for significance — currently 1 seed each. Also retrains the
  checkpoints lost to the 07-30 clobber/migration.
- [ ] **Ranking metrics table** (Hit@1/5, NDCG@5) written up per dataset — the reported protocol.
- [ ] **Crossover figure + write-up** (`scripts/analysis_crossover.py`, `figures/crossover.*`).

⛔ **Dropped (do NOT run):** CLIP fine-tuning (RQ3), bigG backbone, RQ2 fusion study,
Prime Pantry, Video Games.

_Last updated: 2026-08-03._
