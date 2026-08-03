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

### RQ1 / mechanism contrast — bigG (NOW APPENDIX ONLY; ViT-L is the headline backbone)

> Per the 2026-07-30 decision (see `STORY.md`), **ViT-L/14 is the single headline backbone**;
> bigG is kept only as an appendix showing the cold/warm effect is *backbone-dependent*.
> The bigG numbers below stand as that appendix evidence.

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
| clip_inject + fine-tune | vitl14_ft | — | — | — | ⏳ needs per-dataset LoRA fine-tune |

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

## Prime Pantry (3rd dataset, ViT-L/14, concat, 1 seed) — COMPLETE (marginal/muddy)

15,611 users / 7,841 items; fresh data, SASRec trained by us (`train_sasrec.py`), 90.6% image cov.

| Model | Overall | Cold | Warm |
|-------|--------:|-----:|-----:|
| SASRec (CF only) | 0.2568 | 0.0263 | 0.3322 |
| text (+LLM) | 0.2328 | 0.1394 | 0.2633 |
| clip_align | 0.2304 | 0.0861 | 0.2775 |
| clip_inject | 0.2240 | **0.1454** | 0.2497 |

**Cold:** clip_inject vs text **+0.006** (marginal, like All Beauty; grocery = moderate visual).
Notable **mechanism datapoint**: clip_align HURTS cold (−0.053 vs text) and clip_inject recovers
it (+0.059 over align) — the bottleneck genuinely matters here (cleaner align-vs-inject
separation than Luxury). Overall vision slightly negative. Verdict: another marginal result.

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

### Cross-dataset cold-slice vision effect (clip_inject − text) — generation Hit@1

| Dataset | Δ cold | verdict |
|---------|:------:|---------|
| Luxury Beauty | **+0.076** | ✅ helps |
| All Beauty | +0.006 | ⚠️ ~zero |
| Prime Pantry | +0.006 | ⚠️ ~zero |
| Toys (sub) | +0.000 | ⚠️ ~zero |
| AMAZON_FASHION | −0.024 | ❌ hurts |

**Only Luxury Beauty benefits — 4 other datasets (incl. the engineered-favorable Toys) show
zero-to-negative.** Consistent with EXP-3 (the Luxury gain tracks CLIP-text > SBERT, not the
image). Paper thesis tightens to "vision helps only in narrow conditions; here, Luxury Beauty
specifically — and we show why." (These are generation Hit@1; ranking eval pending → re-verify.)

## Proxy result (not recommendation — image↔text retrieval, held-out items)

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
| Luxury Beauty | 9,930 | 6,141 | 90% | 38% | ✅ #1 (strong) |
| AMAZON_FASHION | 3,679 | 7,310 | 83% | 74.8% | ✅ #2 (training) |
| Prime Pantry | 15,611 | 7,841 | 90.6% | 24.6% | ✅ #3 (LOCKED, prep pending) |
| All Beauty | 2,169 | 1,854 | — | — | dropped (muddy/tiny) |
| Video Games | 64,073 | 33,614 | 82% | 20.8% | ✗ ~70h/run |
| Musical Instruments | 40,644 | 30,676 | 69% | 36% | ✗ ~44h/run |
| Arts & Crafts | 86,810 | 64,072 | 55% | 38% | ✗ too big + low img |
| Appliances | 1,568 | 3,473 | 40% | 81% | ✗ image coverage |

**Final 3: Luxury Beauty + AMAZON_FASHION + Prime Pantry.** Fashion's SASRec matched the
existing checkpoint (3679/7310) so it was reusable.

**NEW screening criterion — title coverage.** The whole eval is title-generation-based, so items
need non-empty, mostly-unique titles. A [preprocess bug](../clam_rec/data/preprocess.py) coupled
title extraction to `description`; categories with **no `description` field (AMAZON_FASHION)**
raised KeyError and dropped 88% of titles → empty-title matched empty-title → **inflated Hit@1**
(SASRec Fashion cold showed a fake 0.7547). **Fixed** (title/description now independent);
re-preprocess recovered titles 888 → 7,309/7,310 with interactions byte-identical (alignment intact).
Corrected Fashion SASRec: **overall 0.2505 / cold 0.0710 / warm 0.7825** — Fashion is genuinely
**FAVORABLE** (CF collapses on cold 0.071, even weaker than Luxury's 0.168 → large vision headroom).
Luxury/All Beauty were unaffected (they have descriptions; their SASRec ID≈title, numbers valid).
Fashion ladder re-launched 2026-07-30 with fixed titles. Prime Pantry: prep pending (use fixed
preprocess + verify title coverage — it may also lack descriptions).

## Still to run (not yet done)

Plan (locked 2026-07-30): **ViT-L/14 single backbone**, **RQ2/fusion dropped**, per-dataset
ladder **SASRec → text → clip_align → clip_inject → clip_inject+fine-tune**, all 3 datasets,
1 seed for now. See `STORY.md` / `PROJECT.md §0.0`.

- [x] **Luxury:** SASRec, text, clip_inject (ViT-L zs + ft) done. `clip_align` (ViT-L) 🔄 training.
- [x] **All Beauty:** SASRec done; text / clip_align / clip_inject (ViT-L zs) 🔄 training.
- [ ] **All Beauty RQ3** — per-dataset ViT-L LoRA fine-tune → `clip_inject` (vitl14_ft).
- [ ] **Video Games** — full prep (preprocess → verify → ViT-L extract → ladder). The big one
  (≈27k items / 17k images).
- [ ] **Multi-seed** (≥10) for significance — currently 1 seed each. Also retrains the
  checkpoints lost to the 07-30 clobber/migration.
- [ ] bigG appendix `clip_align` (optional) — completes the bigG mechanism triple.

_Last updated: 2026-07-30._
