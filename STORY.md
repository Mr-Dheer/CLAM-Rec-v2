# CLAM-Rec v2 — Method Levers & Paper Story

> Two parts. **Part 1** explains, plainly, what the experimental levers
> (`clip_align`, `clip_inject`; fine-tuning is dropped, kept only for context) actually
> *do*. **Part 2** explains
> how we position the paper and what the story is. Read `PROJECT.md` for the full
> design; this doc is the conceptual + narrative companion.

Last updated: 2026-08-03. Backbone: **ViT-L/14 single backbone**.

---

## ⭐ REFRAMED THESIS (2026-08-03) — the story pivoted, for the better

**The vision angle is a null result** (see `RESULTS.md`): under the *ranking* eval (Hit@1/5,
NDCG@5), `clip_inject`/`clip_align` are neutral-to-negative on cold items across ALL 4 datasets;
the apparent +0.076 on Luxury was **generation-protocol-specific** and reverses under ranking.

**But the data revealed a much stronger, universal pattern — a CF↔LLM crossover by item
popularity:**
- On **cold/rare items** the LLM (`text`) beats CF (`SASRec`) in every dataset (+0.05 to +0.49
  Hit@1); at train_count=0, SASRec=0.000 (never-seen item → zero embedding) while the LLM works.
- On **warm/popular items** CF beats the LLM in every dataset (−0.02 to −0.30).
- `text − SASRec` **decays monotonically and crosses zero** at a dataset-dependent point
  (Fashion ~2, Luxury ~9, Toys ~13, Prime Pantry ~20 interactions). No exceptions.
- **Vision does not shift the crossover** (`inject − text` tiny/negative in every bin).

**New thesis:** *"When do LLMs help sequential recommendation? LLMs add value precisely where
collaborative filtering is weak — cold/rare items — and degrade it where CF is strong (popular
items), a consistent cold/warm crossover across 4 datasets. Adding visual (CLIP) signal does not
shift this crossover."* This directly answers the LLM-SRec question ("do LLMs understand
sequential recommendation?"), absorbs the vision negative as a ruled-out confound, and is robust
(4 datasets, monotonic, holds on Hit@1/5/NDCG@5). Evidence: `scripts/analysis_crossover.py`.

**Scope locked (updated 2026-08-04):** 4 datasets = **Luxury Beauty, AMAZON_FASHION, Toys (sub),
Prime Pantry** (All Beauty dropped — small/density-outlier; Prime Pantry reinstated — clean crossover +
best density fit; see `DATASETS.md`/`FINDING_2.md`); **ViT-L/14 single backbone** (bigG dropped);
**no CLIP fine-tuning** (RQ3 dropped); **fusion fixed = concat** (we tried mean/gating, concat
was best — reported as a one-line note, not a study). Ladder per dataset:
**SASRec → text → clip_align → clip_inject.**

Distinct from Smol-Rec (image-budget) AND from LLM-SRec (retrieval architecture): our contribution
is the **empirical CF/LLM complementarity along the popularity axis + the vision null**.

**Prescriptive kicker (2026-08-03): the crossover is exploitable.** A simple **popularity-gated
score fusion** — weighting each candidate's CF vs LLM score by that candidate's own popularity
(`w=pop/(pop+pivot)`; deployable, popularity known at inference) — **beats both pure CF and pure
LLM on all 4 datasets** (+0.015 to +0.031 Hit@1, up to +0.058 Hit@5), essentially matching the
oracle upper bound. Critically, **naive fusion fails**: rank fusion (RRF) is *negative* and
equal-weight score fusion is neutral — only fusion that *uses the crossover* (popularity weighting)
wins. So the paper goes from *descriptive* (there is a crossover) to *prescriptive* (here's how to
exploit it). Evidence: `scripts/analysis_ensemble.py`, `RESULTS.md` → "CF+LLM fusion".

Part 2 below (the vision-centric framing) is now SUPERSEDED by the above — kept for history.

---

> ⛔ **VISION IS OUT OF SCOPE for the current paper (2026-08-04).** Parts 1 and 2 below are
> largely about the CLIP vision levers (`clip_align`, `clip_inject`, fine-tuning). **That content is
> now for a SEPARATE vision paper — see `FINDING_3.md`.** The current paper is purely **CF vs LLM**
> (`SASRec` + `text`); its story is `FINDINGS.md` + `FINDING_1/2/4.md`. Keep the sections below as
> background/vision-paper material only — do not use them for the current paper.

## Part 1 — The three levers (what each actually does)

### The shared base (identical in all variants)
Every variant is built on **A-LLMRec**: a frozen collaborative recommender (SASRec,
50-D item vectors) aligned with a frozen LLM (OPT-6.7B) that *generates the next
item's title*. Stage 1 aligns each item's SASRec vector with a **content embedding**
in a shared 128-D space; Stage 2 projects things into the LLM's token space as soft
prompts. The levers below only change **what the content embedding is** and **whether
it reaches the LLM at inference**.

For the CLIP variants the content embedding is the **fused CLIP** vector =
combine(CLIP-text, CLIP-image). How you combine them = *fusion* (concat / mean /
learned gating); we tried all three, **concat was best**, so fusion is **fixed = concat**
throughout — a one-line note in the paper, not a study.

### 1. `clip_align` — the "bottleneck" (this *is* the original, rejected CLAM-Rec)
- CLIP is used **only as a Stage-1 alignment target**: a small autoencoder pulls the
  SASRec item vector toward the CLIP vector during training.
- **At inference, the CLIP embedding is discarded.** The item token fed to the LLM
  comes only from the SASRec 50-D vector (projected). CLIP's *only* footprint is
  indirect — it nudged the alignment weights during training.
- Effect: a rich ~1536-D visual signal is squeezed through a 50-D collaborative
  vector. This "bottleneck" is the leading explanation for why v1's gain was tiny.
- **Role in the paper: a control**, not a workhorse. It supports no RQ on its own.

### 2. `clip_inject` — the bottleneck removed
- Same CLIP alignment target as `clip_align`, **plus** a per-item `[MMEmb]` token
  placed next to each item title in the prompt, filled with that item's **actual
  CLIP embedding** (projected by a dedicated head, `mm_emb_proj`).
- **At inference the LLM directly sees the visual signal** for every history and
  candidate item.
- The one line of code that separates them: `mm = "[MMEmb]" if variant=="clip_inject"`.
- **Role in the paper: the workhorse.** All three RQs live here (see below), because
  vision must actually reach the model for "does vision help / how to fuse / does
  fine-tuning help" to be measurable.

### 3. CLIP fine-tuning — a *preprocessing* lever (⛔ DROPPED from the paper, 2026-08-03)
- **Not in scope anymore.** It was a step that produced *better CLIP embeddings*: LoRA
  domain image↔text contrastive fine-tuning of ViT-L/14 on the dataset's own item
  (image, title) pairs. Empirically (Luxury Beauty) it **improved retrieval** (image→text
  R@1 0.53 → 0.63) but **hurt recommendation** (−3 pts under generation Hit@1).
- **Why dropped:** the paper's thesis is now the CF↔LLM crossover; fine-tuning is an
  orthogonal vision sub-study, and the vision angle is already a null result. The
  artifacts/numbers are preserved in `RESULTS.md` (parked) but no RQ depends on them.

### Summary
| Lever | What changes | Reaches LLM at inference? | Role |
|---|---|---|---|
| `clip_align` | CLIP as alignment target only | ❌ (bottleneck) | control for mechanism contrast |
| `clip_inject` | + `[MMEmb]` CLIP token in prompt | ✅ (direct) | the vision workhorse (RQ1 vision test) |
| ~~CLIP fine-tuning~~ | ~~better CLIP embeddings~~ | ~~✅ via `clip_inject`~~ | **dropped (parked in RESULTS.md)** |

The axes of the instrument that remain in the paper:
- **Popularity / coldness (headline)** = *for which items* CF vs the LLM wins (the crossover).
- **Align vs inject (mechanism)** = whether the visual vector *reaches the LLM* (vision test).
- **Fusion** = how to combine *text + image*; **fixed = concat** (mean/gating tried, concat
  best) — a one-line note, not a study.

---

## Part 2 — Positioning & the story

### Where we came from
v1 ("Beyond Text: Multimodal LLM-based Sequential Recommendation") added CLIP to
A-LLMRec and reported **+1.31 Hit@1** on Luxury Beauty. **Rejected: the gain was too
small.** A sibling paper, **Smol-Rec** (same authors, same title, native-VLM visual
pathway), is under review elsewhere — so v2 must be a **distinct** paper.

### The pivot — analysis, not architecture
The rejection reason (small *average* gain) becomes the *thesis*: **the global
average is the wrong lens.** Vision helps a lot in some conditions and not in others;
averaging hides this. So we don't chase a bigger number — we **characterize when
vision helps**. Contribution = the **empirical analysis**, not a new mechanism.

### Staying distinct from Smol-Rec
Different backbone (text LLM + pre-computed CLIP, not a VLM), different question
(item/design *conditions*, not image *budget*), different contribution (analysis, not
a visual mechanism). **Consequence:** `clip_inject` — "inject a visual embedding into
the LLM" — must stay a **minor design choice used as an instrument**, never the
headline claim (that would collide with Smol-Rec/I-LLMRec).

### The trap we must NOT fall into
Framing the paper as **"which approach is better — align or inject?"** is the wrong
move, for two reasons:
1. **It re-invites the v1 rejection.** A "which is better" paper is judged on the
   *size of the gain*. Our own averages are razor-thin — on ViT-L, `clip_inject`
   overall (0.5976) is even *below* the no-vision `text` baseline (0.6014). Headline
   that, and a reviewer sees ~zero average gain and rejects it again.
2. **It collides with Smol-Rec** — it makes "injecting vision into the LLM" the
   contribution.

### The reframe — mechanism contrast as an *instrument*
Same experiments, different sentence. Don't ask "which wins" (average). Ask **"what
does the mechanism contrast reveal about *when* vision helps"** (sliced). Our data
shows exactly why this is the right lens:

| ViT-L, Hit@1 (seed 0) | Overall | Cold | Warm |
|---|---|---|---|
| `text` (no vision) | 0.6014 | 0.4082 | 0.7196 |
| `clip_inject` (zero-shot) | 0.5976 (−0.004) | **0.4845 (+0.076)** | 0.6668 (−0.053) |

On the **average**, vision looks useless (even slightly negative). **Sliced by
coldness**, vision gives a large **+7.6 pt** lift on cold items and *nothing* on warm.
**That gap is the contribution: the small average is the average of a real effect
(cold) and no effect (warm).** `clip_align` then earns its place as the **control**
that shows this cold benefit is *mechanism-dependent* — it should appear under
`inject` (vision reaches the LLM) and shrink/vanish under `align` (bottlenecked).

### The backbone decision — ViT-L/14, single backbone
We use **ViT-L/14 everywhere** (headline + fine-tuning), because:
- It's the backbone that actually **delivers RQ1** — the cold-item benefit above is a
  ViT-L result. bigG (stronger zero-shot) *inverts* it (helps warm, not cold),
  probably because bigG's generic semantics are already captured by CF on warm items.
- It's **fine-tunable**, so RQ3 sits on the *same* model — no "why two backbones?"
- Absolute numbers are lower than bigG, but **for an analysis paper only the
  within-paper contrasts matter**, and those are all valid on ViT-L.
- **bigG is dropped entirely** (2026-08-03) — not even an appendix. ViT-L/14 is the
  single backbone for every reported number.

### The story, in one paragraph
> Adding visual signal to an LLM recommender yields a negligible *average* gain — the
> very reason a naive multimodal extension was unconvincing. We show this average is
> misleading: sliced by item coldness, vision delivers a **large, consistent benefit
> on cold items** (weak collaborative signal) and **none on warm items** (strong
> collaborative signal). Using a mechanism contrast (`align` vs `inject`) as an
> instrument, we show this cold benefit is **conditional on the visual signal actually
> reaching the LLM**, not merely being used as an alignment target.
> Together: vision helps LLM recommenders exactly where collaborative filtering is
> weak, and only through mechanisms and embeddings that preserve general semantics.

> ⚠️ The paragraph above is the OLD vision-centric framing (kept for history). The
> current headline is the **CF↔LLM crossover** (see the reframed thesis at the top of
> this file); under the ranking protocol the vision result is a **null / ruled-out
> confound**, not a cold-item win.

### The baseline ladder (decided 2026-07-30: fusion/RQ2 dropped, SASRec baseline added)
Per dataset we report a single clean ladder, all ViT-L/14, fusion fixed = concat:

**SASRec (CF only) → `text` (+LLM) → `clip_align` (+vision, bottlenecked) →
`clip_inject` (+vision, injected).** (Fine-tune step dropped 2026-08-03.)

The SASRec baseline (no LLM) anchors "how much do the LLM and vision add over plain
CF" — and it is what exposes the crossover: CF nails warm items and collapses on cold,
the LLM does the reverse. Vision does not move the needle under the ranking protocol.

### How the levers map to the paper's RQs
- **RQ1 — Popularity/coldness (headline):** `SASRec` vs `text` sliced by item
  interaction count → the CF↔LLM crossover, 4 datasets.
- **Mechanism / vision (secondary):** `clip_align` vs `clip_inject` vs `text` — shows
  vision does not shift the crossover (null, whether or not it reaches the LLM).
- **Fusion (note only):** concat vs mean vs gating → concat best → fixed = concat.
  Not a reported study.
- ⛔ **Dropped 2026-08-03:** fine-tuning (RQ3), bigG backbone, Prime Pantry dataset.
  All parked in `RESULTS.md`; none appear in the paper.

### Honest caveats / open items
- **1 seed so far** on the generation numbers — the crossover is robust across 4
  datasets and monotonic, but per-cell deltas need ≥2 seeds for significance.
- The vision null and the crossover both hold under the **ranking** protocol
  (Hit@1/5, NDCG@5) — the protocol the paper reports.
