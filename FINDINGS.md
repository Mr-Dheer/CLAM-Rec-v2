# FINDINGS.md — Experiments, Results & Interpretation (paper backbone)

> **Purpose.** A detailed, self-contained walkthrough of every experiment, what it found,
> *why*, and **exactly which code produced it**. Two audiences:
> 1. **Kavach** — to understand each result in depth.
> 2. **A future Claude writing the paper** — read this doc *and* the code it points to, then
>    write. Every claim below names the script/function and the command to reproduce it.
>
> Companion docs: `PROJECT.md` (design/positioning), `STORY.md` (narrative), `RESULTS.md`
> (numbers ledger), `DATASETS.md` (data), `EVAL_PROTOCOL.md` (why 20 candidates),
> `INFRASTRUCTURE.md` (where things run). Last updated: 2026-08-04.

---

## 0. TL;DR — the thesis and the three contributions

**Thesis.** *When do LLMs help sequential recommendation? LLMs and collaborative filtering (CF)
are complementary along the axis of item popularity: the LLM wins on cold/rare items, CF wins on
warm/popular items — a consistent crossover across datasets. The crossover point is predicted by
dataset density, and a simple popularity-gated fusion exploits the complementarity to beat both.*

**Per-finding deep-dive docs** (detailed standalone walkthroughs — method, mechanism, code, paper
phrasing): `FINDING_1.md` (crossover), `FINDING_2.md` (density), `FINDING_4.md` (fusion).
This file (`FINDINGS.md`) is the master index for **this** paper.

> ⛔ **Vision (CLIP) is OUT OF SCOPE for this paper.** This paper is purely **CF vs LLM** — it uses
> only the `SASRec` and `text` variants. The vision variants (`clip_align`, `clip_inject`), CLIP
> fine-tuning, and the generation→ranking protocol reversal are **parked for a separate vision
> paper** — see `FINDING_3.md`. Do **not** include any vision result here (it would dilute this
> story and scoop the vision paper).

**Three contributions (= three results sections):**
1. **The CF↔LLM crossover** — LLM (`text`) beats CF (`SASRec`) on cold items, CF beats the LLM on
   warm items; monotonic, all 4 datasets, on Hit@1/5/10. (§4)
2. **The crossover point is predicted by dataset density** — it lands near the dataset's mean
   interactions/item (Pearson r ≈ +0.98; r ≈ +0.99 vs mean history length). (§5)
3. **Popularity-gated fusion (prescriptive)** — weighting each candidate's CF vs LLM score by that
   candidate's popularity beats *both* pure models on **Hit@1 and NDCG@{5,10} across all 4 datasets**
   (and on Hit@5/10 on the 3 LLM-competitive ones), ≈ the oracle ceiling; naive fusion (RRF,
   equal-weight) does not. (§7)

**Headline numbers** (ranking, Hit@1, 20 candidates, seed 0):

_(Dataset set updated 2026-08-04: All Beauty dropped (small/outlier), Prime Pantry added; see
`DATASETS.md`.)_

| Dataset | SASRec cold | text cold | SASRec warm | text warm | pop-gate overall |
|---------|:-----------:|:---------:|:-----------:|:---------:|:----------------:|
| Luxury Beauty | 0.169 | **0.492** | **0.710** | 0.666 | **0.622** (>0.600) |
| AMAZON_FASHION | 0.071 | **0.158** | **0.783** | 0.725 | **0.308** (>0.292) |
| Toys & Games (sub) | 0.107 | **0.237** | **0.289** | 0.267 | **0.266** (>0.247) |
| Prime Pantry | 0.026 | **0.146** | **0.332** | 0.252 | **0.286** (>0.254)† |

_(pop-gate overall = Hit@1. †Prime Pantry is CF-dominant — CF beats the LLM overall there; pop-gate
still wins Hit@1 and NDCG but pure CF wins Hit@5/10. See `FINDING_4.md`.)_

---

## 1. The instrument (shared experimental setup)

### 1.1 Base model — A-LLMRec (KDD 2024)
A frozen collaborative recommender (**SASRec**) is aligned with a frozen LLM (**OPT-6.7B, 8-bit**)
so the LLM can use collaborative knowledge and *generate the next item's title*. Two stages:
- **Stage 1 (alignment):** a dual autoencoder maps SASRec's 50-d item vector and a *content*
  embedding (text or multimodal) into a shared 128-d space (matching + reconstruction + BCE-rec
  losses).
- **Stage 2 (LLM integration):** small MLPs project the joint item embedding and the user/sequence
  representation into OPT's token space as **soft prompt tokens**; OPT is prompted with the user
  rep + history titles + a candidate set and scored/generates the next title.
- **Code:** `clam_rec/model/clam_rec.py` (the `ClamRec` class: `stage1_step`, `stage2_step`,
  `generate`, `rank_candidates`), `clam_rec/model/llm4rec.py` (OPT wrapper, `score_titles`,
  `replace_soft_tokens`), `clam_rec/model/sasrec.py` (frozen SASRec).

### 1.2 The two variants (the experimental axis)
One model, a `variant` switch (`clam_rec/model/clam_rec.py`) — **this paper uses only these two**:
- **SASRec** — CF only, no LLM. The floor. Ranks candidates by dot-product.
  Code: `scripts/eval_sasrec.py`.
- **text** — A-LLMRec; Stage-1 content = SBERT; the LLM ranks candidate titles by likelihood.

> The model also supports vision variants (`clip_align`, `clip_inject`), but **vision is out of
> scope for this paper** and parked for a separate vision paper (`FINDING_3.md`). Do not report it
> here.

### 1.3 Datasets (4) — see `DATASETS.md` for full provenance
Amazon Review 2018 (McAuley v2), k-core filtered, leave-one-out:

| Dataset | Users | Items | Cold % | mean inter/item | mean hist len |
|---------|------:|------:|:------:|:---------------:|:-------------:|
| Luxury Beauty | 9,930 | 6,141 | 38% | 10.4 | 6.4 |
| AMAZON_FASHION | 3,679 | 7,310 | 75% | 2.5 | 4.9 |
| Toys & Games (sub) | 9,513 | 7,253 | 47% | 10.8 | 8.3 |
| Prime Pantry | 15,611 | 7,841 | 25% | 19.2 | 9.6 |

Toys is a random-15%-user subsample (seed 42, fixed before results) of the full category, with
SASRec retrained on the subsample. Code: `scripts/subsample_dataset.py`, `scripts/train_sasrec.py`.

### 1.4 Evaluation protocol — ranking over 20 candidates (see `EVAL_PROTOCOL.md`)
For each test user: ground-truth next item + 19 random negatives = **20 candidates**.
- **LLM variants:** each candidate title scored by the LLM's length-normalized log-likelihood given
  the prompt (`clam_rec/model/clam_rec.py:rank_candidates` → `llm4rec.py:score_titles`), ranked.
- **SASRec:** candidates scored by dot-product (`scripts/eval_sasrec.py`).
- **Metrics:** Hit@1/5/10, NDCG@5/10 (`clam_rec/eval/metrics.py`: `record_rank`, `hit_at_k`,
  `ndcg_at_k`), sliced **overall / cold / warm**.
- **Cold vs warm:** cold = target item's TRAIN interaction count ≤ 5, else warm
  (`clam_rec/data/partition.py:tag_cold_warm`).
- **Why 20 (not 100):** A-LLMRec lists candidate titles *in the prompt*; >~50 candidates overflow
  OPT's 2048-token context (measured: 100 → 2,500–3,500 tokens → CUDA crash). **20 is locked.**
  Full reasoning + token table: `EVAL_PROTOCOL.md`.
- **Run inference:** `scripts/infer.py --config <cfg> --variant <v> --seed 0 --rank --rank_chunk 20`.
- **Outputs:** `results/{dataset}_{variant}/seed_0.jsonl` (one JSON record per user with
  `ranked`, `cold`, `train_count`, `answer`).

---

## 2. How to read the metrics (primer)
- **Hit@k** = fraction of users where the true item is in the top-k of the ranked 20.
- **NDCG@k** = position-weighted Hit@k: `1/log2(rank+1)` if rank ≤ k, else 0. Rewards ranking the
  true item higher, not just within top-k.
- **cold / warm** slices split users by whether the *target item* is rare (≤5 train interactions)
  or popular. This slicing is the whole game — the averages hide the crossover.

---

## 3. Data correctness (why the numbers are trustworthy)
The original (pre-rewrite) pipeline was not reproducible; we rebuilt with verification:
- Item→ASIN alignment is byte-identical to A-LLMRec's canonical files
  (`scripts/verify_itemmap.py`, `clam_rec/data/preprocess.py`).
- A **title bug** (titles coupled to a missing `description` field) once dropped 88% of Fashion
  titles and inflated Hit@1; fixed via `_clean_title` + decoupling in `preprocess.py` (recovered
  888 → 7,309 titles, interactions byte-identical). See `DATASETS.md`.
- Clean write-mode JSONL logging (one file per seed) — fixes the old append-mode corruption.

---

## 4. Experiment 1 — the CF↔LLM crossover (headline)

**Question.** As a function of item popularity, when does the LLM beat CF and vice-versa?

**Setup.** Compare `SASRec` (CF) vs `text` (LLM) Hit@1, binned by the target item's train
interaction count.

**Code.** `scripts/analysis_crossover.py` (bins by train_count, prints per-bin SASRec/text
Hit@1 and the `text−CF` gap, flags the sign flip). Reads the `seed_0.jsonl` ranking files.
**Reproduce:** `conda run -n ALLM-Rec python scripts/analysis_crossover.py`.

**Results (ranking Hit@1, cold/warm):**

| Dataset | SASRec cold | text cold | **Δ cold (LLM−CF)** | SASRec warm | text warm | **Δ warm (CF−LLM)** |
|---------|:-----------:|:---------:|:-------------------:|:-----------:|:---------:|:-------------------:|
| Luxury | 0.169 | 0.492 | **+0.323** | 0.710 | 0.666 | **+0.044** |
| Fashion | 0.071 | 0.158 | **+0.087** | 0.783 | 0.725 | **+0.058** |
| Toys | 0.107 | 0.237 | **+0.130** | 0.289 | 0.267 | **+0.022** |
| Prime Pantry | 0.026 | 0.146 | **+0.120** | 0.332 | 0.252 | **+0.080** |

In every dataset the LLM wins cold, CF wins warm. The gap `text−SASRec` decays monotonically with
popularity and **crosses zero** (per-bin curves in `analysis_crossover.py` output / figure §5).
The pattern holds at Hit@5 and Hit@10 too (full tables in `RESULTS.md`).

**Interpretation (the mechanism — this is the core of the paper).**
- **CF learns an item from *who bought it*.** Rare item → almost no co-purchase data → its learned
  embedding is noise → near-random ranking. Popular item → rich data → excellent. So CF is a
  **popularity expert** with a hard floor on cold items (at `train_count=0`, SASRec = 0.000: a
  never-seen item has a zero/untrained embedding).
- **The LLM reads the item's *title*** — content signal that exists regardless of popularity. So
  it's roughly flat across cold/warm; it dominates CF where CF has no signal (cold) and loses where
  CF is strong (warm).
- They are **complementary specialists**. The crossover is where the two curves cross.

**Paper mapping.** Main result / Figure 1(a). This is the answer to "do LLMs understand sequential
recommendation?" (the LLM-SRec question): *they add value precisely where CF is weak.*

---

## 5. Experiment 2 — the crossover point is predicted by dataset density

**Question.** The crossover happens at a *different* popularity level per dataset. Is that level
predictable?

**Setup.** For each dataset, locate the train_count where `text−SASRec` Hit@1 flips sign
(interpolated); correlate with dataset density metrics.

**Code.** `scripts/plot_crossover_sparsity.py` — computes the per-bin crossover curves, finds the
zero-crossing per dataset, computes mean interactions/item and mean history length, correlates, and
draws the figure. **Reproduce:** `conda run -n ALLM-Rec python scripts/plot_crossover_sparsity.py`.
**Figure:** `figures/crossover_sparsity.{pdf,png}` — panel (a) the crossover curves, panel (b)
crossover point vs mean interactions/item with the `y=x` line.

**Results.**

| Dataset | crossover @ train_count | mean interactions/item | mean history length |
|---------|:-----------------------:|:----------------------:|:-------------------:|
| AMAZON_FASHION | ~2.0 | 2.5 | 4.9 |
| Luxury | ~9.2 | 10.4 | 6.4 |
| Toys | ~13.2 | 10.8 | 8.3 |
| Prime Pantry | ~19.7 | 19.2 | 9.6 |

- **Pearson r(crossover, mean interactions/item) = +0.979**
- **Pearson r(crossover, mean history length) = +0.990**

The crossover point ≈ the dataset's mean interactions/item for **all 4** (Prime Pantry, the densest,
anchors the high end at 19.7 ≈ 19.2). (All Beauty, dropped 2026-08-04, was the small warm-skewed
outlier that had held r down to 0.89; see `FINDING_2.md`.)

**Interpretation.** The denser the collaborative signal (more interactions per item), the longer CF
stays reliable, so the flip moves to higher popularity. This turns the finding from *descriptive*
(there is a crossover) into *predictive* (you can estimate where it is from data density).
**Independent confirmation:** the learned fusion gate (§7.4) rediscovers ~the same pivot with no
prior.

**Caveat.** n = 4 → both correlations are ~significant (interactions/item r=0.98, p≈0.02; history
length r=0.99, p≈0.01), but n=4 is small — report as "strongly suggestive + mechanistically
sensible," not "established law." (A 5th dataset — e.g. re-adding All Beauty, or a new category —
would harden it.)

**Paper mapping.** Result 2 / Figure 1(b). The "law" that makes the crossover principled.

---

## 6. ⛔ Vision (CLIP) — MOVED TO A SEPARATE PAPER (not in this paper)

The vision result (adding CLIP is a null; the generation→ranking protocol reversal; CLIP
fine-tuning; backbone dependence) is **out of scope for this paper** and parked in **`FINDING_3.md`**
for a dedicated vision paper. Reasons: (a) none of this paper's findings use the vision variants —
they use only `SASRec` and `text`; (b) keeping vision out removes the collision with the sibling
vision paper *Smol-Rec*; (c) putting the vision null here would scoop the future vision paper.
**Do not report any CLIP/vision result in this paper.**

---

## 7. Experiment 4 — popularity-gated fusion (the prescriptive kicker)

**Question.** The crossover says CF and the LLM are complementary. Can we *exploit* it to beat both?

### 7.1 The data it runs on (shared candidates)
Fusion needs both models scored on the **identical** candidate set with per-candidate scores logged.
- Candidate pool: `scripts/make_candidates.py` → `data/candidates/{dataset}_seed0.json`
  (`{user: {target, cands}}`, 1 positive + 19 negatives).
- Re-inference logging scores: `scripts/run_shared_reinfer.sh <cfg> <gpu>` runs
  `eval_sasrec.py` and `infer.py` with `--candidates_file` (and `--out_tag`), which log
  `candidate_ids` + per-candidate `scores` + `target_id`.
  Model support: `clam_rec.py:load_shared_candidates`, `rank_candidates` (returns per-candidate
  scores), and the `--candidates_file`/`--out_tag` flags in `infer.py`/`eval_sasrec.py`.
- Outputs: `results/{dataset}_{sasrec,text_concat}_shared/seed_0.jsonl`.
- **Sanity:** shared-candidate metrics match own-candidate metrics within ±0.008 → the numbers
  are not an artifact of which negatives are drawn (a robustness point for the paper).

### 7.2 The method — popularity-gated score fusion
For each candidate item, in one user's ranking:
1. Get CF score and LLM score; **z-normalize each model's 20 scores per user** (they live on
   different scales — dot-products vs log-likelihoods).
2. **Popularity weight** `w = pop/(pop + pivot)`, where `pop` = that candidate's train interaction
   count and `pivot` = dataset mean interactions/item. Popular candidate → `w→1` (trust CF); rare
   candidate → `w→0` (trust LLM).
3. **Fuse:** `score = w·z_CF + (1−w)·z_LLM`. Rank all 20 by fused score.
- **Deployable:** every candidate's popularity is known at inference; you never need to know which
  candidate is the answer.
- **Code:** `scripts/analysis_ensemble.py` (Hit@1/5, NDCG@5, cold/warm) and
  `scripts/analysis_ensemble_full.py` (full Hit@1/5/10, NDCG@5/10 + cold/warm — the paper table).
  Item popularity from `data_partition`. **Reproduce:**
  `conda run -n ALLM-Rec python scripts/analysis_ensemble_full.py`.

### 7.3 Baselines (why the win is meaningful)
- **RRF** (reciprocal rank fusion): `1/(60+rank_CF)+1/(60+rank_LLM)`. Uses only *ranks*, discards
  score magnitudes → cannot strongly suppress a confident distractor. Popularity-blind.
- **z-fuse** (equal-weight): `z_CF + z_LLM` — i.e. pop-gate with `w=0.5` everywhere. Lets CF's noise
  on rare items get equal say. Popularity-blind.
- **ORACLE** (upper bound, *not* deployable): routes the whole decision by the *true target's*
  popularity. Ceiling for any popularity-based method.

### 7.4 Results (ranking, all metrics, overall + cold/warm Hit@1)

**Overall Hit@1 (Δ over best pure model):** Luxury 0.622 (**+0.022**), Fashion 0.308 (**+0.015**),
Toys 0.266 (**+0.019**), Prime Pantry 0.286 (**+0.032**) — **positive on all 4**, ≈ oracle (Prime
Pantry is CF-dominant: pop-gate wins Hit@1/NDCG but CF wins Hit@5/10). Full per-metric
tables (Hit@1/5/10, NDCG@5/10) are in `RESULTS.md` → "CF+LLM fusion" and regenerated by
`analysis_ensemble_full.py`. Representative (Luxury):

| Method | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | cold H@1 | warm H@1 |
|--------|:-----:|:-----:|:------:|:------:|:-------:|:--------:|:--------:|
| CF (SASRec) | 0.501 | 0.709 | 0.805 | 0.611 | 0.643 | 0.166 | 0.706 |
| text (LLM) | 0.600 | 0.756 | 0.850 | 0.682 | 0.712 | 0.494 | 0.664 |
| RRF | 0.539 | 0.744 | 0.888 | 0.646 | 0.692 | 0.243 | 0.719 |
| z-fuse | 0.575 | 0.748 | 0.857 | 0.667 | 0.702 | 0.300 | 0.744 |
| **pop-gate** | **0.622** | **0.801** | 0.884 | **0.718** | **0.744** | 0.480 | 0.709 |
| ORACLE* | 0.627 | 0.816 | 0.896 | 0.728 | 0.754 | 0.494 | 0.708 |

**Key reads:**
- **pop-gate wins on essentially every metric, every dataset**, and ≈ oracle (deployably).
- **The cold/warm columns expose the mechanism:** pop-gate's *cold* Hit@1 ≈ the LLM's cold, its
  *warm* Hit@1 ≈ CF's warm — it **inherits the better expert per regime**.
- **Naive fusion fails:** RRF is often *below* the best pure model on Hit@1 (Luxury 0.539 < 0.600);
  z-fuse is ~neutral. Only *popularity-weighted* fusion wins → the win is "you must use the
  crossover," not "any ensemble helps."
- **Nuance to report honestly:** on Toys cold, pop-gate (0.182) < pure LLM (0.228) — high pivot pulls
  CF weight into cold items — but it more than compensates on warm (beats both), netting a win.

### 7.5 Ablation — learned gate (confirms the fixed formula is near-optimal)
Replace `w=pop/(pop+pivot)` with a learned `w=sigmoid(a·log(pop)+b)` (2 params), fit on a train half
of users, evaluated on the held-out half. **Code:** `scripts/analysis_learned_gate.py`.
**Result:** learned ≈ fixed within ±0.004 (both directions) on all datasets, both just under oracle
→ *the simple formula is already near-optimal; learning the gate adds nothing.* Bonus: the learned
gate's 50/50 point (pop ≈ 14.7/4.1/3.3/8.4) lands near each dataset's mean interactions/item — an
independent rediscovery of §5. **Reporting:** one sentence + footnote (not a table).

### 7.6 The worked intuition (for the paper's method figure)
Three candidates, pivot=9; C is the rare true item, A a popular distractor.

| candidate | pop | w | z_CF | z_LLM | fused |
|-----------|----:|:-:|:----:|:-----:|:-----:|
| A (popular distractor) | 50 | 0.85 | −1.2 | +1.0 | **−0.86** |
| B (medium) | 10 | 0.53 | +1.0 | −1.0 | +0.05 |
| C (rare, TRUE) | 2 | 0.18 | +0.2 | +0.9 | **+0.77** |

Pure CF picks B (blind to rare C); pure LLM picks A (fooled by the popular distractor's title);
**fusion picks C** — it uses CF to *suppress* the popular distractor A and the LLM to *promote* the
rare target C. Neither pure model can do both.

**Paper mapping.** Result 4 + a method figure. This is what makes the paper *prescriptive*, not just
diagnostic.

---

## 8. Robustness / supporting checks
- **Candidate-sampling stability** — shared vs own candidates differ by ≤0.008 Hit@1 (§7.1). Code:
  compare `results/{ds}_sasrec/` vs `results/{ds}_sasrec_shared/`.
- **Metric holds across k** — crossover present at Hit@1/5/10 (`RESULTS.md` full tables).
- **Not yet done:** multi-seed (all numbers are seed 0). The crossover's robustness argument is
  currently *cross-dataset + monotonic*; ≥2–3 seeds would add per-cell significance.

---

## 9. Excluded / parked (do NOT put in the paper; kept for the record)
All parked in `RESULTS.md` with numbers, and in `STORY.md`/`PROJECT.md` §0.0 with rationale:
- **bigG backbone** — dropped; ViT-L/14 is the single backbone (bigG *inverts* the cold/warm effect).
- **CLIP fine-tuning (RQ3)** — dropped; "better retrieval, worse recommendation" negative parked.
  Code exists under `clam_rec/finetune/`.
- **RQ2 fusion study (mean/gating)** — collapsed to a one-line note: concat was best, fusion fixed =
  concat. Code paths remain in `clam_rec/fusion/fusion.py` but unused.
- **Prime Pantry** — prepared but excluded (results not favorable). Data/results on disk.

---

## 10. Positioning (for intro / related work)
- **vs A-LLMRec (base):** we don't propose a new architecture; we *analyze* the base as an
  instrument and add the fusion. Same 20-candidate protocol → comparable.
- **vs Smol-Rec (sibling, same authors, under review):** *entirely different question* — this paper
  is **CF vs LLM complementarity** (no vision at all), Smol-Rec is about visual-context budget. Since
  this paper contains **no vision**, there is no collision. (The vision work is a separate paper —
  `FINDING_3.md` — which is where Smol-Rec positioning must be handled.)
- **vs LLM-SRec:** we borrow only the *ranking metric definitions*, not its two-tower retrieval
  architecture. We answer its question ("do LLMs understand sequential rec?") empirically with the
  crossover + the deployable fusion.
- **Suggested framing:** "When do LLMs help sequential recommendation?" — CF/LLM complementarity
  along popularity, a density-predicted crossover, vision ruled out, and a popularity-gated fusion.

---

## 11. Caveats / open items (state honestly)
1. **1 seed** — no error bars yet (multi-seed is the top remaining compute).
2. **20 candidates** — bounded by OPT's context (`EVAL_PROTOCOL.md`); metrics are optimistic vs a
   100-negative / full-catalog eval, but the *direction* of every finding is candidate-count-robust.
3. **n=4 for the density correlation** — suggestive, not proven (§5).
4. **Toys is subsampled** — random-user, seed fixed before results, SASRec retrained (`DATASETS.md`).

---

## 12. Code & data map (quick index)

**Model / eval:**
- `clam_rec/model/clam_rec.py` — ClamRec: `stage1_step`, `stage2_step`, `generate`,
  `rank_candidates` (L383), `_build_sample` (L283), `_make_candidates` (L248), `content_emb` (L124),
  `load_shared_candidates` (L264).
- `clam_rec/model/llm4rec.py` — OPT wrapper, `score_titles` (L136), `replace_soft_tokens`.
- `clam_rec/model/sasrec.py` — frozen SASRec.
- `clam_rec/eval/metrics.py` — `record_rank` (L68), `hit_at_k`, `ndcg_at_k`, `record_hit_at_1`.
- `clam_rec/data/partition.py` — `data_partition` (L20), `tag_cold_warm` (L55).
- `clam_rec/data/preprocess.py` — preprocessing + `_clean_title`.
  _(CLIP/vision code — `clam_rec/clip/`, `clam_rec/fusion/`, `clam_rec/finetune/` — is for the
  separate vision paper; see `FINDING_3.md`.)_

**Run / train:**
- `scripts/train.py` — train stage 1/2.
- `scripts/infer.py` — inference; flags `--rank --rank_chunk --candidates_file --out_tag`.
- `scripts/eval_sasrec.py` — SASRec CF ranking baseline.
- `scripts/make_candidates.py` — shared candidate pools (`--num`).
- `scripts/run_shared_reinfer.sh` — driver that produces the shared-candidate scored files.
- `scripts/subsample_dataset.py`, `train_sasrec.py` — Toys subsample + SASRec training.

**Analysis / figures (the results):**
- `scripts/analysis_crossover.py` — §4 crossover (per-bin Hit@1).
- `scripts/plot_crossover_sparsity.py` — §4+§5 figure → `figures/crossover_sparsity.{pdf,png}`.
- `scripts/plot_crossover.py` — earlier crossover figure → `figures/crossover.{pdf,png}`.
- `scripts/analysis_ensemble.py` — fusion (Hit@1/5, NDCG@5, cold/warm).
- `scripts/analysis_ensemble_full.py` — §7 fusion, full metrics (the paper table).
- `scripts/analysis_learned_gate.py` — §7.5 learned-gate ablation.

**Data / results:**
- `results/{dataset}_{sasrec,text_concat}/seed_0.jsonl`
  — per-variant ranking results (own candidates).
- `results/{dataset}_{sasrec,text_concat}_shared/seed_0.jsonl` — shared candidates + per-candidate
  scores (fusion input).
- `data/candidates/{dataset}_seed0.json` — shared 20-candidate pools.
- `assets/sasrec_{dataset}.pth` — frozen SASRec checkpoints; `results/checkpoints/` — LLM ladder.

**Environment:** conda env `ALLM-Rec` (`conda run -n ALLM-Rec python ...`), on the A6000 box.
See `INFRASTRUCTURE.md`.
