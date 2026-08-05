# CLAM-Rec v2 — project context (READ THIS FIRST)

This file is the entry point for any Claude/Codex session. It gives the full context of the
project and routes you to the detailed docs and code. It is an **index + status**, not a
duplicate of the numbers — trust the docs it points to.

---

## What this project is

An **empirical analysis paper** on **sequential recommendation**: *When do LLMs help, and how do
they relate to collaborative filtering (CF)?* Built on **A-LLMRec** (KDD 2024) — a frozen SASRec
(CF) aligned with a frozen OPT-6.7B LLM that ranks candidate item titles by likelihood.

**The paper's story (three findings):**
1. **CF↔LLM crossover** — the LLM (`text`) beats CF (`SASRec`) on **cold/rare** items, CF beats the
   LLM on **warm/popular** items; the gap decays monotonically and crosses zero in every dataset.
   (Mechanism: SASRec's item embedding is untrained for rare items — at 0 interactions it scores
   exactly 0.000 — while the LLM reads titles and is popularity-invariant.) → `FINDING_1.md`
2. **The crossover point is predicted by dataset density** — it lands near the dataset's mean
   interactions/item (Pearson r ≈ 0.98). Predictive, not just descriptive. → `FINDING_2.md`
3. **Popularity-gated fusion (prescriptive)** — weighting each candidate's CF vs LLM score by that
   candidate's popularity beats *both* pure models on **Hit@1 and NDCG@{5,10} across all 4 datasets**
   (and Hit@5/10 on the 3 LLM-competitive ones), ≈ an oracle upper bound; naive fusion (RRF,
   equal-weight) does not. → `FINDING_4.md`

One-line thesis: *LLMs and CF are complementary along item popularity; the crossover is
predictable from data density and exploitable by a simple popularity-gated fusion.*

---

## ⛔ Scope — what is IN and OUT of this paper

**IN:** the `SASRec` and `text` variants only; **4 datasets** (Luxury Beauty, AMAZON_FASHION,
Toys & Games (sub), Prime Pantry); **likelihood ranking over 20 candidates** (Hit@1/5/10,
NDCG@5/10, sliced cold/warm); ViT-L is irrelevant here (text baseline uses SBERT, not vision).

**OUT (do NOT put in this paper):**
- **Vision / CLIP** — `clip_align`, `clip_inject`, CLIP fine-tuning, the generation→ranking
  reversal, bigG backbone. This is a **separate future paper**; its results + plan live in
  **`FINDING_3.md`**. The vision code (`clam_rec/clip/`, `clam_rec/fusion/`, `clam_rec/finetune/`)
  and `results/*clip*` dirs stay in the repo for that paper — **ignore them for this paper.**
- **All Beauty** dataset (dropped — too small / density outlier).
- **RQ2 fusion study, RQ3 fine-tuning, bigG** — all dropped/parked.

**Hard constraints to respect when writing:**
- **1 seed** (seed 0) everywhere — no multi-seed; the robustness argument is *cross-dataset +
  monotone-across-bins*, not per-cell significance. **We are deliberately NOT running multi-seed.**
  State this honestly as a limitation.
- **20 candidates** is locked — listing candidates in the prompt means >~50 overflow OPT's
  2048-token context (see `EVAL_PROTOCOL.md`).

---

## Current status (2026-08-04)

- Findings 1, 2, 3 (vision, parked), and 4 are **done and written up** — including the Prime Pantry
  fusion (completed 2026-08-04). All 4 datasets have full ranking + fusion numbers.
- **Honest nuance in F4:** pop-gate beats both pure models on **Hit@1 and NDCG@{5,10} in all 4**,
  and on Hit@5/10 on the 3 LLM-competitive datasets. **Prime Pantry is CF-dominant** (CF beats the LLM
  overall) so there pop-gate wins Hit@1/NDCG yet pure CF wins Hit@5/10; the learned gate also beats the
  fixed density pivot there. Captured in `FINDING_4.md` / `RESULTS.md`.
- **No open experiments.** The only deliberate non-item is multi-seed (we are not doing it).
- Everything is committed + pushed to `github.com:Mr-Dheer/CLAM-Rec-v2` (branch `master`).

---

## Reading order (current-paper docs)

1. **`FINDINGS.md`** — master index + TL;DR + the 3 contributions + the instrument, datasets,
   metrics, and a code/data map. Start here.
2. **`FINDING_1.md`**, **`FINDING_2.md`**, **`FINDING_4.md`** — per-finding deep-dives (method,
   mechanism, exact numbers, code pointers, reproduce commands, suggested paper phrasing).
3. **`RESULTS.md`** — the numbers ledger (authoritative for all metrics/tables).
4. **`DATASETS.md`** — per-dataset provenance/prep/selection.
5. **`EVAL_PROTOCOL.md`** — why ranking uses 20 candidates (the OPT-2048 constraint).

**Writing the paper:** the LaTeX project is in **`Cambodia/`** (LNCS template; `samplepaper.tex`).
Read **`Cambodia/WRITING_STYLE.md`** for the target writing style (it points to well-written exemplar
PDFs in `Cambodia/reference-paper/` — read those for *style*, not content). Content/numbers come from
the docs above; Figure 1 is `figures/crossover_sparsity.pdf`. TeX Live is installed
(`~/texlive/2026/bin`); build with `latexmk -pdf` or via Overleaf (repo is synced to it).

**Separate vision paper:** `FINDING_3.md` (do not use in this paper).

**Historical / design context (NOT paper-writing docs — superseded, read only if you need the
project's backstory):** `PROJECT.md` (original design doc; predates the pivots), `STORY.md`
(vision-era narrative + reframed thesis), `OVERNIGHT_2026-07-22.md` (old snapshot),
`INFRASTRUCTURE.md` (where code/data live; useful if running things), `README.md`.

---

## Code map (what runs what)

Package `clam_rec/` (in-scope):
- `model/clam_rec.py` — `ClamRec`: stage1/stage2 train, `generate`, `rank_candidates`
  (likelihood ranking, returns per-candidate scores), `load_shared_candidates`.
- `model/llm4rec.py` — OPT-6.7B wrapper, `score_titles`. `model/sasrec.py` — frozen SASRec.
- `eval/metrics.py` — `record_rank`, `hit_at_k`, `ndcg_at_k`, `record_hit_at_1`.
- `data/partition.py` — `data_partition`, `tag_cold_warm` (cold = train_count ≤ 5).
- _(out-of-scope: `clip/`, `fusion/`, `finetune/` — vision paper.)_

Scripts (in-scope):
- `scripts/eval_sasrec.py` — SASRec CF ranking. `scripts/infer.py` — LLM ranking
  (`--rank`, `--candidates_file`, `--out_tag`, `--nshards/--shard`).
- `scripts/make_candidates.py`, `run_shared_reinfer.sh`, `run_pp_fusion.sh` — shared-candidate
  fusion data.
- **Analyses (produce the paper's results):** `analysis_crossover.py` (F1),
  `plot_crossover_sparsity.py` (F1+F2 figure → `figures/crossover_sparsity.{pdf,png}`),
  `analysis_ensemble.py` / `analysis_ensemble_full.py` (F4 fusion),
  `analysis_learned_gate.py` (F4 ablation).

Results/data:
- `results/{ds}_{sasrec,text_concat}/seed_0.jsonl` — ranking results (own candidates).
- `results/{ds}_{sasrec,text_concat}_shared/seed_0.jsonl` — shared candidates + per-candidate
  scores (fusion input).
- `assets/sasrec_{ds}.pth` — frozen SASRec; `results/checkpoints/` — trained LLM heads.
- `data/candidates/` — shared pools (gitignored; regenerate via `make_candidates.py`, seed 0).

---

## Environment & reproduce

Conda env **`ALLM-Rec`** on an A6000 box: `conda run -n ALLM-Rec python <script>`.
You generally **do not need to re-run** to write the paper — all numbers are in `RESULTS.md` /
the `FINDING_*.md` docs. If you do: the analysis scripts are CPU-only and fast (seconds–minutes);
only `infer.py` (the LLM) needs a GPU. See `INFRASTRUCTURE.md`.

Quick regenerate of the headline results (CPU):
```bash
conda run -n ALLM-Rec python scripts/analysis_crossover.py          # F1 crossover
conda run -n ALLM-Rec python scripts/plot_crossover_sparsity.py     # F1+F2 figure + r
conda run -n ALLM-Rec python scripts/analysis_ensemble_full.py      # F4 fusion table
```

---

## History (why there is vision/design baggage)

The project started as a *vision* paper (add CLIP to A-LLMRec) — hence the CLIP code and the
vision-heavy `PROJECT.md`/`STORY.md`. It pivoted: vision turned out to be a **null** under ranking,
but the data revealed the **CF↔LLM crossover**, which became this paper. Vision was then split off
into a separate future paper (`FINDING_3.md`). Datasets and scope were tightened along the way
(bigG, RQ3 fine-tuning, RQ2 fusion-study, All Beauty, and multi-seed all dropped). When in doubt
about scope, this file + `FINDINGS.md` win over the older `PROJECT.md`/`STORY.md`.
