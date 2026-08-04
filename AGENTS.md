# AGENTS.md — start here (Codex / any agent)

**Full project context is in [`CLAUDE.md`](./CLAUDE.md) — read it first.** This file mirrors the
essentials so you're oriented even before opening it.

## The project
An empirical analysis paper on sequential recommendation, built on **A-LLMRec** (frozen SASRec +
frozen OPT-6.7B). Three findings:
1. **CF↔LLM crossover** — the LLM beats CF on cold/rare items, CF beats the LLM on warm/popular
   items (monotone, all datasets). → `FINDING_1.md`
2. **Crossover point predicted by dataset density** (Pearson r ≈ 0.98). → `FINDING_2.md`
3. **Popularity-gated fusion** beats both pure models (naive fusion doesn't). → `FINDING_4.md`

## Scope — respect this
- **IN:** `SASRec` + `text` variants; 4 datasets (Luxury Beauty, AMAZON_FASHION, Toys sub,
  Prime Pantry); likelihood ranking over 20 candidates (Hit@1/5/10, NDCG@5/10, cold/warm).
- **OUT (do not put in this paper):** vision/CLIP (`clip_align`, `clip_inject`, fine-tuning) — a
  **separate paper**, see `FINDING_3.md`, ignore `clam_rec/clip|fusion|finetune` and `results/*clip*`;
  All Beauty dataset; bigG; RQ2/RQ3.
- **Constraints:** 1 seed everywhere (no multi-seed — a stated limitation); 20 candidates locked
  (OPT 2048-token limit, `EVAL_PROTOCOL.md`).

## Reading order
`FINDINGS.md` (master index) → `FINDING_1.md` / `FINDING_2.md` / `FINDING_4.md` → `RESULTS.md`
(numbers, authoritative) → `DATASETS.md`, `EVAL_PROTOCOL.md`.
Historical/design only (superseded): `PROJECT.md`, `STORY.md`, `OVERNIGHT_*`, `README.md`.
Infra/how-to-run: `INFRASTRUCTURE.md`.

## Working notes
- All numbers are already in the docs — you generally don't need to re-run. If you do, analysis
  scripts are CPU/fast; only `scripts/infer.py` needs a GPU. Env: `conda run -n ALLM-Rec python ...`.
- All 4 findings are complete (incl. Prime Pantry fusion). Honest nuance: pop-gate beats both pure
  models on Hit@1 and NDCG@{5,10} in all 4 (and Hit@5/10 on 3/4), but CF-dominant Prime Pantry has
  CF winning Hit@5/10 — see `FINDING_4.md`. No open experiments.
