# WRITING_STYLE.md — writing-style references (read the PDFs, form your own view)

> **What this is:** pointers to a few *well-written* papers to learn writing **style and structure**
> from — **not** content, facts, or citations (those come only from `../FINDINGS.md`,
> `../FINDING_1/2/4.md`, `../RESULTS.md`). You have a large context window: **read the exemplar PDFs
> yourself and decide what, if anything, to take from each.** Nothing here prescribes which paper
> maps to which section — that's your call.
>
> **Honest caveat:** whoever set up this repo did **not** read these PDFs in depth; the notes below
> are only rough labels. Trust your own reading over them.
>
> **Hard rule:** imitate *how* they write, never *what* they say. Every number/claim in our paper
> comes from our own docs — never import a fact, dataset, or phrasing from these papers.

## Exemplars (in `reference-paper/`)
Read whichever seem useful; ignore the rest. Rough labels only:

- `guo17a-1-8.pdf` — Guo et al., *On Calibration of Modern Neural Networks* (ICML 2017).
- `2001.08361v1-1-10.pdf` — Kaplan et al., *Scaling Laws for Neural Language Models* (first ~10 pp).
- `3298689.3347058.pdf` — Dacrema et al., *Are We Really Making Much Progress?* (RecSys 2019).
- `1611.03530v2-1-9.pdf` — Zhang et al., *Understanding deep learning requires rethinking
  generalization* (ICLR 2017).

(These are broadly: clear empirical-analysis / measurement papers in ML and recommender systems —
the same *genre* as our paper. Same topic isn't available; the CF↔LLM crossover is our novelty.)

## A few general reminders (adapt freely — not rules)
Standard good-writing for an empirical paper; weigh them yourself:
- Prefer clarity and economy; let evidence carry the argument; avoid hype words.
- Define terms/notation before using them; justify method choices briefly.
- Pair numbers with a takeaway; state limitations honestly without undermining the finding.

## What is content, not style (do NOT take from these papers)
- Numbers, method, findings, positioning, citations → `../FINDINGS.md`, `../FINDING_1/2/4.md`,
  `../RESULTS.md`; Figure 1 = `../figures/crossover_sparsity.pdf`; scope (CF vs LLM, **no vision**) =
  `../CLAUDE.md`.
- Do not copy any exemplar's wording, structure-for-its-own-sake, or off-domain apparatus
  (theorems/proofs/scaling-law math we don't have).
