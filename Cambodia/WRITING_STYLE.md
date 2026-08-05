# WRITING_STYLE.md — how to write this paper (style guide for the writing session)

> **Purpose:** teach the *writing style and structure* of a strong empirical-analysis paper —
> **not** content, facts, or citations (those come from `../FINDINGS.md`, `../FINDING_1/2/4.md`,
> `../RESULTS.md`). Four well-written exemplar PDFs live in `reference-paper/`. You have a large
> context window — **read the relevant sections of these PDFs directly** and imitate their *prose,
> structure, and rigor*. Do **not** copy their wording or content.
>
> **Hard rule:** imitate *how they write*, never *what they say*. All numbers/claims for our paper
> come from our own docs. Never import a fact, dataset, or phrasing from these papers.

---

## The exemplars (in `reference-paper/`) and what each teaches

| File | Paper | Read for | Teaches |
|------|-------|----------|---------|
| `guo17a-1-8.pdf` | Guo et al., *On Calibration of Modern Neural Networks* (ICML'17) | **whole paper (short)** | The **structural twin** of our paper: measure a phenomenon across settings → explain it → propose a *dead-simple* fix (temperature scaling). Calm, precise prose. |
| `2001.08361v1-1-10.pdf` | Kaplan et al., *Scaling Laws for Neural LMs* | **§1 Intro, §3 Empirical Results & Basic Power Laws** | How to present an **empirical predictive relationship** (state the law → one clean figure → fitted form → where it holds). |
| `3298689.3347058.pdf` | Dacrema et al., *Are We Really Making Much Progress?* (RecSys'19) | **whole paper** | Same field + genre: honest, evidence-first **empirical analysis** in recommender systems; cross-dataset comparison; measured critical tone. |
| `1611.03530v2-1-9.pdf` | Zhang et al., *Understanding DL requires rethinking generalization* (ICLR'17) | **§1 Intro + the framing of the headline experiments** | How to make a **simple, surprising empirical finding land** with maximum clarity and minimum fuss. |

---

## Which exemplar to imitate for each part of OUR paper

| Our section | Primary model | What to take |
|-------------|---------------|--------------|
| **Abstract / Intro** (headline = the CF↔LLM crossover) | **Zhang** + Kaplan §1 | State the empirical finding up front, plainly and confidently; frame "averages hide it, sliced view reveals it." |
| **Method / Setup** (SASRec + LLM, ranking@20, cold/warm) | **Guo** (method exposition) | Crisp problem/notation setup; define terms (crossover point, cold/warm, popularity) before using them; minimal but complete. |
| **Finding 1 — the crossover** (phenomenon) | **Guo** results + **Zhang** | Present a measured phenomenon across datasets; lead with the mechanism datapoint (SASRec=0.000 at train_count 0); emphasize the *monotone* trend. |
| **Finding 2 — density predicts crossover** (empirical law) | **Kaplan §3** | State the relationship, one clean scatter (`crossover ≈ mean interactions/item`, r=0.98), give the "≈ y=x" reading, note where it holds; hedge n=4 honestly. |
| **Finding 4 — popularity-gated fusion** (simple fix) | **Guo** (calibration→temperature scaling arc) | "The phenomenon implies a trivial, deployable method that beats both" — mirror how Guo motivates and presents temperature scaling. |
| **Honest caveats / limitations** | **Dacrema** + Kaplan | Confident about the pattern, explicit about limits (1 seed, n=4, 20 candidates, the Prime Pantry Hit@5/10 nuance). Evidence-first, no overclaiming. |
| **Overall tone & pacing** | **Dacrema** | Sober, precise, honest; let the data carry the argument. |

---

## Distilled style directives (the "house style" for this paper)

Read the exemplars to *feel* these, but concretely, imitate:
1. **State findings up front.** Put the crossover + the density law in the abstract/intro, not buried.
2. **Formal before informal.** Define the task, notation, and terms (cold/warm, crossover point,
   popularity) *before* the prose that uses them (Guo/Kaplan do this cleanly).
3. **"We do X because Y."** Every method choice gets a one-clause justification (why 20 candidates,
   why z-normalize, why `w = pop/(pop+pivot)`).
4. **One figure carries each finding.** Kaplan-style: a single clean plot per result, referenced
   precisely in text; don't over-figure.
5. **Present tense for method, past for experiments** (standard; the exemplars follow it).
6. **Quantify, then interpret.** Give the number, then one sentence of meaning — never a number
   without a takeaway, never a takeaway without a number.
7. **Honest hedging.** Match Dacrema/Kaplan: state limits plainly (n=4, 1 seed) *without* undermining
   the finding. Confidence + candor.
8. **Economy.** Short sentences, no throat-clearing, no hype words ("novel", "significantly" unless
   statistically). Let evidence do the work.

---

## Anti-patterns — do NOT imitate
- **Their content, datasets, or numbers** — ours come only from `../FINDINGS.md` / `../RESULTS.md`.
- **Off-domain scaffolding** — e.g. Zhang/Kaplan are ML-theory-flavored; don't import heavy
  theorem/proof or scaling-law math machinery we don't have. Take their *clarity*, not their apparatus.
- **Their phrasings** — no distinctive sentences lifted verbatim (style ≠ copy).

---

## Where the *content* comes from (not here)
- Method + findings + numbers: `../FINDINGS.md`, `../FINDING_1.md`, `../FINDING_2.md`, `../FINDING_4.md`, `../RESULTS.md`.
- Figures: `../figures/crossover_sparsity.pdf` (Figure 1).
- Positioning / citations (A-LLMRec, LLM-SRec, SASRec, Smol-Rec): `../FINDINGS.md` §10 + the bibliography.
- Scope (what's in/out, e.g. no vision): `../CLAUDE.md`, `../FINDINGS.md`.
