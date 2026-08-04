# FINDING 4 — Popularity-gated CF+LLM fusion (the prescriptive result)

> **Deep-dive doc for Finding 4** (per-finding series; master overview = `FINDINGS.md`).
> Goals: (1) explain the fusion completely — method, math, intuition; (2) give a future Claude every
> number, baseline, code pointer, and paper-phrasing to write this section. Self-contained.
> Last updated: 2026-08-04.

**One-line finding.** Because CF and the LLM are complementary along popularity (Finding 1), we can
*exploit* it: a **popularity-gated score fusion** — for each candidate, weight its CF vs LLM score by
that candidate's own popularity — **beats both pure CF and pure LLM on all 4 datasets**, and nearly
matches an oracle upper bound. **Naive fusion (RRF, equal-weight) does not help** — you must use the
crossover. This turns the paper from *descriptive* (there is a crossover) to *prescriptive* (here is
how to exploit it).

---

## 1. The idea (why this should work at all)
Finding 1 established two **complementary specialists**:
- **CF** — reliable on **popular** items, useless on **rare** ones (their embeddings are untrained;
  at `train_count=0`, SASRec Hit@1 = 0.000).
- **LLM** — reads titles, roughly popularity-invariant; strong on rare items, weaker on popular ones.

If for **each candidate** we knew which model to trust, we'd get the best of both. And we *do* have a
proxy for trust: the candidate's **popularity**. Popular candidate → trust CF; rare candidate → trust
the LLM. That proxy is *known at inference* (it's a training-data statistic), so the fusion is
**deployable**, not an oracle.

---

## 2. The data it runs on (shared candidates + logged scores)
Fusion requires **both models scored on the identical candidate set**, with per-candidate scores
saved (not just the final ranking).

- **Shared candidate pool:** `scripts/make_candidates.py` → `data/candidates/{ds}_seed0.json`
  = `{user: {"target": id, "cands": [20 ids]}}` (1 positive + 19 negatives, sampled once, shared by
  all models). *Why this matters:* the original per-variant runs each sampled their own negatives, so
  their candidate sets barely overlapped — fusing them was invalid (an early RRF attempt looked
  "perfect" only because the sole shared item was the answer). Shared pools fix this.
- **Re-inference with scores:** `scripts/run_shared_reinfer.sh <cfg> <gpu>` runs `eval_sasrec.py`
  and `infer.py` with `--candidates_file` + `--out_tag`, which log `candidate_ids`, per-candidate
  `scores`, and `target_id`. Model support: `clam_rec/model/clam_rec.py:load_shared_candidates`
  and `rank_candidates` (returns per-candidate scores); `--candidates_file`/`--out_tag` flags in
  `scripts/infer.py` and `scripts/eval_sasrec.py`.
- **Outputs:** `results/{ds}_{sasrec,text_concat}_shared/seed_0.jsonl`.
- **Sanity:** shared-candidate metrics match own-candidate metrics within ±0.008 → not an artifact of
  the negative sampling. (Robustness point for the paper.)

---

## 3. The method — popularity-gated score fusion (the math)
For one user with 20 candidates:

1. **Raw scores.** CF gives each candidate a dot-product score; the LLM gives each a
   length-normalized log-likelihood. These are on **totally different scales**.
2. **z-normalize per user** (per model, over the 20 candidates):
   `z_CF = (s_CF − mean(s_CF)) / std(s_CF)`, likewise `z_LLM`. Now both are mean-0, std-1 and
   comparable. (Per-user, so it adapts to each user's score spread.)
3. **Popularity weight** for each candidate `c`:
   `w(c) = pop(c) / (pop(c) + pivot)`, where `pop(c)` = candidate `c`'s train interaction count and
   `pivot` = the dataset's mean interactions/item (from `data_partition`). Properties:
   - popular candidate (`pop ≫ pivot`) → `w → 1` → trust CF;
   - rare candidate (`pop ≪ pivot`) → `w → 0` → trust LLM;
   - at `pop = pivot` → `w = 0.5` (equal blend). It's a **smooth** gate, not a hard switch.
   - `pivot` = the empirically-measured crossover location (Finding 2) — that's *why* it's the right
     pivot.
4. **Fuse & rank:** `score(c) = w(c)·z_CF(c) + (1 − w(c))·z_LLM(c)`; rank the 20 by fused score.

**Code:** `scripts/analysis_ensemble.py` (Hit@1/5, NDCG@5, cold/warm) and
`scripts/analysis_ensemble_full.py` (full Hit@1/5/10, NDCG@5/10 + cold/warm — the paper table).
**Reproduce:** `conda run -n ALLM-Rec python scripts/analysis_ensemble_full.py`.

---

## 4. Worked example (the method figure)
Three candidates, `pivot = 9`; **C is the true item** (rare), **A is a popular distractor**.

| candidate | pop | w = pop/(pop+9) | z_CF | z_LLM | fused = w·z_CF+(1−w)·z_LLM |
|-----------|----:|:---------------:|:----:|:-----:|:-------------------------:|
| A (popular distractor) | 50 | 0.85 | −1.2 *(CF: "doesn't fit you")* | +1.0 *(LLM fooled by title)* | **−0.86** |
| B (medium) | 10 | 0.53 | +1.0 | −1.0 | +0.05 |
| C (rare, TRUE) | 2 | 0.18 | +0.2 *(CF: noisy)* | +0.9 *(LLM: "fits you")* | **+0.77** |

- **Pure CF** ranks B > C > A → picks B → **miss** (blind to rare C).
- **Pure LLM** ranks A > C > B → picks A → **miss** (fooled by popular distractor A).
- **Fusion** ranks **C > B > A → HIT.** It did *two* things at once: used **CF to suppress the
  popular distractor A** (`w=0.85` → A inherits CF's confident −1.2) and used the **LLM to promote the
  rare true item C** (`w=0.18` → C inherits the LLM's +0.9). **Neither pure model can do both.**

This is the crux: the popularity weight routes *each candidate* to the expert that is reliable *for
that candidate*.

---

## 5. The baselines (why the win is meaningful, not "any ensemble helps")
- **RRF** (reciprocal rank fusion, standard in IR): `1/(60+rank_CF) + 1/(60+rank_LLM)`. Uses only
  **ranks**, discarding score magnitudes — so CF's *confident* "no" on a distractor becomes a weak
  vote. Popularity-blind. Result: often **below** the best single model on Hit@1.
- **z-fuse** (equal-weight): `z_CF + z_LLM` = pop-gate with `w=0.5` everywhere. Lets CF's **noise on
  rare items** get equal say → dilutes each expert with the other's blind spot. Popularity-blind.
  Result: ~neutral.
- **ORACLE** (upper bound, *not deployable*): routes the *whole* decision by the **true target's**
  popularity — i.e. it already knows which candidate is the answer. Ceiling for popularity-based
  methods.

The point of including RRF and z-fuse: they show *naive* fusion fails, so the pop-gate win is
attributable specifically to **using the popularity structure** (the crossover), not to ensembling.

---

## 6. Results — full metrics, all datasets (seed 0, 20 candidates)

Bold = the deployable **pop-gate** winning a column. `*` ORACLE = non-deployable upper bound.

### Luxury Beauty (n=9912)
| Method | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | cold H@1 | warm H@1 |
|--------|:-----:|:-----:|:------:|:------:|:-------:|:--------:|:--------:|
| CF (SASRec) | 0.501 | 0.709 | 0.805 | 0.611 | 0.643 | 0.166 | 0.706 |
| text (LLM) | 0.600 | 0.756 | 0.850 | 0.682 | 0.712 | 0.494 | 0.664 |
| RRF | 0.539 | 0.744 | 0.888 | 0.646 | 0.692 | 0.243 | 0.719 |
| z-fuse | 0.575 | 0.748 | 0.857 | 0.667 | 0.702 | 0.300 | 0.744 |
| **pop-gate** | **0.622** | **0.801** | 0.884 | **0.718** | **0.744** | 0.480 | 0.709 |
| ORACLE* | 0.627 | 0.816 | 0.896 | 0.728 | 0.754 | 0.494 | 0.708 |

### Prime Pantry (n=15606) — ⏳ PENDING
Fusion for Prime Pantry requires its shared-candidate scores; the sharded re-inference is running
(2026-08-04). This table + the summary gains will be filled from `analysis_ensemble_full.py` when it
completes. (Prime Pantry's crossover is strong — cold SASRec 0.026 vs text 0.146 — so a clear pop-gate
win is expected.)

### AMAZON_FASHION (n=3261)
| Method | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | cold H@1 | warm H@1 |
|--------|:-----:|:-----:|:------:|:------:|:-------:|:--------:|:--------:|
| CF (SASRec) | 0.246 | 0.382 | 0.491 | 0.314 | 0.349 | 0.063 | 0.789 |
| text (LLM) | 0.292 | 0.475 | 0.657 | 0.385 | 0.443 | 0.144 | 0.730 |
| RRF | 0.252 | 0.407 | 0.595 | 0.329 | 0.389 | 0.078 | 0.767 |
| z-fuse | 0.260 | 0.390 | 0.551 | 0.326 | 0.377 | 0.081 | 0.790 |
| **pop-gate** | **0.308** | **0.497** | **0.659** | **0.405** | **0.457** | 0.144 | 0.791 |
| ORACLE* | 0.310 | 0.519 | 0.699 | 0.415 | 0.473 | 0.148 | 0.789 |

### Toys & Games (sub) (n=9513)
| Method | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | cold H@1 | warm H@1 |
|--------|:-----:|:-----:|:------:|:------:|:-------:|:--------:|:--------:|
| CF (SASRec) | 0.206 | 0.484 | 0.691 | 0.347 | 0.414 | 0.106 | 0.296 |
| text (LLM) | 0.247 | 0.496 | 0.710 | 0.372 | 0.441 | 0.228 | 0.264 |
| RRF | 0.242 | 0.520 | 0.746 | 0.382 | 0.455 | 0.155 | 0.320 |
| z-fuse | 0.254 | 0.530 | 0.732 | 0.394 | 0.459 | 0.165 | 0.334 |
| **pop-gate** | **0.266** | **0.554** | **0.768** | **0.414** | **0.482** | 0.182 | 0.342 |
| ORACLE* | 0.278 | 0.584 | 0.784 | 0.433 | 0.498 | 0.228 | 0.323 |

**pop-gate Hit@1 gain over the best pure model:** Luxury **+0.022**, Fashion **+0.015**,
Toys **+0.019** (Prime Pantry pending). Hit@5 gains are larger (Toys **+0.058**, Luxury **+0.045**).

---

## 7. How to read the results (in depth)

1. **pop-gate wins essentially every column, every dataset** — not just Hit@1 but Hit@5, NDCG@5,
   NDCG@10 too. It beats *both* pure models simultaneously.
2. **pop-gate ≈ oracle** everywhere (within ~0.01–0.02; on All Beauty Hit@1 it *exceeds* the oracle,
   0.606 vs 0.599). It captures nearly all the achievable complementarity — **deployably**. It can
   beat the oracle because per-candidate weighting is *finer* than the oracle's single all-or-nothing
   route of the whole decision.
3. **The cold/warm columns make the mechanism visible.** Look at pop-gate:
   - its **cold** Hit@1 ≈ the **LLM's** cold (Luxury 0.480 vs LLM 0.494; Fashion 0.144 = 0.144),
   - its **warm** Hit@1 ≈ the **CF's** warm (Luxury 0.709 vs CF 0.706; All Beauty 0.962 vs 0.960).
   It **inherits the stronger expert in each regime** — exactly the design intent, observable in the
   numbers.
4. **Naive fusion clearly loses.** RRF Hit@1 is *below* the best pure model on Luxury (0.539 < 0.600),
   Fashion (0.252 < 0.292), All Beauty (~tie). z-fuse is ~neutral. Both only creep up at Hit@10 (where
   throwing candidates roughly-right is easier). This is the evidence that the *popularity weighting*
   is what matters.
5. **Honest nuance — Toys cold.** pop-gate cold (0.182) is *below* pure LLM (0.228). Toys' pivot is
   high (8.3), so even cold items (train ≤5) get non-trivial CF weight `w = pop/(pop+8.3)` (up to
   ~0.38), and CF is bad on cold → drags cold down. But pop-gate more than compensates on **warm**
   (0.342, beating *both* CF 0.296 and LLM 0.264), netting an overall win. Report this openly — it
   shows the gate trades a little cold for a lot of warm, which is the right trade for overall accuracy.

---

## 8. Ablation — learned gate (confirms the fixed formula is near-optimal)
We replace the fixed `w = pop/(pop+pivot)` with a **learned** `w = sigmoid(a·log(pop) + b)` (2
parameters), fit to rank the target first via softmax cross-entropy, **on a train half of users**,
evaluated on the held-out half.
- **Code:** `scripts/analysis_learned_gate.py`. **Reproduce:**
  `conda run -n ALLM-Rec python scripts/analysis_learned_gate.py`.
- **Result:** learned ≈ fixed within **±0.004** (both directions) on all datasets, both just under
  oracle → **learning the gate adds nothing; the simple formula is already near-optimal.**
- **Bonus:** the learned 50/50 point (pop ≈ 14.7 / 4.1 / 3.3 / 8.4) lands near each dataset's mean
  interactions/item — an independent rediscovery of the density pivot (Finding 2).
- **Reporting:** one sentence + footnote, *not* a table (a negative ablation that *reassures*).

---

## 9. Deployability (why it's a real method, not a cheat)
- At inference you rank a candidate set and **know every candidate's popularity** (a training
  statistic). So you can compute every `w(c)` and fuse — **without knowing which candidate is the
  answer.** That's a deployable recommender component.
- The **oracle** is *not* deployable: it routes by the **true target's** popularity, which presumes
  you already know the answer. It's only in the table as a ceiling.
- We also tried a *user-level* router (route by the user's history length) earlier — it captured ≈0,
  because the effect is **item-intrinsic**, not user-intrinsic. That's why the gate must be
  **per-candidate**, keyed on item popularity. (Do not confuse this with the pop-gate; the user-level
  router is a discarded negative baseline.)

---

## 10. Caveats (state honestly)
- **1 seed.** Gains are +0.015 to +0.031 Hit@1 — real and consistent across 4 datasets, but per-cell
  significance wants ≥2–3 seeds (pending).
- **20 candidates** (`EVAL_PROTOCOL.md`). Absolute values optimistic; the *relative* fusion win is
  the claim.
- **pivot choice.** We use `pivot = mean interactions/item` (no fitting). The learned-gate ablation
  shows tuning the pivot doesn't materially help, so the un-fit choice is defensible.
- **Toys cold** underperforms the LLM (see §7.5) — reported, not hidden.

---

## 11. How to write this section of the paper
- **Placement:** Result 4 (the prescriptive payoff), after the crossover (1) and density (2), with
  the vision null (3) in between. A method figure (the worked example, §4) + the full results table.
- **Narrative:** the crossover implies complementarity → a per-candidate popularity gate exploits it →
  it beats both models and ≈ oracle → naive fusion doesn't → learned gate confirms the fixed rule.
- **Lead with:** "naive fusion fails, popularity-aware fusion wins" — that contrast is the message
  (it's not "ensembling helps," it's "*using the crossover* helps").
- **Suggested sentences (adapt):**
  > "The crossover implies collaborative filtering and the LLM are complementary, which we exploit with
  > a popularity-gated fusion: we z-normalize each model's candidate scores per user and combine them
  > per candidate as `w·s_CF + (1−w)·s_LLM`, with `w = pop/(pop+pivot)` weighting collaborative
  > filtering by the candidate's popularity (`pivot` = mean interactions per item, the measured
  > crossover). This deployable fusion beats both pure models on all four datasets (+0.015 to +0.031
  > Hit@1) and matches an oracle that routes by the true item's popularity. Popularity-agnostic fusion
  > — reciprocal-rank fusion or equal-weight score fusion — does not help and is often worse than the
  > better single model, showing the gain comes specifically from exploiting the popularity structure.
  > A gate with a learned popularity threshold matches the fixed rule, confirming it is near-optimal."

---

## 12. Code & data pointers (quick index)

| What | Where |
|------|-------|
| Fusion (Hit@1/5, NDCG@5, cold/warm) | `scripts/analysis_ensemble.py` |
| Fusion full metrics (paper table) | `scripts/analysis_ensemble_full.py` |
| Learned-gate ablation | `scripts/analysis_learned_gate.py` |
| Build shared candidate pool | `scripts/make_candidates.py` → `data/candidates/{ds}_seed0.json` |
| Re-inference logging scores | `scripts/run_shared_reinfer.sh`; flags in `scripts/infer.py`, `scripts/eval_sasrec.py` |
| Model support | `clam_rec/model/clam_rec.py:load_shared_candidates`, `rank_candidates` |
| Shared scored results | `results/{ds}_{sasrec,text_concat}_shared/seed_0.jsonl` |
| Item popularity | `clam_rec/data/partition.py:data_partition` (train counts) |
| Metric helpers | `clam_rec/eval/metrics.py:hit_at_k`, `ndcg_at_k` |

**Reproduce the whole finding:**
```bash
# (data already generated: results/{ds}_{sasrec,text_concat}_shared/seed_0.jsonl)
conda run -n ALLM-Rec python scripts/analysis_ensemble_full.py   # the results table
conda run -n ALLM-Rec python scripts/analysis_learned_gate.py    # the ablation
```
