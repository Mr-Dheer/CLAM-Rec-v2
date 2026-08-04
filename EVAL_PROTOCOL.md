# EVAL_PROTOCOL.md — Evaluation protocol & the 20-candidate decision

> Why the paper evaluates ranking over **20 candidates** (1 ground-truth + 19 negatives)
> and does **not** go to 50/100. This is a locked decision (2026-08-03) with a hard
> technical reason, not a preference. Read before changing `candidate_num`.

**Decision: `candidate_num = 20`, fixed for all datasets and variants.** Metrics
(Hit@1/5/10, NDCG@5/10) are computed by likelihood-ranking these 20 candidates.

---

## Why not more candidates (the hard constraint)

The base method (A-LLMRec) ranks by building a **text prompt that lists every candidate's
title**, then scoring each candidate's title likelihood given that prompt. The frozen LLM is
**OPT-6.7B, whose context window is hard-capped at 2,048 positions.** Listing more candidates
makes the prompt longer, and past 2,048 tokens the position-embedding lookup goes out of
bounds → an immediate CUDA index assert (`indexSelectLargeIndex: srcIndex < srcSelectDimSize`),
i.e. the run **crashes**, it does not silently degrade.

**Measured prompt lengths** (tokens, OPT tokenizer; the prompt lists the candidate titles —
and *also* the user's history titles, so real prompts are longer still):

| Dataset | N=20 (max) | N=50 (max) | N=100 (max) |
|---------|:----------:|:----------:|:-----------:|
| Luxury Beauty | 627 | 1,411 | **2,842 ✗** |
| AMAZON_FASHION | 692 | 1,580 | **3,125 ✗** |
| Toys & Games (sub) | 586 | 1,298 | **2,490 ✗** |
| Prime Pantry | 703 | 1,623 | **3,088 ✗** |

- **N=100 overflows 2,048 on every dataset** → impossible.
- **N=50 is already the theoretical ceiling** (1,863 on All Beauty), and once the full user
  history is included in the prompt it becomes unsafe for long-history users → crash risk.
- **N=20 fits comfortably with headroom on all datasets** — which is precisely why A-LLMRec
  (the base) uses 20. This is a **property of the base architecture**, verified empirically
  (a 100-candidate probe crashed as predicted, 2026-08-03).

**Getting to 100 would require changing the *method*, not the eval:** score each candidate
title *without* listing candidates in the prompt (so the prompt length no longer scales with
candidate count). That is a genuine protocol change — the models were trained with candidates
listed, so it would need re-validation — and is **out of scope** for this analysis paper.

---

## Why 20 is fine and defensible

1. **It is the base method's protocol.** A-LLMRec (KDD 2024) evaluates over 20 candidates
   (1 + 19). Matching it keeps our results directly comparable and standard for this family.
2. **The metrics are stable across candidate samplings.** Re-running with an independently
   sampled shared candidate set moved Hit@1 by only **~0.003–0.008** on every dataset/slice —
   so which 20 negatives are drawn barely matters; the numbers are not an artifact of the
   sample. (SASRec non-shared vs shared, 2026-08-03.)
3. **The headline finding is robust across k.** The CF↔LLM crossover holds at **Hit@1, Hit@5,
   and Hit@10** in all 4 datasets (see `RESULTS.md`) — it is not a knife-edge of one cutoff.
4. **The thesis does not depend on candidate count.** The crossover is driven by SASRec having
   no signal on cold/rare items vs the LLM's semantic signal — a structural effect, not a
   property of the negative-sampling size.

**Note on saturation (the honest caveat to state in the paper):** at 20 candidates the ranked
metrics sit high (Hit@10 ≈ 0.8–0.98 on the warm slice), because 20 is a small pool. We report
this as a limitation. It does not affect the cold-slice conclusions or the crossover direction;
it only means absolute @k values are optimistic relative to a full-catalog / 100-negative eval.

---

## For the paper (suggested phrasing)

> "Following A-LLMRec, we rank the ground-truth item against 19 sampled negatives (20
> candidates) via the LLM's length-normalized title likelihood. The candidate count is bounded
> by the base model's 2,048-token context, since candidates are enumerated in the prompt; we
> verified that metrics are stable across independent negative samplings (Hit@1 within ±0.008)
> and that the reported crossover holds at Hit@{1,5,10}."

---

## Related docs
- `RESULTS.md` — the ranking tables (Hit@1/5/10, NDCG@5/10) this protocol produced.
- `STORY.md` — the CF↔LLM crossover thesis these metrics support.
- `PROJECT.md` — overall design; §0.0 for current scope.
