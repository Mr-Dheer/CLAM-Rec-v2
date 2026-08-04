# Overnight progress — morning briefing (2026-07-22)

> ⚠️ **HISTORICAL SNAPSHOT — SUPERSEDED.** This is a point-in-time briefing. Since then the
> thesis pivoted to the **CF↔LLM crossover** and CLIP fine-tuning was **dropped** from scope
> (2026-08-03). Do not treat anything below as current. See `PROJECT.md §0.0`, `STORY.md`,
> `RESULTS.md` for current scope.

Hi Kavach — here's what I did while you were away. TL;DR: **CLIP fine-tuning works,
the whole pipeline is now validated end-to-end, and I caught + fixed a bug that
would have crashed the A6000 run.** Everything is committed and turnkey for the cluster.

## What got done

1. **CLIP LoRA fine-tuning (ViT-L/14) — DONE and it works.**
   - Domain image-text contrastive fine-tuning, LoRA (0.75% params), split by item
     (no leakage), gradient checkpointing to fit the 16GB card.
   - **Result: held-out image→text retrieval R@1 went 0.530 → 0.633 (+10.4 pts).**
     Clean generalization on unseen items → fine-tuning genuinely helps the embeddings.

   - **Robustness:** repeated on a 2nd by-item split (different held-out items) —
     seed 0: Δ+0.104, seed 1: Δ+0.100 R@1. The ~+0.10 gain is stable across splits,
     not a lucky seed.

2. **Cold vs warm proxy analysis.**
   - Fair (shared-pool) comparison: fine-tuning helps cold (+0.102 R@1) and warm
     (+0.104) items **about equally** at the representation level.
   - ⚠️ Honest caveat: this is a *retrieval* proxy, not recommendation. Whether
     fine-tuning gives a *cold-specific recommendation* gain is still open — that
     needs the A6000. Do **not** over-claim "fine-tuning rescues cold items" yet.

3. **Fine-tuned + zero-shot embeddings extracted and aligned** — ready to drop into
   the recommendation run (`data/clip/clip_fused_Luxury_Beauty_vitl14_{zeroshot,ft}.npy`).

4. **Caught + fixed a run-blocking bug.** The current env (transformers 4.57 +
   torch 2.5) refuses to load OPT's `.bin` weights (a CVE mitigation), and OPT has
   no safetensors on HF. This **would have crashed the A6000 run immediately.**
   Fixed with a one-time `.bin`→safetensors conversion (`ensure_safetensors.py`).

5. **Validated the ENTIRE pipeline end-to-end** with a tiny stand-in LLM (opt-125m):
   Stage 1 → Stage 2 → generate → sliced Hit@1 report all run correctly and produce
   the cold/warm table the paper needs. No integration surprises left for the cluster.

6. **Made the run flexible + collision-free:** a `clip_variant` switch
   (bigG / vitl14_zeroshot / vitl14_ft) so RQ3 runs alongside RQ1/RQ2 with unique
   output dirs. All 7 experiment combos dry-run-validated.

## What's left (needs the A6000)

Just one thing: run the real recommendation experiments. It's one command:
```bash
cd ~/Dev/CLAM-Rec-v2 && conda activate ALLM-Rec
bash scripts/run_all.sh
```
This trains `text` / `clip_align` / `clip_inject` (bigG) + fusion ablation +
ViT-L zero-shot vs fine-tuned, runs 10 seeds each, and prints the sliced tables
with significance. First run will spend a few minutes converting opt-6.7b to
safetensors.

## Things to decide with me later
- **RQ3 framing:** the proxy says fine-tuning helps cold & warm equally. If the
  recommendation run agrees, the RQ3 story becomes "domain fine-tuning gives a
  uniform lift" rather than "it targets cold items" — still publishable, just a
  different sentence. We'll see the real numbers on the cluster.
- Whether to also run RQ1/RQ2 with ViT-L (not just bigG) for consistency.

## Where to read more
- Full design + method + how-to-run: **`PROJECT.md`** (authoritative).
- This session's detailed results: `results/finetune_vitl14/` (history.json,
  proxy_coldwarm*.json) and git log.

Everything is committed. Nothing is broken. Ping me when you're back and we'll
launch the cluster run (or write the paper draft).
