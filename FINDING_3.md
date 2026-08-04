# FINDING 3 — Vision (CLIP) is a null for LLM-based sequential recommendation

> ⛔⛔ **THIS IS FOR A SEPARATE (FUTURE) PAPER — DO NOT USE IN THE CURRENT PAPER.** ⛔⛔
>
> The current paper is the **CF↔LLM crossover / density / popularity-gated fusion** story
> (see `FINDINGS.md`, `FINDING_1.md`, `FINDING_2.md`, `FINDING_4.md`). That paper uses **only**
> the `SASRec` and `text` variants — **no vision**. Everything in *this* doc (the `clip_align` /
> `clip_inject` variants, CLIP fine-tuning, backbone comparison, and the generation→ranking
> protocol reversal) is **parked for a dedicated vision paper** and must **not** be cited or
> included in the current paper — putting it there would (a) dilute the CF/LLM story and (b) scoop
> the future vision paper. Keep the two entirely separate.
>
> Purpose of this doc: preserve the vision results + mechanism + code pointers so the vision paper
> can be written later without re-deriving anything. Last updated: 2026-08-04.

---

## 0. One-line finding (the vision paper's thesis candidate)

*Adding visual (CLIP) signal to an LLM-based sequential recommender does **not** help under a
ranking protocol — it is neutral-to-negative on cold items across all datasets, whether the visual
signal is used only as a Stage-1 alignment target (`clip_align`) or injected directly into the LLM
(`clip_inject`). Moreover, the apparent benefit under a generation protocol **reverses** under
ranking — a cautionary methodological result.*

---

## 1. The variants (the vision instrument)

Built on A-LLMRec (frozen SASRec + frozen OPT-6.7B). One model, a `variant` switch
(`clam_rec/model/clam_rec.py`):
- **text** — SBERT content, no vision (this is the baseline the *current* paper keeps).
- **clip_align** — content = fused CLIP (ViT-L/14, concat of text+image), used **only** as a
  Stage-1 alignment target. At inference the item token comes from SASRec; **the visual signal
  never reaches the LLM** ("the bottleneck").
- **clip_inject** — same CLIP alignment **plus** a per-item `[MMEmb]` soft token filled with the
  item's CLIP embedding (trained head `mm_emb_proj`), placed next to each title in the prompt →
  **vision reaches the LLM** at inference.
Code: `content_emb`, `_build_sample`, `_load_clip_arrays` in `clam_rec.py`; CLIP features in
`clam_rec/clip/extract.py`; fusion in `clam_rec/fusion/fusion.py`.

---

## 2. Result A — vision is null under ranking (cold-slice Hit@1 vs text)

| Dataset | text cold | clip_align Δ | clip_inject Δ |
|---------|:---------:|:------------:|:-------------:|
| Luxury Beauty | 0.492 | −0.022 | **−0.056** |
| AMAZON_FASHION | 0.158 | −0.014 | −0.028 |
| Toys & Games (sub) | 0.237 | −0.018 | −0.006 |
| Prime Pantry | 0.146 | −0.050 | +0.002 |
| _All Beauty (historical)_ | _0.131_ | _+0.014_ | _−0.001_ |

All deltas ≤ +0.014, mostly negative, both directions across datasets → **no systematic vision
benefit**, whether or not vision reaches the LLM. Overall Hit@1 (ranking) shows the same: `clip_*`
≈ `text` (e.g. Luxury text 0.600 / align 0.610 / inject 0.562; Prime Pantry text 0.226 / align
0.221 / inject 0.222). The full per-variant ranking tables (Hit@1/5/10, NDCG@5/10, cold/warm) are
recoverable from `results/{ds}_{clip_align,clip_inject}_concat_vitl14_zeroshot/seed_0.jsonl` via
the same metric code as the main paper.

**Interpretation.** `text` and all `clip` variants use the same LLM; the only difference is whether
CLIP embeddings are added. Since `text ≈ clip`, the LLM's power comes from **titles (text), not
images**. Adding the picture adds nothing reliable.

---

## 3. Result B — the mechanism (align vs inject) doesn't rescue it

`clip_align` (vision as alignment target only, bottlenecked) and `clip_inject` (vision reaches the
LLM) are **both** null. So the null is not "vision was blocked from the LLM" — even when the visual
embedding is fed directly into the prompt (`inject`), it does not help. This rules out the
"bottleneck" as the explanation and points to the visual signal itself not being discriminative
enough for next-item ranking given the title already present.

---

## 4. Result C — the generation→ranking protocol reversal (the interesting one)

Under the **older generation** protocol (LLM generates one title; Hit@1 by title match), vision
**looked helpful**: on Luxury Beauty, `clip_inject − text` on the cold slice was **+0.076**. Under
the **ranking** (likelihood) protocol it **reverses to −0.056**.

This is a genuine methodological finding worth a section: *a multimodal augmentation that appears to
help under generation-based Hit@1 can be neutral-or-harmful under likelihood ranking* — i.e. the
evaluation protocol, not the model, drove the apparent gain. (Both protocols' numbers are in
`RESULTS.md`; generation tables are marked "secondary/historical" there.)

---

## 5. Result D — CLIP fine-tuning: better retrieval, worse recommendation

LoRA domain image↔text contrastive fine-tuning of ViT-L/14 on each dataset's own (image, title)
pairs (`clam_rec/finetune/`):
- **improves retrieval**: image→text Recall@1 0.53 → 0.63 (+0.10, robust across splits), and
- **degrades recommendation**: `clip_inject` with fine-tuned vs zero-shot CLIP ≈ **−0.03 Hit@1**
  across slices (generation protocol).

A cautionary "representation quality ≠ recommendation utility" negative — fine-tuning makes each
item's image/text more self-consistent but collapses the cross-item structure the recommender needs.

---

## 6. Result E — backbone dependence (bigG vs ViT-L)

The cold/warm vision effect is **backbone-dependent**: CLIP-ViT-**bigG**-14 (stronger zero-shot)
*inverts* the pattern seen with ViT-L/14 (helps warm, slightly hurts cold), probably because bigG's
generic semantics are already captured by CF on warm items. This is why the main analysis fixed a
single backbone (ViT-L/14). Another axis the vision paper can develop.

---

## 7. Positioning for the vision paper (plan for later)
- **vs Smol-Rec (sibling, under review):** Smol-Rec claims vision **helps** via a VLM's *native
  patch-token pathway*. This null is with **pre-computed CLIP embeddings injected as soft tokens** —
  a *different mechanism*. The reconcilable story: "CLIP-embedding injection does not help LLM
  sequential rec, unlike VLM-native visual context." **This tension is the main thing to position
  carefully** before committing to the vision paper.
- Likely framing: a systematic **negative-results / analysis** paper — vision null + the protocol
  reversal (§4) + the fine-tuning negative (§5) + backbone dependence (§6).

---

## 8. Code & data pointers (for the vision paper)

| What | Where |
|------|-------|
| Vision variants (align/inject, `[MMEmb]`) | `clam_rec/model/clam_rec.py` (`content_emb`, `_build_sample`, `mm_emb_proj`) |
| CLIP extraction / images | `clam_rec/clip/extract.py`, `download_images.py` |
| Fusion of CLIP text+image | `clam_rec/fusion/fusion.py` (concat/mean/gating/text_only) |
| CLIP fine-tuning (LoRA) | `clam_rec/finetune/` (finetune_clip, extract_vitl14, proxy_analysis, pairs_dataset) |
| Vision mechanism analysis | `scripts/analysis_mechanism.py` |
| Ranking results (clip variants) | `results/{ds}_{clip_align,clip_inject}_concat_vitl14_zeroshot/seed_0.jsonl` |
| Generation results (protocol reversal) | `results/_backup_gen/`, `RESULTS.md` generation tables |
| bigG results (appendix/backbone) | `results/Luxury_Beauty_clip_inject_concat/` (bigG); `RESULTS.md` parked bigG |
| Fine-tuning artifacts | `results/finetune_vitl14*/` |

**Reminder:** none of the above goes into the current CF↔LLM paper.
