# CLAM-Rec v2 — Project Design Document

**"When Does Vision Help LLM-based Sequential Recommendation?"**

> This is the single source of truth for the project. It explains *why* we are
> doing this, *how* the paper is positioned, *what* the method is, and *how* the
> code implements it. It is written so that (a) a human can read it and understand
> the whole project, and (b) a fresh Claude Code session with no memory of prior
> chats can pick up and continue the work. Read this end-to-end before touching
> anything.

Last updated: 2026-07-21.

---

## 0. TL;DR (read this first)

- We have a **rejected paper** (CLAM-Rec, "Beyond Text: Multimodal LLM-based
  Sequential Recommendation") that added CLIP visual embeddings to A-LLMRec.
  It was **rejected because the improvement was tiny** (+1.31 Hit@1 points).
- We also have a **second, stronger paper** (Smol-Rec) that is **currently under
  review elsewhere**, with the **same title** and overlapping idea. Both papers
  must survive as **distinct** papers, so CLAM-Rec v2 must be clearly different.
- **New plan:** rewrite CLAM-Rec from scratch as an **analysis paper** — *"When
  does vision actually help?"* — targeting a **B/C-tier venue**. The contribution
  is the empirical analysis (cold vs warm items, fusion strategies, and a
  mechanism contrast), not a new architecture.
- The old code was **messy and partly wrong** (couldn't reproduce its own
  numbers; the CLIP pipeline fabricated visual data for 10% of items). We are
  building a **clean, verified** codebase in `~/Dev/CLAM-Rec-v2`.
- **Status:** all data/CLIP/eval/model code is written and validated. The only
  remaining step is running OPT-6.7B training + inference, which needs the A6000
  cluster (arriving in a few days). Everything is launch-ready.

---

## 1. Background: the two papers and why we're here

### 1.1 CLAM-Rec (the rejected paper — what we are rewriting)
- Title: *"Beyond Text: Multimodal LLM-based Sequential Recommendation"* (ICANN26, **rejected**).
- Idea: extend **A-LLMRec** (KDD 2024) by replacing its SBERT text embeddings with
  **concatenated CLIP text+image embeddings** (2560-D from CLIP-ViT-bigG-14).
- Result: Hit@1 on Amazon Luxury Beauty went 0.5840 → 0.5971 = **+1.31 points**.
- **Rejection reason: the improvement was too small.**

### 1.2 Smol-Rec (the sibling paper — DO NOT collide with it)
- **Same title.** 5 authors (Kavach Dheer, Muhammad Ammar Ul Hassan, Peter
  Corcoran, Joseph Lemley, Josephine Griffith — University of Galway).
- **Currently UNDER REVIEW elsewhere.**
- Idea: uses **SmolVLM2-2.2B (a VLM)** and feeds **raw history images through the
  VLM's native visual pathway** (patch tokens) at inference. Headline study =
  **visual-context budget** (how many history images `k` to attach; peaks at k=5,
  degrades after). 3 datasets (All Beauty, Luxury Beauty, Video Games). Bigger
  gains (Luxury Beauty 58.00 → 62.51). It's the stronger paper.

### 1.3 The collision risk (important)
CLAM-Rec and Smol-Rec share the **same title, same intro/related-work framing,
same A-LLMRec base, same "add vision to beauty rec" thesis.** As-is, resubmitting
CLAM-Rec while Smol-Rec is under review is a **dual-submission risk** that could
hurt *both* papers. Therefore CLAM-Rec v2 **must**:
1. Use a **different title** (drop "Beyond Text").
2. Have a **genuinely different research question** that Smol-Rec does not answer.
3. **Rewrite** the intro/related work so the framing does not mirror Smol-Rec.
4. Cite/position against the VLM-native approach rather than competing with it.

---

## 2. The goal and positioning of CLAM-Rec v2

### 2.1 The core idea (the pivot that saves the paper)
The +1.31 **global average** was the rejection reason. Our thesis is that **the
global average is the wrong lens**: vision helps a lot in some conditions and not
at all in others, and averaging hides this. So instead of chasing a bigger average
number, we **characterize when vision helps**.

> **Research question:** *Under what item- and design-level conditions does adding
> visual (CLIP) signal improve LLM-based sequential recommendation?*

This turns the weak result into the *finding*: "vision helps exactly where it
should (cold items) and not where it shouldn't (warm items) — the small average is
the average of a large effect and no effect."

### 2.2 Why this is distinct from Smol-Rec (the safety argument)
| Axis | Smol-Rec | CLAM-Rec v2 |
|------|----------|-------------|
| Backbone | SmolVLM2 (a VLM) | OPT-6.7B (text LLM) |
| Visual input | raw images via VLM native pathway at inference | **pre-computed CLIP embeddings** |
| Headline study | image **budget** `k` (how many images) | item/design **conditions** (when does it help) |
| Contribution | a visual mechanism | an **empirical analysis** |
| Datasets | 3 | Luxury Beauty (for now) |

Smol-Rec asks "how many images?"; we ask "for which items?". Different question,
different backbone, different framing. That is what makes them two papers.

### 2.3 Target venue
**B/C-tier / lower-tier conference.** Analysis papers fit these venues well
(no novel-architecture bar; a solid, well-executed empirical study is enough).

### 2.4 A suggested new title
e.g. *"When Does Vision Help? An Analysis of Multimodal Signal in LLM-based
Sequential Recommendation"* — anything that drops "Beyond Text" and foregrounds
the analysis framing.

---

## 3. The research questions (each = a results section)

- **RQ1 — Item coldness (the headline).** Does visual signal help more for **cold**
  items (few training interactions) than **warm** ones?
  *Hypothesis:* yes — cold items have weak collaborative-filtering signal, so
  content/visual matters more. This is the money result.
- **RQ2 — Fusion strategy.** Does *how* CLIP text+image are fused matter?
  Compare **concat** vs **mean** vs a learned **gating** module.
- **Mechanism contrast (cross-cutting).** The original CLAM-Rec used CLIP only as
  a Stage-1 *alignment target*; the visual signal **never reaches the LLM at
  inference** (see §4.3, the "bottleneck"). We compare that (`clip_align`) against
  a variant that **injects** the CLIP embedding into the LLM at inference
  (`clip_inject`). Does fixing the bottleneck help, and does it help more on cold
  items?

**Optional axes available but not committed** (the code/data support them):
image-present vs image-missing items (now possible because our image coverage is a
correct 90%), and user-sparsity (short vs long history).

---

## 4. The method

### 4.1 The base: A-LLMRec (two stages)
A-LLMRec aligns a frozen collaborative-filtering recommender (SASRec) with a
frozen LLM (OPT-6.7B) so the LLM can use collaborative knowledge.

- **Stage 1 — alignment.** A dual autoencoder maps SASRec item embeddings (50-D)
  and a *content* embedding (text or multimodal) into a shared **128-D** latent
  space, using: a **matching** loss (MSE, pull the two latents together), a
  **reconstruction** loss per autoencoder (prevent collapse), and a **BCE
  recommendation** loss (preserve ranking ability). Output: a 128-D joint item
  embedding.
- **Stage 2 — LLM integration.** Two-layer MLPs project (a) the 128-D joint item
  embedding and (b) the SASRec user/sequence representation into the LLM's token
  embedding space, as **soft prompt tokens**. The frozen OPT-6.7B is prompted with
  the user representation + item titles (each title accompanied by its soft token)
  + a candidate set, and **generates the title of the next item**.
- **Evaluation.** For each user: the ground-truth next item + 19 random negatives
  = 20 candidates. **Hit@1** = fraction of cases where the generated title matches
  the ground-truth title (string / fuzzy match). 10 seeds for negative sampling.

### 4.2 What CLAM-Rec changes
CLAM-Rec replaces the Stage-1 *content* embedding: instead of **SBERT (768-D)**,
it uses **fused CLIP = [CLIP-text (1280) ‖ CLIP-image (1280)] = 2560-D** from
CLIP-ViT-bigG-14. That's the entire original idea.

### 4.3 The bottleneck (why the original gain was tiny — the key insight)
In the original CLAM-Rec, at **inference** the item embedding fed to the LLM comes
**only** from `SASRec.item_emb → mlp` (a 50-D collaborative vector projected to
128-D). The CLIP branch (`mlp2`) is used **only as a target in the Stage-1
matching loss** — it never flows into the LLM. So a rich 2560-D visual signal only
influences the LLM *second-hand*, by nudging how the 50-D CF vector gets reshaped.
**This architecturally caps the possible gain**, and it's the most likely reason
the improvement was only +1.31. (Confirmed by reading the original `opt-clip`
branch inference code: `generate()` never calls the CLIP lookup.)

### 4.4 The three model variants (the experimental instrument)
We implement one model with a `variant` switch:

1. **`text`** — baseline. Stage-1 content target = SBERT. No vision. (= A-LLMRec.)
2. **`clip_align`** — original CLAM-Rec. Stage-1 content target = fused CLIP.
   Vision informs the aligned item embedding but is **NOT** fed to the LLM at
   inference (reproduces the bottleneck).
3. **`clip_inject`** — bottleneck-fixed. Same CLIP alignment target **AND** a
   per-item CLIP soft token (`[MMEmb]`) injected into the LLM prompt next to each
   item title, so the visual signal reaches the LLM directly at inference.

Note: `clip_inject` is deliberately framed as a **minor design choice within the
analysis**, NOT as the paper's headline mechanism — because "inject visual
embedding as a soft token into the LLM" is close to what Smol-Rec/I-LLMRec do, and
we must not make that our central claim. The headline is the **analysis** (RQ1/RQ2).

### 4.5 The fusion strategies (RQ2)
- **concat** — `[text ‖ image]`, 2560-D (each half L2-normalized). Default.
- **mean** — `(text + image)/2`, L2-normalized, 1280-D. Falls back to text if the
  item has no image.
- **gating** — a small learned MLP predicts a per-item gate `g ∈ [0,1]`; output =
  L2(`g·text + (1-g)·image`), 1280-D. Forces `g=1` when the image is missing.

---

## 5. The dataset (Amazon Luxury Beauty)

- 9,930 users, 6,141 items, 63,953 interactions (4-core filtered). Matches the
  paper exactly.
- Very **skewed** item popularity: median 3 interactions/item; 66% of items have
  ≤5 interactions. → rich cold/warm split for RQ1.
- **Image coverage: 5,527/6,141 = 90.0%** (the other 10% have no image URL in the
  Amazon metadata → they get a **zero image half**). This matches the paper's
  stated "~90% have images". (The *old* code buggily had 100% coverage — it
  fabricated visual embeddings for the image-less items; we fixed this.)
- **Cold/warm split** (coldness = the target/test item's interaction count in the
  **training** portion only, to avoid leakage): at threshold ≤5, **3,763 cold /
  6,149 warm** of 9,912 test users. Well-balanced for significance testing.

---

## 6. Data provenance and correctness (why you can trust the numbers)

The old results were **not reproducible** and the old CLIP code was **not
trustworthy**, so we rebuilt with explicit verification at every step:

1. **Item→ASIN alignment is provably correct.** Our clean preprocessing
   (`clam_rec/data/preprocess.py`) reproduces A-LLMRec's `data_preprocess.py` and
   *also* saves the item map. The regenerated interaction file is **byte-identical
   (same SHA-256)** to the canonical `Luxury_Beauty.txt`, and our `id→asin` map
   agrees **100%** (6141/6141) with the old map. → CLIP rows provably line up with
   the SASRec item ids.
2. **CLIP embeddings regenerated from scratch** (`clam_rec/clip/extract.py`) with
   CLIP-ViT-bigG-14. Verified against the old fused file: **image-half cosine =
   1.000**, text-half cosine median = 0.996. The only real difference is the
   **corrected 90% image coverage** (old = buggy 100%).
3. **Clean JSONL logging** for inference (one record per test case, write-mode).
   This fixes the old pipeline's append-mode bug that concatenated multiple runs
   into single seed files and corrupted the counts.
4. **Per-slice paired-t significance** across seeds (scipy).

### Known facts / gotchas discovered
- The old saved `.txt` results give CLIP = 0.5768 vs baseline 0.5809 — i.e. the
  **direction is reversed** vs the paper's claimed +1.31. The paper's numbers
  could not be reproduced from artifacts. **Do not rely on any old results.**
- `data_preprocess.py` pass-2 does NOT re-apply the `overall<3` skip that pass-1
  uses, so ~96 items enter via low-rated reviews. Our preprocessing matches this
  exactly (that's why the byte-match works). The download script's `ASIN set
  mismatch WARNING` (6141 vs 6045) is a benign consequence and is handled (we
  download for the full 6141-item set).

---

## 7. The code (how everything is implemented)

Repo: `~/Dev/CLAM-Rec-v2`. Environment: conda env **`ALLM-Rec`**
(`/home/kavach/Dev/anaconda3/envs/ALLM-Rec/bin/python`) — already has torch 2.5.1
+cu121, open_clip 3.2.0, transformers 4.57, sentence-transformers 3.4.1,
bitsandbytes, scipy, etc. No env setup needed.

### 7.1 Layout
```
clam_rec/
  config.py               # loads configs/*.yaml into a flat namespace
  data/
    preprocess.py         # reproduce itemmap (verified byte-identical)
    partition.py          # leave-one-out + cold/warm tagging (RQ1)
    seq_dataset.py        # SeqDataset (train), SeqDatasetInference (eval)
  clip/
    download_images.py    # download 1 image/ASIN (from Smol-Rec script), 90% cov
    extract.py            # CLIP text+image extract + align + fuse (from scratch)
  fusion/
    fusion.py             # concat / mean / GatingFusion
  model/
    sasrec.py             # frozen SASRec + RecSys loader
    llm4rec.py            # OPT-6.7B wrapper + [MMEmb] soft-token injection
    clam_rec.py           # MAIN model: 3 variants, Stage1/Stage2/generate
  eval/
    metrics.py            # Hit@K/NDCG@K, sliced overall/cold/warm, paired-t
    report.py             # turn per-seed JSONL into paper tables
configs/luxury_beauty.yaml
scripts/
  verify_itemmap.py       # DONE: proves alignment correctness
  train.py                # train one variant/fusion, one stage
  infer.py                # one seed -> results/<variant>_<fusion>/seed_<s>.jsonl
  run_all.sh              # full experiment matrix + report
assets/                   # symlinks to reused inputs (SASRec ckpt, interactions, text dict)
data/
  processed/              # regenerated Luxury_Beauty.txt + itemmap.pkl
  images/Luxury_Beauty/   # 5527 downloaded {asin}.jpg
  clip/                   # clip_{text,image,fused}_Luxury_Beauty.npy
results/                  # checkpoints + per-seed JSONL logs
```

### 7.2 Key files explained
- **`model/clam_rec.py`** — the heart. `ClamRec(cfg)`:
  - loads frozen SASRec, the text dict, and (for CLIP variants) the CLIP arrays;
  - `stage1_step()` runs the alignment losses (bpr + match + 0.5·rc + 0.2·crc);
  - `_build_sample()` constructs the Stage-2 prompt: user rep + history titles
    (each `title[HistoryEmb]` + `[MMEmb]` if inject) + candidate titles; builds
    the projected soft embeddings;
  - `stage2_step()` trains the projections against OPT's next-title loss;
  - `generate()` produces titles at inference; returns `(generated, answers)`.
- **`model/llm4rec.py`** — wraps OPT-6.7B (8-bit via bitsandbytes), registers
  special tokens `[UserRep] [HistoryEmb] [CandidateEmb] [MMEmb]`, and
  `replace_soft_tokens()` overwrites those token positions in the embedded prompt
  with the projected item/multimodal embeddings.
- **`eval/metrics.py`** — `normalize_title` + exact/fuzzy match → Hit@1;
  `evaluate_sliced` splits records into overall/cold/warm; `paired_ttest` does the
  significance test across seeds. Records are dicts:
  `{user, seed, cold, train_count, answer, generated}`.

### 7.3 The config (`configs/luxury_beauty.yaml`)
Single source of run parameters: dataset paths, SASRec dims, CLIP model id, the
`variant` and `fusion` switches, training hyperparameters (batch sizes, 10 epochs,
lr 1e-4), and eval settings (20 candidates, seeds 0–9, fuzzy 0.90, `cold_threshold
= 5`, metrics_k = [1,5,10]). Override `variant`/`fusion`/`stage` from the CLI.

---

## 8. How to run (once the A6000 cluster is available)

OPT-6.7B in 8-bit needs ≈ 24GB+ (the dev box is a 16GB RTX 4080 — too small for
Stage 2/inference; Stage 1 alone fits). On the A6000:

```bash
cd ~/Dev/CLAM-Rec-v2
conda activate ALLM-Rec          # or prefix with: conda run -n ALLM-Rec

# ---- one variant, manually ----
python scripts/train.py --config configs/luxury_beauty.yaml --variant clip_inject --fusion concat --stage 1
python scripts/train.py --config configs/luxury_beauty.yaml --variant clip_inject --fusion concat --stage 2
for s in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/infer.py --config configs/luxury_beauty.yaml --variant clip_inject --fusion concat --seed $s
done

# ---- OR the whole matrix + report ----
bash scripts/run_all.sh
```

`run_all.sh` runs the mechanism contrast (`text`, `clip_align`, `clip_inject` at
fusion=concat) and the RQ2 ablation (`clip_inject` at concat/mean/gating), 10
inference seeds each, then prints the sliced result tables with significance.

### Expected outputs
- Checkpoints: `results/checkpoints/<variant>_<fusion>/{mlp,mlp2,gate,log_proj,item_proj,mm_proj}.pt`
- Logs: `results/<variant>_<fusion>/seed_<s>.jsonl`
- Tables: printed by `python -m clam_rec.eval.report --runs_dir results --baseline text --variants text clip_align clip_inject --ks 1 5 10 --fuzzy 0.90`

### What the tables should show (hypotheses, to be confirmed)
- **RQ1:** on the **cold** slice, `clip_*` >> `text` with significance; on the
  **warm** slice, little/no difference. → "vision helps where CF is weak."
- **Mechanism:** `clip_inject` ≥ `clip_align` (fixing the bottleneck helps),
  especially on cold items.
- **RQ2:** whether gating/mean beat plain concat.

---

## 9. Current status and what's left

| Step | Status |
|------|--------|
| Clean repo scaffold | ✅ |
| Verified item→ASIN alignment | ✅ (byte-identical) |
| Image download (90%) | ✅ |
| Partition + cold/warm tagging | ✅ (3763/6149) |
| Sliced eval harness (+significance) | ✅ (unit-tested) |
| CLIP extraction/align/fuse from scratch | ✅ (verified, bug fixed) |
| Model: 3 variants + fusion | ✅ (Stage 1 + wiring validated on 16GB) |
| **Stage 2 / inference / full matrix** | ⏳ **needs A6000** |
| Paper writing | ⏳ not started |

**Immediate next action when compute is available:** run `scripts/run_all.sh`,
then read the report tables and confirm/adjust the RQ hypotheses. Then write the
paper (new title, rewritten intro/related-work per §2, results per §3).

---

## 10. For a future Claude session (onboarding checklist)

1. Read this file top to bottom. It is the plan.
2. The persistent memory index is at
   `~/.claude/projects/-home-kavach-Dev-Extension-Paper/memory/MEMORY.md`
   (individual notes there mirror this doc but this doc is authoritative).
3. **Constraints you must respect:**
   - CLAM-Rec v2 must stay **distinct from Smol-Rec** (§1.3, §2.2). Do not adopt
     Smol-Rec's title, framing, or make "inject CLIP into the LLM" the headline.
   - Target is a **B/C-tier** venue; the contribution is the **analysis**, not a
     new architecture.
   - **Do not trust old results or old CLIP code** in `~/Dev/Extension-Paper`.
   - Use the **`ALLM-Rec` conda env**. Heavy training needs the A6000 (24GB+).
   - The user prefers **autonomous execution** — don't ask permission for routine
     tool use; reserve questions for genuine research/positioning decisions.
4. Old (untrusted) code for reference only: `~/Dev/Extension-Paper/A-LLMRec`
   (SASRec base, trusted) and `~/Dev/Extension-Paper/Clip` (CLIP pipeline,
   **untrusted** — we rewrote it).
```
