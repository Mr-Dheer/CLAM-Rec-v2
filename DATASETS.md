# DATASETS.md — provenance, preparation, and selection

> How every dataset in this project was built, why it was chosen, and its facts.
> Read this before adding/prepping a dataset. Numbers/results live in `RESULTS.md`;
> the paper story in `STORY.md`; this file is the **data engineering** record.

Source for all: **Amazon Review Data 2018 (McAuley, UCSD), v2**
(`https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/`). Common setup: ViT-L/14
single backbone, `concat` fusion, leave-one-out, `cold_threshold=5` (target item's
TRAIN interaction count ≤ 5 → cold). Last updated: 2026-07-31.

---

## Selection criteria (learned empirically — some the hard way)

A candidate must clear **all** of these:
1. **Item-discriminative images** — the images must identify the specific item (beauty
   product shots ✓; fashion model/flat-lay/packaging soup ✗). This is the *only* strong
   predictor of whether vision helps: Luxury (discriminative) → vision helps; Fashion
   (heterogeneous) → vision hurts. Coverage/size metrics do NOT predict it.
2. **≤ ~15k users** — OPT-6.7B Stage-2 is ~11 h at 10k users, multi-day beyond ~20k.
3. **≥ ~70% image coverage** — else vision is under-supported.
4. **≥ ~80% title coverage AND uniqueness** — the eval is title-generation-based; missing
   or duplicate titles break it (the AMAZON_FASHION bug: 88% of titles were dropped by a
   preprocess coupling bug + HTML/JS garbage titles → fake inflated Hit@1). Fixed in
   `clam_rec/data/preprocess.py` (`_clean_title`, decoupled title/description).
5. **CF weak on cold** (SASRec cold ≪ warm) — the headroom vision might fill. All
   qualifying datasets have this.

---

## Final datasets

| Dataset | Users | Items | Img cov | Title cov | Cold % | SASRec cold/warm | Provenance | Vision Δcold |
|---------|------:|------:|:-------:|:---------:|:------:|:----------------:|------------|:------------:|
| Luxury Beauty | 9,930 | 6,141 | 90% | 96.5% | 38% | 0.168 / 0.714 | reused (Extension-Paper) | **+0.076** ✅ |
| All Beauty | 2,169 | 1,854 | 70%* | 99.5% | 45% | 0.086 / 0.963 | reused SASRec + fresh preprocess | +0.006 ⚠️ |
| AMAZON_FASHION | 3,679 | 7,310 | 83% | 100%† | 75% | 0.071 / 0.783 | reused SASRec + fresh data | **−0.024** ❌ |
| Prime Pantry | 15,611 | 7,841 | 90.6% | 100% | 25% | 0.026 / 0.332 | fresh; SASRec trained by us | *training* |
| Toys & Games (sub) | 9,513 | 7,253 | 91.7% | 100% | 47% | 0.107 / 0.289 | fresh; **subsampled**; SASRec trained | *training* |

\* All Beauty image cov not re-measured post-fix. † AMAZON_FASHION titles recovered
888 → 7,309 after the preprocess bug fix.

---

## Toys & Games (subsample) — the chosen 5th, in detail

**Why chosen.** After Fashion (negative) and the muddy All Beauty, we needed a dataset with
**item-discriminative images** — the property that made Luxury work. Toys fits: toys have
distinctive appearances. We downloaded + screened Toys, CDs&Vinyl, and Grocery; CDs failed
(35% image coverage), Grocery is packaging-visual (moderate, and overlaps Prime Pantry), so
**Toys was the best domain bet**. Full Toys is far too big (374k users), so we subsampled.

**Why subsampling is valid (and how we kept it honest).** Standard practice for large
categories. Rules followed: (a) subsample by **user** (keep full histories), not by
interaction; (b) **fixed seed (42) chosen before seeing any result** — no cherry-picking the
subsample that "works"; (c) re-apply 5-core; (d) **retrain SASRec** on the subsample (can't
reuse the full-data model). Reported transparently here.

**Method** (`scripts/subsample_dataset.py`):
1. Preprocess full Toys to its k-core (`scripts/analyze_dataset.py`).
2. Random 15% of users (seed 42) → **iterative 5-core** (drop users <5 items / items <5
   users until stable) → reindex users/items to 1..N, keeping item→asin so images/CLIP align.
   - Note: frac=0.06 collapsed to 0 users (sparse dataset); **frac=0.15 → Luxury-sized**.
3. Result: **9,513 users / 7,253 items**, avg seq 8.27, 5-core.

**Facts of the subsample** (image coverage jumps because surviving items are more popular):
- image coverage **91.7%** (6,650/7,253 have URLs; 6,445 downloaded) — best of any dataset.
- title coverage **100%** (cleaned; 3 garbage titles dropped).
- **47.3% cold** — large cold population (good headroom).
- SASRec baseline **cold 0.107 ≪ warm 0.289** — CF genuinely weak on cold (favorable).

**Prep done** (all verified, pre-flight ALL GREEN):
- Meta filtered to subsample items (7.2 GB → small `data/raw/meta_Toys_and_Games_sub.json`).
- Titles built with the fixed `_clean_title` (HTML-strip + cap 150 + drop garbage).
- SASRec trained on the subsample (`scripts/train_sasrec.py`, 200 ep) → `assets/sasrec_Toys_and_Games_sub.pth`.
- 6,445 images downloaded (`clam_rec/clip/download_images.py`).
- ViT-L zero-shot embeddings extracted → `data/clip/clip_{text,image}_Toys_and_Games_sub_vitl14_zeroshot.npy` (7254×768).
- Config: `configs/toys_and_games_sub.yaml`.

**Status (2026-07-31):** `clip_inject` training (GPU 2). `text` + `clip_align` queued for when
Prime Pantry frees GPUs 0/1/3. Toys is our best-designed shot at a **second clear vision win**.

---

## Candidates screened and REJECTED

| Dataset | Reason |
|---------|--------|
| Video Games | 64k users → ~70 h/run (too big); box art only moderately visual |
| Musical Instruments | 41k users (too big) |
| Arts & Crafts | 87k users + 55% image coverage |
| CDs & Vinyl | 35% image coverage (surprising for covers, but metadata lacks URLs) |
| Appliances | 40% image coverage |
| Magazine Subscriptions | tiny + 54% image + **3.9% unique titles** (same mag re-subscribed) |
| Gift Cards | tiny + non-visual + 43% image |
| Software | 43% image coverage (boxes, not products) |
| Digital Music | 2.8% image, 7.3% title (digital goods have no product metadata), too big |
| Grocery & Gourmet | usable (69% img) but packaging-visual + overlaps Prime Pantry; held in reserve |

**Takeaway:** the manageable-size *strongly-visual* space in Amazon-v2 is essentially
{Luxury Beauty, All Beauty, AMAZON_FASHION} at native size; anything else favorable requires
**subsampling a big visual category** (Toys is the template — repeat for Home & Kitchen, Pet
Supplies, etc. if more datasets are wanted).
