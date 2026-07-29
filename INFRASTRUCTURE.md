# INFRASTRUCTURE.md — Where the code and data live (READ THIS FIRST if confused)

> This file exists because the project spans **multiple machines and directories**,
> and that has repeatedly caused confusion. If you are a future Claude session (or
> Kavach) and you're unsure "where is the code / why does X differ between here and
> there / where do I run things" — this document answers it. Read it before touching
> anything on a remote box or moving files around.

Last updated: 2026-07-29.

---

## TL;DR

- There is **ONE clean codebase** (`clam_rec/` package + scripts + configs), **mirrored
  in 3 places** and kept in sync via **git** (for code) and **rsync** (for data).
- It is **NOT** scattered across many directories. The *same* code lives in 3 locations;
  the split is about **which machine does which job**, not about the code being fragmented.
- A separate, **OLD/untrusted** project also exists on the local machine — do not confuse
  it with the clean one.

---

## The 4 locations (and what each is for)

### 1. OLD project (messy, untrusted) — LOCAL: `~/Dev/Extension-Paper/`
- The **original** work Kavach wrote before the rewrite. Contains:
  - `A-LLMRec/` — original A-LLMRec code (SASRec base = trusted; we reuse it).
  - `Clip/` — the OLD CLIP pipeline (**untrusted**, had bugs: fabricated image
    embeddings for image-less items, append-mode logging, unreproducible results).
  - The paper PDFs (both CLAM-Rec and Smol-Rec).
- **Role:** reference only. We reused the SASRec checkpoint + data from here, but
  **rewrote all CLIP code from scratch**. Do NOT trust or copy the old CLIP code.
- Assets we pulled from here into the clean repo (as `assets/` symlinks locally):
  SASRec checkpoint, `Luxury_Beauty.txt`, text-name-dict, and the source images/metadata.

### 2. CLEAN project — LOCAL: `~/Dev/CLAM-Rec-v2/`  ← YOU ARE HERE
- The fresh, clean rewrite. **All new work lives here.** This is the dev machine
  (an RTX 4080, 16GB — fine for CLIP extraction / fine-tuning / verification, but
  **too small for OPT-6.7B training**).
- Layout:
  ```
  clam_rec/     the Python package (data, clip, model, fusion, eval, finetune)
  configs/      config yaml files
  scripts/      run scripts (train.py, infer.py, run_*.sh, verify_itemmap.py)
  assets/       SASRec ckpt, interactions, text dict (symlinks to Extension-Paper locally)
  data/         images/, clip/ (embeddings .npy), processed/ (itemmap, interactions)
  results/      checkpoints/ + per-seed inference .jsonl
  logs/         run logs (mostly on remote)
  PROJECT.md    authoritative design doc (goal, method, positioning)
  RESULTS.md    factual results ledger (every run + metrics)
  INFRASTRUCTURE.md  ← this file
  ```

### 3. GitHub — `github.com/Mr-Dheer/CLAM-Rec-v2` (PUBLIC)
- A **copy of #2's code**, pushed. This is the transfer hub / source of truth for code.
- Contains code + docs + small artifacts (LoRA adapter configs, history JSONs).
- Does **NOT** contain large data (embeddings `.npy`, images, checkpoints) — those are
  `.gitignore`d and moved via rsync instead.

### 4. REMOTE GPU box — `~/CLAM-Rec-v2/` on `dreal_gpu`
- Host: `d-real-gpu2.cs.universityofgalway.ie`, user `kavach_d`. 4× RTX A6000 (49GB).
- A **git clone of #3 (GitHub)** + data rsynced from #2.
- **This is where heavy training/inference actually runs** (OPT-6.7B needs a big GPU).
- SSH: `ssh dreal_gpu` (key-based auth set up: `~/.ssh/id_ed25519_dreal`).
- Conda env: `A-LLMRec` (`source ~/anaconda3/etc/profile.d/conda.sh && conda activate ALLM-Rec`).
- GPU policy: pin `CUDA_VISIBLE_DEVICES` explicitly. Historically "only GPU 2"; GPU 0
  was later also cleared. Always check `nvidia-smi` first (shared machine).

---

## How the 3 code copies stay in sync

```
   LOCAL ~/Dev/CLAM-Rec-v2  --git push-->  GitHub  --git clone/pull-->  REMOTE ~/CLAM-Rec-v2
        (edit here)                     (Mr-Dheer/CLAM-Rec-v2)              (run here)

   LOCAL data/ (images, embeddings)  --rsync-->  REMOTE data/   (git-ignored, not on GitHub)
```

- **Code**: edit locally → `git commit` → `git push` → on remote `git pull` (or
  `git checkout origin/master -- <files>` to grab specific files without a full pull).
- **Data/embeddings/checkpoints**: transferred with `rsync` (they're too big / are
  git-ignored). e.g. `rsync -a data/clip/ dreal_gpu:~/CLAM-Rec-v2/data/clip/`.
- **Assets are symlinks locally** (into `~/Dev/Extension-Paper/`), so rsync them with
  `-L` (dereference) when copying to the remote.

---

## WHY things drift (the recurring confusion) and how to fix it

**The trap:** scripts/configs sometimes get created *directly on the remote* during a
run (faster than edit-locally-commit-push-pull). Then the remote has files the local
repo + GitHub don't — the three copies **drift**.

Concretely, this happened with: `configs/luxury_beauty_rq3.yaml`, `scripts/run_bigG.sh`,
`run_rq3.sh`, `run_rq3_resume.sh`, `run_baseline_text.sh` (all born on the remote), and a
stray edit to the base `configs/luxury_beauty.yaml`.

**Fix pattern (used 2026-07-29):**
1. On remote: `git status --short` to see remote-only / modified files.
2. `rsync` the remote-only files back to LOCAL.
3. Revert any accidental edits to shared/base files (e.g. base `luxury_beauty.yaml`
   was reset to its original `batch_size2=1, batch_size_infer=8`).
4. Local: `git add -A && git commit && git push`.
5. On remote: `git fetch origin master && git checkout origin/master -- <those files>`
   so the remote's git recognizes them as tracked/clean.

**Rule of thumb to avoid drift:** prefer editing locally + push/pull. If you must create
a script on the remote for speed, remember to rsync it back and commit it afterward.

---

## Config convention (important, avoids the batch-size mess)

- **Base config `configs/luxury_beauty.yaml`** stays at ORIGINAL A-LLMRec defaults
  (`batch_size2=1`, `batch_size_infer=8`). Do NOT edit it for run tweaks — Kavach asked
  for this explicitly.
- **Run-specific tweaks live in a SEPARATE config** `configs/luxury_beauty_rq3.yaml`
  (currently `batch_size2=4`, `batch_size_infer=16` — the values that don't OOM on the
  A6000; see `PROJECT.md` §11.2 for the OOM lessons). The `run_*.sh` scripts point at
  this config.

---

## Quick "where do I ..." reference

| I want to... | Where / how |
|--------------|-------------|
| Edit model/eval/CLIP code | LOCAL `~/Dev/CLAM-Rec-v2`, then commit+push |
| Extract / regenerate CLIP embeddings | LOCAL (has `open_clip`, images, bigG cached); remote does NOT have open_clip or images |
| Fine-tune CLIP | LOCAL (fits 16GB with LoRA + grad checkpointing) |
| Train/infer OPT-6.7B (the real runs) | REMOTE `dreal_gpu`, on GPU 2 or 0, via `scripts/run_*.sh` detached |
| Get new code onto the remote | push to GitHub, then `git pull`/`checkout` on remote |
| Get new embeddings onto the remote | `rsync` local `data/clip/*.npy` → remote |
| See results | `RESULTS.md` (ledger) + remote `results/*/seed_*.jsonl` + `logs/*.log` |
| Understand the project goal/method | `PROJECT.md` (authoritative) |

---

## Related docs
- `PROJECT.md` — the authoritative design doc (goal, method, positioning vs Smol-Rec, run instructions, OOM lessons).
- `RESULTS.md` — the growing results ledger (every run + metrics).
- `OVERNIGHT_2026-07-22.md` — a point-in-time progress snapshot.
