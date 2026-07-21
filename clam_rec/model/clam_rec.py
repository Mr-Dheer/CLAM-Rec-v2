"""
CLAM-Rec model with three variants (the mechanism-contrast axis) and a fusion
switch (RQ2). Built on the trusted A-LLMRec two-stage design.

Variants (self.variant):
  - "text"        : baseline. Stage-1 alignment target = SBERT text embedding
                    (no vision). Matches A-LLMRec.
  - "clip_align"  : original CLAM-Rec. Stage-1 alignment target = fused CLIP
                    (text+image). Vision informs the aligned item embedding but
                    is NOT fed to the LLM at inference (the bottleneck).
  - "clip_inject" : bottleneck-fixed. Same CLIP alignment target AND a per-item
                    CLIP soft token ([MMEmb]) injected into the LLM at inference.

Stage 1: dual autoencoder aligns SASRec item emb <-> (text|clip) emb in a 128-D
latent, with matching + reconstruction + BCE-rec losses (unchanged from A-LLMRec).
Stage 2: project the aligned item embedding and SASRec user rep into OPT token
space as soft prompts; frozen OPT generates the next-item title.
"""

import pickle
import random

import numpy as np
import torch
import torch.nn as nn

from clam_rec.model.sasrec import RecSys
from clam_rec.model.llm4rec import llm4rec
from clam_rec.fusion.fusion import GatingFusion, static_fuse, fused_dim


class TwoLayerMLP(nn.Module):
    """Encoder+decoder pair used for alignment (returns latent, reconstruction)."""

    def __init__(self, dims, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(dims, hidden)
        self.fc2 = nn.Linear(hidden, dims)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.sigmoid(self.fc1(x))
        return z, self.fc2(z)


class ClamRec(nn.Module):
    def __init__(self, cfg):
        """cfg: a simple namespace/dict with fields used below (see configs)."""
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.variant = cfg.variant           # text | clip_align | clip_inject
        self.fusion = cfg.fusion             # concat | mean | gating
        self.maxlen = cfg.maxlen
        self.dataset = cfg.dataset

        with open(cfg.text_name_dict, "rb") as f:
            self.text_name_dict = pickle.load(f)

        self.recsys = RecSys(cfg.sasrec_checkpoint, self.device)
        self.item_num = self.recsys.item_num
        self.rec_dim = self.recsys.hidden_units

        self.mlp = TwoLayerMLP(self.rec_dim)          # SASRec-item autoencoder
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

        # ---- content branch (the alignment target) --------------------------
        self.uses_clip = self.variant in ("clip_align", "clip_inject")
        if cfg.stage == 1:
            self._build_content_branch(cfg)

        # ---- Stage-2 projections into the LLM token space -------------------
        if cfg.stage == 2 or cfg.stage == "inference":
            self.llm = llm4rec(device=self.device, llm_model=cfg.llm,
                               load_in_8bit=getattr(cfg, "load_in_8bit", True))
            H = self.llm.llm_model.config.hidden_size
            self.log_emb_proj = self._proj(self.rec_dim, H, act="leaky")
            self.item_emb_proj = self._proj(128, H, act="gelu")
            if self.variant == "clip_inject":
                # projects the per-item fused CLIP embedding into token space
                self.mm_emb_proj = self._proj(self._content_dim(cfg), H, act="gelu")
            self._load_clip_arrays(cfg)   # need raw CLIP at inference for [MMEmb]

    # ------------------------------------------------------------------ #
    def _proj(self, din, dout, act="gelu"):
        a = nn.GELU() if act == "gelu" else nn.LeakyReLU()
        m = nn.Sequential(nn.Linear(din, dout), nn.LayerNorm(dout), a, nn.Linear(dout, dout))
        nn.init.xavier_normal_(m[0].weight)
        nn.init.xavier_normal_(m[3].weight)
        return m

    def _content_dim(self, cfg):
        if self.variant == "text":
            return 768  # SBERT
        return fused_dim(self.fusion, cfg.clip_dim)

    def _build_content_branch(self, cfg):
        if self.variant == "text":
            from sentence_transformers import SentenceTransformer
            self.sbert = SentenceTransformer("nq-distilbert-base-v1")
            self.mlp2 = TwoLayerMLP(768)
        else:
            self._load_clip_arrays(cfg)
            if self.fusion == "gating":
                self.gate = GatingFusion(cfg.clip_dim)
            self.mlp2 = TwoLayerMLP(self._content_dim(cfg))

    def _load_clip_arrays(self, cfg):
        """Load aligned text/image CLIP arrays and precompute static fusion."""
        if not self.uses_clip:
            return
        text = np.load(cfg.clip_text_npy)
        image = np.load(cfg.clip_image_npy)
        self.clip_text = torch.tensor(text, dtype=torch.float32)
        self.clip_image = torch.tensor(image, dtype=torch.float32)
        if self.fusion in ("concat", "mean"):
            fused = static_fuse(self.fusion, text, image)
            self.clip_fused = torch.tensor(fused, dtype=torch.float32)
        else:
            self.clip_fused = None  # gating computes on the fly

    # ------------------------------------------------------------------ #
    def content_emb(self, item_ids):
        """Per-item content (alignment-target) embedding for given ids."""
        ids = torch.as_tensor(item_ids, dtype=torch.long)
        if self.fusion == "gating":
            t = self.clip_text[ids].to(self.device)
            im = self.clip_image[ids].to(self.device)
            return self.gate(t, im)
        return self.clip_fused[ids].to(self.device)

    def get_item_emb(self, item_ids):
        """Aligned 128-D item latent (SASRec-item -> mlp encoder)."""
        with torch.no_grad():
            e = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device))
            e, _ = self.mlp(e)
        return e

    # ------------------------------------------------------------------ #
    # text helpers (titles for prompt + matching)
    # ------------------------------------------------------------------ #
    def _title(self, item, with_desc=False):
        t = self.text_name_dict["title"].get(item, "No Title")
        if with_desc:
            d = self.text_name_dict["description"].get(item, "No Description")
            return f'"{t}, {d}"'
        return f'"{t}"'

    def find_item_text(self, items, with_desc=False):
        return [self._title(i, with_desc) for i in items]

    def save_stage1(self, path_prefix):
        torch.save(self.mlp.state_dict(), path_prefix + "mlp.pt")
        torch.save(self.mlp2.state_dict(), path_prefix + "mlp2.pt")
        if self.variant != "text" and self.fusion == "gating":
            torch.save(self.gate.state_dict(), path_prefix + "gate.pt")

    def load_stage1(self, path_prefix, freeze=True):
        self.mlp.load_state_dict(torch.load(path_prefix + "mlp.pt", map_location=self.device))
        if freeze:
            for p in self.mlp.parameters():
                p.requires_grad = False
        if self.variant != "text" and self.fusion == "gating" and hasattr(self, "gate"):
            self.gate.load_state_dict(torch.load(path_prefix + "gate.pt", map_location=self.device))

    def save_stage2(self, path_prefix):
        torch.save(self.log_emb_proj.state_dict(), path_prefix + "log_proj.pt")
        torch.save(self.item_emb_proj.state_dict(), path_prefix + "item_proj.pt")
        if self.variant == "clip_inject":
            torch.save(self.mm_emb_proj.state_dict(), path_prefix + "mm_proj.pt")

    def load_stage2(self, path_prefix):
        self.log_emb_proj.load_state_dict(torch.load(path_prefix + "log_proj.pt", map_location=self.device))
        self.item_emb_proj.load_state_dict(torch.load(path_prefix + "item_proj.pt", map_location=self.device))
        if self.variant == "clip_inject":
            self.mm_emb_proj.load_state_dict(torch.load(path_prefix + "mm_proj.pt", map_location=self.device))

    # ================================================================== #
    # Stage 1: multimodal/text <-> SASRec alignment
    # ================================================================== #
    def _content_target(self, item_ids):
        """The alignment target embedding for a batch of items."""
        if self.variant == "text":
            texts = self.find_item_text(item_ids, with_desc=True)
            tok = self.sbert.tokenize(texts)
            return self.sbert({"input_ids": tok["input_ids"].to(self.device),
                               "attention_mask": tok["attention_mask"].to(self.device)}
                              )["sentence_embedding"]
        return self.content_emb(item_ids)

    def stage1_step(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        if self.variant == "text":
            self.sbert.train()
        optimizer.zero_grad()

        u, seq, pos, neg = data
        idxs = [self.maxlen * (i + 1) - 1 for i in range(u.shape[0])]
        with torch.no_grad():
            log_emb, pos_emb, neg_emb = self.recsys.model(u, seq, pos, neg, mode="item")
        log_emb_, pos_emb_, neg_emb_ = log_emb[idxs], pos_emb[idxs], neg_emb[idxs]
        pos_ = pos.reshape(pos.size)[idxs]
        neg_ = neg.reshape(neg.size)[idxs]

        start, end, it = 0, 60, 0
        totals = dict(loss=0, bpr=0, match=0, rc=0, crc=0)
        while start < len(log_emb_):
            le, pe, ne = log_emb_[start:end], pos_emb_[start:end], neg_emb_[start:end]
            pp, nn_ = pos_[start:end], neg_[start:end]
            start, end, it = end, end + 60, it + 1

            pos_ct = self._content_target(pp)
            neg_ct = self._content_target(nn_)

            pos_match, pos_proj = self.mlp(pe)
            neg_match, neg_proj = self.mlp(ne)
            pos_ct_match, pos_ct_proj = self.mlp2(pos_ct)
            neg_ct_match, neg_ct_proj = self.mlp2(neg_ct)

            pos_logits = (le * pos_proj).mean(1)
            neg_logits = (le * neg_proj).mean(1)
            bpr = self.bce(pos_logits, torch.ones_like(pos_logits)) + \
                  self.bce(neg_logits, torch.zeros_like(neg_logits))
            match = self.mse(pos_match, pos_ct_match) + self.mse(neg_match, neg_ct_match)
            rc = self.mse(pos_proj, pe) + self.mse(neg_proj, ne)
            crc = self.mse(pos_ct_proj, pos_ct.data) + self.mse(neg_ct_proj, neg_ct.data)

            total = bpr + match + 0.5 * rc + 0.2 * crc
            total.backward()
            optimizer.step()
            optimizer.zero_grad()
            for k, v in zip(totals, [total, bpr, match, rc, crc]):
                totals[k] += v.item()
        print(f"[stage1 {self.variant}/{self.fusion}] ep {epoch}/{total_epoch} "
              f"step {step}/{total_step} loss={totals['loss']/it:.4f} bpr={totals['bpr']/it:.4f} "
              f"match={totals['match']/it:.4f} rc={totals['rc']/it:.4f} crc={totals['crc']/it:.4f}")

    # ================================================================== #
    # prompt construction (shared by stage 2 + inference)
    # ================================================================== #
    def _make_interact(self, interact_ids, max_num=10):
        titles = self.find_item_text(interact_ids)[-max_num:]
        mm = "[MMEmb]" if self.variant == "clip_inject" else ""
        text = ",".join(t + "[HistoryEmb]" + mm for t in titles)
        return text, interact_ids[-max_num:]

    def _make_candidates(self, interact_ids, num, target_id, target_title):
        neg = []
        while len(neg) < 50:
            t = np.random.randint(1, self.item_num + 1)
            if t not in interact_ids and t not in neg:
                neg.append(t)
        random.shuffle(neg)
        mm = "[MMEmb]" if self.variant == "clip_inject" else ""
        cand_ids = [target_id]
        cand_txt = [target_title + "[CandidateEmb]" + mm]
        for c in neg[:num - 1]:
            cand_txt.append(self._title(c) + "[CandidateEmb]" + mm)
            cand_ids.append(c)
        perm = np.random.permutation(len(cand_txt))
        return ",".join(np.array(cand_txt)[perm]), np.array(cand_ids)[perm]

    def _domain_prompt(self, interact_text, candidate_text):
        p = " is a user representation.This user has bought " + interact_text
        p += (" in the previous. Recommend one next item for this user to buy next "
              "from the following item title set, ")
        p += candidate_text + ". The recommendation is "
        return p

    def _build_sample(self, u, seq, pos, i):
        target_id = pos[i][-1] if pos.ndim > 1 else pos[i]
        target_title = self._title(target_id)
        interact_text, interact_ids = self._make_interact(seq[i][seq[i] > 0])
        candidate_text, candidate_ids = self._make_candidates(
            seq[i][seq[i] > 0], self.cfg.candidate_num, target_id, target_title)
        prompt = self._domain_prompt(interact_text, candidate_text)

        interact_emb = self.item_emb_proj(self.get_item_emb(interact_ids))
        candidate_emb = self.item_emb_proj(self.get_item_emb(candidate_ids))
        out = dict(prompt=prompt, target_title=target_title,
                   interact_emb=interact_emb, candidate_emb=candidate_emb)
        if self.variant == "clip_inject":
            out["interact_mm"] = self.mm_emb_proj(self.content_emb(interact_ids))
            out["candidate_mm"] = self.mm_emb_proj(self.content_emb(candidate_ids))
        return out

    # ================================================================== #
    # Stage 2 training
    # ================================================================== #
    def stage2_step(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        optimizer.zero_grad()
        u, seq, pos, neg = data
        self.llm.eval()
        with torch.no_grad():
            log_emb = self.recsys.model(u, seq, pos, neg, mode="log_only")

        samples = dict(text_input=[], text_output=[], interact=[], candidate=[])
        if self.variant == "clip_inject":
            samples["interact_mm"], samples["candidate_mm"] = [], []
        for i in range(len(u)):
            s = self._build_sample(u, seq, pos, i)
            samples["text_input"].append(s["prompt"])
            samples["text_output"].append(s["target_title"])
            samples["interact"].append(s["interact_emb"])
            samples["candidate"].append(s["candidate_emb"])
            if self.variant == "clip_inject":
                samples["interact_mm"].append(s["interact_mm"])
                samples["candidate_mm"].append(s["candidate_mm"])

        log_emb = self.log_emb_proj(log_emb)
        loss = self.llm(log_emb, samples)
        loss.backward()
        optimizer.step()
        print(f"[stage2 {self.variant}/{self.fusion}] ep {epoch}/{total_epoch} "
              f"step {step}/{total_step} loss={loss.item():.4f}")

    # ================================================================== #
    # inference (generation)
    # ================================================================== #
    @torch.no_grad()
    def generate(self, data):
        u, seq, pos, neg = data
        answers, prompts = [], []
        interact_embs, candidate_embs = [], []
        interact_mms, candidate_mms = [], []
        log_emb = self.recsys.model(u, seq, pos, neg, mode="log_only")
        for i in range(len(u)):
            s = self._build_sample(u, seq, pos, i)
            answers.append(s["target_title"])
            prompts.append(s["prompt"])
            interact_embs.append(s["interact_emb"])
            candidate_embs.append(s["candidate_emb"])
            if self.variant == "clip_inject":
                interact_mms.append(s["interact_mm"])
                candidate_mms.append(s["candidate_mm"])

        log_emb = self.log_emb_proj(log_emb)
        atts = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device).unsqueeze(1)
        log_emb = log_emb.unsqueeze(1)

        self.llm.llm_tokenizer.padding_side = "left"
        toks = self.llm.llm_tokenizer(prompts, padding="longest", return_tensors="pt").to(self.device)
        with torch.cuda.amp.autocast():
            inputs_embeds = self.llm.llm_model.get_input_embeddings()(toks.input_ids)
            toks, inputs_embeds = self.llm.replace_soft_tokens(
                toks, inputs_embeds, interact_embs, candidate_embs,
                interact_mms if self.variant == "clip_inject" else None,
                candidate_mms if self.variant == "clip_inject" else None)
            inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts, toks.attention_mask], dim=1)
            outputs = self.llm.llm_model.generate(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                do_sample=False, num_beams=1, max_new_tokens=64, min_length=1,
                pad_token_id=self.llm.llm_tokenizer.eos_token_id,
                repetition_penalty=1.5)
        outputs[outputs == 0] = 2
        text = self.llm.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [t.strip() for t in text], answers
