"""
Frozen OPT-6.7B wrapper for recommendation, ported from A-LLMRec's llm4rec.py.

Adds a new special token [MMEmb] for the clip_inject variant: a per-item
multimodal (CLIP) soft token placed next to each item title, so the visual
signal reaches the LLM directly at inference (fixing the alignment-only
bottleneck of the original CLAM-Rec).

Special tokens: [UserRep], [HistoryEmb], [CandidateEmb], [MMEmb].
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, OPTForCausalLM, BitsAndBytesConfig

from clam_rec.model.ensure_safetensors import ensure_safetensors


class llm4rec(nn.Module):
    def __init__(self, device, llm_model="opt", max_output_txt_len=256, load_in_8bit=True):
        super().__init__()
        self.device = device

        if llm_model in ("opt", "opt_small"):
            # opt_small (opt-125m) is ONLY for testing the pipeline end-to-end on a
            # small GPU; real experiments use opt (opt-6.7b). Same architecture.
            model_name = "facebook/opt-125m" if llm_model == "opt_small" else "facebook/opt-6.7b"
            # This env (transformers 4.57 + torch 2.5) blocks .bin loading; OPT ships
            # as .bin only, so convert to local safetensors once. See ensure_safetensors.
            local_path = ensure_safetensors(model_name)
            if load_in_8bit and llm_model == "opt":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                self.llm_model = OPTForCausalLM.from_pretrained(
                    local_path, quantization_config=bnb_config,
                    dtype=torch.float16, use_safetensors=True,
                    device_map=self.device)
            else:
                self.llm_model = OPTForCausalLM.from_pretrained(
                    local_path, dtype=torch.float16, device_map=self.device)
            self.llm_tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False)
        else:
            raise ValueError(f"{llm_model} is not supported")

        self.llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llm_tokenizer.add_special_tokens({"bos_token": "</s>"})
        self.llm_tokenizer.add_special_tokens({"eos_token": "</s>"})
        self.llm_tokenizer.add_special_tokens({"unk_token": "</s>"})
        self.llm_tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[UserRep]", "[HistoryEmb]", "[CandidateEmb]", "[MMEmb]"]})
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for _, p in self.llm_model.named_parameters():
            p.requires_grad = False
        self.max_output_txt_len = max_output_txt_len

    def _tok_id(self, tok):
        return self.llm_tokenizer(tok, return_tensors="pt",
                                  add_special_tokens=False).input_ids.item()

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens["input_ids"].append(torch.cat([
                input_ids[i][:this_input_ones], output_ids[i][1:],
                input_ids[i][this_input_ones:]]))
            llm_tokens["attention_mask"].append(torch.cat([
                input_atts[i][:this_input_ones], output_atts[i][1:],
                input_atts[i][this_input_ones:]]))
        llm_tokens["input_ids"] = torch.stack(llm_tokens["input_ids"])
        llm_tokens["attention_mask"] = torch.stack(llm_tokens["attention_mask"])
        return llm_tokens, input_part_targets_len

    def replace_soft_tokens(self, llm_tokens, inputs_embeds,
                            interact_embs, candidate_embs,
                            interact_mm=None, candidate_mm=None):
        """Overwrite [HistoryEmb]/[CandidateEmb] (and optionally [MMEmb]) token
        positions with the provided per-item embeddings."""
        if len(interact_embs) == 0:
            return llm_tokens, inputs_embeds
        hist_id = self._tok_id("[HistoryEmb]")
        cand_id = self._tok_id("[CandidateEmb]")
        mm_id = self._tok_id("[MMEmb]")

        for b in range(len(llm_tokens["input_ids"])):
            ids = llm_tokens["input_ids"][b]
            for tok_id, embs in [(hist_id, interact_embs[b]), (cand_id, candidate_embs[b])]:
                idxs = (ids == tok_id).nonzero().view(-1)
                for idx, emb in zip(idxs, embs):
                    inputs_embeds[b][idx] = emb
            if interact_mm is not None and candidate_mm is not None:
                mm_seq = list(interact_mm[b]) + list(candidate_mm[b])
                idxs = (ids == mm_id).nonzero().view(-1)
                for idx, emb in zip(idxs, mm_seq):
                    inputs_embeds[b][idx] = emb
        return llm_tokens, inputs_embeds

    def forward(self, log_emb, samples):
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device).unsqueeze(1)

        out_tok = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples["text_output"]],
            return_tensors="pt", padding="longest", truncation=False).to(self.device)
        in_tok = self.llm_tokenizer(
            samples["text_input"], return_tensors="pt",
            padding="longest", truncation=False).to(self.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            in_tok.input_ids, in_tok.attention_mask, out_tok.input_ids, out_tok.attention_mask)

        targets = llm_tokens["input_ids"].masked_fill(
            llm_tokens["input_ids"] == self.llm_tokenizer.pad_token_id, -100)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100
        empty_targets = torch.ones(atts_llm.size(), dtype=torch.long).to(self.device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens["input_ids"])
        llm_tokens, inputs_embeds = self.replace_soft_tokens(
            llm_tokens, inputs_embeds, samples["interact"], samples["candidate"],
            samples.get("interact_mm"), samples.get("candidate_mm"))
        attention_mask = llm_tokens["attention_mask"]

        inputs_embeds = torch.cat([log_emb.unsqueeze(1), inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, attention_mask], dim=1)

        with torch.cuda.amp.autocast():
            outputs = self.llm_model(inputs_embeds=inputs_embeds,
                                     attention_mask=attention_mask,
                                     return_dict=True, labels=targets)
        return outputs.loss
