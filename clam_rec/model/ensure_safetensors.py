"""
Ensure an OPT checkpoint is available as local safetensors.

Why this exists
---------------
transformers >= ~4.5x + torch < 2.6 refuse to load `.bin` checkpoints via
torch.load (CVE-2025-32434). facebook/opt-6.7b and opt-125m are distributed as
`.bin` only, so `from_pretrained(... use_safetensors=True)` FAILS in this env.
This converts the `.bin` weights to a local safetensors directory ONCE, so the
model loads cleanly. Handles OPT's tied lm_head/embed weights.

Used by llm4rec so both the real opt-6.7b run and the opt-125m test path work on
the current ALLM-Rec env (transformers 4.57, torch 2.5). If the env is later
upgraded to torch>=2.6 the direct `.bin` path would work and this becomes a no-op.
"""

import os

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

CACHE_ROOT = os.path.expanduser("~/.cache/clam_rec/opt_safetensors")


def ensure_safetensors(model_id: str) -> str:
    """Return a local dir containing `model_id` as safetensors (converting if needed)."""
    out = os.path.join(CACHE_ROOT, model_id.replace("/", "__"))
    marker = os.path.join(out, "model.safetensors")
    single_marker = os.path.join(out, "model-00001-of-00002.safetensors")
    if os.path.exists(marker) or os.path.exists(single_marker):
        return out
    os.makedirs(out, exist_ok=True)

    AutoConfig.from_pretrained(model_id).save_pretrained(out)
    AutoTokenizer.from_pretrained(model_id, use_fast=False).save_pretrained(out)

    files = list_repo_files(model_id)
    bin_files = sorted(f for f in files if f.endswith(".bin"))
    if not bin_files:
        # already safetensors upstream; let from_pretrained handle it
        return model_id

    # merge all shards into one state dict, drop tied duplicate, save single file
    sd = {}
    for bf in bin_files:
        part = torch.load(hf_hub_download(model_id, bf), map_location="cpu", weights_only=True)
        sd.update(part)
    sd.pop("lm_head.weight", None)  # tied to embed_tokens; transformers re-ties on load
    sd = {k: v.contiguous() for k, v in sd.items()}
    save_file(sd, marker, metadata={"format": "pt"})
    return out
