"""Config loading: merge a YAML file into a flat namespace the model consumes."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml


def load_config(yaml_path, **overrides):
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    root = Path(yaml_path).resolve().parents[1]

    def p(rel):
        rel = str(rel)
        return rel if rel.startswith("/") else str(root / rel)

    ns = SimpleNamespace(
        dataset=raw["dataset"]["name"],
        interactions=p(raw["dataset"]["interactions"]),
        text_name_dict=p(raw["dataset"]["text_name_dict"]),
        meta_json=raw["dataset"]["meta_json"],
        maxlen=raw["dataset"]["maxlen"],
        sasrec_checkpoint=p(raw["sasrec"]["checkpoint"]),
        clip_dim=raw["clip"]["dim"],
        clip_text_npy=p(f"data/clip/clip_text_{raw['dataset']['name']}.npy"),
        clip_image_npy=p(f"data/clip/clip_image_{raw['dataset']['name']}.npy"),
        fusion=raw["clip"]["fusion"],
        llm=raw["model"]["llm"],
        variant=raw["model"]["variant"],
        candidate_num=raw["eval"]["candidate_num"],
        seeds=raw["eval"]["seeds"],
        fuzzy_threshold=raw["eval"]["fuzzy_threshold"],
        cold_threshold=raw["eval"]["cold_threshold"],
        metrics_k=raw["eval"]["metrics_k"],
        batch_size1=raw["train"]["batch_size1"],
        batch_size2=raw["train"]["batch_size2"],
        batch_size_infer=raw["train"]["batch_size_infer"],
        num_epochs=raw["train"]["num_epochs"],
        stage1_lr=raw["train"]["stage1_lr"],
        stage2_lr=raw["train"]["stage2_lr"],
        device="cuda:0",
        stage=1,
        load_in_8bit=True,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns
