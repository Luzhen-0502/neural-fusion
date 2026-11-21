# -*- coding: utf-8 -*-
"""export_merge.py —— 权重合并与导出（见 docs/EXPORT.md）

流程：
    ckpt/（只存 130M 适配层）
        → merge_lora_to_backbone()   把 LoRA 数学合并进主干（需要 64GB+ 内存）
        → requantize_4bit()          重新量化成 4bit
        → write_manifest()           写加载声明
        → export_dit_quant()         DiT 单独量化
    产物：一个目录 + manifest.json，对外表现为「一个模型」

⚠️ 本机 32GB 内存做不了 merge_lora（bf16 主干要 40GB），要在云上跑。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def merge_lora_to_backbone(ckpt_dir: str, out_dir: str) -> None:
    """把 LoRA 合并回主干（W' = W + (B@A)·alpha/r，精确等价）。"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    backbone = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        torch_dtype=torch.bfloat16,
        device_map="cpu",           # 合并不需要 GPU，但需要 ≥48GB 内存
    )
    lora_path = Path(ckpt_dir) / "step-final" / "backbone_lora"
    peft_model = PeftModel.from_pretrained(backbone, str(lora_path))
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    print("主干 LoRA 已合并 →", out_dir)


def requantize_4bit(src: str, out: str) -> None:
    """合并后的 bf16 主干重新量化成 4bit NF4。"""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    m = AutoModelForCausalLM.from_pretrained(src, quantization_config=bnb)
    m.save_pretrained(out)
    print("主干预已量化 →", out)


def export_connectors(ckpt_dir: str, out_dir: str) -> None:
    """把 Connector / Merger / GenHead 拷成独立文件。"""
    src = Path(ckpt_dir) / "step-final"
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("connector.pt", "merger.pt", "gen_adapter.pt"):
        p = src / name
        if p.exists():
            torch.save(torch.load(p, map_location="cpu"), dst / name)
            print("导出", name, f"{p.stat().st_size/1e6:.1f}MB")


def export_dit_quant(dit_ckpt: str, out_dir: str) -> None:
    """DiT 量化导出（4bit）。真实环境用 optimum-quanto。"""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    try:
        from optimum.quanto import freeze, qint4, quantize
        from diffusers import FluxTransformer2DModel

        dit = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", subfolder="transformer",
            torch_dtype=torch.bfloat16)
        quantize(dit, weights=qint4)
        freeze(dit)
        dit.save_pretrained(d / "dit-q4")
        print("DiT 已量化导出 →", d / "dit-q4")
    except ImportError:
        print("没装 optimum-quanto，跳过（pip install optimum-quanto）")


def write_manifest(out_dir: str) -> None:
    """写加载声明：一个目录 = 一个模型。"""
    manifest = {
        "name": "NF-1",
        "version": "1.0.0",
        "description": "GPT-OSS-20B + Qwen2.5-VL ViT + FLUX.1-Schnell 的神经融合统一模型",
        "parts": [
            {"role": "backbone", "path": "nf1-backbone-4bit", "dtype": "nf4", "frozen": True},
            {"role": "connector", "path": "connector.pt", "dtype": "bf16", "trainable": True},
            {"role": "gen_adapter", "path": "gen_adapter.pt", "dtype": "bf16", "trainable": True},
            {"role": "dit", "path": "dit-q4", "dtype": "qint4", "frozen": True},
        ],
        "fusion": {
            "vision_to_backbone": {"in": 3584, "out": 2880, "module": "connector"},
            "backbone_to_dit": {"in": 2880, "out": 4096, "module": "gen_adapter"},
        },
        "load_order": ["backbone", "connector", "gen_adapter", "dit"],
    }
    p = Path(out_dir) / "manifest.json"
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("manifest 已写 →", p)


def verify(out_dir: str) -> bool:
    """导出校验：文件在不在、大小对不对。"""
    d = Path(out_dir)
    checks = {
        "manifest.json": 0,
        "connector.pt": 10e6,
        "gen_adapter.pt": 5e6,
    }
    ok = True
    for name, min_size in checks.items():
        p = d / name
        if not p.exists():
            print(f"✗ 缺 {name}")
            ok = False
            continue
        size = p.stat().st_size
        flag = "✓" if size >= min_size else "✗"
        if size < min_size:
            ok = False
        print(f"{flag} {name}: {size/1e6:.2f}MB")
    return ok


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="训练输出目录")
    ap.add_argument("--out", required=True, help="导出目录")
    ap.add_argument("--skip-merge", action="store_true", help="跳过主干合并（本机跑不了）")
    a = ap.parse_args()

    if not a.skip_merge:
        merged = str(Path(a.out) / "nf1-backbone")
        merge_lora_to_backbone(a.ckpt, merged)
        requantize_4bit(merged, str(Path(a.out) / "nf1-backbone-4bit"))

    export_connectors(a.ckpt, a.out)
    export_dit_quant(a.ckpt, a.out)
    write_manifest(a.out)

    print()
    print("校验：", "通过 ✓" if verify(a.out) else "有问题 ✗")
