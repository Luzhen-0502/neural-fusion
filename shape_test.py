# -*- coding: utf-8 -*-
"""shape_test.py —— 本机 CPU 形状验证

不需要 GPU、不需要真实权重。用 tiny 配置把整条链路跑一遍，
确认每一层的张量维度都对得上。

运行：python shape_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import NFConfig, tiny  # noqa: E402
from rotary import Rotary1D, Rotary2D  # noqa: E402
from unified_model import UnifiedModel  # noqa: E402


def check(name: str, got: tuple, want: tuple) -> bool:
    ok = tuple(got) == tuple(want)
    flag = "✓" if ok else "✗"
    print(f"  {flag} {name}: {tuple(got)}" + ("" if ok else f"  期望 {tuple(want)}"))
    return ok


def main() -> int:
    torch.manual_seed(0)
    cfg = tiny()
    print(cfg.summary())
    print()

    m = UnifiedModel(cfg)
    m.freeze_pretrained()
    m.eval()

    B, T = 2, 9
    img = torch.randn(B, 3, cfg.vision.image, cfg.vision.image)
    ids = torch.randint(0, cfg.backbone.vocab, (B, T))
    vis_n = cfg.vision.tokens_after_merger
    H = cfg.backbone.hidden
    d = cfg.dit

    all_ok = True
    print("【1】视觉桥")
    with torch.no_grad():
        vis = m.vision(img)
    all_ok &= check("视觉 token", vis.shape, (B, vis_n, H))

    print("【2】图文理解（融合点 1）")
    with torch.no_grad():
        logits = m.understand(img, ids)
    all_ok &= check("文本 logits", logits.shape, (B, vis_n + T, cfg.backbone.vocab))

    print("【3】生成条件（融合点 2）")
    with torch.no_grad():
        h = torch.randn(B, H)
        cond = m.encode_condition(h)
    all_ok &= check("条件序列", cond.shape, (B, cfg.gen_seq_len, d.cond_dim))

    print("【4】文生图")
    with torch.no_grad():
        lat = m.generate(ids, steps=2)
    all_ok &= check("生成 latent", lat.shape, (B, d.latent_ch, d.latent_size, d.latent_size))

    print("【5】图生图")
    src = torch.randn(B, d.latent_ch, d.latent_size, d.latent_size)
    with torch.no_grad():
        out = m.edit(img, ids, src, strength=0.5)
    all_ok &= check("编辑 latent", out.shape, (B, d.latent_ch, d.latent_size, d.latent_size))

    print("【6】RoPE")
    r1 = Rotary1D(head_dim=cfg.backbone.head_dim, max_len=64)
    x = torch.randn(1, 10, 4, cfg.backbone.head_dim)
    y = r1(x)
    all_ok &= check("1D RoPE", y.shape, x.shape)
    norm_ok = torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)
    print(f"  {'✓' if norm_ok else '✗'} 旋转保模长")
    all_ok &= norm_ok

    r2 = Rotary2D(head_dim=cfg.vision.hidden, grid=cfg.vision.grid)
    x2 = torch.randn(1, cfg.vision.tokens_before_merger, cfg.vision.hidden)
    y2 = r2(x2)
    all_ok &= check("2D RoPE", y2.shape, x2.shape)

    print("【7】冻结 / 可训练划分")
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.trainable_parameters())
    print(f"  总参数 {total/1e6:.2f}M，可训 {train/1e6:.2f}M（{train/total*100:.1f}%）")
    frozen_ok = total > train > 0
    print(f"  {'✓' if frozen_ok else '✗'} 主干已冻结、连接器可训")
    all_ok &= frozen_ok

    print()
    if all_ok:
        print("全部形状验证通过 ✓")
        return 0
    print("有维度对不上，看上面 ✗ 的行")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
