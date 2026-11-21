# -*- coding: utf-8 -*-
"""rotary.py —— 统一 RoPE 实现

★ 设计要点（对应 ARCHITECTURE.md 4.2）：
  RoPE 的**实现**统一（同一套频率公式、同一 dtype、同一数值精度），
  但三个模块的**频率表各自独立**（不共享权重，否则破坏预训练）。

提供三种位置形态：
  - 1D：文本（主干）
  - 2D：图像网格（ViT）
  - 3D：latent 空间（DiT，含时间表达）
"""

from __future__ import annotations

import torch
from torch import nn


def build_rope_freqs(head_dim: int, seq_len: int, theta: float = 10000.0,
                     device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (cos, sin)，形状都是 [seq_len, head_dim]。"""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(pos, freqs)                 # [L, half]
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    return cos, sin


class Rotary1D(nn.Module):
    """文本用的 1D RoPE。频率表是 buffer，不参与训练。"""

    def __init__(self, head_dim: int, max_len: int = 8192, theta: float = 150000.0):
        super().__init__()
        cos, sin = build_rope_freqs(head_dim, max_len, theta)
        # 拼成 [max_len, head_dim/2] 的两份，roformer 风格需要重复一次
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., L, head_dim]；返回同 shape 的旋转后张量。"""
        L = x.shape[-2]
        return apply_rope(x, self.cos[:L].to(x.dtype), self.sin[:L].to(x.dtype))


class Rotary2D(nn.Module):
    """图像用的 2D RoPE：head_dim 前半段编码 h，后半段编码 w。"""

    def __init__(self, head_dim: int, grid: int, theta: float = 10000.0):
        super().__init__()
        half = head_dim // 2
        # build_rope_freqs(d, L) 返回 [L, d/2]
        cos_h, sin_h = build_rope_freqs(half, grid, theta)
        cos_w, sin_w = build_rope_freqs(half, grid, theta)
        self.register_buffer("cos_h", cos_h, persistent=False)
        self.register_buffer("sin_h", sin_h, persistent=False)
        self.register_buffer("cos_w", cos_w, persistent=False)
        self.register_buffer("sin_w", sin_w, persistent=False)
        self.grid = grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, grid*grid, head_dim] → 同 shape"""
        B, N, D = x.shape
        half = D // 2
        idx = torch.arange(N, device=x.device)
        hi = idx // self.grid
        wi = idx % self.grid
        h = _rope1d(x[..., :half],
                    self.cos_h[hi].to(x.dtype).unsqueeze(0),
                    self.sin_h[hi].to(x.dtype).unsqueeze(0))
        w = _rope1d(x[..., half:],
                    self.cos_w[wi].to(x.dtype).unsqueeze(0),
                    self.sin_w[wi].to(x.dtype).unsqueeze(0))
        return torch.cat([h, w], dim=-1)


def _rope1d(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """t: [B,N,d]，cos/sin: [1,N,d/2] → 相邻半区配对旋转。"""
    d = t.shape[-1]
    t1, t2 = t[..., : d // 2], t[..., d // 2:]
    return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对最后一维做旋转。x 最后两维是 [..., head_dim]，cos/sin 广播到同样形状。"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    # 相邻配对旋转（LLaMA 风格）
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


if __name__ == "__main__":
    # 自检：旋转后模长不变
    r = Rotary1D(head_dim=8, max_len=16)
    x = torch.randn(2, 5, 4, 8)
    y = r(x)
    print("输入 shape:", tuple(x.shape))
    print("输出 shape:", tuple(y.shape))
    print("模长保持:", torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5))
