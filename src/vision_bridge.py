# -*- coding: utf-8 -*-
"""vision_bridge.py —— 视觉 → 主干 的神经接线（本设计第一个融合点）

数据流：
    图像 → ViT(冻结) → 1024 token/1280维 → Merger → 256 token/3584维
        → VisualConnector ★可训练 → 256 token/2880维 → 拼进主干序列

这里最关键的就是 VisualConnector：它把视觉特征拉进 GPT-OSS 的词向量空间。
训练时只训它，就能让主干「看懂图」。
"""

from __future__ import annotations

import torch
from torch import nn

from config import NFConfig


class SpatialMerger(nn.Module):
    """每个 2×2 相邻 patch 拼成一个，再降维。

    对应 Qwen2.5-VL 的 merger：
        [B, 1024, 1280] → [B, 256, 5120] → Linear → [B, 256, 3584]
    """

    def __init__(self, hidden: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden * 4, out_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, grid: int) -> torch.Tensor:
        """x: [B, grid*grid, hidden] → [B, (grid/2)^2, out_dim]"""
        B, N, H = x.shape
        half = grid // 2
        x = x.view(B, grid, grid, H)
        # 2×2 抽样合并
        x = x.view(B, half, 2, half, 2, H).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, half * half, 4 * H)
        return self.act(self.fc(x))


class VisualConnector(nn.Module):
    """★ 核心接线：3584 → 2880（本设计新增，唯一需要训的视觉部分）"""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return self.out_norm(self.drop(h))


class VisionBridge(nn.Module):
    """把 ViT + Merger + Connector 串成一个整体。

    真实模型里 `vit` 传进来的是 `Qwen2_VisionTransformer`（冻结）：
        from transformers import Qwen2VLForConditionalGeneration
        vit = Qwen2VLForConditionalGeneration.from_pretrained(...).model.visual
    这里用可替换接口，方便本机用 tiny 配置做形状验证。
    """

    def __init__(self, cfg: NFConfig, vit: nn.Module | None = None):
        super().__init__()
        v = cfg.vision
        self.grid = v.grid
        self.vit = vit if vit is not None else DummyViT(v)
        self.merger = SpatialMerger(v.hidden, v.merger_out)
        self.connector = VisualConnector(v.merger_out, cfg.backbone.hidden,
                                         cfg.connector_dropout)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel: [B,3,H,W] → [B, 256, 2880]（视觉 token，已对齐主干空间）"""
        feat = self.vit(pixel_values)           # [B,1024,1280]
        merged = self.merger(feat, self.grid)   # [B,256,3584]
        return self.connector(merged)           # [B,256,2880]

    def trainable_parameters(self):
        return list(self.merger.parameters()) + list(self.connector.parameters())


class DummyViT(nn.Module):
    """本机形状验证用的假 ViT（结构对，权重随机）。"""

    def __init__(self, v):
        super().__init__()
        self.patch = nn.Conv2d(3, v.hidden, kernel_size=v.patch, stride=v.patch)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=v.hidden, nhead=v.heads,
                                       dim_feedforward=v.hidden * 4,
                                       batch_first=True, activation="gelu")
            for _ in range(v.layers)
        ])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch(pixel_values)                 # [B,H,P,P]
        B, C, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)             # [B,P*P,C]
        for blk in self.blocks:
            x = blk(x)
        return x


if __name__ == "__main__":
    from config import tiny

    cfg = tiny()
    bridge = VisionBridge(cfg)
    img = torch.randn(1, 3, cfg.vision.image, cfg.vision.image)
    out = bridge(img)
    print("图像:", tuple(img.shape))
    print("视觉 token:", tuple(out.shape), "（期望 256 ×", cfg.backbone.hidden, "）")
    n = sum(p.numel() for p in bridge.trainable_parameters())
    print(f"可训参数（merger+connector）: {n/1e6:.2f}M")
