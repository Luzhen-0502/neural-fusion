# -*- coding: utf-8 -*-
"""unified_model.py —— NF-1 组装：把三块接成一个模型

对外只暴露三个方法：
    understand(pixel, text_ids)  → 文本 logits（图文理解）
    generate(text_hidden)        → 图像 latent（文生图）
    edit(pixel, text_hidden)     → 新图 latent（图生图）

内部所有数据流都在同一张计算图里，不是「先调 A 再调 B」。
"""

from __future__ import annotations

import torch
from torch import nn

from config import NFConfig
from gen_bridge import DummyDiT, GenHeadAdapter
from vision_bridge import VisionBridge


class DummyBackbone(nn.Module):
    """本机形状验证用的假主干（结构对，权重随机）。

    真实加载（需要 GPU + 4bit）：
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        backbone = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b", quantization_config=bnb, device_map="auto")
        for p in backbone.parameters(): p.requires_grad = False
    """

    def __init__(self, cfg: NFConfig):
        super().__init__()
        b = cfg.backbone
        self.embed = nn.Embedding(b.vocab, b.hidden)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=b.hidden, nhead=b.heads,
                                       dim_feedforward=b.hidden * 4,
                                       batch_first=True, activation="gelu")
            for _ in range(b.layers)
        ])
        self.norm = nn.LayerNorm(b.hidden)
        self.lm_head = nn.Linear(b.hidden, b.vocab, bias=False)

    def forward(self, input_ids=None, inputs_embeds=None) -> torch.Tensor:
        x = inputs_embeds if inputs_embeds is not None else self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class UnifiedModel(nn.Module):
    """NF-1 统一模型。"""

    def __init__(self, cfg: NFConfig, *,
                 backbone: nn.Module | None = None,
                 vit: nn.Module | None = None,
                 dit: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone if backbone is not None else DummyBackbone(cfg)
        self.vision = VisionBridge(cfg, vit=vit)
        self.gen_adapter = GenHeadAdapter(cfg.backbone.hidden, cfg.dit.cond_dim,
                                          cfg.gen_seq_len)
        self.dit = dit if dit is not None else DummyDiT(cfg)

    # ------------------------------------------------------------------
    # 冻结 / 可训练
    # ------------------------------------------------------------------
    def freeze_pretrained(self) -> None:
        """冻结主干和 ViT（真实场景下主干是 4bit 量化的）。"""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.vision.vit.parameters():
            p.requires_grad = False
        for p in self.dit.parameters():
            p.requires_grad = False
        # 只放开两个连接器 + merger
        for p in self.vision.merger.parameters():
            p.requires_grad = True
        for p in self.vision.connector.parameters():
            p.requires_grad = True
        for p in self.gen_adapter.parameters():
            p.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # 融合点 1：图文理解
    # ------------------------------------------------------------------
    def understand(self, pixel_values: torch.Tensor,
                   text_ids: torch.Tensor) -> torch.Tensor:
        """视觉 token 与文本 token 拼进同一条序列，一起过主干。

        pixel:   [B,3,448,448]
        text:    [B,T]（input_ids）
        返回:     [B, 256+T, vocab]
        """
        vis = self.vision(pixel_values)                 # [B,256,2880]
        txt = self.backbone.embed(text_ids)             # [B,T,2880]
        seq = torch.cat([vis, txt], dim=1)              # ★ 真正的神经融合在这里
        hidden = self.backbone(inputs_embeds=seq)       # [B,256+T,2880]
        return self.backbone.lm_head(hidden)            # [B,256+T,vocab]

    # ------------------------------------------------------------------
    # 融合点 2：生成条件
    # ------------------------------------------------------------------
    def encode_condition(self, hidden: torch.Tensor) -> torch.Tensor:
        """主干隐状态 → DiT 条件。hidden: [B,L,2880] 或 [B,2880]"""
        return self.gen_adapter(hidden)                 # [B,77,4096]

    def generate(self, text_ids: torch.Tensor,
                 latent_init: torch.Tensor | None = None,
                 steps: int = 4) -> torch.Tensor:
        """文生图：主干理解 prompt → 条件驱动 DiT 去噪。

        text_ids: [B,T]
        返回:     [B,16,64,64]（VAE latent）
        """
        h = self.backbone(input_ids=text_ids)           # [B,T,2880]
        cond = self.encode_condition(h[:, -1, :])       # [B,77,4096]
        d = self.cfg.dit
        x = latent_init if latent_init is not None else \
            torch.randn(text_ids.shape[0], d.latent_ch, d.latent_size, d.latent_size,
                        device=text_ids.device)
        # 简化的 Euler 去噪（真实 FLUX 用 rectified flow 调度）
        for i in range(steps):
            t = 1.0 - i / steps
            v = self.dit(x, cond)
            x = x - (1.0 / steps) * v
        return x

    # ------------------------------------------------------------------
    # 图生图（最能体现融合优越性的一条路径）
    # ------------------------------------------------------------------
    def edit(self, pixel_values: torch.Tensor, text_ids: torch.Tensor,
             src_latent: torch.Tensor, strength: float = 0.6) -> torch.Tensor:
        """先看懂原图，再在理解后的条件下去噪。

        pixel:      [B,3,448,448]
        text_ids:   [B,T]
        src_latent: [B,16,64,64]（原图 VAE 编码）
        """
        vis = self.vision(pixel_values)
        txt = self.backbone.embed(text_ids)
        seq = torch.cat([vis, txt], dim=1)
        h = self.backbone(inputs_embeds=seq)
        cond = self.encode_condition(h[:, -1, :])       # 已经「看过原图」的条件下

        # 加噪到 strength，再去噪
        noise = torch.randn_like(src_latent)
        x = src_latent * (1 - strength) + noise * strength
        steps = max(1, int(4 * strength))
        for _ in range(steps):
            x = x - (1.0 / steps) * self.dit(x, cond)
        return x


if __name__ == "__main__":
    from config import tiny

    cfg = tiny()
    m = UnifiedModel(cfg)
    m.freeze_pretrained()

    B, T = 1, 7
    img = torch.randn(B, 3, cfg.vision.image, cfg.vision.image)
    ids = torch.randint(0, cfg.backbone.vocab, (B, T))

    logits = m.understand(img, ids)
    print("【图文理解】")
    print("  图像:", tuple(img.shape), "文本:", tuple(ids.shape))
    print("  输出:", tuple(logits.shape),
          "（期望 [1,", cfg.vision.tokens_after_merger, "+", T, ",", cfg.backbone.vocab, "]）")

    lat = m.generate(ids)
    print("【文生图】latent:", tuple(lat.shape))

    src = torch.randn(B, cfg.dit.latent_ch, cfg.dit.latent_size, cfg.dit.latent_size)
    out = m.edit(img, ids, src)
    print("【图生图】latent:", tuple(out.shape))

    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.trainable_parameters())
    print(f"\n总参数: {total/1e6:.2f}M  可训: {train/1e6:.2f}M  "
          f"({train/total*100:.1f}%)")
