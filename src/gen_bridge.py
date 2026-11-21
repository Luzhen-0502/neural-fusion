# -*- coding: utf-8 -*-
"""gen_bridge.py —— 主干 → DiT 的神经接线（本设计第二个融合点）

数据流：
    主干最后隐状态 [B,2880] → GenHeadAdapter ★可训练 → [B,77,4096]
        → 替换 FLUX 原本的 T5 文本条件 → 送进 DiT 的 cross-attention

为什么是 4096：FLUX 的文本条件来自 T5-XXL（4096 维）。
我们把主干隐状态直接投到这个空间，DiT 就当它是「文本条件」用——但语义上
它已经包含了「模型对整段对话/图像的理解」，比单纯 T5 编码 prompt 强得多。
"""

from __future__ import annotations

import torch
from torch import nn

from config import NFConfig


class GenHeadAdapter(nn.Module):
    """★ 核心接线：2880 → 4096，并广播成 DiT 需要的 77 token 序列。"""

    def __init__(self, in_dim: int, out_dim: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, in_dim] 或 [B, L, in_dim]（取最后位置）→ [B, seq_len, out_dim]"""
        if h.dim() == 3:
            h = h[:, -1, :]
        v = self.norm(self.fc(h))                     # [B, 4096]
        return v.unsqueeze(1).expand(-1, self.seq_len, -1).contiguous()


class DitConditionInjector(nn.Module):
    """把主干条件注入 DiT。

    真实 DiT 的 cross-attention 里：
        k = k_proj(cond), v = v_proj(cond)
    我们替换 cond 的来源即可。这里给出「条件投影 + 残差融合」的可训练适配层，
    对应 EXPORT/TRAINING 里说的 DiT LoRA（r=64）挂载点。
    """

    def __init__(self, cond_dim: int, hidden: int):
        super().__init__()
        self.cond_to_hidden = nn.Linear(cond_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.gate = nn.Parameter(torch.zeros(1))      # 初值 0：训练初期不干扰原 DiT

    def forward(self, cond: torch.Tensor, dit_hidden: torch.Tensor) -> torch.Tensor:
        """cond: [B,77,4096]；dit_hidden: [B,N,3072] → 融合后的 [B,N,3072]"""
        c = self.norm(self.cond_to_hidden(cond))      # [B,77,3072]
        # 用 DiT 隐状态去 attend 条件（等价于 cross-attn 的输出残差）
        attn = torch.softmax(
            torch.einsum("bnd,bkd->bnk", dit_hidden, c) / (c.shape[-1] ** 0.5), dim=-1)
        fused = torch.einsum("bnk,bkd->bnd", attn, c)
        return dit_hidden + self.gate.tanh() * fused


class DummyDiT(nn.Module):
    """本机形状验证用的假 DiT（结构对，权重随机）。

    真实模型：FLUX.1-schnell 的 MMDiT（19 双流 + 38 单流）。
    这里用简化 Transformer 模拟「图像 latent + 条件 → 速度场」这条通路。
    """

    def __init__(self, cfg: NFConfig):
        super().__init__()
        d = cfg.dit
        self.in_proj = nn.Linear(d.latent_ch, d.hidden)
        self.inject = DitConditionInjector(d.cond_dim, d.hidden)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d.hidden, nhead=4,
                                       dim_feedforward=d.hidden * 2,
                                       batch_first=True, activation="gelu")
            for _ in range(d.double_blocks + d.single_blocks)
        ])
        self.out_proj = nn.Linear(d.hidden, d.latent_ch)

    def forward(self, latent: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """latent: [B,16,64,64]；cond: [B,77,4096] → 预测速度 [B,16,64,64]"""
        B, C, H, W = latent.shape
        x = latent.flatten(2).transpose(1, 2)          # [B,N,C]
        x = self.in_proj(x)                            # [B,N,hidden]
        x = self.inject(cond, x)
        for blk in self.blocks:
            x = blk(x)
        x = self.out_proj(x)                           # [B,N,C]
        return x.transpose(1, 2).view(B, C, H, W)


if __name__ == "__main__":
    from config import tiny

    cfg = tiny()
    d = cfg.dit
    h = torch.randn(1, cfg.backbone.hidden)
    adapter = GenHeadAdapter(cfg.backbone.hidden, d.cond_dim, d.cond_len)
    cond = adapter(h)
    print("主干隐状态:", tuple(h.shape))
    print("生成条件  :", tuple(cond.shape), "（期望 [1,", d.cond_len, ",", d.cond_dim, "]）")

    dit = DummyDiT(cfg)
    latent = torch.randn(1, d.latent_ch, d.latent_size, d.latent_size)
    v = dit(latent, cond)
    print("latent    :", tuple(latent.shape))
    print("速度场    :", tuple(v.shape), "（应与 latent 同形）")
    n = sum(p.numel() for p in adapter.parameters())
    print(f"可训参数（GenHead）: {n/1e6:.2f}M")
