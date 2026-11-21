# -*- coding: utf-8 -*-
"""config.py —— NF-1 全部维度常量（唯一真相源）

所有模块都从这里取尺寸，改一处就全变。
数值取自三个模型的公开配置：
  - GPT-OSS-20B：openai/gpt-oss-20b  config.json
  - Qwen2.5-VL-7B 的 ViT：Qwen/Qwen2.5-VL-7B-Instruct
  - FLUX.1-Schnell：black-forest-labs/FLUX.1-schnell
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BackboneCfg:
    """GPT-OSS-20B 主干"""

    hidden: int = 2880
    layers: int = 24
    heads: int = 64
    kv_heads: int = 8
    head_dim: int = 64
    vocab: int = 201088
    rope_theta: float = 150000.0
    num_experts: int = 32
    experts_top_k: int = 4


@dataclass
class VisionCfg:
    """Qwen2.5-VL-7B 的 ViT 视觉塔"""

    hidden: int = 1280
    layers: int = 32
    heads: int = 16
    patch: int = 14
    image: int = 448
    merger_out: int = 3584       # merger 之后每 token 的维度

    @property
    def grid(self) -> int:
        return self.image // self.patch          # 32

    @property
    def tokens_before_merger(self) -> int:
        return self.grid * self.grid             # 1024

    @property
    def tokens_after_merger(self) -> int:
        return self.tokens_before_merger // 4    # 256（2×2 合并）


@dataclass
class DitCfg:
    """FLUX.1-Schnell 的 MMDiT 生成头"""

    hidden: int = 3072
    double_blocks: int = 19
    single_blocks: int = 38
    cond_dim: int = 4096         # T5-XXL 的维度，也是 GenHead 的输出维度
    latent_ch: int = 16
    latent_size: int = 64        # 512 / 8
    cond_len: int = 77           # 文本条件序列长度
    rope_theta: float = 10000.0

    @property
    def image_len(self) -> int:
        return self.latent_size * self.latent_size


@dataclass
class NFConfig:
    """整个统一模型的配置"""

    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    vision: VisionCfg = field(default_factory=VisionCfg)
    dit: DitCfg = field(default_factory=DitCfg)

    # ---- 融合点（本设计的核心）----
    connector_layers: int = 2
    connector_dropout: float = 0.0
    gen_seq_len: int = 77        # 主干隐状态广播成多少 token 给 DiT

    # ---- 训练 ----
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    def summary(self) -> str:
        b, v, d = self.backbone, self.vision, self.dit
        return (
            f"NF-1 配置\n"
            f"  主干 : hidden={b.hidden} layers={b.layers} heads={b.heads} "
            f"kv={b.kv_heads} head_dim={b.head_dim} vocab={b.vocab}\n"
            f"  视觉 : hidden={v.hidden} layers={v.layers} patch={v.patch} "
            f"image={v.image} → tokens {v.tokens_before_merger}→{v.tokens_after_merger} "
            f"dim {v.merger_out}→{b.hidden}\n"
            f"  生成 : hidden={d.hidden} double={d.double_blocks} single={d.single_blocks} "
            f"cond {b.hidden}→{d.cond_dim} latent={d.latent_ch}×{d.latent_size}×{d.latent_size}"
        )


# 小尺寸配置：本机 CPU 跑形状测试用（结构完全一样，只是缩小）
def tiny() -> NFConfig:
    c = NFConfig()
    c.backbone = BackboneCfg(hidden=128, layers=2, heads=4, kv_heads=2, head_dim=32, vocab=1000)
    c.vision = VisionCfg(hidden=64, layers=2, heads=4, patch=14, image=112, merger_out=192)
    c.dit = DitCfg(hidden=96, double_blocks=1, single_blocks=2, cond_dim=256,
                   latent_ch=16, latent_size=8, cond_len=16)
    c.gen_seq_len = 16
    return c


if __name__ == "__main__":
    print(NFConfig().summary())
    print()
    print(tiny().summary())
