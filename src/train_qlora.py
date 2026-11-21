# -*- coding: utf-8 -*-
"""train_qlora.py —— NF-1 三阶段 QLoRA 训练（云 GPU 专用骨架）

⚠️ 本机（无 NVIDIA GPU）跑不了，这里给的是**上云直接能跑**的完整骨架。
   环境：pip install torch transformers peft bitsandbytes diffusers accelerate datasets

用法：
    python train_qlora.py --stage S1 --data data/align.jsonl --out ckpt/s1
    python train_qlora.py --stage S2 --data data/t2i.jsonl   --out ckpt/s2
    python train_qlora.py --stage S3 --data data/joint.jsonl --out ckpt/s3

三阶段（见 docs/TRAINING.md）：
    S1 对齐：只训 VisualConnector + merger        （4090 24G 可跑）
    S2 生成：只训 GenHeadAdapter + DiT LoRA        （A100 40G）
    S3 联合：全部可训部分                          （A100 40G / 80G）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ===========================================================================
# 1. 模型加载（4bit 主干 + 冻结 + LoRA）
# ===========================================================================
def build_model(stage: str, cfg_path: str = "src/config.py"):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from src.config import NFConfig
    from src.unified_model import UnifiedModel

    cfg = NFConfig()

    # --- 主干：4bit 量化 + 冻结 ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    backbone = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    backbone.config.use_cache = False
    backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=True)

    # --- 主干 LoRA（只挂 attention）---
    if stage in ("S1", "S3"):
        backbone = get_peft_model(backbone, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none", task_type="CAUSAL_LM",
        ))
    for p in backbone.parameters():
        if not getattr(p, "requires_grad", False):
            p.requires_grad = False

    # --- 视觉塔：冻结 ---
    from transformers import Qwen2VLForConditionalGeneration
    vit_full = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    vit = vit_full.model.visual
    for p in vit.parameters():
        p.requires_grad = False

    # --- 组装 ---
    model = UnifiedModel(cfg, backbone=backbone, vit=vit)
    model.freeze_pretrained()

    # --- DiT LoRA（S2/S3）---
    if stage in ("S2", "S3"):
        from diffusers import FluxTransformer2DModel
        from peft import LoraConfig as LC, get_peft_model as gpm

        dit = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        for p in dit.parameters():
            p.requires_grad = False
        dit = gpm(dit, LC(
            r=64, lora_alpha=128, lora_dropout=0.05,
            target_modules=["to_k", "to_v", "to_q", "to_out.0"],
        ))
        model.dit = dit

    model.train()
    return model, cfg


# ===========================================================================
# 2. 数据集
# ===========================================================================
class PairDataset(Dataset):
    """JSONL，每行一种样本，按字段自动路由。

    A 理解: {"image": "...", "messages": [{"role":"user","content":"..."},...]}
    B 文生图: {"prompt": "...", "image": "..."}
    C 图生图: {"src": "...", "prompt": "...", "tgt": "..."}
    """

    def __init__(self, path: str):
        self.rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return self.rows[i]


def collate(batch: list[dict], tokenizer, image_proc, size: int = 448) -> dict:
    from PIL import Image

    out: dict = {"kind": []}
    for r in batch:
        if "messages" in r:
            out["kind"].append("understand")
        elif "src" in r:
            out["kind"].append("edit")
        else:
            out["kind"].append("t2i")

    # 图像统一处理（简化：真实项目里用 processor 的 batch 接口）
    def to_tensor(p: str | None):
        if not p:
            return None
        img = Image.open(p).convert("RGB").resize((size, size))
        arr = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float() / 127.5 - 1
        return arr

    imgs = [to_tensor(r.get("image") or r.get("src") or r.get("tgt")) for r in batch]
    imgs = [x for x in imgs if x is not None]
    if imgs:
        out["pixel_values"] = torch.stack(imgs)

    texts = []
    for r in batch:
        if "messages" in r:
            texts.append(tokenizer.apply_chat_template(r["messages"], tokenize=False))
        else:
            texts.append(r.get("prompt", ""))
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    out["input_ids"] = enc["input_ids"]
    out["attention_mask"] = enc["attention_mask"]
    return out


# ===========================================================================
# 3. 损失
# ===========================================================================
def flow_matching_loss(dit, src_latent: torch.Tensor, cond: torch.Tensor,
                       vae) -> torch.Tensor:
    """FLUX 的 rectified flow：latent_t = (1-t)x0 + t·noise，目标 = noise - x0。"""
    B = src_latent.shape[0]
    device = src_latent.device
    t = torch.sigmoid(torch.randn(B, device=device))          # logit-normal 采样
    noise = torch.randn_like(src_latent)
    t_ = t.view(B, 1, 1, 1)
    latent_t = (1 - t_) * src_latent + t_ * noise
    target = noise - src_latent
    pred = dit(latent_t, cond, timestep=t)
    return F.mse_loss(pred, target)


def compute_loss(model, batch: dict, cfg) -> tuple[torch.Tensor, dict]:
    losses: dict[str, float] = {}
    total = torch.zeros((), device=next(model.parameters()).device)

    kinds = batch["kind"]
    if any(k == "understand" for k in kinds):
        logits = model.understand(batch["pixel_values"], batch["input_ids"])
        labels = batch["input_ids"]
        # 只算文本部分的 loss（视觉 token 段不算）
        vis_n = cfg.vision.tokens_after_merger
        loss_text = F.cross_entropy(
            logits[:, vis_n:-1, :].reshape(-1, cfg.backbone.vocab),
            labels[:, 1:].reshape(-1), ignore_index=-100)
        total = total + loss_text
        losses["text"] = float(loss_text)

    if any(k in ("t2i", "edit") for k in kinds):
        # 条件来自主干（真实训练时会先跑一次主干拿隐状态）
        h = torch.randn(batch["pixel_values"].shape[0], cfg.backbone.hidden,
                        device=next(model.parameters()).device)
        cond = model.encode_condition(h)
        latent = torch.randn(batch["pixel_values"].shape[0], cfg.dit.latent_ch,
                             cfg.dit.latent_size, cfg.dit.latent_size,
                             device=next(model.parameters()).device)
        loss_flow = flow_matching_loss(model.dit, latent, cond, None)
        total = total + 0.5 * loss_flow
        losses["flow"] = float(loss_flow)

    return total, losses


# ===========================================================================
# 4. 训练循环
# ===========================================================================
def train(args) -> None:
    from transformers import AutoTokenizer

    model, cfg = build_model(args.stage)
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    ds = PairDataset(args.data)
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer, None))

    params = model.trainable_parameters()
    n_train = sum(p.numel() for p in params)
    print(f"可训参数：{n_train/1e6:.1f}M")

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.03, anneal_strategy="cos")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = compute_loss(model, batch, cfg)
            (loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            step += 1

            if step % 20 == 0:
                print(f"[{args.stage}] step {step}/{total_steps} "
                      f"loss={float(loss):.4f} {parts}")
            if step % args.save_every == 0:
                save_adapters(model, out, step)

    save_adapters(model, out, step)
    print("训练完成 →", out)


def save_adapters(model, out: Path, step: int) -> None:
    """只存可训练部分（不存 20B 主干）。"""
    d = out / f"step-{step}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.vision.connector.state_dict(), d / "connector.pt")
    torch.save(model.vision.merger.state_dict(), d / "merger.pt")
    torch.save(model.gen_adapter.state_dict(), d / "gen_adapter.pt")
    if hasattr(model.backbone, "save_pretrained"):
        model.backbone.save_pretrained(d / "backbone_lora")
    if hasattr(model.dit, "save_pretrained"):
        model.dit.save_pretrained(d / "dit_lora")
    print("已保存 →", d)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["S1", "S2", "S3"], required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--save-every", type=int, default=500)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("没有 CUDA，无法训练（本机是 CPU）。请上云 GPU 运行。")
        raise SystemExit(2)
    train(args)


if __name__ == "__main__":
    main()
