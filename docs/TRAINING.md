# 训练方案：数据集 · 损失函数 · QLoRA 全流程

> 配套 `ARCHITECTURE.md` 阅读。这里只讲「怎么训出来」。

---

## 1. 数据集设计

### 1.1 四类数据，缺一不可

| 类型 | 格式 | 量级（S1/S2/S3） | 作用 | 来源建议 |
|---|---|---|---|---|
| **A 图文理解对** | `(image, question, answer)` | 50万 / — / 50万 | 训 Connector，让主干看懂图 | LLaVA-Instruct、ShareGPT4V、自采游戏截图 |
| **B 文生图对** | `(prompt, image)` | — / 30万 / 30万 | 训 GenHead + DiT LoRA | LAION-aesthetic、COYO、美术资产 |
| **C 图生图编辑对** | `(src_image, edit_prompt, tgt_image)` | — / — / 20万 | 训「理解后修改」 | InstructPix2Pix、自建编辑对 |
| **D 领域专项** | 见 1.2 | 5万 | 你要求的枪械/建筑/游戏资产 | 自采 + 标注 |

### 1.2 领域专项数据集（你点名的方向）

| 子集 | 内容 | 标注要求 | 建议量 |
|---|---|---|---|
| **枪械** | 各型号侧视/持握/第一人称 | 型号名、类别（步枪/手枪/霰弹）、部件（弹匣/瞄具/枪管） | 2万张 |
| **建筑** | 内外景、风格（赛博/中世纪/现代） | 风格标签、结构要素（拱顶/玻璃幕墙/塔楼） | 2万张 |
| **游戏资产** | 道具图标、场景、角色 | 资产类型、材质、用途 | 1万张 |
| **图文配对** | 上述图像的描述文本 | 中英双语，≥3 句细节描述 | 与图像 1:1 |

> **合规提示**：枪械图像数据集只用于「识别与描述」任务，不要用于生成真实武器图纸或可制造内容。训练时可在 prompt 里加安全前缀，避免模型输出可操作的危险细节。

### 1.3 数据配比（S3 联合阶段）

```
A 图文理解 : B 文生图 : C 图生图 : D 领域 = 4 : 3 : 2 : 1
```

**为什么不是 1:1:1:1**：Connector 是薄弱环节，需要更多图文对喂；生成头有 FLUX 的强先验，少量数据即可适配。

---

## 2. 损失函数设计

### 2.1 三个损失，各有各的职责

```python
# ① 语言建模损失（文本回答 / 图文问答）
L_text = CrossEntropy(logits[:, :-1], labels[:, 1:])

# ② 图像生成损失（FLUX 的 rectified flow matching）
#    训练时：latent = (1-t)*x0 + t*noise，目标速度 v = noise - x0
L_flow = MSE(DiT(latent_t, t, cond), noise - x0)

# ③ 视觉-语言对齐损失（可选，S1 阶段用）
#    让 Connector 输出的视觉特征，靠近主干里「对应文本描述」的隐状态
L_align = 1 - cosine_sim(visual_feat_pooled, text_feat_pooled)
```

### 2.2 联合损失

```python
L_total = L_text + λ_flow * L_flow + λ_align * L_align
```

| 阶段 | λ_flow | λ_align | 说明 |
|---|---|---|---|
| S1 对齐 | 0 | 0.1 | 只训 Connector，`L_text` 起主导 |
| S2 生成 | 1.0 | 0 | 只训 GenHead + DiT LoRA |
| S3 联合 | 0.5 | 0.05 | 平衡，防止生成压过理解 |

### 2.3 关键细节：L_flow 的 t 采样

FLUX 用 **logit-normal 采样**（不是均匀采样），中段 t 权重更高：

```python
def sample_t(batch, device, mu=0.0, sigma=1.0):
    t = torch.sigmoid(torch.randn(batch, device=device) * sigma + mu)
    return t  # 落在 (0,1)，集中在 0.5 附近
```

### 2.4 损失监控（怎么判断训崩了）

| 指标 | 健康范围 | 异常含义 |
|---|---|---|
| `L_text` | 1.8 ~ 2.6 | <1.2 说明过拟合；>3.5 说明 Connector 没对齐 |
| `L_flow` | 0.4 ~ 0.9 | >1.5 说明 DiT LoRA 学习率太大 |
| 生成图 CLIP 分 | >0.28 | <0.2 说明条件注入没生效 |
| 主干困惑度变化 | 波动 <5% | 波动大说明 LoRA 侵入了主干 |

---

## 3. QLoRA 完整流程

### 3.1 环境

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.0 peft==0.12.0 bitsandbytes==0.43.3
pip install diffusers==0.30.0 accelerate==0.33.0 datasets==2.20.0
```

> 版本要锁死。`transformers 5.x` 和 `diffusers 0.40` 的 API 与这套方案不兼容（本机装的是 5.15/0.40，只能做形状验证，不能训练）。

### 3.2 量化加载主干

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

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
for p in backbone.parameters():
    p.requires_grad = False
```

### 3.3 挂 LoRA（只挂 attention）

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

backbone = prepare_model_for_kbit_training(backbone)  # 开梯度检查点、转 fp32 的 layernorm

lora_cfg = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
backbone = get_peft_model(backbone, lora_cfg)
backbone.print_trainable_parameters()
# 预期：trainable 40M / total 20B ≈ 0.2%
```

### 3.4 组装统一模型

```python
from src.unified_model import UnifiedModel

model = UnifiedModel(
    backbone=backbone,          # 已量化 + LoRA
    vit=vit,                    # 冻结
    connector=Connector(3584, 2880),      # 可训
    gen_adapter=GenHeadAdapter(2880, 4096),  # 可训
    dit=dit,                    # 4bit + LoRA
    vae=vae,                    # 冻结
)
```

### 3.5 优化器与超参

| 参数 | S1 对齐 | S2 生成 | S3 联合 |
|---|---|---|---|
| 优化器 | AdamW 8bit | AdamW 8bit | AdamW 8bit |
| Connector lr | 1e-4 | — | 5e-5 |
| GenHead lr | — | 2e-4 | 1e-4 |
| LoRA lr | 1e-4 | 2e-4 | 1e-4 |
| batch size | 8 | 4 | 4 |
| grad accum | 4 | 8 | 8 |
| 有效 batch | 32 | 32 | 32 |
| warmup | 3% | 5% | 3% |
| scheduler | cosine | cosine | cosine |
| epochs | 1 | 2 | 1 |
| bf16 | ✅ | ✅ | ✅ |
| grad checkpoint | ✅ | ✅ | ✅ |

### 3.6 显存账单（关键，决定你能不能租到卡）

| 阶段 | 权重(4bit) | 激活 | 优化器 | 总计 | 最小卡 |
|---|---|---|---|---|---|
| S1 | 12.5 GB | ~6 GB | ~0.3 GB | **~19 GB** | RTX 4090 24G ✅ |
| S2 | 18 GB | ~10 GB | ~0.5 GB | **~29 GB** | A100 40G ✅ |
| S3 | 18.5 GB | ~16 GB | ~0.6 GB | **~35 GB** | A100 40G / 80G |

> **结论**：S1 你自己租一张 4090（约 ¥2/小时）就能跑通，先验证接线对不对。S2/S3 必须 A100。

### 3.7 训练循环骨架（伪代码）

```python
for step, batch in enumerate(loader):
    # 文本/理解分支
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.forward_text(batch)          # 视觉 token 拼进主干
        loss_text = F.cross_entropy(out.logits, batch.labels)

    # 生成分支（S2/S3）
    if stage in ("S2", "S3"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cond = model.encode_condition(batch.prompt)   # 主干隐状态 → 4096
            loss_flow = flow_matching_loss(model.dit, batch.image, cond)

    loss = loss_text + lam * loss_flow
    accelerator.backward(loss)
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()
```

### 3.8 检查点策略

- 每 500 步存一次 `connector.pt` + `lora_*.pt`（**不存主干**，几十 GB 没必要）
- 只存可训练部分，最终导出时再合并（见 EXPORT.md）
- 同时存 `optimizer.pt` 以便断点续训

---

## 4. 评估（训完怎么判断成不成）

| 维度 | 指标 | 及格线 |
|---|---|---|
| 视觉理解 | 自建 500 题 VQA 准确率 | >75% |
| 文本能力保持 | MMLU 子集 / HumanEval 通过率 | 相对基线下降 <3% |
| 生成质量 | FID（512×512） | <30 |
| 条件一致性 | CLIP-Score | >0.28 |
| 图生图保真 | 与原图的 LPIPS（越低越像） | <0.25 |

**最重要的一条**：训练前后各跑一次 HumanEval，**如果代码能力掉了超过 3%，立刻回滚 LoRA 学习率**——这是「保住主干能力」的硬指标。

---

## 5. 本机能做什么、不能做什么

| 任务 | 本机（32GB, 无 GPU） | 说明 |
|---|---|---|
| 代码形状验证 | ✅ | `shape_test.py`，随机张量跑前向 |
| 数据预处理/清洗 | ✅ | |
| 小模型冒烟训练 | ⚠️ 勉强 | 用 tiny 配置（hidden 256、2 层）验证流程 |
| S1 对齐训练 | ❌ | 需 19GB 显存 |
| S2/S3 联合训练 | ❌ | 需 35GB 显存 |
| 推理（4bit） | ⚠️ 慢 | 主干能跑，DiT 在 CPU 上极慢 |
