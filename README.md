# NF-1 · 神经层级深度融合统一多模态模型

> GPT-OSS-20B（语言主干） + Qwen2.5-VL-7B 的 ViT（视觉塔） + FLUX.1-Schnell（DiT 生成头）
> **融成一个网络内部的信息流**，不是三个模型轮着调度。

---

## ⚡ 想直接看东西：下这个 exe（10MB，双击即开）

```powershell
irm https://github.com/Luzhen-0502/neural-fusion/releases/latest/download/install.ps1 | iex
```

桌面会出现「NF-1 架构检查器」，双击就能看：
- 完整数据流：图像 → ViT → Merger → Connector → 主干 → GenHead → FLUX DiT → 图像
- 每一层的输入/输出张量维度、参数量
- 一键「形状验证」，9 项检查全过（纯数学，不联网、不用 GPU、不装 PyTorch）

### ⚠️ 这个 exe 里**没有模型**

| 有 | 没有 |
|---|---|
| 架构可视化 + 维度推导 | ❌ 任何模型权重 |
| 9 项形状验证 | ❌ 聊天能力 |
| 10MB，免安装 | ❌ 画图能力 |

**为什么？** 三个底座模型合计约 **80GB**（GPT-OSS-20B ≈40GB bf16、Qwen2.5-VL ≈16GB、FLUX.1-Schnell ≈24GB），而且**融合后的 NF-1 权重还没训练出来**——训练需要 A100 级 GPU，本机跑不了。

**「融合模型」不是下载来的，是训出来的。** 本仓库给的是「怎么搭 + 怎么训 + 怎么导出」的全套方案。

---

## 三类人怎么用

| 你是 | 做什么 | 链接 |
|---|---|---|
| **只想看架构长啥样** | 跑上面那条命令，下 10MB 的检查器 | [Release](https://github.com/Luzhen-0502/neural-fusion/releases/latest) |
| **想看代码/验证形状** | clone 仓库，`python shape_test.py` | 见下方 |
| **想自己真训练** | 读 `docs/TRAINING.md`，准备云 GPU，从 HuggingFace 下三个底座 | 见「底座模型从哪来」 |

### 底座模型从哪来（给了三步）

```bash
# 1) 装依赖（需要 Python 3.10+ 和 CUDA 环境）
pip install torch transformers diffusers peft bitsandbytes accelerate datasets

# 2) 下三个底座（合计约 80GB，必须自己下）
huggingface-cli download openai/gpt-oss-20b                 --local-dir models/gpt-oss-20b
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct        --local-dir models/qwen2.5-vl
huggingface-cli download black-forest-labs/FLUX.1-schnell   --local-dir models/flux-schnell

# 3) 从 S1 阶段开始训（先只训 Connector，4090 就能跑）
python src/train_qlora.py --stage S1 --data data/align.jsonl --out ckpt/s1
```

> 国内下载慢：`export HF_ENDPOINT=https://hf-mirror.com` 再执行上面的命令。

---

## 这是什么

不是「调三个模型的脚本」，而是**两处神经接线**：

```
图像 ─► ViT(冻结) ─► Merger ─► Connector ★可训 ─► 视觉 token(2880维)
                                                      │
文本 token ─► Embed(2880) ──────────────────────────┴─► 同一条序列
                                                          │
                                              ┌───────────┴───────────┐
                                              │  GPT-OSS-20B 主干      │
                                              │  24 层 MoE Transformer │
                                              │  ★权重冻结，只挂 LoRA   │
                                              └───────────┬───────────┘
                                                          │
                              ┌───────────────────────────┴─────────┐
                              │                                     │
                          LM Head                              GenHead ★可训
                          (文本输出)                          2880 → 4096
                                                                    │
                                                          FLUX MMDiT (DiT)
                                                          19双流 + 38单流
                                                                    │
                                                              latent → VAE
                                                                    │
                                                                  图像
```

**两个融合点**：
1. **输入端**：视觉 token 投影到 2880 维，和文本 token 拼进同一条序列，一起过 24 层注意力。
2. **输出端**：主干最后隐状态投影成 4096 维，直接当 DiT 的文本条件。

---

## 目录

```
neural-fusion/
├── docs/
│   ├── ARCHITECTURE.md   架构图 + 每层张量维度 + 数据流 + 维度对齐
│   ├── TRAINING.md       数据集类型 + 损失函数 + QLoRA 三阶段全流程
│   └── EXPORT.md         权重合并 + 4bit 重量化 + 单文件导出
├── src/
│   ├── config.py         全部维度常量（唯一真相源）
│   ├── rotary.py         统一 RoPE（1D 文本 / 2D 图像）
│   ├── vision_bridge.py  ViT + Merger + VisualConnector ★
│   ├── gen_bridge.py     GenHeadAdapter + DiT 条件注入 ★
│   ├── unified_model.py  组装 + 三个前向口
│   ├── train_qlora.py    三阶段 QLoRA 训练（云 GPU）
│   └── export_merge.py   合并导出
└── shape_test.py         本机 CPU 形状验证（不用真权重）
```

---

## 先验证（本机就能跑，不需要 GPU）

```bash
pip install torch
cd neural-fusion
python shape_test.py
```

输出（实测通过）：

```
【1】视觉桥        ✓ 视觉 token: (2, 16, 128)
【2】图文理解      ✓ 文本 logits: (2, 25, 1000)
【3】生成条件      ✓ 条件序列: (2, 16, 256)
【4】文生图        ✓ 生成 latent: (2, 16, 8, 8)
【5】图生图        ✓ 编辑 latent: (2, 16, 8, 8)
【6】RoPE          ✓ 1D/2D 旋转保模长
【7】冻结/可训      ✓ 主干冻结、连接器可训
全部形状验证通过 ✓
```

这个测试用 `tiny` 配置（hidden 128、2 层），**结构和大模型一模一样**，只是尺寸缩小——用来证明每一层的张量维度都对得上。

---

## 硬件账（不吹牛）

| 任务 | 本机（32GB / Intel Arc / 无 CUDA） | 需要 |
|---|---|---|
| 形状验证 | ✅ 能跑 | 无 |
| 数据清洗预处理 | ✅ 能跑 | 无 |
| **S1 对齐训练** | ❌ | RTX 4090 24G（约 ¥2/小时） |
| **S2 生成训练** | ❌ | A100 40G |
| **S3 联合训练** | ❌ | A100 40G / 80G |
| 4bit 推理 | ⚠️ 很慢 | CPU 上 DiT 一张图要几分钟 |

**可训参数只有 130M（占全模型 0.4%）**，这就是「最大限度保住主干能力」的兑现。

---

## 关键结论（先看这三条再动手）

1. **物理上三个模型不能合成一个权重文件**——层数、宽度、归一化都不同。能合的是「加载目录 + manifest」，对外表现为一个模型。
2. **RoPE 不能跨模态共享频率表**，只能统一实现 + 保持各自独立频率表；真正需要对齐的是「视觉 token 插入主干序列时的位置编号」。
3. **训练本机做不了**，必须云 GPU。但代码骨架、维度设计、损失函数、导出方案都是完整可执行的。

---

## 文档索引

- 架构（张量维度逐层列）：[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- 训练（数据集 + 损失 + QLoRA）：[docs/TRAINING.md](docs/TRAINING.md)
- 导出（合并 + 量化 + manifest）：[docs/EXPORT.md](docs/EXPORT.md)

## 许可

代码 MIT。三个底座模型各自遵循原许可证（GPT-OSS: Apache-2.0；Qwen2.5-VL: Apache-2.0；FLUX.1-Schnell: Apache-2.0）。
