# 权重合并与导出：怎么把三块拼成一个文件

> 训练完成后，把「主干 + Connector + GenHead + LoRA」合并成一个可分发的一体化模型。
> 核心原则：**训练时分开存（只有 130M 可训参数），分发时合并成一个包**。

---

## 1. 导出总览

```
训练产出（小）                        导出产物（大）
├── connector.pt        18.6M  ─┐
├── gen_adapter.pt      11.8M   ├─► 合并 ─► nf1-unified.safetensors (~11.5GB)
├── backbone_lora.pt    40M    │         （主干4bit权重 + 所有适配层）
├── dit_lora.pt         60M   ─┘
└── （主干原权重不改）              ─► + dit_int4.safetensors (~6.5GB)
```

**关键决策：模型权重固化进 HuggingFace 目录结构，不塞进 exe。**
理由：11.5GB + 6.5GB，任何 exe 打包方案都会让启动解包超过十分钟。

---

## 2. 第一步：LoRA 合并回主干

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 注意：要把 4bit 主干反量化成 bf16 才能合并（内存要 40GB+，在云上做）
backbone = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    torch_dtype=torch.bfloat16,
    device_map="cpu",     # 合并不需要 GPU
)
peft_model = PeftModel.from_pretrained(backbone, "ckpt/backbone_lora")
merged = peft_model.merge_and_unload()          # ★ LoRA 数学合并
merged.save_pretrained("out/nf1-backbone", safe_serialization=True)
```

> **合并的数学**：`W' = W + (B @ A) * (alpha / r)`。这是精确等价，不是近似。

**显存/内存要求**：bf16 主干 = 40GB。本机 32GB 不够，必须在 64GB+ 的机器上做（或者用 `device_map="auto"` + CPU offload 慢速合并）。

---

## 3. 第二步：合并后的主干重新量化成 4bit

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained("out/nf1-backbone", quantization_config=bnb)
model.save_pretrained("out/nf1-backbone-4bit")
```

> 重新量化后约 11.5GB。这一步是为了让推理端能直接加载。

---

## 4. 第三步：把适配层焊进主干

**两种做法，选一种：**

### 方案 A：外挂（简单，推荐先用）

保持三个 `.safetensors` 分开，用一个 `manifest.json` 声明加载顺序：

```json
{
  "name": "NF-1",
  "version": "1.0.0",
  "parts": [
    {"role": "backbone", "path": "nf1-backbone-4bit", "dtype": "nf4", "frozen": true},
    {"role": "connector", "path": "connector.pt", "dtype": "bf16", "trainable": true},
    {"role": "gen_adapter", "path": "gen_adapter.pt", "dtype": "bf16", "trainable": true},
    {"role": "dit", "path": "dit-int4.safetensors", "dtype": "nf4", "frozen": true},
    {"role": "vae", "path": "vae.safetensors", "dtype": "fp16", "frozen": true}
  ],
  "fusion": {
    "vision_to_backbone": {"in": 3584, "out": 2880, "module": "connector"},
    "backbone_to_dit": {"in": 2880, "out": 4096, "module": "gen_adapter"}
  }
}
```

### 方案 B：物理焊死（真·单文件）

```python
import torch, safetensors.torch as st

# 1. 读主干分片
backbone_sd = {}
for f in Path("out/nf1-backbone-4bit").glob("*.safetensors"):
    backbone_sd.update(st.load_file(str(f)))

# 2. 读适配层并加前缀
backbone_sd["fusion.connector.weight"] = torch.load("ckpt/connector.pt")["weight"]
backbone_sd["fusion.connector.bias"]   = torch.load("ckpt/connector.pt")["bias"]
backbone_sd["fusion.gen_adapter.weight"] = torch.load("ckpt/gen_adapter.pt")["weight"]

# 3. 写成一个文件
st.save_file(backbone_sd, "out/nf1-unified.safetensors", metadata={"format": "nf1"})
```

> **注意**：4bit 量化权重是 bitsandbytes 的特殊格式，**不能直接和 bf16 的 adapter 拼在同一个 safetensors 里**（dtype 冲突）。
> 所以「真·单文件」只能用方案 A 的 bf16 版本（40GB），或者接受「一个目录 + manifest」。
> **本设计推荐方案 A**：对外仍然是一个模型目录、一个加载入口，用户无感。

---

## 5. 第四步：DiT 导出（GGUF / safetensors）

FLUX DiT 要上 CPU 推理（你的 32GB 场景），必须量化：

```bash
# 用 ComfyUI 的 GGUF 工具，或者 diffusers 的 optimum-quanto
python -m optimum.quanto.quantize \
  --model black-forest-labs/FLUX.1-schnell \
  --weights qint4 \
  --output out/flux-schnell-q4
```

> 12B DiT 4bit ≈ 6.5GB。CPU 上 512×512、4 步，约 3~8 分钟/张（比原来 UNet 的 76 秒慢，但这是 DiT 的代价）。

---

## 6. 最终分发结构

```
NF-1/
├── manifest.json                 # 加载声明（上面的方案 A）
├── nf1-backbone-4bit/            # 11.5GB
├── connector.pt                  # 37MB
├── gen_adapter.pt                # 24MB
├── dit-int4.safetensors          # 6.5GB
├── vae.safetensors               # 330MB
└── tokenizer/                    # 主干分词器
```

**总大小 ≈ 18.4GB**，符合「≤24GB 运行内存」目标（加载后常驻约 18GB）。

---

## 7. 导出校验清单

| 检查项 | 方法 | 通过标准 |
|---|---|---|
| 合并后文本能力 | 跑 50 条固定 prompt | 输出与合并前一致（±2% 困惑度） |
| Connector 接线 | 同一张图喂两次 | 视觉特征 cosine >0.99 |
| 生成头条件 | 不同 prompt 生成 | CLIP-score 差异明显（说明条件生效） |
| 文件完整性 | sha256 校验 | 与导出时记录一致 |
| 加载顺序 | 按 manifest 冷启动 | 峰值内存 ≤ 20GB |

---

## 8. 一句话总结

**训练存 130M，分发存 18.4GB。**
LoRA 数学合并、4bit 重量化、适配层加前缀、manifest 声明——四步，就能把「三个模型」对外表现为「一个模型目录」。
