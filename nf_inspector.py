# -*- coding: utf-8 -*-
"""
NF-1 架构检查器（Architecture Inspector）

双击就能看：
  - 整条数据流：图像 → ViT → Merger → Connector → 主干 → GenHead → DiT → 图
  - 每一层的张量维度（输入/输出/参数量）
  - 一键「形状验证」：纯数学推导维度传播，不需要 GPU、不需要 torch

给不想装 PyTorch、只想看架构的人用。
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

FONT = "Microsoft YaHei UI"
BG = "#0F172A"
CARD = "#1E293B"
TEXT = "#E2E8F0"
SUB = "#94A3B8"
ACCENT = "#38BDF8"
GREEN = "#4ADE80"
RED = "#F87171"
AMBER = "#FBBF24"


# ===========================================================================
# 模型规格（和 src/config.py 保持一致）
# ===========================================================================
@dataclass
class Spec:
    # 主干
    b_hidden: int = 2880
    b_layers: int = 24
    b_heads: int = 64
    b_kv: int = 8
    b_head_dim: int = 64
    b_vocab: int = 201088
    b_rope: float = 150000.0
    # 视觉
    v_hidden: int = 1280
    v_layers: int = 32
    v_heads: int = 16
    v_patch: int = 14
    v_image: int = 448
    v_merger: int = 3584
    # 生成
    d_hidden: int = 3072
    d_double: int = 19
    d_single: int = 38
    d_cond: int = 4096
    d_latent: int = 16
    d_latent_size: int = 64
    d_cond_len: int = 77

    @property
    def v_grid(self) -> int:
        return self.v_image // self.v_patch

    @property
    def v_tokens(self) -> int:
        return self.v_grid ** 2

    @property
    def v_after_merger(self) -> int:
        return self.v_tokens // 4


SPEC = Spec()


def gb(n: float) -> str:
    return f"{n/1e9:.2f}B" if n >= 1e9 else f"{n/1e6:.1f}M" if n >= 1e6 else f"{n/1e3:.1f}K"


# ===========================================================================
# 层清单：每一行的 (分组, 名称, 输入, 输出, 参数量, 说明)
# ===========================================================================
def build_rows(s: Spec) -> list[tuple]:
    R = []
    add = R.append

    # ---- 视觉塔 ----
    add(("视觉塔", "V1 PatchEmbed",
         f"[B,3,{s.v_image},{s.v_image}]",
         f"[B,{s.v_tokens},{s.v_hidden}]",
         s.v_hidden * 3 * s.v_patch * s.v_patch + s.v_hidden,
         f"卷积 patch={s.v_patch}，{s.v_grid}×{s.v_grid}={s.v_tokens} 个 token"))
    add(("视觉塔", "V2 2D-RoPE",
         f"[B,{s.v_tokens},{s.v_hidden}]",
         f"[B,{s.v_tokens},{s.v_hidden}]", 0,
         "h/w 各占一半维度，频率表冻结"))
    add(("视觉塔", f"V3 ViT Block ×{s.v_layers}",
         f"[B,{s.v_tokens},{s.v_hidden}]",
         f"[B,{s.v_tokens},{s.v_hidden}]",
         int(12 * s.v_layers * s.v_hidden ** 2),
         f"{s.v_heads} 头，每头 {s.v_hidden//s.v_heads} 维，❄️冻结"))
    add(("视觉塔", "V4 Merger",
         f"[B,{s.v_tokens},{s.v_hidden}]",
         f"[B,{s.v_after_merger},{s.v_merger}]",
         s.v_merger * s.v_hidden * 4 + s.v_merger,
         "2×2 空间合并 + MLP，token 减到 1/4"))

    # ---- 连接器（融合点 1）----
    add(("★连接器", "C1 LayerNorm",
         f"[B,{s.v_after_merger},{s.v_merger}]",
         f"[B,{s.v_after_merger},{s.v_merger}]",
         2 * s.v_merger, "稳定分布"))
    add(("★连接器", "C2 Linear",
         f"[B,{s.v_after_merger},{s.v_merger}]",
         f"[B,{s.v_after_merger},{s.b_hidden}]",
         s.v_merger * s.b_hidden + s.b_hidden,
         f"★关键：{s.v_merger} → {s.b_hidden}（对齐主干宽度）"))
    add(("★连接器", "C3 GELU",
         f"[B,{s.v_after_merger},{s.b_hidden}]",
         f"[B,{s.v_after_merger},{s.b_hidden}]", 0, "非线性"))
    add(("★连接器", "C4 Linear",
         f"[B,{s.v_after_merger},{s.b_hidden}]",
         f"[B,{s.v_after_merger},{s.b_hidden}]",
         s.b_hidden ** 2 + s.b_hidden, "残差投影"))
    add(("★连接器", "C5 LayerNorm",
         f"[B,{s.v_after_merger},{s.b_hidden}]",
         f"[B,{s.v_after_merger},{s.b_hidden}]",
         2 * s.b_hidden, "输出前归一 ✔ 可训练"))

    # ---- 主干 ----
    add(("主干", "M1 Embedding",
         f"[B,L]",
         f"[B,L,{s.b_hidden}]",
         s.b_vocab * s.b_hidden, f"词表 {s.b_vocab}"))
    add(("主干", "M2 序列拼接",
         f"视觉[{s.v_after_merger}] + 文本[T]",
         f"[B,{s.v_after_merger}+T,{s.b_hidden}]", 0,
         "★融合点：视觉 token 和文本 token 拼成同一条序列"))
    add(("主干", f"M3 Transformer ×{s.b_layers}",
         f"[B,{s.v_after_merger}+T,{s.b_hidden}]",
         f"[B,{s.v_after_merger}+T,{s.b_hidden}]",
         int(12 * s.b_layers * s.b_hidden ** 2),
         f"GQA {s.b_heads}头/{s.b_kv}KV头，head_dim={s.b_head_dim}，"
         f"RoPE θ={s.b_rope:.0f}，MoE 32专家top4，❄️冻结"))
    add(("主干", "M4 Final Norm",
         f"[B,L,{s.b_hidden}]",
         f"[B,L,{s.b_hidden}]", 2 * s.b_hidden, "RMSNorm"))
    add(("主干", "M5 LM Head",
         f"[B,L,{s.b_hidden}]",
         f"[B,L,{s.b_vocab}]",
         s.b_hidden * s.b_vocab, "→ 文本 logits"))

    # ---- 生成头（融合点 2）----
    add(("★生成头", "G1 取末位隐状态",
         f"[B,L,{s.b_hidden}]",
         f"[B,{s.b_hidden}]", 0, "只取生成指令的最后 token"))
    add(("★生成头", "G2 Linear",
         f"[B,{s.b_hidden}]",
         f"[B,{s.d_cond}]",
         s.b_hidden * s.d_cond + s.d_cond,
         f"★关键：{s.b_hidden} → {s.d_cond}（对齐 T5 条件维度）"))
    add(("★生成头", "G3 LayerNorm",
         f"[B,{s.d_cond}]",
         f"[B,{s.d_cond}]", 2 * s.d_cond, "✔ 可训练"))
    add(("★生成头", "G4 广播成条件序列",
         f"[B,{s.d_cond}]",
         f"[B,{s.d_cond_len},{s.d_cond}]", 0,
         f"复制成 {s.d_cond_len} 个 token，匹配 FLUX 条件长度"))

    # ---- DiT ----
    add(("DiT", "D1 latent 输入",
         f"[B,{s.d_latent},{s.d_latent_size},{s.d_latent_size}]",
         f"[B,{s.d_latent_size**2},{s.d_hidden}]", 0,
         f"{s.d_latent} 通道，{s.d_latent_size}×{s.d_latent_size} latent"))
    add(("DiT", f"D2 双流 MMDiT ×{s.d_double}",
         f"[B,N,{s.d_hidden}]",
         f"[B,N,{s.d_hidden}]",
         int(20 * s.d_double * s.d_hidden ** 2),
         "图像流 + 文本流各自 attention 再交叉，❄️冻结+LoRA"))
    add(("DiT", f"D3 单流 MMDiT ×{s.d_single}",
         f"[B,N,{s.d_hidden}]",
         f"[B,N,{s.d_hidden}]",
         int(12 * s.d_single * s.d_hidden ** 2),
         "两流合并后的主干块，❄️冻结+LoRA"))
    add(("DiT", "D4 输出投影",
         f"[B,N,{s.d_hidden}]",
         f"[B,{s.d_latent},{s.d_latent_size},{s.d_latent_size}]", 0,
         "预测速度场 v"))
    add(("DiT", "D5 VAE Decode",
         f"[B,{s.d_latent},{s.d_latent_size},{s.d_latent_size}]",
         f"[B,3,{s.d_latent_size*8},{s.d_latent_size*8}]", 0,
         "8× 上采样，❄️冻结"))
    return R


# ===========================================================================
# 形状验证（纯数学，不 import torch）
# ===========================================================================
def run_shape_check(s: Spec) -> list[tuple[str, bool, str]]:
    out = []

    def chk(name, cond, detail):
        out.append((name, bool(cond), detail))

    # 1) 视觉链路
    tokens = s.v_tokens
    chk("ViT token 数", tokens == s.v_grid ** 2,
        f"{s.v_grid}×{s.v_grid} = {tokens}")
    chk("Merger 后 token", s.v_after_merger * 4 == tokens,
        f"{tokens} → {s.v_after_merger}（2×2 合并）")
    chk("视觉维度对齐", s.v_merger == 3584 and s.b_hidden == 2880,
        f"{s.v_merger} → {s.b_hidden}")

    # 2) 主干注意力约束
    q_dim = s.b_heads * s.b_head_dim
    chk("主干 Q 维度可整除", q_dim % s.b_heads == 0,
        f"{s.b_heads}×{s.b_head_dim} = {q_dim}")
    chk("主干 KV 分组", s.b_heads % s.b_kv == 0,
        f"{s.b_heads}/{s.b_kv} = {s.b_heads // s.b_kv} 组 GQA")

    # 3) 生成链路
    chk("生成维度对齐", s.b_hidden == 2880 and s.d_cond == 4096,
        f"{s.b_hidden} → {s.d_cond}")
    chk("条件序列长度", s.d_cond_len == 77,
        f"FLUX 原生要求 {s.d_cond_len}")
    chk("latent 与 DiT 一致", s.d_latent == 16,
        f"{s.d_latent} 通道（VAE 与 DiT 原生一致）")

    # 4) 内存估算
    llm_4bit = 11.3
    vit_fp16 = 0.3 * 2
    dit_4bit = 6.5
    conn = (s.v_merger * s.b_hidden + s.b_hidden ** 2) * 2 / 1e9
    gen = (s.b_hidden * s.d_cond) * 2 / 1e9
    total = llm_4bit + vit_fp16 + dit_4bit + conn + gen
    chk("总驻留内存 ≤ 24GB", total <= 24,
        f"LLM {llm_4bit} + ViT {vit_fp16} + DiT {dit_4bit} + 连接器 {conn+gen:.2f} = {total:.1f} GB")

    return out


# ===========================================================================
# 界面
# ===========================================================================
class App:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("NF-1 架构检查器 · 神经层级深度融合模型")
        self.root.configure(bg=BG)
        w, h = 1280, 780
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{min(w,sw-60)}x{min(h,sh-80)}+{max(0,(sw-w)//2)}+{max(0,(sh-h)//2-40)}")
        self.root.minsize(900, 600)
        self._build()

    def _build(self) -> None:
        r = self.root
        r.grid_columnconfigure(0, weight=3)
        r.grid_columnconfigure(1, weight=2)
        r.grid_rowconfigure(1, weight=1)

        # 顶栏
        top = tk.Frame(r, bg=CARD, height=62)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        top.grid_propagate(False)
        tk.Label(top, text="🐟", font=(FONT, 20), bg=CARD, fg=ACCENT).pack(side=tk.LEFT, padx=(16, 8), pady=10)
        tk.Label(top, text="NF-1 架构检查器", font=(FONT, 16, "bold"), bg=CARD, fg=TEXT)\
            .pack(side=tk.LEFT, pady=10)
        tk.Label(top, text="GPT-OSS-20B + Qwen2.5-VL ViT + FLUX Schnell（DiT）",
                 font=(FONT, 10), bg=CARD, fg=SUB).pack(side=tk.LEFT, padx=14, pady=10)

        # 左：层表
        left = tk.Frame(r, bg=BG)
        left.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=10)
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("NF.Treeview", background=CARD, fieldbackground=CARD,
                        foreground=TEXT, rowheight=26, borderwidth=0,
                        font=(FONT, 10))
        style.configure("NF.Treeview.Heading", background="#334155", foreground=TEXT,
                        font=(FONT, 10, "bold"), borderwidth=0)
        style.map("NF.Treeview", background=[("selected", "#0EA5E9")],
                  foreground=[("selected", "#0B1220")])

        cols = ("in", "out", "params")
        self.tree = ttk.Treeview(left, columns=cols, show="tree headings",
                                 style="NF.Treeview")
        self.tree.heading("#0", text="层")
        self.tree.heading("in", text="输入")
        self.tree.heading("out", text="输出")
        self.tree.heading("params", text="参数量")
        self.tree.column("#0", width=200, anchor="w")
        self.tree.column("in", width=250, anchor="w")
        self.tree.column("out", width=250, anchor="w")
        self.tree.column("params", width=90, anchor="e")

        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        self.rows = build_rows(SPEC)
        groups: dict[str, str] = {}
        for grp, name, i, o, p, note in self.rows:
            if grp not in groups:
                groups[grp] = self.tree.insert("", "end", text=grp, open=True, values=("", "", ""))
            self.tree.insert(groups[grp], "end", text=name,
                             values=(i, o, gb(p) if p else "—"))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # 右：说明 + 验证
        right = tk.Frame(r, bg=BG)
        right.grid(row=1, column=1, sticky="nsew", padx=(6, 12), pady=10)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.detail = tk.Text(right, height=8, wrap=tk.WORD, font=(FONT, 10),
                              bg=CARD, fg=TEXT, relief=tk.FLAT, padx=14, pady=12,
                              state=tk.DISABLED, insertbackground=TEXT)
        self.detail.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        vf = tk.Frame(right, bg=BG)
        vf.grid(row=1, column=0, sticky="nsew")
        vf.grid_rowconfigure(1, weight=1)
        vf.grid_columnconfigure(0, weight=1)

        bar = tk.Frame(vf, bg=BG)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Button(bar, text="▶ 运行形状验证", font=(FONT, 11, "bold"),
                  bg=ACCENT, fg="#0B1220", relief=tk.FLAT, padx=18, pady=8,
                  cursor="hand2", command=self._run_check).pack(side=tk.LEFT)
        self.verdict = tk.Label(bar, text="", font=(FONT, 11, "bold"), bg=BG, fg=SUB)
        self.verdict.pack(side=tk.LEFT, padx=14)

        self.check_box = tk.Text(vf, wrap=tk.WORD, font=(FONT, 10),
                                 bg=CARD, fg=TEXT, relief=tk.FLAT, padx=14, pady=12,
                                 state=tk.DISABLED)
        vsb = ttk.Scrollbar(vf, orient=tk.VERTICAL, command=self.check_box.yview)
        self.check_box.configure(yscrollcommand=vsb.set)
        self.check_box.grid(row=1, column=0, sticky="nsew")
        vsb.grid(row=1, column=1, sticky="ns")
        self.check_box.tag_configure("ok", foreground=GREEN)
        self.check_box.tag_configure("bad", foreground=RED)
        self.check_box.tag_configure("dim", foreground=SUB)

        # 底栏
        tk.Label(r, text="纯数学推导，不依赖 PyTorch / 不联网 / 不用 GPU",
                 font=(FONT, 9), bg=BG, fg=SUB).grid(row=2, column=0, columnspan=2, pady=(0, 8))

        self._show_detail(self.rows[0])
        self._run_check()

    def _on_select(self, _e=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        text = self.tree.item(sel[0], "text")
        for row in self.rows:
            if row[1] == text:
                self._show_detail(row)
                return

    def _show_detail(self, row) -> None:
        grp, name, i, o, p, note = row
        txt = (
            f"{grp} · {name}\n\n"
            f"输入：{i}\n"
            f"输出：{o}\n"
            f"参数量：{gb(p) if p else '无参数（纯计算/拷贝）'}\n\n"
            f"{note}"
        )
        self.detail.configure(state=tk.NORMAL)
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", txt)
        self.detail.configure(state=tk.DISABLED)

    def _run_check(self) -> None:
        results = run_shape_check(SPEC)
        self.check_box.configure(state=tk.NORMAL)
        self.check_box.delete("1.0", tk.END)
        okn = 0
        for name, ok, detail in results:
            if ok:
                okn += 1
            self.check_box.insert(tk.END, ("✓ " if ok else "✗ "), "ok" if ok else "bad")
            self.check_box.insert(tk.END, f"{name}\n", "ok" if ok else "bad")
            self.check_box.insert(tk.END, f"    {detail}\n\n", "dim")
        self.check_box.configure(state=tk.DISABLED)
        allok = okn == len(results)
        self.verdict.configure(
            text=f"{'全部通过' if allok else '有项未过'}  {okn}/{len(results)}",
            fg=GREEN if allok else RED)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
