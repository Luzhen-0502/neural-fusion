# -*- coding: utf-8 -*-
"""
nf_downloader.py —— NF-1 底座模型下载器（给 exe 用的一键下载模块）

设计要点：
  1. 纯 curl 驱动，不依赖 huggingface_hub / torch，exe 打出来才小
  2. 多镜像自动切换：一个源失败自动换下一个
  3. 断点续传：curl -C -，断了重开接着下
  4. 实时进度：文件级 + 字节级，后台线程 + 队列回主线程

⚠️ 这里用的是**合法镜像源**（HF 官方国内镜像 hf-mirror.com、GitHub 加速 CDN），
   不是 VPN / 翻墙工具。镜像只是「同一份文件的另一个加速下载地址」。
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 镜像源（合法 CDN，按顺序尝试）
# ---------------------------------------------------------------------------
HF_MIRRORS = [
    "https://hf-mirror.com",           # HF 官方推荐的国内镜像
    "https://huggingface.co",          # 直连（海外/有合规网络时更快）
]

GH_MIRRORS = [
    "",                                 # 直连
    "https://gh-proxy.com/",           # GitHub 文件加速
    "https://ghproxy.net/",            # 备用加速
    "https://ghproxy.cc/",             # 备用加速
]


@dataclass
class ModelSpec:
    key: str
    title: str
    repo: str
    note: str
    approx_gb: float
    files: list[str] = field(default_factory=list)


MODELS = [
    ModelSpec("llm", "GPT-OSS-20B（语言主干）", "openai/gpt-oss-20b",
              "整个系统的中枢，代码/推理/规划都靠它", 40.0),
    ModelSpec("vit", "Qwen2.5-VL-7B（视觉塔）", "Qwen/Qwen2.5-VL-7B-Instruct",
              "看图、识别物体/场景，只取它的 ViT", 16.0),
    ModelSpec("dit", "FLUX.1-Schnell（DiT 生成头）", "black-forest-labs/FLUX.1-schnell",
              "文生图/图生图，Transformer 架构的扩散模型", 24.0),
]

# 只下权重和配置，跳过演示图/README 之类的杂物
SKIP_SUFFIX = (".md", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webp", ".svg")
SKIP_NAMES = {".gitattributes", "README.md", "LICENSE", "LICENSE.md", "not_here.txt"}


class Downloader:
    """后台下载器：多镜像 + 断点续传 + 进度回调。"""

    def __init__(self, emit) -> None:
        self.emit = emit                     # emit(kind, payload)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.total_bytes = 0
        self.done_bytes = 0
        self.current = ""

    # -- 对外 ------------------------------------------------------------
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self._stop.set()

    def start(self, model_keys: list[str], out_dir: Path) -> None:
        if self.running():
            return
        self._stop.clear()
        self.total_bytes = 0
        self.done_bytes = 0
        self._thread = threading.Thread(
            target=self._run, args=(model_keys, out_dir), daemon=True, name="nf-dl")
        self._thread.start()

    # -- 内部 ------------------------------------------------------------
    def _emit(self, kind: str, payload=None) -> None:
        try:
            self.emit(kind, payload)
        except Exception:
            pass

    def _probe(self, url: str, timeout: int = 20) -> bool:
        try:
            req = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": "NF1-Inspector"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return 200 <= r.status < 400
        except Exception:
            return False

    def _list_files(self, repo: str) -> tuple[list[str], str]:
        """列出仓库文件（返回 文件列表 + 命中的镜像）。"""
        path = f"/api/models/{repo}/tree/main?recursive=true"
        for mirror in HF_MIRRORS:
            url = mirror + path
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "NF1-Inspector"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read())
                files = []
                for item in data:
                    if item.get("type") != "file":
                        continue
                    p = item.get("path", "")
                    name = Path(p).name
                    if not p or name in SKIP_NAMES:
                        continue
                    if name.lower().endswith(SKIP_SUFFIX):
                        continue
                    files.append(p)
                if files:
                    return files, mirror
            except Exception:
                continue
        raise RuntimeError(f"列不出 {repo} 的文件（所有镜像都失败）")

    def _run(self, model_keys: list[str], out_dir: Path) -> None:
        try:
            chosen = [m for m in MODELS if m.key in model_keys]
            if not chosen:
                self._emit("dl_error", "没选任何模型")
                return

            # 1) 先列文件、算总量
            self._emit("dl_status", "正在获取文件清单……")
            plan: list[tuple[ModelSpec, list[str], str]] = []
            for m in chosen:
                if self._stop.is_set():
                    return
                self._emit("dl_status", f"列出 {m.title} 的文件……")
                try:
                    files, mirror = self._list_files(m.repo)
                except Exception as e:  # noqa: BLE001
                    self._emit("dl_log", f"✗ {m.title}：{e}")
                    continue
                plan.append((m, files, mirror))
                self._emit("dl_log", f"✓ {m.title}：{len(files)} 个文件（镜像 {mirror}）")

            total_files = sum(len(f) for _, f, _ in plan)
            idx = 0
            for m, files, mirror in plan:
                target = out_dir / m.repo.split("/")[-1]
                target.mkdir(parents=True, exist_ok=True)
                for rel in files:
                    if self._stop.is_set():
                        self._emit("dl_status", "已停止")
                        return
                    idx += 1
                    dst = target / rel
                    if dst.exists() and dst.stat().st_size > 0:
                        self._emit("dl_progress", (idx, total_files, rel, dst.stat().st_size))
                        continue
                    self.current = rel
                    self._download_file(m, rel, dst, mirror, idx, total_files)

            if self._stop.is_set():
                self._emit("dl_status", "已停止")
                return
            self._emit("dl_status", "全部下载完成 ✓")
            self._emit("dl_done", str(out_dir))
        except Exception as e:  # noqa: BLE001
            self._emit("dl_error", f"{type(e).__name__}: {e}")

    def _download_file(self, model: ModelSpec, rel: str, dst: Path,
                       preferred: str, idx: int, total: int) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        # 构建候选 URL：首选镜像 → 其余镜像
        mirrors = [preferred] + [m for m in HF_MIRRORS if m != preferred]
        for mirror in mirrors:
            if self._stop.is_set():
                return
            url = f"{mirror}/{model.repo}/resolve/main/{rel}"
            self._emit("dl_status", f"[{idx}/{total}] {rel}")
            ok = self._curl(url, dst, idx, total, rel)
            if ok:
                return
        self._emit("dl_log", f"✗ 放弃：{rel}（所有镜像都失败）")

    def _curl(self, url: str, dst: Path, idx: int, total: int, rel: str) -> bool:
        cmd = [
            "curl.exe", "-L", "-C", "-", "--retry", "5", "--retry-delay", "2",
            "--retry-all-errors", "--connect-timeout", "20",
            "--speed-time", "60", "--speed-limit", "1024",   # 60秒低于1KB/s就重连
            "-sS", "-o", str(dst), url,
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception as e:  # noqa: BLE001
            self._emit("dl_log", f"  启动 curl 失败：{e}")
            return False

        while proc.poll() is None:
            if self._stop.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                return False
            size = dst.stat().st_size if dst.exists() else 0
            self._emit("dl_progress", (idx, total, rel, size))
            time.sleep(0.4)

        rc = proc.returncode
        if rc == 0 and dst.exists() and dst.stat().st_size > 0:
            return True
        self._emit("dl_log", f"  curl 返回 {rc}，换镜像重试：{rel}")
        return False


def find_curl() -> str | None:
    return shutil.which("curl.exe") or shutil.which("curl")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"
