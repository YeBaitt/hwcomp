# 可控扩散图像增强流水线 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 两周内构建一套端到端的、Diffusion 架构的 4K 同分辨率可控图像增强流水线，处理官方 100 张测试图并打包为 `my_work.zip`。

**Architecture:** DiffBIR 式两阶段——Stage-1 NAFNet（保真打底，PSNR 锚点）+ Stage-2 潜空间扩散细化（SD2.1 条件适配，生成主干，流水线始终执行）。可控性由 λ/n/w/α 四个旋钮实现。4K 推理走分块 + 重叠 cos² 融合（像素域），训练在 ImagePairs 真实双相机对上以 2K 主 + 3.5K 混训。

**Tech Stack:** Python 3.12（`hwcomp` 专属 conda 环境，从零安装）、PyTorch 2.2 + CUDA、`diffusers`、`transformers`、`accelerate`、`opencv-python`、`scikit-image`、`pyiqa`、`yaml`。

## Global Constraints

- 一律使用项目专属环境 `hwcomp` 执行：`/home/liaitong/miniconda3/envs/hwcomp/bin/python`、`.../bin/pip`、`.../bin/pytest`。**该环境是全新创建的（Python 3.12，无任何包），Task 1 Step 4 一次性装好全部依赖后再用；绝不污染 `relags`。**
- 遵循 `docs/代码规范.md`：导入全部置顶、函数/类必须中文 docstring（描述做什么）、嵌套 ≤3 层、`except` 非 `Exception` 时注释触发条件、禁止装饰分隔线、每模块单测 + `if __name__ == "__main__"` 示例。
- **计划中代码示例仅为示意**：示例的 import 顺序与空行若与 `docs/代码规范.md` 冲突，**以代码规范为准**——import 置顶且 stdlib→第三方→项目内；顶层定义之间单空行、禁止连续两个空行。
- 不引入无必要依赖；能用标准库/已有库解决的不新增。
- 环境 Python 3.12（≥3.11，规范无需 PEP 563）；前向引用一律用字符串注解，不加 `from __future__ import annotations`。
- 数据（`dataset/`）、权重（`checkpoints/`）、产物（`output_dir/`、`*.zip`）不入 git（已在 `.gitignore`）。
- 单张 RTX 4090 24GB；可联网下载预训练权重。
- 提交命名：`output_dir/case1.jpg`…`case100.jpg`，与测试图严格同名、无中文符号，压缩为 `my_work.zip`（内含 `output_dir/` 目录）。
- 关键时间线：P4 结束（约第 9-10 天）必须具备可提交流水线；8-27 课程汇报前产出 100 张结果。
- 旋钮校准**只允许**在官方 5 对 val 上进行，绝不在 test 上调参。

---

## 文件结构

```
hw_comp/
├── pyproject.toml                # 包元数据（src 布局，pip install -e .）
├── config.yaml                   # 全部超参
├── src/enhance/
│   ├── __init__.py
│   ├── config.py                 # Config.from_yaml 加载 config.yaml
│   ├── data/
│   │   ├── __init__.py
│   │   ├── pairs.py              # find_pairs / to_same_res（2K、3.5K）
│   │   ├── profile.py            # estimate_profile / apply_color_normalize
│   │   └── dataset.py            # EnhancementDataset + _shared_augment
│   ├── evaluate/
│   │   ├── __init__.py
│   │   └── metrics.py            # psnr / ssim / lpips / niqe / brisque / musiq
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── tiler.py              # tiles_for / tile_weights / stitch / finalize
│   │   └── engine.py             # EnhancementEngine（stage1→stage2→旋钮→缝合→TTA）
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── knobs.py              # KnobConfig / blend / reinject_hf / apply_knobs
│   ├── model/
│   │   ├── __init__.py
│   │   ├── stage1.py             # load_nafnet（vendor 集成）
│   │   └── stage2.py             # stage2_refine（DiffBIR IRControlNet 封装，subprocess 回退）
│   ├── submit/
│   │   ├── __init__.py
│   │   └── package.py            # validate_names / package
│   └── train/
│       ├── __init__.py
│       ├── train_stage1.py       # NAFNet 微调入口
│       └── train_stage2.py       # 扩散适配微调入口
├── tests/
│   ├── conftest.py               # 公共 fixture（临时目录、合成图）
│   ├── test_config.py
│   ├── test_pairs.py
│   ├── test_profile.py
│   ├── test_dataset.py
│   ├── test_tiler.py
│   ├── test_knobs.py
│   ├── test_metrics.py
│   ├── test_package.py
│   └── test_smoke.py             # 端到端冒烟（GPU，P3 前先标记 skip）
├── vendor/                        # 第三方仓库（不入 git 权重）
│   ├── DiffBIR/
│   └── NAFNet/
└── scripts/
    ├── download_weights.sh       # 下载预训练权重
    ├── smoke_e2e.py              # 1 张 val 全链路
    ├── run_inference.py          # 100 张 test → output_dir
    ├── tune_knobs.py             # val 上 λ/n/w/α 网格搜索
    └── report.py                 # val 全指标表 + 汇总
```

接口约定：所有图像 `numpy.ndarray` 为 float32，范围 [0,1]，RGB 通道序（H,W,3）。`torch.Tensor` 为 (B,3,H,W)。

---

## Task 1: 项目脚手架 + 配置加载

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: `src/enhance/__init__.py`, `src/enhance/config.py`
- Create: `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: 无
- Produces: `Config.from_yaml(path: Path) -> Config`；Config 字段（`data_root`, `image_pairs_train_dir`, `val_dir`, `test_dir`, `out_root`, `ckpt_root`, `submit_dir`, `zip_path`, `patch_size`, `batch_size`, `num_workers`, `seed`, `kind_2k_weight`, `tile_size`, `overlap`, `steps`, `scheduler`, `batch_tiles`, `use_tta`）

- [ ] **Step 1: 写失败测试**

```python
# tests/conftest.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
```
```python
# tests/test_config.py
from pathlib import Path

from enhance.config import Config

def test_from_yaml(tmp_path):
    yaml_text = """
data:
  root: dataset
  image_pairs_train_dir: dataset/ImagePairs/train
  val_dir: dataset/huawei/val
  test_dir: dataset/huawei/test
output:
  out_root: output
  ckpt_root: checkpoints
  submit_dir: output_dir
  zip_path: my_work.zip
training:
  patch_size: 256
  batch_size: 8
  num_workers: 4
  seed: 42
  kind_2k_weight: 0.7
  stage1_epochs: 150
inference:
  tile_size: 512
  overlap: 128
  steps: 20
  scheduler: dpm
  batch_tiles: 8
  use_tta: false
model:
  stage1_pretrained: vendor/NAFNet/weights/sidd.pth
knobs:
  lam: 0.3
  n: 0.15
  w: 2.0
  alpha: 0.4
"""
    p = tmp_path / "config.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    cfg = Config.from_yaml(p)
    assert cfg.patch_size == 256
    assert cfg.kind_2k_weight == 0.7
    assert cfg.stage1_epochs == 150
    assert cfg.tile_size == 512
    assert cfg.lam == 0.3
    assert cfg.w == 2.0
    assert cfg.zip_path == Path("my_work.zip")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'enhance'`

- [ ] **Step 3: 创建脚手架文件**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "enhance"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "pyyaml",
    "numpy<2",                      # torch 2.2 基于 numpy 1.x 编译，numpy 2.x 触发 crash 警告
    "opencv-python",
    "scikit-image",
    "torch",                        # CUDA 版由 Task 1 Step 4 显式安装（cu121）
    "diffusers>=0.24,<0.30",
    "transformers>=4.31,<4.41",     # 5.x 要求 torch>=2.5，与 torch 2.2 冲突
    "accelerate>=0.25,<0.34",
    "safetensors",
    "pyiqa",
    "pytest",
]

[tool.setuptools.packages.find]
where = ["src"]
```
```yaml
# config.yaml
data:
  root: dataset
  image_pairs_train_dir: dataset/ImagePairs/train
  val_dir: dataset/huawei/val
  test_dir: dataset/huawei/test
output:
  out_root: output
  ckpt_root: checkpoints
  submit_dir: output_dir
  zip_path: my_work.zip
training:
  patch_size: 256
  batch_size: 8
  num_workers: 4
  seed: 42
  kind_2k_weight: 0.7
  stage1_epochs: 150
inference:
  tile_size: 512
  overlap: 128
  steps: 20
  scheduler: dpm
  batch_tiles: 8
  use_tta: false
model:
  stage1_pretrained: vendor/NAFNet/weights/sidd.pth
knobs:
  lam: 0.3
  n: 0.15
  w: 2.0
  alpha: 0.4
```
```python
# src/enhance/__init__.py
"""可控扩散图像增强流水线。"""
```
```python
# src/enhance/config.py
"""配置加载：从 config.yaml 读取并包装为 dataclass。"""
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_root: Path
    image_pairs_train_dir: Path
    val_dir: Path
    test_dir: Path
    out_root: Path
    ckpt_root: Path
    submit_dir: Path
    zip_path: Path
    patch_size: int
    batch_size: int
    num_workers: int
    seed: int
    kind_2k_weight: float
    stage1_epochs: int
    tile_size: int
    overlap: int
    steps: int
    scheduler: str
    batch_tiles: int
    use_tta: bool
    stage1_pretrained: str
    lam: float
    n: float
    w: float
    alpha: float

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        out = d["output"]
        return cls(
            data_root=Path(d["data"]["root"]),
            image_pairs_train_dir=Path(d["data"]["image_pairs_train_dir"]),
            val_dir=Path(d["data"]["val_dir"]),
            test_dir=Path(d["data"]["test_dir"]),
            out_root=Path(out["out_root"]),
            ckpt_root=Path(out["ckpt_root"]),
            submit_dir=Path(out["submit_dir"]),
            zip_path=Path(out["zip_path"]),
            patch_size=d["training"]["patch_size"],
            batch_size=d["training"]["batch_size"],
            num_workers=d["training"]["num_workers"],
            seed=d["training"]["seed"],
            kind_2k_weight=d["training"]["kind_2k_weight"],
            stage1_epochs=d["training"]["stage1_epochs"],
            tile_size=d["inference"]["tile_size"],
            overlap=d["inference"]["overlap"],
            steps=d["inference"]["steps"],
            scheduler=d["inference"]["scheduler"],
            batch_tiles=d["inference"]["batch_tiles"],
            use_tta=d["inference"]["use_tta"],
            stage1_pretrained=d["model"]["stage1_pretrained"],
            lam=d["knobs"]["lam"],
            n=d["knobs"]["n"],
            w=d["knobs"]["w"],
            alpha=d["knobs"]["alpha"],
        )
```

- [ ] **Step 4: 一次性初始化 `hwcomp` 环境 + 安装依赖 + 跑测试**

先装 CUDA 版 torch（否则 pip 在 Linux 上默认装 CPU-only 轮子），再装其余依赖：
```bash
PIP=/home/liaitong/miniconda3/envs/hwcomp/bin/pip
$PIP install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121
cd /home/liaitong/hw_comp && $PIP install -e .
/home/liaitong/miniconda3/envs/hwcomp/bin/python -c "import torch, diffusers, cv2, pyiqa; print('cuda', torch.cuda.is_available(), 'torch', torch.__version__)"
```
然后跑测试：`cd /home/liaitong/hw_comp && /home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_config.py -v`
Expected: 依赖装好（`cuda True`）、测试 PASS

- [ ] **Step 5: 提交**

```bash
git add pyproject.toml config.yaml src/ tests/
git commit -m "feat: 项目脚手架与配置加载"
```

---

## Task 2: 数据配对构建（ImagePairs → 2K/3.5K）

**Files:**
- Create: `src/enhance/data/__init__.py`, `src/enhance/data/pairs.py`
- Test: `tests/test_pairs.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `Pair` dataclass（`lq_path: Path`, `hr_path: Path`）
  - `find_pairs(root: Path) -> list[Pair]` —— 扫描 `root` 下 `*.png`，配对 `X_ARC.png` 与 `X_ARC_gt.png`
  - `load_lq_hr(pair: Pair) -> tuple[np.ndarray, np.ndarray]` —— 返回 RGB float32 [0,1]
  - `to_same_res(lq: np.ndarray, hr: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]` —— `kind in {"2k","35k"}`，两图输出同尺寸

- [ ] **Step 1: 写失败测试**

```python
# tests/test_pairs.py
import cv2
import numpy as np
import pytest

from enhance.data.pairs import find_pairs, to_same_res


def _img(h, w):
    # 生成值域 [0,1] 的确定性图案：sin*cos∈[-1,1]，平移归一化以满足 inp 的范围断言
    yy, xx = np.mgrid[0:h, 0:w]
    return (((np.sin(yy / 5) * np.cos(xx / 4)) + 1) / 2).astype(np.float32)[..., None].repeat(3, axis=-1)


def test_to_same_res_2k_shape_and_value():
    lq = _img(32, 48)
    hr = _img(64, 96)
    inp, target = to_same_res(lq, hr, "2k")
    assert inp.shape == target.shape == (32, 48, 3)
    assert inp.dtype == target.dtype == np.float32
    # 2k 语义：target = hr 双三次缩到 lq 尺寸（锁定语义，防实现漂移）
    expected = cv2.resize(hr, (48, 32), interpolation=cv2.INTER_CUBIC)
    assert np.allclose(target, expected, atol=1e-6)
    assert 0.0 <= inp.min() and inp.max() <= 1.0


def test_to_same_res_35k_shape():
    lq = _img(32, 48)
    hr = _img(64, 96)
    inp, target = to_same_res(lq, hr, "35k")
    assert inp.shape == target.shape == (64, 96, 3)


def test_to_same_res_unknown_kind_raises():
    lq = _img(8, 8)
    hr = _img(16, 16)
    with pytest.raises(ValueError):
        to_same_res(lq, hr, "4k")


def test_find_pairs(tmp_path):
    for i in range(3):
        (tmp_path / f"a{i}_ARC.png").write_bytes(b"x")
        (tmp_path / f"a{i}_ARC_gt.png").write_bytes(b"x")
    (tmp_path / "lonely.png").write_bytes(b"x")
    pairs = find_pairs(tmp_path)
    assert len(pairs) == 3
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_pairs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'enhance.data'`

- [ ] **Step 3: 实现**

```python
# src/enhance/data/__init__.py
"""数据管线：配对解析、退化画像、数据集。"""
```
```python
# src/enhance/data/pairs.py
"""ImagePairs 真实双相机对解析与 2K/3.5K 同分辨率构造。"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class Pair:
    lq_path: Path
    hr_path: Path


def find_pairs(root: Path) -> list:
    pairs = []
    for lq in sorted(root.rglob("*.png")):
        if lq.name.endswith("_gt.png"):
            continue
        gt = lq.with_name(lq.stem + "_gt.png")
        if gt.exists():
            pairs.append(Pair(lq_path=lq, hr_path=gt))
    return pairs


def load_lq_hr(pair: Pair) -> Tuple[np.ndarray, np.ndarray]:
    lq = cv2.imread(str(pair.lq_path))
    hr = cv2.imread(str(pair.hr_path))
    lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return lq, hr


def to_same_res(lq: np.ndarray, hr: np.ndarray, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    if kind == "2k":
        target = cv2.resize(hr, (lq.shape[1], lq.shape[0]), interpolation=cv2.INTER_CUBIC)
        return lq, target
    if kind == "35k":
        inp = cv2.resize(lq, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
        return inp, hr
    raise ValueError(f"未知 kind: {kind}")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_pairs.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/enhance/data/ tests/test_pairs.py
git commit -m "feat: ImagePairs 配对解析与 2K/3.5K 同分辨率构造"
```

---

## Task 3: 退化画像 + LQ→GT 色彩归一化

**Files:**
- Create: `src/enhance/data/profile.py`
- Test: `tests/test_profile.py`

**Interfaces:**
- Consumes: 无（独立模块）
- Produces:
  - `Profile` dataclass（`noise_std: float`, `color_gain: np.ndarray (3,)`, `color_offset: np.ndarray (3,)`）
  - `estimate_profile(lq: np.ndarray, hr: np.ndarray) -> Profile`
  - `apply_color_normalize(lq: np.ndarray, p: Profile) -> np.ndarray`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_profile.py
import numpy as np

from enhance.data.profile import apply_color_normalize, estimate_profile


def _ramp(h, w):
    yy, xx = np.mgrid[0:h, 0:w] / 255.0
    base = np.stack([xx, yy, (xx + yy) / 2], axis=-1).astype(np.float32)
    return base


def test_estimate_recovers_gain_offset():
    rng = np.random.default_rng(0)
    hr = _ramp(64, 64)
    gain = np.array([1.1, 0.9, 1.0], dtype=np.float32)
    off = np.array([0.05, -0.03, 0.0], dtype=np.float32)
    lq = hr * gain + off  # 构造退化：GT→LQ
    p = estimate_profile(lq, hr)
    # Profile 记录 LQ→GT 增益/偏移（设计文档 §4.2），故恢复的是 1/gain 与 -off/gain
    assert np.allclose(p.color_gain, 1.0 / gain, atol=1e-2)
    assert np.allclose(p.color_offset, -off / gain, atol=1e-2)


def test_apply_normalize_aligns_mean():
    rng = np.random.default_rng(1)
    hr = _ramp(48, 64)
    lq = hr * 1.15 + 0.06
    p = estimate_profile(lq, hr)
    out = apply_color_normalize(lq, p)
    assert abs(out.mean() - hr.mean()) < 1e-3
    assert out.min() >= 0.0 and out.max() <= 1.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_profile.py -v`
Expected: FAIL with import error

- [ ] **Step 3: 实现**

```python
# src/enhance/data/profile.py
"""逐对退化画像：估计噪声、色彩偏移，并提供 LQ→GT 色彩归一化。"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Profile:
    """逐对退化画像：噪声水平与逐通道色彩增益/偏移。"""
    noise_std: float
    color_gain: np.ndarray
    color_offset: np.ndarray


def _laplacian_detail(img: np.ndarray) -> float:
    """返回图像平均拉普拉斯绝对值，用于粗估高频细节量。"""
    g = img.mean(axis=2)
    lap = 4 * g - (np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1))
    return float(np.abs(lap).mean())


def estimate_profile(lq: np.ndarray, hr: np.ndarray) -> Profile:
    """估计一对 (LQ, HR) 的噪声水平与逐通道色彩增益/偏移（最小二乘）。"""
    # 噪声粗估：LQ 高频细节多于 HR 的部分
    noise_std = max(0.0, (_laplacian_detail(lq) - _laplacian_detail(hr)))
    gain = np.zeros(3, dtype=np.float64)
    off = np.zeros(3, dtype=np.float64)
    for c in range(3):
        x = lq[..., c].ravel().astype(np.float64)
        y = hr[..., c].ravel().astype(np.float64)
        a = np.vstack([x, np.ones_like(x)]).T
        g, o = np.linalg.lstsq(a, y, rcond=None)[0]
        gain[c], off[c] = g, o
    return Profile(noise_std=noise_std, color_gain=gain.astype(np.float32), color_offset=off.astype(np.float32))


def apply_color_normalize(lq: np.ndarray, p: Profile) -> np.ndarray:
    """按画像的增益/偏移把 LQ 归一化到 HR 的色彩空间。"""
    return np.clip(lq * p.color_gain + p.color_offset, 0.0, 1.0)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_profile.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/enhance/data/profile.py tests/test_profile.py
git commit -m "feat: 退化画像与色彩归一化"
```

---

## Task 4: PatchDataset + 共享增强

**Files:**
- Create: `src/enhance/data/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `find_pairs`, `load_lq_hr`, `to_same_res`（Task 2）
- Produces:
  - `_shared_augment(lq: np.ndarray, hr: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]`
  - `EnhancementDataset(pairs_root: Path, patch_size: int = 256, kind_2k_weight: float = 0.7, seed: int = 42)`，`__getitem__(idx) -> tuple[torch.Tensor, torch.Tensor]`（(3,H,W) float）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_dataset.py
import numpy as np
import torch

from enhance.data.dataset import EnhancementDataset, _shared_augment


def _img(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return (np.stack([xx, yy, (xx + yy) / 2], axis=-1) / 255.0)


def test_shared_augment_preserves_geometry():
    # 构造确定性关系 HR = LQ * 0.5；共享增强必须对两图施加同一几何变换（线性变换保持该关系）
    lq = np.zeros((16, 20, 3), dtype=np.float32)
    lq[:5] = 1.0
    hr = lq * 0.5
    for seed in range(8):
        rng = np.random.default_rng(seed)
        a, b = _shared_augment(lq.copy(), hr.copy(), rng)
        assert a.shape == b.shape
        assert a.shape in {(16, 20, 3), (20, 16, 3)}  # rot90 会交换 H/W
        assert np.allclose(a, b * 2.0, atol=1e-5)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_dataset.py -v`
Expected: FAIL with import error

- [ ] **Step 3: 实现**

```python
# src/enhance/data/dataset.py
"""PatchDataset：2K/3.5K 混合采样 + 共享增强，返回 (input, target) 张量。"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .pairs import find_pairs, load_lq_hr, to_same_res

def _shared_augment(lq: np.ndarray, hr: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        lq, hr = lq[:, ::-1], hr[:, ::-1]
    if rng.random() < 0.5:
        lq, hr = lq[::-1], hr[::-1]
    k = int(rng.integers(0, 4))
    if k:
        lq, hr = np.rot90(lq, k), np.rot90(hr, k)
    return np.ascontiguousarray(lq), np.ascontiguousarray(hr)

class EnhancementDataset(Dataset):
    def __init__(self, pairs_root: Path, patch_size: int = 256, kind_2k_weight: float = 0.7, seed: int = 42):
        self.pairs = find_pairs(pairs_root)
        self.patch_size = patch_size
        self.kind_2k_weight = kind_2k_weight
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return max(64, len(self.pairs) * 64)

    def _crop(self, img: np.ndarray, y: int, x: int, ps: int) -> np.ndarray:
        """按给定裁剪位置取 ps×ps 子块（输入与目标必须用同一位置）。"""
        return img[y:y + ps, x:x + ps]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[idx % len(self.pairs)]
        lq, hr = load_lq_hr(pair)
        kind = "2k" if self.rng.random() < self.kind_2k_weight else "35k"
        inp, target = to_same_res(lq, hr, kind)
        inp, target = _shared_augment(inp, target, self.rng)
        h, w = inp.shape[:2]
        ps = min(self.patch_size, h, w)
        y = int(self.rng.integers(0, h - ps + 1))
        x = int(self.rng.integers(0, w - ps + 1))
        inp, target = self._crop(inp, y, x, ps), self._crop(target, y, x, ps)
        inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).float()
        target_t = torch.from_numpy(target.transpose(2, 0, 1)).float()
        return inp_t, target_t
```

- [ ] **Step 4: 运行测试确认通过 + 真实数据冒烟**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_dataset.py -v`，再运行真实数据检查：
```bash
/home/liaitong/miniconda3/envs/hwcomp/bin/python -c "
from pathlib import Path
from torch.utils.data import DataLoader
from enhance.data.dataset import EnhancementDataset
ds = EnhancementDataset(Path('dataset/ImagePairs/train'), patch_size=256)
dl = DataLoader(ds, batch_size=4, num_workers=2)
x, y = next(iter(dl))
print('batch shapes:', x.shape, y.shape, 'range:', float(x.min()), float(x.max()))
"
```
Expected: 测试 PASS；真实数据输出 `(4,3,256,256) (4,3,256,256)`，值域在 [0,1]

- [ ] **Step 5: 提交**

```bash
git add src/enhance/data/dataset.py tests/test_dataset.py
git commit -m "feat: PatchDataset 与共享增强"
```

---

## Task 5: 评估指标

**Files:**
- Create: `src/enhance/evaluate/__init__.py`, `src/enhance/evaluate/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `psnr(pred: np.ndarray, ref: np.ndarray, max_val: float = 1.0) -> float`
  - `ssim(pred: np.ndarray, ref: np.ndarray) -> float`（亮度通道）
  - `niqe(img: np.ndarray) -> float`、`brisque(img: np.ndarray) -> float`、`musiq(img: np.ndarray) -> float`（内部用 pyiqa，惰性加载，输入 RGB [0,1]）
  - `report(pred: np.ndarray, ref: np.ndarray | None, device: str = "cpu") -> dict`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_metrics.py
import numpy as np

from enhance.evaluate.metrics import psnr, ssim


def test_psnr_identical_is_inf():
    a = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32)
    assert psnr(a, a) == float("inf")


def test_psnr_noisy_smaller():
    rng = np.random.default_rng(1)
    a = rng.random((32, 32, 3), dtype=np.float32)
    b = np.clip(a + 0.05, 0, 1)
    assert psnr(a, b) < psnr(a, a)
    assert 10 < psnr(a, b) < 40


def test_ssim_range():
    rng = np.random.default_rng(2)
    a = rng.random((32, 32, 3), dtype=np.float32)
    b = np.clip(a + 0.05, 0, 1)
    s = ssim(a, b)
    assert 0.0 <= s <= 1.0
    assert ssim(a, a) > ssim(a, b)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_metrics.py -v`
Expected: FAIL with import error

- [ ] **Step 3: 实现**

```python
# src/enhance/evaluate/__init__.py
"""质量评估：有参考 + 无参考指标。"""
```
```python
# src/enhance/evaluate/metrics.py
"""图像质量指标：PSNR/SSIM 纯实现，LPIPS/NIQE/BRISQUE/MUSIQ 用 pyiqa 惰性加载。"""
import numpy as np
import torch
from skimage.metrics import structural_similarity
from typing import Optional


def psnr(pred: np.ndarray, ref: np.ndarray, max_val: float = 1.0) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    mse = float(np.mean((pred - ref) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10 * np.log10(max_val ** 2 / mse))


def ssim(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.ndim == 3:
        pred, ref = pred.mean(axis=2), ref.mean(axis=2)
    return float(structural_similarity(pred, ref, data_range=1.0))


def _np_to_pyiqa(img: np.ndarray, device: str) -> torch.Tensor:
    # pyiqa 输入要求 (1,3,H,W) RGB [0,1]
    return torch.from_numpy(img.transpose(2, 0, 1))[None].to(device)


def niqe(img: np.ndarray, device: str = "cpu") -> float:
    import pyiqa
    m = pyiqa.create_metric("niqe")
    with torch.no_grad():
        return float(m(_np_to_pyiqa(img, device)).mean().item())


def brisque(img: np.ndarray, device: str = "cpu") -> float:
    import pyiqa
    m = pyiqa.create_metric("brisque")
    with torch.no_grad():
        return float(m(_np_to_pyiqa(img, device)).mean().item())


def musiq(img: np.ndarray, device: str = "cpu") -> float:
    import pyiqa
    m = pyiqa.create_metric("musiq")
    with torch.no_grad():
        return float(m(_np_to_pyiqa(img, device)).mean().item())


def report(pred: np.ndarray, ref: Optional[np.ndarray] = None, device: str = "cpu") -> dict:
    """返回全部指标 dict；有 ref 时含 PSNR/SSIM，无 ref 时仅无参考指标。"""
    out = {}
    if ref is not None:
        out["psnr"] = psnr(pred, ref)
        out["ssim"] = ssim(pred, ref)
    out["niqe"] = niqe(pred, device)
    out["brisque"] = brisque(pred, device)
    out["musiq"] = musiq(pred, device)
    return out
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/enhance/evaluate/ tests/test_metrics.py
git commit -m "feat: 评估指标（PSNR/SSIM + pyiqa 无参考）"
```

---

## Task 6: 分块器 + 融合旋钮

**Files:**
- Create: `src/enhance/inference/__init__.py`, `src/enhance/inference/tiler.py`
- Create: `src/enhance/fusion/__init__.py`, `src/enhance/fusion/knobs.py`
- Test: `tests/test_tiler.py`, `tests/test_knobs.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `tiles_for(height: int, width: int, tile_size: int, overlap: int) -> list[tuple[int,int,int,int]]`（(y0,y1,x0,x1)）
  - `tile_weights(tiles: list[tuple[int,int,int,int]], image_shape: tuple[int,int]) -> list[np.ndarray]`（每个 tile 一个 partition-of-unity 权重块；接缝处线性互补，贴图像边界一侧恒为 1）
  - `accumulate_tile(canvas: np.ndarray, weight_sum: np.ndarray, rect, tile: np.ndarray, weights: np.ndarray) -> None`（就地累加）
  - `finalize(canvas: np.ndarray, weight_sum: np.ndarray) -> np.ndarray`（归一化）
  - `KnobConfig` dataclass（`lam=0.3, n=0.15, w=2.0, alpha=0.4`）
  - `blend(stage1, diffused, lam) -> np.ndarray`、`reinject_hf(diffused, input_img, alpha) -> np.ndarray`、`apply_knobs(stage1, diffused, input_img, cfg) -> np.ndarray`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tiler.py
import numpy as np

from enhance.inference.tiler import accumulate_tile, finalize, tile_weights, tiles_for


def test_tiles_cover_every_pixel():
    tiles = tiles_for(64, 96, tile_size=32, overlap=8)
    covered = np.zeros((64, 96), dtype=bool)
    for y0, y1, x0, x1 in tiles:
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_stitch_reconstructs_constant():
    # 常量图经任意网格缝合后逐像素还原 → partition-of-unity 成立（含图像边界与贴底块的大重叠区）
    img = np.full((64, 96, 3), 0.7, dtype=np.float32)
    canvas = np.zeros_like(img)
    ws = np.zeros((64, 96), dtype=np.float32)
    tiles = tiles_for(64, 96, 32, 8)
    for rect, w in zip(tiles, tile_weights(tiles, (64, 96))):
        y0, y1, x0, x1 = rect
        accumulate_tile(canvas, ws, rect, img[y0:y1, x0:x1], w)
    out = finalize(canvas, ws)
    assert np.abs(out - 0.7).max() < 1e-4
    assert np.abs(out).max() == 0.7  # 无 NaN/越界


def test_tile_weights_boundary_one():
    tiles = tiles_for(64, 96, 32, 8)
    for rect, w in zip(tiles, tile_weights(tiles, (64, 96))):
        y0, y1, x0, x1 = rect
        if y0 == 0:
            assert np.allclose(w[0, :], 1.0)   # 贴顶边 → 权重 1，避免黑边
        if y1 == 64:
            assert np.allclose(w[-1, :], 1.0)  # 贴底边 → 权重 1


def test_tile_weights_internal_ramp_zero():
    tiles = tiles_for(64, 96, 32, 8)
    for rect, w in zip(tiles, tile_weights(tiles, (64, 96))):
        y0, y1, x0, x1 = rect
        if y0 > 0 and y1 < 64 and x0 > 0 and x1 < 96:
            assert w[0, 0] == 0.0 and w[-1, -1] == 0.0  # 纯内部块四角在接缝处权重为 0
```
```python
# tests/test_knobs.py
import numpy as np

from enhance.fusion.knobs import KnobConfig, apply_knobs, blend, reinject_hf


def test_blend_endpoints():
    a = np.zeros((8, 8, 3), dtype=np.float32)
    b = np.ones((8, 8, 3), dtype=np.float32)
    assert np.allclose(blend(a, b, 0.0), a)
    assert np.allclose(blend(a, b, 1.0), b)


def test_reinject_zero_alpha_identity():
    a = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32)
    assert np.allclose(reinject_hf(a, a, 0.0), a, atol=1e-6)


def test_apply_knobs_output_range():
    s1 = np.zeros((16, 16, 3), dtype=np.float32)
    d = np.ones((16, 16, 3), dtype=np.float32)
    inp = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = apply_knobs(s1, d, inp, KnobConfig(lam=0.5, alpha=0.5))
    assert out.min() >= 0.0 and out.max() <= 1.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_tiler.py tests/test_knobs.py -v`
Expected: FAIL with import error

- [ ] **Step 3: 实现**

```python
# src/enhance/inference/__init__.py
"""推理：分块、引擎。"""
```
```python
# src/enhance/inference/tiler.py
"""4K 分块与重叠加权融合（像素域，partition-of-unity）。"""
import numpy as np
from typing import List, Tuple


def tiles_for(height: int, width: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """固定步长铺块，最后一块对齐图像边缘；返回 (y0,y1,x0,x1) 列表。"""
    def spans(L: int, T: int) -> List[Tuple[int, int]]:
        step = max(1, T - overlap)
        starts = list(range(0, max(1, L - T), step))
        if L - T not in starts:  # 最后一块对齐图像边缘（去重：L==T 时 range 已含 0）
            starts.append(L - T)
        return [(s, min(s + T, L)) for s in starts]

    th, tw = min(tile_size, height), min(tile_size, width)
    ys, xs = spans(height, th), spans(width, tw)
    return [(y0, y1, x0, x1) for (y0, y1) in ys for (x0, x1) in xs]


def tile_weights(tiles: List[Tuple[int, int, int, int]], image_shape: Tuple[int, int]) -> List[np.ndarray]:
    """为每个 tile 生成 partition-of-unity 权重块。

    相邻两块在重叠区线性互补（前块 1→0、后块 0→1），
    贴图像边界的一侧权重恒为 1，避免黑边。重叠宽度按实际相邻起点差自适应。
    """
    T = tiles[0][1] - tiles[0][0]
    ystarts = sorted({t[0] for t in tiles})
    xstarts = sorted({t[2] for t in tiles})
    return [_tile_axis_weight(y0, y1, ystarts, T)[:, None] * _tile_axis_weight(x0, x1, xstarts, T)[None, :]
            for (y0, y1, x0, x1) in tiles]


def _tile_axis_weight(s: int, e: int, starts: List[int], T: int) -> np.ndarray:
    """单个 tile 的一维权重：与上块重叠处 0→1、与下块重叠处 1→0，其余为 1。"""
    w = np.ones(e - s)
    idx = starts.index(s)
    if idx > 0:
        o = starts[idx - 1] + T - s
        w[:o] = np.linspace(0.0, 1.0, o)
    if idx < len(starts) - 1:
        o = s + T - starts[idx + 1]
        w[-o:] = np.linspace(1.0, 0.0, o)
    return w


def accumulate_tile(canvas: np.ndarray, weight_sum: np.ndarray, rect: Tuple[int, int, int, int],
                    tile: np.ndarray, weights: np.ndarray) -> None:
    y0, y1, x0, x1 = rect
    canvas[y0:y1, x0:x1] += tile * weights[..., None]
    weight_sum[y0:y1, x0:x1] += weights


def finalize(canvas: np.ndarray, weight_sum: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(canvas, weight_sum[..., None], out=canvas, where=weight_sum[..., None] > 0)
    return out
```
```python
# src/enhance/fusion/__init__.py
"""可控融合旋钮。"""
```
```python
# src/enhance/fusion/knobs.py
"""可控融合旋钮：λ 混合 + 高频残差回注。"""
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class KnobConfig:
    lam: float = 0.3
    n: float = 0.15
    w: float = 2.0
    alpha: float = 0.4


def blend(stage1: np.ndarray, diffused: np.ndarray, lam: float) -> np.ndarray:
    return stage1 * (1.0 - lam) + diffused * lam


def reinject_hf(diffused: np.ndarray, input_img: np.ndarray, alpha: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(input_img, (0, 0), sigmaX=3.0)
    hf = input_img - blurred
    return np.clip(diffused + alpha * hf, 0.0, 1.0)


def apply_knobs(stage1: np.ndarray, diffused: np.ndarray, input_img: np.ndarray, cfg: KnobConfig) -> np.ndarray:
    out = blend(stage1, diffused, cfg.lam)
    return reinject_hf(out, input_img, cfg.alpha)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_tiler.py tests/test_knobs.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/enhance/inference/ src/enhance/fusion/ tests/test_tiler.py tests/test_knobs.py
git commit -m "feat: 4K 分块缝合与融合旋钮"
```

---

## Task 7: 依赖下载 + DiffBIR/NAFNet vendor 集成 + API 确认

**Files:**
- Create: `scripts/download_weights.sh`
- Create: `vendor/`（git 子模块或直接 clone，不入 git 大权重）

**Interfaces:**
- Consumes: 无
- Produces:
  - `vendor/DiffBIR/`（含 SD2.1 基础权重 + IRControlNet 权重，权重放 `vendor/DiffBIR/weights/`）
  - `vendor/NAFNet/`（含预训练权重 `vendor/NAFNet/weights/`）
  - `src/enhance/model/__init__.py`

- [ ] **Step 1: 写下载脚本**

```bash
# scripts/download_weights.sh
#!/usr/bin/env bash
set -euo pipefail
# DiffBIR 权重：SD2.1 基础 + IRControlNet（v2 同分辨率通用权重）
# 来源：XPixelGroup/DiffBIR README（HuggingFace / OpenXLab / 百度网盘）
mkdir -p vendor/DiffBIR vendor/NAFNet
echo "克隆仓库（若未存在）"
[ -d vendor/DiffBIR/.git ] || git clone https://github.com/XPixelGroup/DiffBIR.git vendor/DiffBIR
[ -d vendor/NAFNet/.git ] || git clone https://github.com/megvii-research/NAFNet.git vendor/NAFNet
echo "下载 DiffBIR 权重（见 README 链接，放入 vendor/DiffBIR/weights/）"
echo "下载 NAFNet 预训练权重（放入 vendor/NAFNet/weights/）"
```

- [ ] **Step 2: 执行下载并确认 API**

Run:
```bash
chmod +x scripts/download_weights.sh
# 手动按 DiffBIR README 填入权重下载链接后执行：
# bash scripts/download_weights.sh
```
然后**发现并确认 API**（写入 `docs/api-notes.md`，供后续任务引用）：
```bash
cd vendor/DiffBIR && /home/liaitong/miniconda3/envs/hwcomp/bin/python -c "
import diffusers, transformers
print('diffusers', diffusers.__version__)
"
# 阅读 DiffBIR README 与源码，确认：
#  1. 同分辨率恢复的入口命令（--upscale 1、--tiled、--tile_size、--tile_stride、--cfg_scale、--g_scale）
#  2. Stage-1 模型如何加载/替换（确认能否用自定义 stage-1 输出作为 control 输入）
#  3. 权重的确切路径与文件名
```
将确认结果记录到 `docs/api-notes.md`（此文件不入 git，或入 git 均可）。

- [ ] **Step 3: 验证权重可加载**

Run（用 DiffBIR 仓库自带示例脚本对 `dataset/huawei/val/case1_lq.jpg` 跑一次恢复，确认能出图）：
```bash
cd vendor/DiffBIR
/home/liaitong/miniconda3/envs/hwcomp/bin/python scripts/restoration.py \
  --input /home/liaitong/hw_comp/dataset/huawei/val/case1_lq.jpg \
  --output /tmp/diffbir_case1.png \
  --upscale 1 --tiled --tile_size 512 --tile_stride 256 --aligned 1
```
Expected: `/tmp/diffbir_case1.png` 生成，尺寸 4096×3072，肉眼/指标可见增强（与 LQ 相比 PSNR 有提升或 NIQE 下降）

- [ ] **Step 4: 提交脚本**

```bash
git add scripts/download_weights.sh
git commit -m "feat: 预训练权重下载脚本"
```

> 说明：若 DiffBIR 的 `restoration.py` 不直接支持"外部 stage-1 控制输入"，则 Task 9 的 `stage2.py` 采用 subprocess 回退路径：以我们 NAFNet 的输出作为 `--input` 传入，并通过 DiffBIR 源码中把内部 stage-1 替换为直接读取该输入的方式（在 vendor 内改一处，见 Task 9 具体实现）。

---

## Task 8: Stage-1 NAFNet 封装 + 微调

**Files:**
- Create: `src/enhance/model/__init__.py`, `src/enhance/model/stage1.py`
- Create: `src/enhance/train/__init__.py`, `src/enhance/train/train_stage1.py`

**Interfaces:**
- Consumes: `Config`（Task 1）、`EnhancementDataset`（Task 4）
- Produces:
  - `load_nafnet(weights_path: str, device: str) -> torch.nn.Module`
  - `train_stage1(cfg: Config) -> Path`（训练并返回 best checkpoint 路径）

- [ ] **Step 1: 实现 stage1 模块与训练脚本**

```python
# src/enhance/model/__init__.py
"""模型：Stage-1 NAFNet、Stage-2 扩散。"""
```
```python
# src/enhance/model/stage1.py
"""Stage-1：NAFNet 保真打底，加载与推理。"""
import sys
from pathlib import Path

import numpy as np
import torch

_VENDOR = Path(__file__).resolve().parents[3] / "vendor"


def load_nafnet(weights_path: str, device: str = "cuda") -> torch.nn.Module:
    sys.path.insert(0, str(_VENDOR / "NAFNet"))
    from models.NAFNet_arch import NAFNet  # 延迟导入 vendor 子模块

    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()
```
```python
# src/enhance/train/train_stage1.py
"""Stage-1 NAFNet 微调：L1 损失 + AMP + checkpoint。"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vendor" / "NAFNet"))
from models.NAFNet_arch import NAFNet

from enhance.config import Config
from enhance.data.dataset import EnhancementDataset


def _build_model() -> torch.nn.Module:
    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    return model.cuda()


def train_stage1(cfg: Config) -> Path:
    ds = EnhancementDataset(cfg.image_pairs_train_dir, patch_size=cfg.patch_size,
                            kind_2k_weight=cfg.kind_2k_weight, seed=cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    model = _build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.stage1_epochs)
    model.train()
    for epoch in range(cfg.stage1_epochs):
        total = 0.0
        for i, (x, y) in enumerate(dl):
            x, y = x.cuda(), y.cuda()
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = torch.nn.functional.l1_loss(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss)
            if i % 50 == 0:
                print(f"epoch {epoch} step {i} loss {float(loss):.4f}")
        sched.step()
    ckpt_path = cfg.ckpt_root / "stage1_best.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    return ckpt_path


if __name__ == "__main__":
    cfg = Config.from_yaml(Path("config.yaml"))
    train_stage1(cfg)
```
> `stage1_epochs` 已由 Task 1 的 `Config` 提供；引擎（Task 10）固定加载 `stage1_best.pth`。训练前 P0 冒烟阶段可用 `cfg.stage1_pretrained` 指向的预训练权重先跑通（引擎自动回退，见 Task 10）。

- [ ] **Step 2: 短训验证（10 步，loss 下降）**

Run:
```bash
/home/liaitong/miniconda3/envs/hwcomp/bin/python -c "
from pathlib import Path
import torch
from enhance.config import Config
from enhance.data.dataset import EnhancementDataset
from enhance.train.train_stage1 import _build_model
from torch.utils.data import DataLoader
cfg = Config.from_yaml(Path('config.yaml'))
ds = EnhancementDataset(cfg.image_pairs_train_dir, patch_size=128, kind_2k_weight=0.7, seed=0)
dl = DataLoader(ds, batch_size=4, num_workers=0)
model = _build_model(); opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
x, y = next(iter(dl)); x, y = x.cuda(), y.cuda()
l0 = float(torch.nn.functional.l1_loss(model(x), y))
for _ in range(10):
    opt.zero_grad()
    loss = torch.nn.functional.l1_loss(model(x), y)
    loss.backward(); opt.step()
print('loss before/after:', l0, float(torch.nn.functional.l1_loss(model(x), y)))
"
```
Expected: 打印 `loss before/after:` 且 after < before

- [ ] **Step 3: 提交**

```bash
git add src/enhance/model/ src/enhance/train/ config.yaml src/enhance/config.py tests/
git commit -m "feat: Stage-1 NAFNet 微调"
```

---

## Task 9: Stage-2 扩散封装（DiffBIR IRControlNet + subprocess 回退）

**Files:**
- Create: `src/enhance/model/stage2.py`

**Interfaces:**
- Consumes: `docs/api-notes.md`（Task 7 确认的 DiffBIR API）
- Produces:
  - `stage2_refine(control_image: np.ndarray, out_path: Path, steps: int = 20, guidance: float = 2.0, tile_size: int = 512, stride: int = 256) -> np.ndarray`（对 control 图做潜空间扩散细化，返回 RGB [0,1]）

- [ ] **Step 1: 写失败测试（接口级）**

```python
# tests/test_stage2.py
import numpy as np
import pytest

from enhance.model.stage2 import stage2_refine


@pytest.mark.gpu
def test_stage2_refine_shapes(tmp_path):
    img = np.random.default_rng(0).random((256, 256, 3), dtype=np.float32)
    out = stage2_refine(img, tmp_path / "out.png", steps=5, guidance=1.0, tile_size=256, stride=128)
    assert out.shape == img.shape
    assert 0.0 <= out.min() and out.max() <= 1.0
```
（`pytest.mark.gpu` 由 `conftest.py` 定义；未标记 `--gpu` 时自动 skip。）

- [ ] **Step 2: 在 conftest 增加 gpu 标记**

```python
# tests/conftest.py 追加
import pytest

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="需要 GPU")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
```
在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 增加：
```toml
[tool.pytest.ini_options]
addopts = "--gpu"  # 本项目运行在有 GPU 的机器上，默认开 GPU
markers = ["gpu: 需要 GPU 的测试"]
```

- [ ] **Step 3: 实现 stage2（首选 Python API + subprocess 回退）**

```python
# src/enhance/model/stage2.py
"""Stage-2：DiffBIR IRControlNet 潜空间扩散细化。

优先走 DiffBIR 的 Python API；若 API 与 docs/api-notes.md 记录不符，
回退到 subprocess 调用 DiffBIR CLI（--upscale 1 同分辨率模式）。
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

_VENDOR = Path(__file__).resolve().parents[3] / "vendor"
_DIFFBIR = _VENDOR / "DiffBIR"


def stage2_refine(control_image: np.ndarray, out_path: Path, steps: int = 20,
                  guidance: float = 2.0, tile_size: int = 512, stride: int = 256) -> np.ndarray:
    inp_path = out_path.with_name(out_path.stem + "_ctl.png")
    _save_uint8(control_image, inp_path)
    _run_via_cli(inp_path, out_path, steps, guidance, tile_size, stride)
    out = _load_uint8(out_path)
    return out


def _run_via_cli(inp: Path, out: Path, steps: int, guidance: float, tile_size: int, stride: int) -> None:
    """subprocess 回退：DiffBIR CLI 同分辨率恢复（--upscale 1）。"""
    cmd = [
        sys.executable, str(_DIFFBIR / "scripts" / "restoration.py"),
        "--input", str(inp), "--output", str(out),
        "--upscale", "1", "--tiled", "--tile_size", str(tile_size),
        "--tile_stride", str(stride), "--aligned", "1",
        "--cfg_scale", str(guidance),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _save_uint8(img: np.ndarray, path: Path) -> None:
    bgr = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _load_uint8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
```

> **升级路径（替换 DiffBIR 内部 stage-1 为我们 NAFNet）：** 在 `vendor/DiffBIR` 内新增驱动脚本 `driver_stage2.py`（不入 git 或入 git 均可），加载 `NAFNet`（Task 8 的权重）对输入跑 stage-1，得到 control 图；再调用 DiffBIR 的 `IRControlNet` 采样（以 control 图作为 condition，不做内部 stage-1 重跑）。具体调用方式以 `docs/api-notes.md` 中确认的为准；若无法做到"跳过内部 stage-1"，则维持 CLI 路径并把我们的 NAFNet 输出作为 `--input`（此时 DiffBIR 内部 stage-1 会对 NAFNet 输出再处理一次，结果仍可用，仅不如直接控制精细）。

- [ ] **Step 4: 跑 GPU 冒烟测试**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_stage2.py -v -m gpu`
Expected: PASS（out shape == input shape，值域 [0,1]）

- [ ] **Step 5: 提交**

```bash
git add src/enhance/model/stage2.py tests/test_stage2.py tests/conftest.py pyproject.toml
git commit -m "feat: Stage-2 扩散细化封装（含 subprocess 回退）"
```

---

## Task 10: 推理引擎 + 端到端冒烟

**Files:**
- Create: `src/enhance/inference/engine.py`
- Create: `scripts/smoke_e2e.py`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: `Config`（Task 1）、`tiles_for/tile_weights/accumulate_tile/finalize`（Task 6）、`KnobConfig/apply_knobs`（Task 6）、`load_nafnet`（Task 8）、`stage2_refine`（Task 9）
- Produces:
  - `EnhancementEngine(cfg: Config)`；`engine.enhance(img: np.ndarray) -> np.ndarray`；`engine.enhance_path(in_path: Path, out_path: Path) -> None`

- [ ] **Step 1: 实现引擎**

```python
# src/enhance/inference/engine.py
"""4K 增强推理引擎：stage1 → stage2 分块 → 缝合 → 旋钮融合。"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from enhance.config import Config
from enhance.fusion.knobs import KnobConfig, apply_knobs
from enhance.inference.tiler import accumulate_tile, finalize, tile_weights, tiles_for
from enhance.model.stage1 import load_nafnet
from enhance.model.stage2 import stage2_refine


class EnhancementEngine:
    """4K 增强推理引擎。Stage-1 模型惰性加载并缓存，避免重复载权。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.knobs = KnobConfig(lam=cfg.lam, n=cfg.n, w=cfg.w, alpha=cfg.alpha)
        self._stage1_model = None

    def _stage1_ckpt(self) -> Path:
        best = Path(self.cfg.ckpt_root) / "stage1_best.pth"
        if best.exists():
            return best
        # 尚未训练时用预训练权重跑通冒烟
        return Path(self.cfg.stage1_pretrained)

    def _stage1(self, img: np.ndarray) -> np.ndarray:
        if self._stage1_model is None:
            self._stage1_model = load_nafnet(str(self._stage1_ckpt()), self.device)
        x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(self.device)
        with torch.no_grad():
            y = self._stage1_model(x)
        return y[0].permute(1, 2, 0).cpu().numpy()

    def _stage2_tiled(self, img: np.ndarray, tmp: Path) -> np.ndarray:
        h, w = img.shape[:2]
        canvas = np.zeros_like(img)
        ws = np.zeros((h, w), dtype=np.float32)
        tiles = tiles_for(h, w, self.cfg.tile_size, self.cfg.overlap)
        for rect, w in zip(tiles, tile_weights(tiles, (h, w))):
            y0, y1, x0, x1 = rect
            tile = img[y0:y1, x0:x1]
            out = stage2_refine(tile, tmp / f"{y0}_{x0}.png",
                                steps=self.cfg.steps, guidance=self.knobs.w,
                                tile_size=self.cfg.tile_size, stride=self.cfg.tile_size - self.cfg.overlap)
            accumulate_tile(canvas, ws, rect, out, w)
        return finalize(canvas, ws)

    def enhance(self, img: np.ndarray) -> np.ndarray:
        tmp = Path(tempfile.mkdtemp())
        s1 = self._stage1(img)
        d = self._stage2_tiled(s1, tmp)
        return apply_knobs(s1, d, img, self.knobs)

    def enhance_path(self, in_path: Path, out_path: Path) -> None:
        bgr = cv2.imread(str(in_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = self.enhance(rgb)
        out_bgr = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
```
> 注意：`enhance()` 对每块调用一次 `stage2_refine`（每次起 subprocess 很慢）。**性能优化（必须）**：将 stage2 改为 Python API 批处理（Task 9 的升级路径）；若仍用 subprocess，应把整张图一次传给 CLI（`--tiled`），而不是逐块调用。上方的逐块实现仅作为可运行的保底，正式跑 100 张前必须换用批处理路径（见 Task 12 的性能清单）。

- [ ] **Step 2: 冒烟脚本**

```python
# scripts/smoke_e2e.py
"""端到端冒烟：1 张 val 图走通 增强→指标，输出对比表。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from enhance.config import Config
from enhance.evaluate.metrics import report
from enhance.inference.engine import EnhancementEngine


def main():
    cfg = Config.from_yaml(Path("config.yaml"))
    engine = EnhancementEngine(cfg)
    val_dir = Path(cfg.val_dir)
    lq = cv2.cvtColor(cv2.imread(str(val_dir / "case1_lq.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gt = cv2.cvtColor(cv2.imread(str(val_dir / "case1_gt.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = engine.enhance(lq)
    base = report(lq, gt)
    enh = report(out, gt)
    print("LQ   :", {k: round(v, 3) for k, v in base.items()})
    print("增强后:", {k: round(v, 3) for k, v in enh.items()})


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 端到端冒烟验证（GPU）**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/python scripts/smoke_e2e.py`
Expected: 打印 LQ 与增强后的 PSNR/SSIM/NIQE 等；**增强后 PSNR/SSIM 高于 LQ**（这是"有效"的判据；若低于则调大 λ 或检查 stage-2）

- [ ] **Step 4: 提交**

```bash
git add src/enhance/inference/engine.py scripts/smoke_e2e.py
git commit -m "feat: 4K 增强推理引擎与端到端冒烟"
```

---

## Task 11: 旋钮网格调优 + val 全指标报告

**Files:**
- Create: `scripts/tune_knobs.py`, `scripts/report.py`

**Interfaces:**
- Consumes: `EnhancementEngine`（Task 10）、`report`（Task 5）
- Produces: `output/knob_grid.csv`（λ/n/w/α → val 指标）

- [ ] **Step 1: 实现网格调优脚本**

```python
# scripts/tune_knobs.py
"""在 5 对 val 上网格搜索 (λ, α)，输出 CSV。只在 val 上调参。"""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from enhance.config import Config
from enhance.evaluate.metrics import report
from enhance.fusion.knobs import KnobConfig, apply_knobs
from enhance.inference.engine import EnhancementEngine


def main():
    cfg = Config.from_yaml(Path("config.yaml"))
    engine = EnhancementEngine(cfg)
    vals = [(f"case{i}", f"case{i}_lq.jpg", f"case{i}_gt.jpg") for i in range(1, 6)]
    imgs = {}
    for name, lq_name, gt_name in vals:
        vdir = Path(cfg.val_dir)
        lq = cv2.cvtColor(cv2.imread(str(vdir / lq_name)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt = cv2.cvtColor(cv2.imread(str(vdir / gt_name)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs[name] = (lq, gt)
    # 每个 val 图先跑一次 stage1+stage2，缓存，避免重复推理
    tmp = Path(tempfile.mkdtemp())
    s1 = {n: engine._stage1(lq) for n, (lq, _) in imgs.items()}
    base = {n: engine._stage2_tiled(s1[n], tmp) for n in imgs}
    rows = []
    for lam in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for alpha in [0.0, 0.3, 0.5]:
            ps, ss, nq = [], [], []
            for n, (lq, gt) in imgs.items():
                out = apply_knobs(s1[n], base[n], lq, KnobConfig(lam=lam, alpha=alpha))
                r = report(out, gt)
                ps.append(r["psnr"]); ss.append(r["ssim"]); nq.append(r["niqe"])
            rows.append({"lam": lam, "alpha": alpha,
                         "psnr": round(float(np.mean(ps)), 3),
                         "ssim": round(float(np.mean(ss)), 3),
                         "niqe": round(float(np.mean(nq)), 3)})
    out_csv = Path(cfg.out_root) / "knob_grid.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lam", "alpha", "psnr", "ssim", "niqe"])
        w.writeheader(); w.writerows(rows)
    print("写至", out_csv)
    print("最优 PSNR 行:", max(rows, key=lambda r: r["psnr"]))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 实现报告脚本**

```python
# scripts/report.py
"""对 output_dir 下结果与 val 输出打印全指标汇总。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from enhance.config import Config
from enhance.evaluate.metrics import report


def main():
    cfg = Config.from_yaml(Path("config.yaml"))
    vdir = Path(cfg.val_dir)
    totals = {k: [] for k in ["psnr", "ssim", "niqe", "brisque", "musiq"]}
    for i in range(1, 6):
        lq = cv2.cvtColor(cv2.imread(str(vdir / f"case{i}_lq.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt = cv2.cvtColor(cv2.imread(str(vdir / f"case{i}_gt.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r = report(lq, gt)
        print(f"case{i} LQ:", {k: round(v, 3) for k, v in r.items()})
        for k in totals:
            totals[k].append(r[k])
    print("均值:", {k: round(float(np.mean(v)), 3) for k, v in totals.items()})


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行并记录基线**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/python scripts/report.py`，再运行 `/home/liaitong/miniconda3/envs/hwcomp/bin/python scripts/tune_knobs.py`
Expected: 打印 val 5 对的 LQ 基线指标与均值；生成 `output/knob_grid.csv`，记录最优 λ/α

- [ ] **Step 4: 提交**

```bash
git add scripts/tune_knobs.py scripts/report.py
git commit -m "feat: val 旋钮网格调优与指标报告"
```

---

## Task 12: 提交打包 + 命名校验

**Files:**
- Create: `src/enhance/submit/__init__.py`, `src/enhance/submit/package.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `validate_names(images: list[Path], expected: list[str]) -> list[str]`
  - `package(submit_dir: Path, zip_path: Path) -> None`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_package.py
import zipfile

from enhance.submit.package import package, validate_names


def _make_dir(tmp_path):
    d = tmp_path / "output_dir"
    d.mkdir()
    for i in range(1, 4):
        (d / f"case{i}.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    (d / "case5.jpg").write_bytes(b"\xff\xd8\xff\xe0")  # 多余文件
    (d / "bad_name.png").write_bytes(b"\xff\xd8\xff\xe0")  # 非 jpg
    return d


def test_validate_names(tmp_path):
    d = _make_dir(tmp_path)
    errs = validate_names(list(d.glob("*.jpg")), ["case1.jpg", "case2.jpg", "case3.jpg"])
    # case5 多余、case1-3 齐全 → 报出 case5 多余；bad_name.png 不在 jpg 扫描内
    assert any("case5.jpg" in e for e in errs)
    assert not any("case1.jpg" in e for e in errs)


def test_package_zip_structure(tmp_path):
    d = _make_dir(tmp_path)
    z = tmp_path / "my_work.zip"
    package(d, z)
    assert z.exists()
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    assert "output_dir/case1.jpg" in names
    assert "output_dir/case2.jpg" in names
```

- [ ] **Step 3: 运行测试确认失败**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_package.py -v`
Expected: FAIL with import error

- [ ] **Step 4: 实现**

```python
# src/enhance/submit/__init__.py
"""提交：命名校验与打包。"""
```
```python
# src/enhance/submit/package.py
"""提交命名校验与打包为 zip（内含 output_dir/）。"""
import zipfile
from pathlib import Path
from typing import List


def validate_names(images: List[Path], expected: List[str]) -> List[str]:
    errors = []
    names = {p.name for p in images}
    for e in expected:
        if e not in names:
            errors.append(f"缺少 {e}")
    for n in sorted(names - set(expected)):
        errors.append(f"多余 {n}")
    return errors


def package(submit_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for img in sorted(submit_dir.glob("*.jpg")):
            z.write(img, f"output_dir/{img.name}")
```

- [ ] **Step 5: 运行测试确认通过 + 提交**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/pytest tests/test_package.py -v`
Expected: PASS
```bash
git add src/enhance/submit/ tests/test_package.py
git commit -m "feat: 提交命名校验与打包"
```

---

## Task 13: 100 张 test 推理 + 打包 + 性能优化清单

**Files:**
- Create: `scripts/run_inference.py`

**Interfaces:**
- Consumes: `EnhancementEngine`（Task 10）、`package`/`validate_names`（Task 12）、`Config`
- Produces: `output_dir/case1.jpg`…`case100.jpg` + `my_work.zip`

- [ ] **Step 1: 性能优化（运行 100 张前必须完成）**

检查清单（逐项确认后打勾）：
- [ ] stage2 改为 **Python API 批处理**（一次进程、按块 batch 推理），不用逐块 subprocess——4K 100 张估算 2-5h
- [ ] stage1 全图/大块推理，避免逐块重复加载权重
- [ ] 断点续跑：`run_inference.py` 跳过 `output_dir` 中已存在的 `caseXX.jpg`
- [ ] OOM 保护：`tile_size` 自动减半重试一次

- [ ] **Step 2: 实现推理脚本**

```python
# scripts/run_inference.py
"""100 张 test 图推理 → output_dir/caseXX.jpg，支持断点续跑。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from enhance.config import Config
from enhance.inference.engine import EnhancementEngine
from enhance.submit.package import package, validate_names


def main():
    cfg = Config.from_yaml(Path("config.yaml"))
    engine = EnhancementEngine(cfg)
    test_dir = Path(cfg.test_dir)
    out_dir = Path(cfg.submit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(test_dir.glob("case*.jpg"))
    for i, p in enumerate(cases):
        # 输出名 = 输入 stem 去掉 _lq 后缀（若存在），保证与官方 caseN.jpg 严格对应
        out_name = p.stem[:-3] if p.stem.endswith("_lq") else p.stem
        out = out_dir / f"{out_name}.jpg"
        if out.exists():
            print(f"[{i + 1}/{len(cases)}] 跳过已存在 {out.name}")
            continue
        print(f"[{i + 1}/{len(cases)}] 处理 {p.name}")
        engine.enhance_path(p, out)
    # 命名校验
    expected = [f"case{i}.jpg" for i in range(1, len(cases) + 1)]
    errs = validate_names(list(out_dir.glob("*.jpg")), expected)
    if errs:
        print("命名错误：", errs)
        raise SystemExit(1)
    package(out_dir, Path(cfg.zip_path))
    print("打包完成：", cfg.zip_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行 100 张推理 + 打包**

Run: `/home/liaitong/miniconda3/envs/hwcomp/bin/python scripts/run_inference.py`
Expected: `output_dir/` 下 100 张 jpg；`my_work.zip` 生成；`unzip -l my_work.zip | head` 显示 `output_dir/case1.jpg` 等

- [ ] **Step 4: 质量抽查**

Run: 用 `scripts/report.py` 之外的抽查——从 test 里挑 5 张，肉眼/无参考指标对比 LQ 与增强结果（`NIQE/BRISQUE/MUSIQ` 应优于 LQ）。
Expected: 无参考指标整体改善；若有变差，回 Task 11 调整旋钮并重跑。

- [ ] **Step 5: 提交**

```bash
git add scripts/run_inference.py
git commit -m "feat: 100 张 test 推理与打包"
```

---

## Task 14（加码，可选）：TTA / 数据补全重训 / 回退链

**Files:**
- Modify: `src/enhance/inference/engine.py`（加 `use_tta` 分支）
- Modify: `config.yaml`（`stage1_epochs`、`stage2_steps` 加大）

- [ ] **Step 1: TTA 8 向自集成（仅最终定稿用）**

```python
# engine.py 内新增方法；同时把 `from itertools import product` 加进 engine.py 顶部导入
def enhance_tta(self, img: np.ndarray) -> np.ndarray:
    outs = []
    for flip_y, flip_x, k in product([False, True], [False, True], range(2)):
        t = img
        if flip_y:
            t = t[::-1]
        if flip_x:
            t = t[:, ::-1]
        t = np.rot90(t, k)
        out = self.enhance(t)
        out = np.rot90(out, -k)
        if flip_x:
            out = out[:, ::-1]
        if flip_y:
            out = out[::-1]
        outs.append(out)
    return np.mean(outs, axis=0)
```
验证：val 上 PSNR 是否提升（预期 +0.03~0.13 dB）；提升不达预期则关闭。

- [ ] **Step 2: 数据补全后重训**

ImagePairs 下载补全后，直接复用 Task 8/9 的脚本重训（`stage1_epochs`、`stage2_steps` 加大），在 val 上验证提升。

- [ ] **Step 3: 回退链**

若 Stage-2 在 4K 上 PSNR 停滞或出现伪影，按设计文档第 5.4 节切换 IR-SDE / ControlNet-SDEdit / StableSR。切换以 Task 10 的 `engine.enhance` 接口为边界，不破坏下游。

- [ ] **Step 4: 逐类特化**

test 五类场景（小脸/文字/绿植/钟表/鸟类）各抽代表图，若某类 val/无参考指标明显差，针对性调旋钮（λ 或 n）为该类生成专属配置，逐类推理。

- [ ] **Step 5: 提交**

```bash
git add src/enhance/inference/engine.py config.yaml
git commit -m "feat: TTA 与加码配置"
```

---

## Self-Review 记录

- **Spec 覆盖**：数据策略（2K/3.5K 混训）→ Task 2/4；退化画像 → Task 3；Stage-1/Stage-2 → Task 8/9；可控旋钮 → Task 6/11；4K 分块 → Task 6/10；评估 → Task 5/11；提交打包 → Task 12/13；回退链/TTA → Task 14；代码规范 → 每任务 Global Constraints。
- **占位符扫描**：Task 7 的 `docs/api-notes.md` 与 Task 9 的 vendor 内部改动是**明确标注的执行期发现任务**（外部仓库 API 需实测确认），并提供了始终可执行的 subprocess 回退路径，非悬空占位。
- **类型一致性**：`to_same_res`、`tile_weights`（返回 list，消费方 Task 10 已同步）、`stage2_refine`、`enhance`、`report`、`package` 等签名在定义任务与消费任务间一致；`Config` 已包含 `stage1_epochs` / `stage1_pretrained` / `lam/n/w/alpha`，供 Task 8/10 直接使用，无跨任务字段漂移。
- **实现正确性复核**（分块器）：原 `tile_weights` 在图像边缘 ramp 到 0 会产生黑边，且最后一块贴底造成重叠区大于 ramp 区、partition-of-unity 不成立。已改为按相邻块实际重叠自适应线性互补、贴边一侧恒为 1 的方案，并用常量图逐像素还原测试锁定（`test_stitch_reconstructs_constant`）。
- **测试稳健性**：Task 2/4 的弱断言（`mean` 差值、随机几何变换）已替换为语义锁定或确定性种子断言，避免 flaky。
- **环境**：`relags` 弃用，改用用户新建的 `hwcomp` 专属环境（Python 3.12，空环境），Task 1 Step 4 一次性安装 CUDA torch + 全部依赖。
