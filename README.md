# 两阶段可控扩散图像增强系统

**第七届 CSIG 图像图形技术挑战赛 × 华为终端 · 赛道一**（Tianchi 532499）

对华为手机拍摄的低质量（LQ）图像做**同分辨率**增强（不放大），输出保真且自然的干净图像。系统为"两阶段工厂"：

- **Stage-1 NAFNet（保真打底）**：去除噪声 / 模糊 / 压缩伪影，恢复基本轮廓与颜色，输出忠实原图的初步干净图。
- **Stage-2 SD2.1 潜空间扩散（感知精修）**：以 Stage-1 输出为 ControlNet 条件图，用 DiffBIR IRControlNet v2.1 做潜空间扩散细化，补细节纹理。
- **旋钮融合**：λ（混合比例）、n（SDEdit 噪声，预留）、w（CFG 强度）、α（高频回注）、β（低频锚定）、σ 控制最终输出在"保真 ↔ 感知"之间的取向。

> 更详细的通俗讲解见 [docs/2026-08-13-two-stage-model-architecture.md](docs/2026-08-13-two-stage-model-architecture.md)（面向零基础），技术总结见 [docs/2026-08-13-diffusion-enhancement-summary.md](docs/2026-08-13-diffusion-enhancement-summary.md)。

---

## 目录结构

```
.
├── config.yaml                    # 唯一主配置：数据路径 / 训练超参 / 推理 / 旋钮
├── pyproject.toml                 # 包定义（src layout，pip install -e . 安装）
├── environment.yaml               # conda 环境（对齐开发机 hwcomp）
├── src/enhance/                   # 主包
│   ├── config.py                  # Config dataclass（从 config.yaml 扁平化加载）
│   ├── data/
│   │   ├── dataset.py             # EnhancementDataset：2K/3.5K 混合采样 + 共享增强 + LRU 缓存
│   │   ├── pairs.py               # 图片对解析（*_ARC.png ↔ *_ARC_gt.png）与 npz 缓存键
│   │   ├── preprocess.py          # 预处理
│   │   └── profile.py             # 数据画像
│   ├── evaluate/metrics.py        # PSNR / SSIM / NIQE / BRISQUE / MUSIQ
│   ├── fusion/knobs.py            # 旋钮融合（λ/n/w/α/β/σ）
│   ├── inference/
│   │   ├── engine.py              # 4K 增强引擎：stage1 → stage2 → 旋钮（主入口）
│   │   └── tiler.py               # 分块推理 + 重叠加权缝合
│   ├── model/
│   │   ├── stage1.py              # NAFNet 加载/推理
│   │   └── stage2.py              # DiffBIR 进程内扩散细化（fp16 autocast，CLI 回退）
│   ├── submit/package.py          # 提交命名校验与打包
│   └── train/train_stage1.py      # ★ Stage-1 微调训练（L1 + bf16 AMP + warmup/cosine + 真实 val 早停）
├── scripts/
│   ├── download_weights.sh        # 下载 NAFNet / DiffBIR 预训练权重（幂等，支持 HF 镜像）
│   ├── synth_pairs.py             # 生成标定退化合成训练对（blur+噪声+JPEG，val 标定参数）
│   ├── build_npz_cache.py         # 预解压 npz uint8 缓存，加速 stage1 训练解码
│   ├── precompute_conds.py        # ★ Stage-2 微调前：预计算 stage1(lq) 条件图
│   ├── eval_ft_ckpt.py            # ★ 微调 ckpt 的 gate 评测（baseline + 多 step × 多 knob）
│   ├── eval_val_crop.py           # 快速 val 检查（中心 1024² 全管线）
│   ├── eval_full_val.py           # 端到端 val 验收（完整 4K 全管线）
│   ├── eval_stage1_val.py         # 单独评估 stage1 checkpoint 在真实 val 上的 PSNR
│   ├── run_inference.py           # 100 张 test 推理 → output_dir + 自动打包
│   ├── build_full_pipeline.py     # 全管线提交构建（--knobs 指定旋钮 → zip）
│   ├── smoke_e2e.py               # 端到端冒烟（1 张 val 图）
│   ├── tune_knobs.py              # val 旋钮网格搜索（λ×α）
│   └── report.py                  # val LQ 基线全指标汇总
├── tests/                         # pytest 单测（conftest 提供 --gpu 标记；pyproject addopts 默认开启 GPU 测试）
├── vendor/                        # 第三方代码 + 预训练权重（gitignore 排除，不入库）
│   ├── NAFNet/                    # NAFNet 源码（basicsr arch）+ weights/NAFNet-SIDD-width64.pth
│   └── DiffBIR/                   # DiffBIR 源码 + weights/（sd2.1 + IRControlNet v2.1）
│       ├── train_stage2.py        # ★ Stage-2 微调训练入口
│       └── configs/train/train_stage2_ft.yaml  # ★ Stage-2 微调配置
├── dataset/                       # 数据（gitignore 排除，不入库）
│   ├── div8k_syn/                 # Stage-1 训练对（DIV8K 生成，1500 对）
│   ├── syn4k/                     # Stage-2 微调对（ImagePairs 生成，500 对）+ cond/
│   └── huawei/                    # 比赛数据：val/（5 对）、test/（100 张）
├── checkpoints/                   # 训练权重（gitignore 排除）
│   ├── stage1_best.pth            # Stage-1 定稿
│   └── stage2_ft/                 # Stage-2 微调实验目录
└── docs/                          # 设计 / 讲解 / API 笔记 / 代码规范
```

---

## 环境搭建

### 1. 创建 conda 环境

```bash
conda env create -f environment.yaml
conda activate hwcomp
```

`environment.yaml` 对齐开发机（Python 3.12 + torch 2.2.0 cu121 + RTX 4090 24GB）。若 torch 拉成 CPU 版，手动指定：

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

### 2. 安装本项目

```bash
pip install -e .          # src layout，把 enhance 包装为 editable
pytest                    # 跑单测（48 个；GPU 被推理占用时用 pytest -o addopts="" 跳过 GPU 测试）
```

### 3. 下载预训练权重

```bash
bash scripts/download_weights.sh
```

自动下载到：

| 权重 | 路径 |
|---|---|
| NAFNet SIDD | `vendor/NAFNet/weights/NAFNet-SIDD-width64.pth` |
| SD2.1 基础模型 | `vendor/DiffBIR/weights/sd2.1-base-zsnr-laionaes5.ckpt` |
| IRControlNet v2.1 | `vendor/DiffBIR/weights/DiffBIR_v2.1.pt` |
| SwinIR（DiffBIR 内部依赖） | `vendor/DiffBIR/weights/realesrgan_s4_swinir_100k.pth` |

### 4. 数据就位

数据集**不入库**（`dataset/` 在 `.gitignore`），队友需自行放置：

```
dataset/
├── huawei/
│   ├── val/      # case1_lq.jpg ~ case5_lq.jpg + 同名 _gt.jpg（真实手机退化对）
│   └── test/     # 100 张待增强图
├── div8k_syn/    # Stage-1 训练对（见下"数据准备"生成）
└── syn4k/        # Stage-2 微调对（见下"数据准备"生成）
```

---

## 数据准备

训练对 = 每对 `synNNNNNN_ARC.png`（LQ 退化图）+ `synNNNNNN_ARC_gt.png`（干净 GT），退化配方已在华为 val 上标定：**blur σ~U(2.5,5) + 高斯噪声 std~U(0.008,0.04) + JPEG q~U(60,95)**。

```bash
# 1) 从干净源（DIV8K 目录 / ImagePairs gt）生成合成退化对
python scripts/synth_pairs.py --src dataset/DIV8K --pattern "*.png" --out dataset/div8k_syn --n 1500
python scripts/synth_pairs.py --src dataset/ImagePairs/train --pattern "*_gt.png" --out dataset/syn4k --n 500

# 2) 预解压 npz 缓存，加速 stage1 训练解码（可选但推荐）
python scripts/build_npz_cache.py --pairs dataset/div8k_syn/train --cache dataset/div8k_syn/cache

# 3) Stage-2 微调前：对每对合成数据预计算 stage1(lq) 条件图（部署对齐，见下）
python scripts/precompute_conds.py --pairs dataset/syn4k/train --cond dataset/syn4k/cond \
    --ckpt checkpoints/stage1_best.pth
```

`config.yaml` 的 `data:` 段需要按你的数据实际位置改好。

---

## Stage-1 训练（NAFNet 微调）

入口：[src/enhance/train/train_stage1.py](src/enhance/train/train_stage1.py)，配置在 [config.yaml](config.yaml) 的 `training:` 段。

```bash
conda activate hwcomp
cd <repo_root>
python -m src.enhance.train.train_stage1
# 等价：PYTHONPATH=src python src/enhance/train/train_stage1.py
```

**做了什么**：

- 加载 `dataset/div8k_syn/train` 的合成对，切 512² patch，2K/3.5K 双分辨率混合训练（`kind_2k_weight=0.7`）；
- L1 损失 + bf16 AMP + 输出 clamp[0,1]（防 loss 尖峰）+ AdamW(lr=2e-5) + warmup 3 epoch → cosine；
- 若 `model.stage1_pretrained` 存在则 warm-start 续训；否则随机初始化；
- **保存判据 = 真实 huawei val 中心 384 PSNR**（`_eval_real_val`，读 `dataset/huawei/val/case{i}_lq/gt.jpg`），`early_stop_patience=8` 早停；
- 合成代理 val 仅作日志（真实 val 与合成退化分布分叉，代理不能作保存判据）。

**输出**：`checkpoints/stage1_best.pth`（state_dict）。

**关键配置**（`config.yaml`）：

| 键 | 默认 | 说明 |
|---|---|---|
| `training.patch_size / batch_size / grad_accum` | 512 / 2 / 2 | patch 512 下 batch 2 显存安全，累积 2 步 ≈ 等效 batch 4 |
| `training.stage1_lr` | 2e-5 | 微调稳定 LR；实测 1e-4 会把已适配模型推出稳定态 |
| `training.stage1_epochs` | 60 | 余弦总周期 |
| `training.num_workers` | 2 | DataLoader worker 会 fork 持有 LRU 缓存副本，必须调小 |
| `training.cache_pairs` | 4 | 每 worker 缓存峰值 ~cache_pairs×240MB，避免整机 OOM |
| `model.stage1_pretrained` | `checkpoints/stage1_best.pth` | warm-start 权重（不存在则随机初始化） |

---

## Stage-2 微调（DiffBIR IRControlNet 微调）

> 原理：部署时 Stage-2 的条件 = **NAFNet Stage-1 输出**。原版 IRControlNet v2.1 用 SwinIR 输出训练，条件分布不匹配 → 幻觉。微调就是让 ControlNet 学会"以 stage1 输出为条件"去还原干净图，消除幻觉。

**顺序**：① 预计算条件 → ② 改配置路径 → ③ 启动训练。

### ① 预计算条件图（必须与部署对齐）

```bash
python scripts/precompute_conds.py --pairs dataset/syn4k/train --cond dataset/syn4k/cond \
    --ckpt checkpoints/stage1_best.pth --tile_size 512 --overlap 128
# 产出 dataset/syn4k/cond/synNNNNNN_ARC_cond.png（stage1(lq) 输出，与 gt 同尺寸）
```

### ② 修改微调配置

编辑 [vendor/DiffBIR/configs/train/train_stage2_ft.yaml](vendor/DiffBIR/configs/train/train_stage2_ft.yaml)，把绝对路径改成你自己的机器：

```yaml
dataset:
  train:
    params:
      pairs_root: /绝对路径/dataset/syn4k/train
      cond_root:  /绝对路径/dataset/syn4k/cond
train:
  sd_path:       /绝对路径/vendor/DiffBIR/weights/sd2.1-base-zsnr-laionaes5.ckpt
  exp_dir:       /绝对路径/checkpoints/stage2_ft
  swinir_path:   /绝对路径/vendor/DiffBIR/weights/realesrgan_s4_swinir_100k.pth
  resume:        /绝对路径/vendor/DiffBIR/weights/DiffBIR_v2.1.pt
```

> ⚠️ **不要改动** `model.diffusion.params` 的 `zero_snr: True` 和 `parameterization: v`——必须与部署配置（`configs/inference/diffusion_v2.1.yaml`）逐位一致，否则微调产物作废（曾因此重训 1hr）。

### ③ 启动训练

```bash
cd vendor/DiffBIR
python train_stage2.py --config configs/train/train_stage2_ft.yaml
```

**做了什么**（[train_stage2.py](vendor/DiffBIR/train_stage2.py)）：

- 加载 SD2.1 权重 + `resume` 的 IRControlNet v2.1 权重，**只训练 ControlNet**（`AdamW(cldm.controlnet.parameters())`，UNet / VAE / CLIP / SwinIR 冻结）；
- 数据：`SynthFtDataset` 读 (gt 干净[-1,1], stage1 条件[0,1]) 同位置 512² 随机裁剪，prompt=""（对齐部署）；
- lr 2e-5、batch 4、`train_steps: 3000`（默认，可改），每 `ckpt_every` 步存一个 controlnet state_dict。

**输出与部署**：

```
checkpoints/stage2_ft/checkpoints/0001000.pt
checkpoints/stage2_ft/checkpoints/0002000.pt
checkpoints/stage2_ft/checkpoints/0003000.pt
```

微调 ckpt 是 **controlnet state_dict**（与 `DiffBIR_v2.1.pt` 同构）。部署时备份原权重后直接替换：

```bash
cd vendor/DiffBIR/weights
cp DiffBIR_v2.1.pt DiffBIR_v2.1_orig.pt          # 备份（原权重 md5=ea976b4ba586954aed9c2c3ac39ed5b9）
cp /绝对路径/checkpoints/stage2_ft/checkpoints/0003000.pt DiffBIR_v2.1.pt
```

### 微调评估 gate

微调后必须过 gate（防 ΔPSNR 单指标选到退化），用 [scripts/eval_ft_ckpt.py](scripts/eval_ft_ckpt.py) 批量评测各 step × 各 knob：

```bash
python scripts/eval_ft_ckpt.py --baseline --steps 1000,2000,3000 \
    --knobs "0.4,1.0,1.0,8.0" "1.0,0.0,0.0,8.0" --crop 1024
```

gate 判据（节选自内存/审查）：

- **A 反幻觉**：case1（近 GT）ΔPSNR 回正、max|ΔPSNR|≤1.5、mean ΔPSNR≥-1.0；
- **B 拒病态**：mean ΔSSIM < -0.08 或 case1 ΔSSIM < -0.08（过平滑）→ 拒；identity-collapse → 拒；
- **C 感知**：NIQE(crop) ≤ baseline+0.3；
- 尽早 step 通过 gate 优先（防合成退化过拟合）。

---

## 评估

```bash
# val 端到端快速检查（中心 1024²，~30s/图，微调迭代用）
python scripts/eval_val_crop.py --knobs 0.4,1.0,1.0,8.0 --tag ft1

# val 完整 4K 全管线验收（stage2 扩散 ~分钟/图，仅提交前用）
python scripts/eval_full_val.py --knobs 0.4,1.0,1.0,8.0 --tag myft

# 单独评估 stage1 checkpoint（真实 val 中心 384 PSNR，与"输入基线"同口径）
python scripts/eval_stage1_val.py checkpoints/stage1_best.pth

# 旋钮网格调参（λ×α，→ output/knob_grid.csv）与 val LQ 基线指标
python scripts/tune_knobs.py
python scripts/report.py
```

`--knobs` 参数格式：`lam,alpha,beta,sigma`。

---

## 推理与提交

```bash
# 100 张 test 图推理 → output_dir/case*.jpg + 自动打包 my_work.zip
# （断点续跑 / 原子写 / 空目录 guard / 命名校验）
python scripts/run_inference.py

# 或指定旋钮的全管线构建
python scripts/build_full_pipeline.py --knobs 0.4,1.0,1.0,8.0 --tag submit_new [--limit 10]
```

当前定稿旋钮（`config.yaml` `knobs:` 段）：`lam=0.2 alpha=1.0 beta=1.0 sigma=8`（LF 锚定 β 为 Pareto 改进，PSNR 25.79→27.02 不掉 NIQE；感知向可调 `lam=0.4`）。

---

## 完整训练管线（从零到提交）

```
Stage-1 收敛（真实 val 判据，→ checkpoints/stage1_best.pth）
   │
   ▼
precompute conds（scripts/precompute_conds.py，条件 = stage1 输出）
   │
   ▼
Stage-2 微调（vendor/DiffBIR/train_stage2.py --config train_stage2_ft.yaml）
   │        （zero_snr: True / parameterization: v 必须与部署一致）
   ▼
Gate 评测（scripts/eval_ft_ckpt.py --baseline + 多 step × 多 knob）
   │
   ▼
选最优 step → 替换 vendor/DiffBIR/weights/DiffBIR_v2.1.pt
   │
   ▼
全管线提交（scripts/build_full_pipeline.py 或 run_inference.py → my_work.zip）
```

---

## 迁移到新机器的注意事项

1. **多个脚本/配置里硬编码了绝对路径 `/home/liaitong/hw_comp`**，队友需改为自己的路径：
   - `vendor/DiffBIR/configs/train/train_stage2_ft.yaml`（数据 / 权重 / exp_dir，见上文）；
   - `scripts/precompute_conds.py`、`scripts/eval_ft_ckpt.py`、`scripts/eval_full_val.py`、`scripts/eval_stage1_val.py`、`scripts/eval_val_crop.py`、`scripts/build_full_pipeline.py` 顶部有 `sys.path.insert(0, "/home/liaitong/hw_comp")` 及绝对路径常量。
2. **`config.yaml` 的 `data:` / `training.cache_root` 需指向你机器上的数据集位置**。
3. **不要提交大文件**：`dataset/`、`checkpoints/`、`output*`、`*.pth`、`*.safetensors`、`*.zip`、`vendor` 权重已在 `.gitignore` 排除。队友需自行 `scripts/download_weights.sh` 下载权重并放置数据。
4. **Stage-2 全程 `torch.autocast("cuda", float16)`**（`model/stage2.py` 内）是 4K 不 OOM/不卡死的前提，不可移除。

---

## 常见问题（FAQ）

- **Stage-1 val 读数反复横跳？** 10 对 holdout 天然高方差（完整图 std ~3dB）。用真实 val + 早停，别追单次尖峰。
- **Stage-2 权重换了但结果没变？** 检查 `vendor/DiffBIR/weights/DiffBIR_v2.1.pt` 是否真的被替换、md5 是否与预期一致（`_orig.pt` 备份 md5=`ea976b4ba586954aed9c2c3ac39ed5b9`）。
- **找不到 DiffBIR 权重？** `stage2.py` 构建 pipeline 时会 `chdir` 到 `vendor/DiffBIR`，权重路径相对该目录解析。
- **CPU-only torch 被装上？** 按"环境搭建"里的 cu121 index 重新安装 torch/torchvision。
- **训练时显存/内存 OOM？** Stage-1 调小 `num_workers`（worker fork 复制 LRU 缓存）与 `cache_pairs`；Stage-2 检查是否包了 fp16 autocast。
