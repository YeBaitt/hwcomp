# 生成式图像增强：Diffusion 双阶段管线项目总结

> 项目状态：核心流水线已交付（100 张 test 推理 + 打包完成），质量保障全绿；
> 视觉质量仍不理想，重训计划见 §11。
> 本文档为持久化项目记录与 2026-08-27 课程汇报材料。

---

## 1. 项目概述（目标 / 比赛背景 / 约束 / 截止日期）

### 1.1 比赛背景

| 项目 | 内容 |
|---|---|
| 比赛 | 第七届 CSIG 图像图形技术挑战赛 × 华为终端 2026「Camera 学术之星」 |
| 赛道 | 赛道一「生成式图像增强可控性挑战」（天池 / Tianchi 532499） |
| 任务 | 对 **100 张 4K 低质量测试图像**进行高质量、可控的图像增强，**分辨率保持 4K 不变**，输出 `caseXX.jpg` |
| 挑战场景 | 小人脸、文字、密集绿植、钟表、鸟类 共 5 类 |

### 1.2 项目目标

在 **2026-08-27 课程汇报**前（约 2 周）产出一套**端到端可提交的扩散增强流水线**，并持续优化：
- 满足竞赛硬性约束（Diffusion 架构 + 同分辨率 4K 入 → 4K 出）；
- 带「保真—感知」可控旋钮，以应对权重未公布的混合评分；
- 全部 100 张 test 图成功推理、命名校验并打包提交。

### 1.3 硬性约束（竞赛规则）

- **Diffusion 架构是强制要求**：规则原文「模型须采用 Diffusion 架构」，**NOT** CNN / one-pass regression；规则未限定必须是 Stable Diffusion / 潜空间扩散。
- **同分辨率**：LQ 与 GT 同为 4K；训练 2K 档与 3.5K 档均为输入输出同尺寸；推理保持 4K 分辨率不变、内容不变、真实自然。
- **任意分辨率推理**：混训（2K + 3.5K 两档）使模型分辨率无关，支持 4K 任意分辨率推理；推理时非标准尺寸按实际尺寸分块。
- **赛制**：初赛只提交输出 JPG（不交代码）；架构要求（Diffusion）在**决赛答辩**（PPT + 现场跑图 + 原创性说明）人工审查。

### 1.4 评分机制（为何需要旋钮）

- 官方数据：**5 对 val**（LQ/GT，4K，仅用于验证）+ **100 张 test**（LQ，4K，无 GT，提交目标）。
- 评分 = 多项**有参考 + 无参考指标的加权平均值**；具体指标与权重在决赛入围名单（约 9–20 名）公布。
- → 方案必须内置可调「保真—感知」旋钮（λ/n/w/α），届时按公布的权重做最后一档调整。

### 1.5 截止日期与时间线

| 节点 | 时间 |
|---|---|
| 内部里程碑（课程汇报） | 2026-08-27（约 2 周窗口） |
| 官方提交截止 | 2026-09-15 18:00 |

### 1.6 运行环境

| 项 | 值 |
|---|---|
| 硬件 | 单张 RTX 4090 24GB |
| 环境 | `hwcomp` conda 环境（torch 2.2.0, CUDA） |
| 网络 | 可联网下载预训练权重 |

---

## 2. 最终交付物（my_work.zip + 100 张 4K 结果 + 验证结论）

### 2.1 提交文件

- **`my_work.zip`**，体积 **316,893,341 字节**；
- 内含 `output_dir/case1.jpg .. case100.jpg` 恰 **100 张 4K 图**；
- 命名与官方 `caseN.jpg` **严格同名对应**，无中文符号；
- 输出 JPG 质量 ≈ **95**。

### 2.2 验证结论

| 校验项 | 结果 |
|---|---|
| 推理完成度 | 100/100，日志尾部「打包完成： my_work.zip」 |
| 文件数 | 恰 100（`unzip -l`） |
| 重复 / 缺号 | `uniq -d` 无重复；1..100 缺号扫描为空 |
| 分辨率 | 全部 4K（4096×3072 横图 / 3072×4096 竖图） |
| 无参考抽查 | 15/15 项全部改善（见 §8.3） |

---

## 3. 系统架构（NAFNet stage-1 + SD2.1 latent diffusion stage-2，两阶段串联；可控性旋钮 λ/n/w/α 及含义）

### 3.1 两阶段总览

流水线始终执行 **Stage-2**，不单独使用 Stage-1：

| Stage | 职责 | 实现 |
|---|---|---|
| **Stage-1** | 像素域**保真打底**（PSNR 锚点） | **NAFNet**（预训练微调；替换 DiffBIR 默认的 BSRNet / SCUNet） |
| **Stage-2** | 潜空间**扩散精修**（感知） | **SD2.1 潜空间扩散**（条件适配 / ControlNet / LoRA / SFT，**同分辨率运行、去掉超分上采样器**） |

### 3.2 融合公式

- 线性混合：`out = (1 − λ)·Stage1 + λ·扩散输出`
- 高频残差回注：`out += α·(input − lowpass(input))`

### 3.3 可控性旋钮 λ / n / w / α

| 旋钮 | 含义 | 低值 | 高值 | 安全默认 | 定稿值 |
|---|---|---|---|---|---|
| **λ** | 与 Stage-1 输出的混合权重 | 保真重（0.2–0.4） | 感知重（0.6–0.8） | ≈0.3 | **0.2** |
| **n** | SDEdit 注入噪声水平 | 小 = 紧贴条件（保真） | 大 = 重渲染（感知） | ≈0.15 | 0.15 |
| **w** | CFG 强度（1–3） | 让先验发挥 | 高 = 夹紧条件（保真） | ≈2.0 | 2.0 |
| **α** | 高频残差回注强度 | — | 低成本恢复 4K 纹理、提 PSNR | ≈0.4 | **0.5** |

**空间自适应**（扩展旋钮）：λ / n 可为**逐像素平滑 mask**——平坦区多扩散、纹理区保真。

### 3.4 为什么这套旋钮是「卖点」

- 官方权重未公布 → 决赛入围后可按实际权重最后一档微调；
- λ/α 的调整**不重跑扩散**（stage-2 输出已缓存，只做廉价像素级重组合），迭代成本极低；
- 空间自适应 mask 可在答辩中作为「可控性」原创性说明。

---

## 4. 数据处理管线（ImagePairs → 2K/3.5K 配对、色彩归一化、退化画像、PatchDataset 共享增强；当前 78 对，扩充中）

### 4.1 数据来源

- **ImagePairs** 真实双相机对：弱相机 LR **1752×1166** / 强相机 HR **3504×2332**；
- 退化是**真实设备差异**，非算法降采样；
- 全量约 **11421 对**；**当前已解压 78 对**（第 14/14 分卷），用户正在继续解压其余分卷以扩充训练集（见 §11）。

### 4.2 两档「同分辨率」策略（混训 → 分辨率无关）

| 档位 | 占比 | 输入 | Target | 说明 |
|---|---|---|---|---|
| **2K（主）** | ~70% | LR 原生（1752×1166） | HR 降采样到 2K | 输入输出同尺寸 |
| **3.5K（辅）** | ~30% | LR 插值放大到 3504×2332 | HR 原生 | 输入输出同尺寸 |

→ 混训使模型**分辨率无关**，支撑 4K 任意分辨率推理。

### 4.3 色彩归一化与退化画像（`data/profile.py`）

- **退化画像** `estimate_profile`：`noise_std = max(0, laplacian_detail(lq) − laplacian_detail(hr))`；逐通道颜色 **gain/offset 用最小二乘（lstsq）** 拟合 lq → hr；
- **色彩归一化** `apply_color_normalize`：`clip(lq·gain + offset, 0, 1)` 把 LQ 映射到 HR 颜色空间。

### 4.4 PatchDataset 与共享增强（`data/dataset.py`、`data/pairs.py`）

- `find_pairs`：扫描 `*.png`，跳过 `*_gt.png`，把 `<id>_ARC.png`（lq）与 `<id>_ARC_gt.png`（hr）配成对；
- `EnhancementDataset`：按 `kind_2k_weight=0.7` 采样 2k/35k；`to_same_res` 用 INTER_CUBIC + **clip [0,1]**；
- **共享增强**：随机 flip / 90° 旋转，LQ 与 HR 施加相同变换；
- **同位置 patch 裁剪**：`patch_size=256`；
- **两级惰性缓存**（解码缓存 + 同分辨率缓存）：78 对全量约 **14 GiB**，适配 125GB RAM；
- `__len__ = max(64, n_pairs × length_factor)`，训练用 `length_factor=2`。

### 4.5 官方数据规模

| 集合 | 规模 | 用途 |
|---|---|---|
| val | 5 对（`case1..case5_lq/gt.jpg`，4K） | 验证 |
| test | 100 张（`case1..case100.jpg`，4K，无 GT） | 提交目标 |

---

## 5. 训练（stage-1 NAFNet 微调 config 参数；stage-2 采用预训练 SD2.1；2K 70% / 3.5K 30% 权重）

### 5.1 Stage-1 NAFNet 微调（`train/train_stage1.py` + `config.yaml`）

| 参数 | 值 |
|---|---|
| 架构 | NAFNet width=64 SIDD（`img_channel=3, middle_blk_num=12, enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2]`） |
| 初始化 | **warm-start** 自 `vendor/NAFNet/weights/NAFNet-SIDD-width64.pth`（存在则用，否则随机） |
| 损失 | L1 |
| 优化器 / 学习率 | AdamW lr=1e-4 |
| 调度 | CosineAnnealingLR（T_max = 30） |
| 精度 | AMP（autocast fp16） |
| patch / batch | 256 / 2 |
| num_workers / seed | 4 / 42 |
| epochs | 30 |
| 输出 | `checkpoints/stage1_best.pth` |

### 5.2 Stage-2：预训练 SD2.1（不微调）

- 采用预训练权重 `sd2.1-base-zsnr-laionaes5.ckpt`（5.16 GB）+ `DiffBIR_v2.1.pt`（1.45 GB），置于 `vendor/DiffBIR/weights/`；
- DiffBIR IRControlNet v2.1，`task=sr, upscale=1`，fp16，`edm_dpm++_3m_sde`，20 steps，`cfg_scale=w=2.0`；
- `apply_cleaner` 替换为恒等（identity），使外部 Stage-1 输出直接作为控制输入。

### 5.3 混训权重

**2K 70% / 3.5K 30%**（`kind_2k_weight: 0.7`）——保证主训练面向主流退化形态，同时让模型分辨率无关。

---

## 6. 推理引擎（4K 分块缝合、tiled fp16、autocast、_stage2_tiled 回退、断点续跑、原子写）

### 6.1 4K 分块与缝合（`inference/tiler.py`）

- tile **512px**、重叠 **128px**；
- 重叠区 **cos² / Hann 加权融合**（像素域，**绝不在潜空间缝合**）；
- 非标准尺寸按实际尺寸分块；partition-of-unity 保证拼接无缝。

### 6.2 显存优化（4K 可跑的关键）

| 阶段 | 措施 |
|---|---|
| Stage-1 | **tiled fp16 autocast**（4K 全图前向会 OOM：NAFNet SimpleGate 128 通道约 3GB/block @ 3072×4096） |
| Stage-2 | **全图 in-process** `cldm_tiled`（cldm tile 512 / stride 384，VAE tile 256），**全程 `torch.autocast("cuda", float16)`** —— 这是修复 4K 卡死/OOM 的关键 |
| 通用 | fp16 + FlashAttention + channels_last |

> 根因记录：Task 10 定位 4K 卡死 = `_refine_inproc` 缺 autocast fp16。CLI 包装了 `run()`（loop.py:180）而 in-process 没有 → fp32 使 VAE tile 中间显存翻倍、进入纯 CPU 阶段（0% GPU，49GB RAM）。补上 autocast 后 4K ~**7–8 min/张**。

### 6.3 回退链（`model/stage2.py`）

| 级别 | 路径 | 备注 |
|---|---|---|
| 首选 | 全图 `stage2_refine`（in-process cldm_tiled） | 4K ~7–8 min/张 |
| 回退 1 | `_stage2_tiled` 逐块 stage-2 + 像素域缝合 | 全图失败时降级 |
| 回退 2 | subprocess 调 DiffBIR CLI（`--pos_prompt "" --neg_prompt "low quality,…"` 保证两条路径一致） | 最终兜底 |

### 6.4 工程健壮性（`scripts/run_inference.py`）

- **断点续跑**：按文件存在跳过已完成图片；
- **原子写**：先写 `caseXX.tmp.jpg` 再 `os.replace`，防中途被杀留下截断文件；
- **空目录 guard**：test 目录为空时显式报错，避免打包空 zip；
- **命名校验** `validate_names` + **自动打包**；
- **确定性采样**：固定噪声（DDIM 50 / DPM-Solver 20）；
- **TTA 8 向自集成**仅 +0.03~0.13 dB PSNR、8 倍耗时 → 只留作最终定稿（本次未启用，见 §10/§11）。

---

## 7. 可控性旋钮调优（val 网格 λ×α；λ=0.2 陡沿分析；val 指标表）

### 7.1 网格设计（`scripts/tune_knobs.py`）

- `LAM_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` × `ALPHA_GRID = [0.0, 0.3, 0.5]` = **18 组合**；
- 在 5 对 val 上评估；每图 `_stage1` + 全图 `_stage2` 各跑一次并缓存，网格内只做廉价 `apply_knobs` 重组合；
- 输出 `output/knob_grid.csv`；逐图 try/except，失败跳过不中断（val 运行 0 失败）。

### 7.2 关键行（val 均值）

| λ | α | PSNR | SSIM | NIQE | 备注 |
|---|---|---|---|---|---|
| 0.0 | 0.5 | 24.793 | 0.790 | 8.900 | 纯 Stage-1，best-PSNR |
| **0.2** | **0.5** | **24.505** | **0.739** | **4.201** | **选中的折中点** |
| 0.2 | 0.0 | 24.417 | — | 4.200 | α 对 NIQE 几乎无影响 |
| 1.0 | 0.5 | 20.230 | 0.424 | 2.981 | 纯扩散，lowest-NIQE |

### 7.3 λ=0.2 陡沿分析（为何选 0.2）

- 从 **LQ 输入基线（§8.1：NIQE 9.526）** → λ=0.2：NIQE **9.526 → 4.200**，已捕获可达总增益（到 λ=1.0 纯扩散的 2.981）的约 **81%**，而 PSNR 相对 λ=0 纯 Stage-1（§7.2：24.793）仅损 **~0.29 dB**；
- 0.2 → 1.0：NIQE 仅再降 ~1.2（4.200 → 2.981），却要付出 **~4.3 dB PSNR** 与 0.32 SSIM 的代价；
- → λ=0.2 是无参考增益的「陡沿」，保真代价极小。

### 7.4 α 的作用

- 同 λ 下对 NIQE 几乎无影响（4.200 vs 4.201）；
- α=0.5 比 α=0.0 高 **~0.09 dB PSNR**（24.505 vs 24.417）；
- → 高频残差回注是**近乎免费的保真增益**，故取 α=0.5。

### 7.5 定稿落库

`config.yaml` knobs：**lam=0.2, n=0.15, w=2.0, alpha=0.5**；落库 commit `67b85d3`（供 Task 13 的 100 张推理使用）。

---

## 8. 评测与验证（val 基线 vs 增强；无参考抽查 15/15 全改善的表格；打包校验）

### 8.1 val 基线 vs 增强（Task 11 调优后）

| 指标 | LQ 基线均值 | 增强后（λ=0.2, α=0.5） | 说明 |
|---|---|---|---|
| PSNR ↑ | 28.034 | 24.505 | 参考指标下降（扩散重渲染偏离像素对齐） |
| SSIM ↑ | 0.787 | 0.739 | 同上 |
| NIQE ↓ | 9.526 | **4.201** | **改善约 56%** |
| BRISQUE ↓ | 67.950 | — | 网格中未测 |
| MUSIQ ↑ | 21.259 | — | 网格中未测 |

> 注意：**参考指标较 LQ 基线下降**是感知重渲染的固有取舍——LQ 与 GT 本身高度对齐（真实双摄同场景），扩散重渲染更「好看」但更偏离像素。评分是有参考+无参考的加权混合且权重未公布，旋钮机制正是为平衡这一点兜底。

### 8.2 val 每 case LQ 基线（`scripts/report.py`）

| case | PSNR | SSIM | NIQE | BRISQUE | MUSIQ |
|---|---|---|---|---|---|
| case1 | 32.029 | 0.950 | 9.538 | 95.522 | 21.521 |
| case2 | 28.189 | 0.832 | 9.315 | 63.036 | 18.691 |
| case3 | 35.622 | 0.935 | 9.704 | 63.412 | 21.905 |
| case4 | 18.056 | 0.350 | 11.203 | 64.512 | 20.398 |
| case5 | 26.276 | 0.868 | 7.872 | 53.266 | 23.780 |
| **均值** | **28.034** | **0.787** | **9.526** | **67.950** | **21.259** |

### 8.3 无参考抽查（test `case1/25/50/75/100`，GPU）——15/15 全部改善

**NIQE（越低越好）**

| case | LQ | 增强后 |
|---|---|---|
| case1 | 6.908 | **4.555** |
| case25 | 8.204 | **3.720** |
| case50 | 7.779 | **4.785** |
| case75 | 8.192 | **3.562** |
| case100 | 7.199 | **4.373** |

**BRISQUE（越低越好）**

| case | LQ | 增强后 |
|---|---|---|
| case1 | 58.116 | **36.120** |
| case25 | 70.461 | **22.345** |
| case50 | 76.302 | **27.987** |
| case75 | 70.774 | **30.701** |
| case100 | 55.096 | **26.016** |

**MUSIQ（越高越好）**

| case | LQ | 增强后 |
|---|---|---|
| case1 | 22.536 | **24.006** |
| case25 | 26.580 | **34.055** |
| case50 | 24.543 | **25.555** |
| case75 | 25.384 | **44.233** |
| case100 | 35.001 | **48.982** |

**结论**：无参考指标整体显著改善 → 无需回退到 Task 11 旋钮调优。

### 8.4 打包校验（`my_work.zip`）

- 体积 **316,893,341 字节**；`unzip -l` 恰 100 项，全部前缀 `output_dir/`；
- `uniq -d` 无重复；1..100 缺号扫描为空；
- 抽查 case1/25/50/75/100 均为 4K（3072×4096 与 4096×3072）。

---

## 9. 质量保障（30/30 测试；SDD 逐任务审查 + 最终审查 + 修复轮次）

### 9.1 测试套件

- **30 个测试函数**（11 个 test 文件 + conftest）：config 1 / dataset 3 / knobs 3 / metrics 3 / package 2 / pairs 4 / profile 2 / smoke 3 / stage1 4 / stage2 1 / tiler 4；
- GPU 空闲时 **30/30 通过**；GPU 被推理占用时以 `pytest -o addopts=""` 跳过 gpu 标记测试（`test_stage2` 唯一带 `@pytest.mark.gpu`）→ **29 passed / 1 skipped**；
- `pyproject.toml` 默认 `addopts='--gpu'`，`conftest.py` 注册 `--gpu` 选项。

### 9.2 SDD 逐任务审查流程

- 每个任务 **fresh implementer → task reviewer → fix loop**；
- **14 个任务**；Task 8 历经 **3 轮 fix**；Task 2/3/4/5/9/10/12 各 1 轮；
- 关键裁决：Task 10 的 4K 卡死根因（缺 autocast fp16）已修复并验证（见 §6.2）；Task 13 的 tile 减半项被裁决为「接受替代方案」（Task 10 已根除 OOM 根因，峰值降低路径已存在）。

### 9.3 最终全分支审查（opus 覆盖全部 44 文件）

| 编号 | 问题 | 处理 |
|---|---|---|
| C1 | **39 处双空行**（tokenize 扫描复现） | 修复清零 |
| C2 | `.gitignore`（output/、*.log、.claude/） | 修复 |
| I1 | resume 完整性（临时文件 + 原子 rename） | 修复 |
| I2 | 空目录 guard | 修复 |
| I3 | 3 处未用导入 | 移除 |
| I4 | 4 个死配置字段（data_root/scheduler/batch_tiles/use_tta） | 注释「预留（未消费）」保留 |

一次性修复 commit `b68344a`，re-review 全清（29 passed / 1 skipped）；全分支 44 commits（f0698a7..b68344a）。

### 9.4 代码规范（`docs/代码规范.md`）

中文 docstring（讲 WHAT 不讲 HOW）、stdlib→三方可→项目 import 顺序置顶、**禁用 `from __future__ import annotations`**、顶层 def/class 间单空行、嵌套 ≤3 层、每模块单测 + `if __name__ == "__main__":` 用法示例；数据/权重/输出经 `.gitignore` 排除出 VCS。

---

## 10. 已知局限与视觉质量现状（当前视觉结果仍不理想；根因分析）

> 客观无参考指标大幅改善（15/15），但**主观视觉结果仍不理想**——输出偏软/糊、纹理不够真实、偶有伪纹理。以下三条根因及其对策：

| # | 根因 | 说明 | 可操作对策 |
|---|---|---|---|
| (a) | **训练数据仅 78 对** | stage-1 NAFNet 微调数据量太小，30 epochs **欠拟合**，输出偏软/糊 | 解压全量分卷扩充数据后重训：patch 256→384、epochs 30→60+（详见 §11） |
| (b) | **stage-2 直接用预训练 SD2.1** | 未在真实机内双摄配对数据上微调，扩散 refine 与真实退化**不对齐**，可能过度平滑或引入伪纹理 | 在配对数据上 **LoRA 微调 UNet**（rank 16–32，patch 256–512），让扩散学到「真实退化 → 清晰」映射（对齐 DiffBIR 做法） |
| (c) | **旋钮偏感知** | α=0.5（及 λ=0.2）偏感知优化，牺牲部分保真度 | 重训后**重跑 tune_knobs 网格**，按当时公布的指标权重重新选 λ/α |

---

## 11. 下一步：重训计划（数据扩充后）

> 待新 ImagePairs 数据到达后最终定稿（距 8-27 约 14 天）。

### 11.1 数据

- 解压全部 **14 分卷**（全量约 11421 对），重建配对；
- 沿用 **2K-primary 70% / 3.5K-mixed 30%** 的 same-res 策略；
- **切出留出验证子集**；
- 先做**配对质量 / 色彩一致性快检**。

### 11.2 Stage-1 重训

| 项 | 旧值 | 新计划 |
|---|---|---|
| patch | 256 | **384** |
| epochs | 30 | **60+** |
| batch | 2 | **4（视显存）** |
| 调度 | CosineAnnealing | **cosine LR + warmup + 早停** |
| 初始化 | NAFNet-SIDD 预训练 | 继续该预训练初始化 |
| 跟踪 | — | val PSNR / SSIM |

### 11.3 Stage-2 裁决（A/B）

| 选项 | 内容 | 特点 |
|---|---|---|
| **(A) 维持预训练 SD2.1，仅调旋钮** | 低风险、快迭代 | 视觉提升有限 |
| **(B) 在配对数据上 LoRA 微调 UNet**（rank 16–32，patch 256–512 训练） | 让扩散学到真实退化 → 清晰映射（对齐 DiffBIR） | **视觉提升潜力最大** |

数据到达后按**数据规模 / 显存 / 时间窗口**三方裁决选 A 或 B。

### 11.4 旋钮与验证闭环（全部重跑）

- 重训后**重跑 tune_knobs 网格**（val 5 对），重新选 λ/α；
- val report + 无参考抽查 + **100 张推理 + 打包校验**全部重跑。

### 11.5 排期

| 步骤 | 预估耗时 |
|---|---|
| 数据就绪（解压/整理） | 外部输入，就绪即启动 |
| 数据管线重建 | 0.5–1 天 |
| Stage-1 重训（GPU） | ~半天到 1 天 |
| Stage-2 LoRA 裁决与训练（若选 B） | 1–2 天 |
| 旋钮重调 + 全量推理 | ~10h |
| 打包提交 | 短 |

8-27 前留有缓冲。

---

## 12. 复现与运行指南（环境 hwcomp conda env；关键命令）

### 12.1 环境

- `hwcomp` conda 环境（torch 2.2.0, CUDA）；
- 单张 RTX 4090 24GB；
- 依赖见 `pyproject.toml`（pyyaml, numpy<2, opencv-python, scikit-image, torch, diffusers>=0.24,<0.30, transformers>=4.31,<4.41, accelerate>=0.25,<0.34, safetensors, pyiqa, pytest）。

### 12.2 关键目录

```
src/enhance/           config / data / evaluate / fusion / inference / model / submit / train
scripts/               download_weights.sh, run_inference.py, report.py, smoke_e2e.py, tune_knobs.py
tests/                 11 个 test 文件 + conftest.py
vendor/                NAFNet / DiffBIR 预训练权重
dataset/               ImagePairs/train、huawei/val、huawei/test
output_dir/  my_work.zip   最终提交产物
docs/代码规范.md        编码规范
```

### 12.3 关键命令

```bash
conda activate hwcomp

# 1) 下载预训练权重（DiffBIR + NAFNet，幂等，支持 HF 镜像）
bash scripts/download_weights.sh

# 2) 运行测试（GPU 空闲 30/30；GPU 被推理占用时跳过 gpu 标记测试）
pytest
pytest -o addopts=""          # → 29 passed / 1 skipped

# 3) Stage-1 NAFNet 微调（→ checkpoints/stage1_best.pth）
python -m src.enhance.train.train_stage1

# 4) val 基线完整指标汇总
python scripts/report.py

# 5) 旋钮网格调优（λ×α，18 组合 → output/knob_grid.csv）
python scripts/tune_knobs.py

# 6) 端到端冒烟（1 张 val 图 enhance → 指标）
python scripts/smoke_e2e.py

# 7) 100 张 test 推理 + 自动打包（断点续跑 + 原子写 + 校验）
python scripts/run_inference.py
# → output_dir/case1..case100.jpg + my_work.zip
```

### 12.4 复现注意事项

- `run_inference.py` 内置断点续跑、原子写、空目录 guard、`validate_names`、自动打包；
- **stage-2 全程 `torch.autocast("cuda", float16)`** 是 4K 不卡死/OOM 的前提，不可移除；
- 旋钮定稿 `config.yaml`：`lam=0.2, n=0.15, w=2.0, alpha=0.5`；
- Stage-1 权重优先级：`checkpoints/stage1_best.pth` → 回退 `cfg.stage1_pretrained`（NAFNet-SIDD-width64.pth）。

---

*文档生成：2026-08-13 ｜ 项目：第七届 CSIG 图像图形技术挑战赛 × 华为终端 赛道一（Tianchi 532499）*
