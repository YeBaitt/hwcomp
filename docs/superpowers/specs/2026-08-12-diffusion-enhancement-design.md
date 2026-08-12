# 可控扩散图像增强流水线 —— 设计文档

Status: 已批准（2026-08-12）
作者: 参赛团队
关联规范: `docs/代码规范.md`

## 1. 背景与目标

参与 **第七届 CSIG 图像图形技术挑战赛 × 华为终端 2026 "Camera学术之星"——赛道一"生成式图像增强可控性挑战"**（天池 532499）。

任务：对 100 张 4K 低质量测试图像（覆盖小人脸、文字、密集绿植、钟表、鸟类五类挑战场景）进行高质量、**可控**的图像增强，分辨率保持 4K 不变，内容一致、真实自然。提交处理结果 `caseXX.jpg`。

目标：在 **2026-08-27 课程汇报** 前（约 2 周）产出一套**端到端可提交**的扩散增强流水线，并持续优化。

## 2. 约束与关键事实

- **模型必须采用 Diffusion 架构**（规则原文："模型须采用Diffusion架构"）。规则未限定必须是 Stable Diffusion / 潜空间扩散；初赛只提交输出 JPG（不交代码），架构要求在**决赛答辩**（PPT + 现场跑图 + 原创性说明）人工审查。
- 评分 = 多项**有参考** + **无参考**指标的加权平均值；具体指标与权重在**决赛入围名单（约 9-20）公布**，因此方案必须带可调"保真-感知"旋钮。
- 官方数据：5 对 val（LQ/GT，4K，仅用于验证）+ 100 张 test（LQ，4K，无 GT，提交目标）。
- 训练数据：ImagePairs 真实双相机对（弱相机 LR 1752×1166 / 强相机 HR 3504×2332，当前 78 对、陆续补全，共约 11421 对）。退化是真实设备差异，非算法降采样。
- 官方退化 ≈ 弱相机设备差异（有效细节约 4x 损失 + 模糊/噪声/压缩），LQ 与 GT 同为 4K。
- 环境：单张 RTX 4090 24GB；`relags` conda 环境（torch 2.2.0, CUDA）；可联网下载预训练权重。
- 官方提交截止 2026-09-15 18:00；本项目以 2026-08-27 课程汇报为内部里程碑。

## 3. 总体架构

```
训练：ImagePairs 真实双相机对（LR 1752×1166 ↔ HR 3504×2332，78 对起、陆续补全）
      │  每对先做退化画像（模糊核/噪声/色彩gamma偏移）+ LQ→GT 色彩归一化
      │
      ├─ 2K 档（主，~70%）：输入=LR 原生 → target=HR 降采样到 2K   （同分辨率对）
      └─ 3.5K 档（辅，~30%）：输入=LR 插值放大到 3504×2332 → target=HR 原生 （同分辨率对）
      │  （随机裁剪 patch 训练，分辨率无关）
      ▼
模型：DiffBIR 式两阶段（同分辨率变体）
      Stage-1  NAFNet（预训练微调，像素域）→ 保真打底（PSNR 锚点）
      Stage-2  潜空间扩散（SD2.1 + 条件适配，去掉超分上采样）→ 生成主干
      （流水线始终执行 Stage-2，不单独使用 Stage-1）
      ▼
可控融合：out = (1-λ)·Stage1 + λ·扩散输出
      辅助旋钮：SDEdit 噪声水平 n / CFG 强度 w / 高频残差回注 α
      ▼
推理：4K 分块(tile) + 重叠 cos² 融合（像素域，优先 MultiDiffusion 每步融合）
      → 确定性采样（DDIM 50 / DPM-Solver 20）→ 可选 TTA（仅最终定稿）
      ▼
输出：output_dir/caseXX.jpg → my_work.zip（命名校验）

验证：官方 5 对 val → PSNR/SSIM/LPIPS + NIQE/MUSIQ 评分表，仅用于旋钮校准
```

## 4. 数据策略

### 4.1 配对构造

- **2K 档（主）**：输入 = LR 原生（1752×1166），target = HR 双三次降采样到 1752×1166。输入输出同尺寸。
- **3.5K 档（辅）**：输入 = LR 双三次插值放大到 3504×2332，target = HR 原生（3504×2332）。输入输出同尺寸。
- 目的：保留真实双相机退化的同时，覆盖 2K 与 3.5K 两档频段，避免 4K 输出偏软；混训使模型分辨率无关，支持 4K 任意分辨率推理。

### 4.2 预处理：逐对退化画像

对每一对 (LQ, GT) 估计并记录：
- 模糊核（各向异性高斯/运动核近似）
- 噪声水平（σ）
- 色彩 / gamma 偏移（LQ→GT 的逐通道增益与偏移）

推理/训练前将 LQ 归一化到 GT 的色彩空间（低成本、稳定先验）。

### 4.3 增强

- 必选：H/V 翻转、随机裁剪、90° 旋转。
- LR 侧轻度高斯噪声 + JPEG 压缩（增强对真实噪声/压缩的鲁棒性）。
- 色彩/亮度抖动仅轻微，且对 LQ/GT 两路**完全一致**。

### 4.4 数据规模

78 对 × stride-128 滑动窗口（256px）≈ 7800 有效 patch，× 翻转旋转 ≈ 3-6 万样本，满足扩散从零训练可行性下限；数据下载补全后规模进一步扩大。

## 5. 模型设计

### 5.1 Stage-1：NAFNet（保真锚点）

- 预训练 NAFNet（图像恢复，megvii-research NAFNet 仓库权重），**替换 DiffBIR 默认的 Stage-1（BSRNet/SCUNet）**，接入其 Stage-2 条件适配。
- 在 ImagePairs 真实对上微调（可选：辅以轻度退化增强）。
- 训练：512×512 随机裁剪，batch 16-32，共享增强；~100-200 epoch（单 4090 约 1-2 GPU-天）。
- 作用：确定性 PSNR 锚点，吸收大部分退化残差。

### 5.2 Stage-2：潜空间扩散（SD2.1 条件适配）

- SD2.1 基础 U-Net + 条件适配分支（ControlNet / LoRA / SFT 式），条件 = Stage-1 输出的潜码。
- 同分辨率运行，**不使用超分上采样器**。
- 训练：512 裁剪编码到潜空间，条件 + GT 潜码监督；适配器保持小（78 对数据不足以完整微调 U-Net）。
- 保真-感知平衡内建于拓扑：lambda 混合为主旋钮。

### 5.3 可控性旋钮（决赛答辩核心卖点）

| 旋钮 | 含义 | 保真→感知方向 |
|---|---|---|
| λ | 与 Stage-1 输出的混合权重 | λ≈0.2-0.4 保真重；0.6-0.8 感知重 |
| n | SDEdit 注入噪声水平 | 小=紧贴条件（保真），大=重渲染（感知） |
| w | CFG 强度（1-3） | 高=夹紧条件（保真），低=让先验发挥 |
| α | 高频残差回注：`out += α·(input−lowpass(input))` | 低成本恢复 4K 纹理、提 PSNR |
| 空间自适应 | λ/n 可为逐像素平滑 mask | 平坦区多扩散、纹理区保真 |

### 5.4 回退链（按触发条件切换）

1. **IR-SDE**（像素域均值回归扩散，微调）：触发——第 8 天前，4K 上扩散阶段 PSNR ≤ NAFNet-only 或出现潜空间伪影/接缝鬼影。优点：保留原生 4K 纹理，max_sigma 是更干净的保真旋钮。
2. **ControlNet-SDEdit**：触发——扩散适配微调发散，或 val LPIPS 变差（数据太薄）。数据效率更高。
3. **StableSR 直接替换**：触发——第 11 天仍无可收敛的提交流水线。牺牲部分保真换保证的扩散提交。

## 6. 推理流水线（4K）

- 分块：512-640px，重叠 128px；**像素域 cos²/Hann 加权融合**（partition-of-unity + 权重归一化），绝不在潜空间缝合。
- 优先 **MultiDiffusion 每步融合**（各块每步噪声预测按 cos² 权重混合 + 单一全局 scheduler.step），保证全局统一去噪轨迹、避免内容漂移。
- 确定性采样器（DDIM 50 / DPM-Solver 20，固定噪声）；先在小样本上校验步数质量再全量跑。
- fp16 + FlashAttention + channels_last；分块 batch 处理（512² batch-8 ≈ 10GB）。
- 吞吐估算：100 张 4K ≈ 2h（batch-1 DPM20），batch-8 可压至 ~30-50min。
- TTA 8 向自集成：仅 +0.03~0.13 dB PSNR 但 8 倍耗时，**只留作最终定稿**。
- 鲁棒性：非标准尺寸按实际尺寸分块；OOM 自动降块重试；推理中断可断点续跑。
- 输出：JPG quality≈95，命名 `caseXX.jpg` 严格一一对应。

## 7. 评估与验证

- 官方 5 对 val：PSNR / SSIM（Y 通道 + RGB）、LPIPS、NIQE、BRISQUE、MUSIQ。
- 旋钮网格搜索（λ, n, w, α）仅在 5 对 val 上进行，目标 = PSNR 与 LPIPS 联合最优；安全默认值 λ≈0.3, n≈0.15, w≈2, α≈0.4。
- **绝不在 test 上调参**。
- 决赛指标公布日（约 9-20）按官方权重重新校准旋钮（汇报日之后的可选项）。

## 8. 模块划分与代码结构

```
project/
├── config.yaml               # 全部超参
├── data/
│   ├── dataset.py            # 解析 ImagePairs，构建 2K/3.5K 对，PatchDataset，增强
│   └── profile.py            # 退化画像（模糊核/噪声/色彩gamma）+ LQ→GT 归一化
├── model/
│   ├── stage1.py             # NAFNet 微调
│   ├── stage2.py             # 扩散适配（ControlNet/LoRA/SFT）
│   └── sampler.py            # DDIM/DPM-Solver 采样，支持起始噪声步长 t0
├── fusion/
│   └── knobs.py              # λ/n/w/α 旋钮、空间自适应 mask
├── inference/
│   └── engine.py             # 4K 分块 + 重叠 cos² 融合 + MultiDiffusion + TTA + 断点续跑
├── evaluate/
│   └── metrics.py            # PSNR/SSIM/LPIPS/NIQE/BRISQUE/MUSIQ
├── submit/
│   └── package.py            # 命名校验 + 打包 output_dir → my_work.zip
└── tests/                    # 各模块单元测试
```

代码遵循 `docs/代码规范.md`：导入置顶、中文 docstring（描述做什么）、嵌套 ≤3 层、异常标注触发条件、每模块单测 + `if __name__ == "__main__"` 示例、不引入不必要依赖。

## 9. 阶段计划（至 2026-08-27）

| 阶段 | 内容 | 时长 |
|---|---|---|
| P0 | 环境确认 + 下载预训练权重（NAFNet / DiffBIR / SD2.1）+ 数据解析验证 + 1 张 val 冒烟 I/O | 0.5 天 |
| P1 | 数据管线 + 退化画像 + 训练/推理脚本 + 单测 | 2 天 |
| P2 | Stage-1 微调 + Stage-2 适配微调，val 建基线（PSNR/SSIM/LPIPS/NIQE） | 4 天 |
| P3 | 4K 分块推理 + λ/n/w/α 网格调优（仅 val） | 2 天 |
| P4 | 100 张 test 推理 + 命名校验 + 打包 + 汇报材料 | 1-2 天 |
| P5 | 加码：数据补全重训 / TTA / 回退链 / 逐类特化 / 集成 | 剩余 |

## 10. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 78 对数据对扩散微调过拟合 | 小适配器（LoRA/SFT/ControlNet 分支）、激进共享增强、伪验证集早停；λ→0 退回纯保真 |
| 潜空间平滑 4K 纹理 → PSNR 上限、边缘软 | 高频残差回注 + λ 混合；4K 仍不佳则第 8 天前切 IR-SDE |
| 分块接缝 / 跨块不一致 / OOM | 128px 重叠 + cos² 像素域融合 + MultiDiffusion 每步融合 + 确定性采样 + 自动降块 |
| 真实设备退化与先验假设不符 | 逐对退化画像 + LQ→GT 色彩归一化前置；NAFNet 吸收残差 |
| 仅 5 对 val 调旋钮与 test 失配 | 旋钮网格仅 val、联合 PSNR+LPIPS；保留安全默认；绝不在 test 调参 |
| 2 周超时 | 硬里程碑：P4 结束即具备可提交流水线；第 11 天 StableSR 兜底 |

## 11. 提交规格

- 材料存放于 `output_dir/`，100 张 `case1.jpg` ~ `case100.jpg`，与测试图严格同名对应、无中文符号。
- 压缩为 `my_work.zip`（外层目录结构：`my_work.zip → output_dir/caseXX.jpg`）。
- 作品命名规范：技术领域 + 作品名称 + 团队/个人名称 + 联系方式。

## 12. 参考来源

- 比赛公告：CSIG（csig.org.cn/22/202607/53572.html）、天池 532499
- DiffBIR：https://github.com/XPixelGroup/DiffBIR （Apache-2.0）
- IR-SDE：https://github.com/supersupercong/image-restoration-sde
- StableSR：https://github.com/IceClear/StableSR
- MultiDiffusion：arXiv:2302.08113
- OSEDiff：https://github.com/cswry/OSEDiff
- Patch Diffusion（小数据从零训练可行性）：arXiv:2304.12526
