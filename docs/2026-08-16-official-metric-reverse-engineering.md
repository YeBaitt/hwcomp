# 官方评分反推：指标相关性分析（2026-08-16）

> 用两次真实提交（probe_lam04_b10 得分 **3.188**，my_work 得分 **2.82**）计算全部可复现指标，
> 判断官方"有参考+无参考加权平均（权重未公布）"与哪些指标正相关、哪个影响更大。

## 1. 两份提交与官方口径

| 提交 | 配置 | 官方得分 | 差异 |
|---|---|---|---|
| probe_lam04_b10.zip | λ0.4 / α1.0 / β1.0（CPU 恒等式重建，逐字节同源） | **3.188** | +0.368 |
| my_work.zip | λ0.2 / α0.5 / β0（stage1+stage2 全管线） | 2.82 | — |

- 官方数据：5 对 val（LQ/GT，4K，仅验证）+ 100 张 test（LQ，4K，**无 GT**，打分对象）。
- 评分 = 多项**有参考 + 无参考指标的加权平均值**，具体指标与权重决赛入围后公布。

## 2. 结论（test 100 张直接算 + val 代理参考指标，方向完全一致）

**官方 +0.37 分只可能来自"无参考感知指标"这一簇；PSNR/SSIM/LPIPS 与官方分不相关甚至负相关。**

### 2.1 test 100 张 —— 无参考指标（官方打分的就是这批输出）

| 指标（↑=高好 ↓=低好） | probe | my_work | Δ(probe−my_work) | probe 胜场/100 |
|---|---|---|---|---|
| **NIQE** ↓ | **3.33** | 4.51 | **−1.17** | **100/100** |
| BRISQUE ↓ | **18.2** | 30.9 | **−12.7** | 97/100 |
| PIQE ↓ | **26.1** | 34.0 | **−8.0** | 90/100 |
| ILNIQE ↓ | **26.5** | 27.3 | −0.8 | 62/100 |
| MUSIQ ↑ | **36.6** | 31.4 | **+5.2** | 85/100 |
| MANIQA ↑ | **0.286** | 0.260 | +0.026 | 81/100 |
| NIMA ↑ | **4.98** | 4.90 | +0.075 | 84/100 |
| TOPIQ-nr ↑ | **0.390** | 0.316 | **+0.074** | **98/100** |
| CLIPIQA ↑ | **0.519** | 0.373 | **+0.146** | **99/100** |
| 锐度(lapvar) ↑ | **310.7** | 106.6 | +204 | — |
| JPEG 体积 | 3.93MB | 3.12MB | +0.81MB | — |

输入原图基线：NIQE 8.25 / BRISQUE 65.7 / MUSIQ 25.8（增强后全线大幅改善）。
**9 个无参考感知指标全部指向 probe 更好**；NIQE 全胜（100/100）且相对变化最大（−26%），
CLIPIQA（99/100）/ TOPIQ-nr（98/100）/ BRISQUE（97/100）接近一边倒。

### 2.2 val 代理 —— 参考指标（test 无 GT，同配置在 5 对 val 复现，1024² 中心裁剪）

| 指标（vs GT） | LQ 基线 | my_work | probe | 方向 |
|---|---|---|---|---|
| **PSNR** ↑ | 27.35 | 25.79 | 25.71 | 几乎持平（probe 略差 0.08） |
| **SSIM** ↑ | 0.763 | 0.716 | **0.635** | probe 明显更差 |
| **LPIPS** ↓ | 0.405 | 0.475 | **0.515** | probe 明显更差 |
| NIQE ↓ | 9.77 | 5.48 | **4.55** | probe 更好 |
| BRISQUE ↓ | 68.4 | 12.9 | **6.5** | probe 更好 |
| MUSIQ ↑ | 27.95 | 48.0 | **57.3** | probe 更好 |

关键反证：**probe 的 SSIM/LPIPS 明显变差、PSNR 持平，官方分却更高 +0.37**。
若官方加权里有可观的参考指标权重，probe 不可能涨分。（注：LQ 本身 PSNR 27.35 已高于两个
增强输出，纯 PSNR 视角"什么都不做"最高——官方显然不这么评。）

### 2.3 "哪个影响更大"

- **方向最一致、区分度最强**：NIQE（100/100 全胜）> CLIPIQA/TOPIQ-nr/BRISQUE（97–99 胜）> MUSIQ/MANIQA/NIMA（81–85 胜）> PIQE（90 胜）> ILNIQE（62 胜）。
- 特别是**基于学习的现代 IQA（CLIPIQA/TOPIQ-nr，99/98 胜）**与官方方向高度一致，进一步佐证"感知侧"结论。
- 但各无参考指标本次同向变化、彼此相关，**两个官方分点无法拆分精确权重**——只能说官方吃
  "无参考感知质量"这一簇，NIQE 是最可信的单一代理。

## 3. 局限（诚实声明）

1. **test 无 GT** → PSNR/SSIM/LPIPS 只能在 val 上算代理；配置与提交逐字节同源，方向可靠，绝对值≠test。
2. **只有 2 个官方数据点** → 能判方向、不能拟合权重。
3. 官方若对输出下采样再算 NIQE，数值会变，但**两提交的相对方向稳定**。

## 4. 可操作建议

- **继续往感知方向推大概率涨分**：λ 上调（0.5–0.6）+ α 保持 1.0，NIQE/MUSIQ 大概率继续改善；
  用 val 盯住 PSNR 别掉太多，并留足 anti-cheat 像素距离（meanΔ）余量。
- **留保真档后路**：决赛权重公布若含较大参考指标权重，需回退 λ0.2–0.3 + β1.0 档。手里同时备两档。

## 5. 如何启动指标测评（复现脚本）

前置：hwcomp conda 环境（含 pyiqa / lpips / skimage），GPU 空闲。

```bash
# 0) 解压两份提交（zip 前缀 output_dir/）
mkdir -p /tmp/sub_probe /tmp/sub_mywork
unzip -o -q probe_lam04_b10.zip -d /tmp/sub_probe && mv /tmp/sub_probe/output_dir/* /tmp/sub_probe/ && rm -rf /tmp/sub_probe/output_dir
unzip -o -q my_work.zip -d /tmp/sub_mywork && mv /tmp/sub_mywork/output_dir/* /tmp/sub_mywork/ && rm -rf /tmp/sub_mywork/output_dir

# 1) test 100 张无参考指标（probe/my_work/原图）→ /tmp/test_nr_metrics.csv（约 15 分钟）
python scripts/analyze_test_nr.py --probe /tmp/sub_probe --mywork /tmp/sub_mywork \
    --input dataset/huawei/test --out /tmp/test_nr_metrics.csv
#    （9 指标：NIQE/BRISQUE/PIQE/ILNIQE/MUSIQ/MANIQA/NIMA/TOPIQ-nr/CLIPIQA）

# 2) 汇总：均值表 + 配对差 + 胜场数
python scripts/analyze_nr_summary.py --csv /tmp/test_nr_metrics.csv

# 3) val 代理参考指标（需一次性 GPU 缓存：val 5 图 lq/gt/s1/s2_w2.0 npy 在 /tmp/knob_sweep）
python scripts/analyze_val_ref.py --cache /tmp/knob_sweep --out /tmp/val_metrics.json
```

三个脚本：
- `scripts/analyze_test_nr.py` — 官方打分同批 test 输出的无参考指标。权重文件缓存在
  `~/.cache/torch/hub/pyiqa/`，首次跑会联网下载（几百 MB），之后本地复用。
- `scripts/analyze_nr_summary.py` — 汇总 CSV，输出配对差与胜场数。
- `scripts/analyze_val_ref.py` — val 上重建两档配置，算 PSNR/SSIM/LPIPS↔GT（依赖 knob 调优缓存 npy）。

指标与方向：NIQE/BRISQUE/PIQE/ILNIQE 越低越好；MUSIQ/MANIQA/NIMA/TOPIQ-nr/CLIPIQA 越高越好。
（CLIPIQA 依赖 openai/CLIP，其 `import clip` 需要 `pkg_resources`——若报
`ModuleNotFoundError: pkg_resources`，把 setuptools 降到 80.x：`pip install 'setuptools<81'`。）
