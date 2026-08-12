# DiffBIR API 确认笔记

Task 7 对 DiffBIR 推理 API 的三项确认结果。

## 1. 同分辨率恢复入口命令

**入口文件：** `vendor/DiffBIR/inference.py`（非 `scripts/restoration.py`，该文件不存在）

**同分辨率恢复命令（v2.1，4K 输入）：**

```bash
cd vendor/DiffBIR
python inference.py \
  --task sr --upscale 1 --version v2.1 \
  --captioner none \
  --pos_prompt '' \
  --neg_prompt 'low quality, blurry, low-resolution, noisy, unsharp, weird textures' \
  --cfg_scale 6.0 --sampler edm_dpm++_3m_sde --steps 10 \
  --precision fp16 \
  --cleaner_tiled --cleaner_tile_size 512 --cleaner_tile_stride 256 \
  --vae_encoder_tiled --vae_encoder_tile_size 256 \
  --vae_decoder_tiled --vae_decoder_tile_size 256 \
  --cldm_tiled --cldm_tile_size 512 --cldm_tile_stride 256 \
  --input /path/to/input/dir/ --output /path/to/output/dir/
```

**关键标志说明（来源：`inference.py` 第 55-287 行）：**

| 标志 | 默认值 | 说明 |
|------|--------|------|
| `--task` | `sr` | 任务类型：sr/face/denoise/unaligned_face |
| `--upscale` | `4` | 放大倍率，`1` 表示同分辨率 |
| `--version` | `v2.1` | 模型版本：v1/v2/v2.1/custom |
| `--cfg_scale` | `6.0` | Classifier-free guidance scale，控制生成与条件的匹配度 |
| `--g_scale` | `0.0` | 恢复引导（restoration guidance）的学习率，仅在 `--guidance` 启用时生效 |
| `--guidance` | False | 是否启用恢复引导（需配合 `--g_loss`） |
| `--sampler` | `edm_dpm++_3m_sde` | 采样器类型 |
| `--steps` | `10` | 采样步数 |
| `--strength` | `1` | ControlNet 控制强度，越小越有创造力 |
| `--start_point_type` | `noise` | 采样起点：noise（随机噪声）或 cond（扩散后条件图） |
| `--precision` | `fp16` | 推理精度：fp32/fp16/bf16 |

**注意：源码中不存在 `--tiled` 统一开关和 `--aligned` 标志。** 平铺推理需要分别指定四个独立开关：
- `--cleaner_tiled` / `--cleaner_tile_size` / `--cleaner_tile_stride`（Stage-1 去噪）
- `--vae_encoder_tiled` / `--vae_encoder_tile_size`（VAE 编码）
- `--vae_decoder_tiled` / `--vae_decoder_tile_size`（VAE 解码）
- `--cldm_tiled` / `--cldm_tile_size` / `--cldm_tile_stride`（扩散采样）

`--input` 必须是目录（不是单文件），`--output` 也是目录。

## 2. Stage-1 替换 / 外部控制输入

**数据流（来源：`diffbir/pipeline.py` 第 236-321 行）：**

```
Pipeline.run(lq: np.ndarray)
  → lq_tensor (B,3,H,W) in [0,1]
  → self.apply_cleaner(lq_tensor) → cond_img  (Stage-1 输出)
  → self.apply_cldm(cond_img, ...)           (Stage-2 扩散: VAE编码→UNet+ControlNet→VAE解码)
  → wavelet_reconstruction(sample, cond_img)  (小波重建融合)
```

**结论：可以直接用自定义 Stage-1 输出作为 ControlNet 的输入。**

`apply_cldm()` 接收的 `cond_img` 只是一个 `torch.Tensor`（B, 3, H, W）像素值，在方法内部通过 `self.cldm.prepare_condition()` 进行 VAE 编码得到潜在控制信号。而 `apply_cleaner()` 完全独立可替换。

**Task 9 两种可选策略：**

### 策略 A：进程内（in-process）-- 推荐
```python
# 加载 SwinIRPipeline（但不用它的 apply_cleaner）
pipeline = loop.pipeline
cond_img = our_nafnet_output_tensor  # (B,3,H,W) in [0,1]
sample = pipeline.apply_cldm(cond_img, ...)
# 然后自己做 wavelet_reconstruction 和颜色转换
```

### 策略 B：子进程回退
保存 NAFNet 输出为 PNG，传入 `--input`，并将 `SwinIRPipeline.apply_cleaner`（`diffbir/pipeline.py` 第 371 行）改为直接返回输入（恒等函数），一行 patch：
```python
def apply_cleaner(self, lq, tiled, tile_size, tile_stride):
    return lq  # NAFNet 输出已作为 --input 传入，不需要再做 stage-1
```

## 2b. 进程内 `pipeline.run()` 必须包 `torch.autocast`（4K 卡死的根因）

**Task 10 实证（2026-08-12）：** CLI 路径在 `diffbir/inference/loop.py:180` 用
`with torch.autocast(self.args.device, torch.float16)` 包裹 `pipeline.run()`，但 Task 9 的进程内
路径直接调 `pipeline.run()` 没包 autocast，导致全部按 fp32 跑：

- VAE 平铺中间态在 CPU 上暂存（`diffbir/utils/tilevae/tilevae.py` `tiles[i] = tile.cpu()`），fp32 使
  内存翻倍；
- 4K 下 VAE 共 192 个 256 平铺块，GroupNorm 统计阶段进入 CPU-only 长停滞：**0% GPU、~49GB RAM、
  看似卡死**（实际是 fp32 慢 + CPU 内存压力）；
- 证据：2048² 两步 134.6s→66.2s；3072×4096 两步 >480s（超时）→218.4s（加 autocast 后）。

**结论（binding）：** 任何进程内 `pipeline.run()` 调用都必须写成：
```python
with torch.autocast("cuda", torch.float16):
    sample = pipeline.run(ctrl[None], ...)   # 返回 (N,H,W,3) uint8
```
修后 4K（20 步）约 7-8 分钟/张，与 CLI 相当。`apply_cleaner` 若赋为实例属性，签名是
`lambda lq, tiled, tile_size, tile_stride: lq`（实例属性不绑 self，4 个位置参数原样传入）。

## 3. 权重的确切路径与文件名

所有权重通过 `load_model_from_url()`（`diffbir/utils/common.py` 第 113-120 行）从 `weights/` 目录（相对于 CWD）加载。因此必须从 `vendor/DiffBIR/` 运行推理脚本。

| 用途 | 文件名 | 路径 | 大小 | 来源 |
|------|--------|------|------|------|
| SD2.1 基础（zsnr，v2.1） | `sd2.1-base-zsnr-laionaes5.ckpt` | `vendor/DiffBIR/weights/` | 5.16 GB | `lxq007/DiffBIR-v2` HF |
| SD2.1 基础（标准，v1/v2） | `v2-1_512-ema-pruned.ckpt` | `vendor/DiffBIR/weights/` | ~2.5 GB | `stabilityai/stable-diffusion-2-1-base` HF |
| IRControlNet v2.1 | `DiffBIR_v2.1.pt` | `vendor/DiffBIR/weights/` | 1.45 GB | `lxq007/DiffBIR-v2` HF |
| IRControlNet v2 | `v2.pth` | `vendor/DiffBIR/weights/` | ~1.3 GB | `lxq007/DiffBIR-v2` HF |
| SwinIR Stage-1（v2.1） | `realesrgan_s4_swinir_100k.pth` | `vendor/DiffBIR/weights/` | 87 MB | `lxq007/DiffBIR-v2` HF |
| BPE 词汇表 | `bpe_simple_vocab_16e6.txt.gz` | `diffbir/model/open_clip/` | 1.3 MB | Git 仓库自带 |

当前已下载 v2.1 所需的最小权重集合：sd2.1-zsnr + IRControlNet v2.1 + SwinIR。

## 兼容性补丁

**无需任何 diffusers 版本补丁。** DiffBIR 使用自己的 `ControlLDM` 模型加载 SD 权重（`diffbir/model/cldm.py`），不依赖 `diffusers` 包进行模型加载。当前环境 diffusers 0.29.2 完全兼容。

xformers 未安装，代码自动回退到 PyTorch SDP attention（`use sdp attention as default`），功能正常。

## 烟雾测试结果

- **命令：** 同上节命令（输入 `/tmp/diffbir_test_input/`，输出 `/tmp/diffbir_test_output/`）
- **输入：** `case1_lq.jpg`，4096x3072
- **输出：** `case1_lq.png`，4096x3072，RGB，23.4 MB
- **耗时：** 6 分 44 秒（壁钟时间）
- **峰值内存：** ~40 GB RSS（含 CUDA 统一内存映射），实际 GPU < 20 GB
- **可见增强：** 输出图像有明显去噪和增强效果（与原始 LQ 对比可见纹理更清晰）
