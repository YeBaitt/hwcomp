"""Stage-2：DiffBIR IRControlNet 潜空间扩散细化。

优先走 DiffBIR 的 Python API：进程内构建 pipeline、把外部 stage-1 输出
（control 图）直接送入采样（api-notes.md 策略 B 进程内版，apply_cleaner
替换为恒等）；若进程内加载/运行失败，回退到 subprocess 调用 DiffBIR CLI
（--upscale 1 同分辨率模式）。
"""
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

_VENDOR = Path(__file__).resolve().parents[3] / "vendor"
_DIFFBIR = _VENDOR / "DiffBIR"

_pipeline = None  # 惰性单例：进程内 DiffBIR pipeline，避免每次重复载权（~5GB SD 权重）


def stage2_refine(control_image: np.ndarray, out_path: Path, steps: int = 20,
                  guidance: float = 2.0, tile_size: int = 512, stride: int = 256) -> np.ndarray:
    """对 control 图（stage-1 输出）做潜空间扩散细化，保存 out_path，返回 RGB [0,1] float32。

    内部把输入补齐到 >=512 且为 512 的整数倍（DiffBIR 断言 control >=512），
    采样后裁回原尺寸，保证任意分辨率输入可用。
    """
    h, w = control_image.shape[:2]
    pad_h = max(512, int(math.ceil(h / 512.0)) * 512) - h
    pad_w = max(512, int(math.ceil(w / 512.0)) * 512) - w
    padded = np.pad(control_image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    try:
        out = _refine_inproc(padded, steps, guidance, tile_size, stride)
    except Exception as exc:
        print(f"stage2 进程内路径失败（{type(exc).__name__}: {exc}），回退 CLI")
        out = _refine_via_cli(padded, steps, guidance, tile_size, stride)
    out = out[:h, :w]
    _save_uint8(out, out_path)
    return out


def _build_pipeline():
    """构建 DiffBIR pipeline（须 cwd=vendor/DiffBIR，权重路径相对 CWD）。

    复用 BSRInferenceLoop（diffbir/inference/bsr_loop.py:18）：构造 args
    （对照 inference.py parse_args 的默认值，覆盖 version="v2.1"、task="sr"、
    upscale=1、device、captioner="none"、precision="fp16"、guidance=False），
    loop.pipeline 即含 run/apply_cldm 的 SR pipeline。
    """
    sys.path.insert(0, str(_DIFFBIR))
    # 延迟导入：DiffBIR 依赖需要在 vendor 目录上下文中加载，且避免污染模块级命名空间
    from diffbir.inference.bsr_loop import BSRInferenceLoop

    args = _make_args()  # 依据 inference.py parse_args 默认值构造 v2.1 args
    return BSRInferenceLoop(args).pipeline


def _make_args():
    """按 inference.py parse_args 默认构造 v2.1 SR 配置（逐字段对齐源码）。"""
    # 延迟导入：仅在 _build_pipeline 调用时需要，避免模块级依赖 argparse
    import argparse

    args = argparse.Namespace()
    args.version = "v2.1"
    args.task = "sr"
    args.upscale = 1
    args.device = "cuda"
    args.captioner = "none"
    args.precision = "fp16"    # load_cldm 需要，默认值来自 parse_args:254
    args.guidance = False      # load_cond_fn 需要，默认值来自 parse_args:224
    args.seed = 231            # 默认值来自 parse_args:250
    return args


def _refine_inproc(control_image: np.ndarray, steps: int, guidance: float,
                   tile_size: int, stride: int) -> np.ndarray:
    """进程内：apply_cleaner 替换为恒等，外部 control 直接进 pipeline.run()。

    返回 float32 RGB [0,1]。
    """
    global _pipeline
    if _pipeline is None:
        cwd = os.getcwd()
        os.chdir(str(_DIFFBIR))  # DiffBIR 权重/配置路径相对 CWD
        try:
            _pipeline = _build_pipeline()
            # 不再做内部 stage-1：直接把外部 NAFNet 输出作为 control 送入扩散
            _pipeline.apply_cleaner = lambda lq, tiled, tile_size, tile_stride: lq
        finally:
            os.chdir(cwd)
    # pipeline.run() 内部: lq 是 (N,H,W,3) uint8 → tensor [0,1] → apply_cleaner → apply_cldm → wavelet_reconstruction
    # 返回 (N,H,W,3) uint8，此处传入 batch=1
    ctrl = _to_uint8(control_image)
    sample = _pipeline.run(
        ctrl[None], steps=steps, strength=1.0,
        cleaner_tiled=False, cleaner_tile_size=0, cleaner_tile_stride=0,
        vae_encoder_tiled=True, vae_encoder_tile_size=256,
        vae_decoder_tiled=True, vae_decoder_tile_size=256,
        cldm_tiled=True, cldm_tile_size=tile_size, cldm_tile_stride=stride,
        pos_prompt="", neg_prompt="low quality, blurry, low-resolution, noisy, unsharp, weird textures",
        cfg_scale=guidance, start_point_type="noise", sampler_type="edm_dpm++_3m_sde",
        noise_aug=0, rescale_cfg=False, s_churn=0, s_tmin=0, s_tmax=0, s_noise=1, eta=0, order=3,
    )
    # sample 形状 (1,H,W,3) uint8 → 转换为 float32 [0,1] 并去掉 batch 维
    return sample[0].astype(np.float32) / 255.0


def _refine_via_cli(control_image: np.ndarray, steps: int, guidance: float,
                    tile_size: int, stride: int) -> np.ndarray:
    """subprocess 回退：DiffBIR CLI 同分辨率恢复（api-notes.md 入口命令）。

    返回 float32 RGB [0,1]。
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        inp, out = tmp / "in", tmp / "out"
        inp.mkdir(), out.mkdir()
        _save_uint8(control_image, inp / "ctl.png")
        cmd = [sys.executable, str(_DIFFBIR / "inference.py"),
               "--task", "sr", "--upscale", "1", "--version", "v2.1",
               "--captioner", "none", "--sampler", "edm_dpm++_3m_sde",
               "--cfg_scale", str(guidance), "--steps", str(steps),
               "--precision", "fp16",
               "--pos_prompt", "", "--neg_prompt", "low quality, blurry, low-resolution, noisy, unsharp, weird textures",
               "--cleaner_tiled", "--cleaner_tile_size", "512", "--cleaner_tile_stride", "256",
               "--vae_encoder_tiled", "--vae_encoder_tile_size", "256",
               "--vae_decoder_tiled", "--vae_decoder_tile_size", "256",
               "--cldm_tiled", "--cldm_tile_size", str(tile_size), "--cldm_tile_stride", str(stride),
               "--input", str(inp), "--output", str(out)]
        subprocess.run(cmd, cwd=str(_DIFFBIR), check=True, capture_output=True)
        return _load_uint8(out / "ctl.png")


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """RGB [0,1] → uint8。"""
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _save_uint8(img: np.ndarray, path: Path) -> None:
    """把 RGB float32 [0,1] 图像写为 PNG（cv2 用 BGR）。"""
    bgr = cv2.cvtColor(_to_uint8(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def _load_uint8(path: Path) -> np.ndarray:
    """读回 PNG 为 RGB float32 [0,1]。"""
    bgr = cv2.imread(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


if __name__ == "__main__":
    # 使用示例：随机 control 图走一次 5 步采样（进程内路径）
    img = np.random.default_rng(0).random((256, 256, 3), dtype=np.float32)
    out = stage2_refine(img, Path("/tmp/stage2_demo.png"), steps=5, guidance=1.0,
                        tile_size=256, stride=128)
    print(f"stage2 demo: in={img.shape} out={out.shape} range=[{out.min():.3f},{out.max():.3f}]")
