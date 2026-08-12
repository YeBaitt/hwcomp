"""图像质量指标：PSNR/SSIM 纯实现，NIQE/BRISQUE/MUSIQ 用 pyiqa 惰性加载。"""
from typing import Optional

import numpy as np
import pyiqa
import torch
from skimage.metrics import structural_similarity

def psnr(pred: np.ndarray, ref: np.ndarray, max_val: float = 1.0) -> float:
    """计算预测图与参考图之间的峰值信噪比（PSNR），输入 RGB [0,1] 数组。"""
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    mse = float(np.mean((pred - ref) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10 * np.log10(max_val ** 2 / mse))

def ssim(pred: np.ndarray, ref: np.ndarray) -> float:
    """计算预测图与参考图之间的结构相似度（SSIM），使用亮度通道。"""
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.ndim == 3:
        pred, ref = pred.mean(axis=2), ref.mean(axis=2)
    return float(structural_similarity(pred, ref, data_range=1.0))

def _np_to_pyiqa(img: np.ndarray, device: str) -> torch.Tensor:
    """把 RGB [0,1] 的 (H,W,3) 数组转成 pyiqa 需要的 (1,3,H,W) 张量。"""
    return torch.from_numpy(img.transpose(2, 0, 1))[None].to(device)

def _pyiqa_metric(name: str, img: np.ndarray, device: str) -> float:
    """用 pyiqa 计算指定无参考指标（首次调用会联网下载权重）。"""
    m = pyiqa.create_metric(name)
    with torch.no_grad():
        return float(m(_np_to_pyiqa(img, device)).mean().item())

def niqe(img: np.ndarray, device: str = "cpu") -> float:
    """计算无参考图像质量指标 NIQE，输入 RGB [0,1] 数组。"""
    return _pyiqa_metric("niqe", img, device)

def brisque(img: np.ndarray, device: str = "cpu") -> float:
    """计算无参考图像质量指标 BRISQUE，输入 RGB [0,1] 数组。"""
    return _pyiqa_metric("brisque", img, device)

def musiq(img: np.ndarray, device: str = "cpu") -> float:
    """计算无参考图像质量指标 MUSIQ，输入 RGB [0,1] 数组。"""
    return _pyiqa_metric("musiq", img, device)

def report(pred: np.ndarray, ref: Optional[np.ndarray] = None, device: str = "cpu") -> dict:
    """返回全部指标 dict；有 ref 时含 PSNR/SSIM，无 ref 时仅无参考指标。首次调用会联网下载 NIQE/BRISQUE/MUSIQ 权重。"""
    out = {}
    if ref is not None:
        out["psnr"] = psnr(pred, ref)
        out["ssim"] = ssim(pred, ref)
    out["niqe"] = niqe(pred, device)
    out["brisque"] = brisque(pred, device)
    out["musiq"] = musiq(pred, device)
    return out

if __name__ == "__main__":
    # 使用示例：合成数组演示 PSNR/SSIM（不触发 pyiqa 权重下载）
    rng = np.random.default_rng(0)
    a = rng.random((32, 32, 3), dtype=np.float32)
    b = np.clip(a + 0.05, 0, 1)
    print(f"psnr(a, b) = {psnr(a, b):.2f} dB")
    print(f"ssim(a, b) = {ssim(a, b):.4f}")
