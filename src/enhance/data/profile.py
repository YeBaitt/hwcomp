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


if __name__ == "__main__":
    # 使用示例：估计一对 LQ/HR 的退化画像并做色彩归一化
    yy, xx = np.mgrid[0:64, 0:64] / 255.0
    hr = np.stack([xx, yy, (xx + yy) / 2], axis=-1).astype(np.float32)
    gain = np.array([1.1, 0.9, 1.0], dtype=np.float32)
    off = np.array([0.05, -0.03, 0.0], dtype=np.float32)
    lq = hr * gain + off
    p = estimate_profile(lq, hr)
    out = apply_color_normalize(lq, p)
    print(f"gain={p.color_gain} offset={p.color_offset} noise_std={p.noise_std:.4f}")
    print(f"归一化后 mean={out.mean():.4f} (hr mean={hr.mean():.4f})")
