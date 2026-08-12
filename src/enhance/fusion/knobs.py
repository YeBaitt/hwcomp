"""可控融合旋钮：λ 混合 + 高频残差回注。"""
from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class KnobConfig:
    """可控融合参数：lam 混合强度、alpha 高频回注强度；n/w 预留给后续任务的噪声与提示权重。"""
    lam: float = 0.3
    n: float = 0.15
    w: float = 2.0
    alpha: float = 0.4

def blend(stage1: np.ndarray, diffused: np.ndarray, lam: float) -> np.ndarray:
    """按 lam 把 stage1 与扩散结果线性混合（lam=0 取 stage1，lam=1 取扩散结果）。"""
    return stage1 * (1.0 - lam) + diffused * lam

def reinject_hf(diffused: np.ndarray, input_img: np.ndarray, alpha: float) -> np.ndarray:
    """向扩散结果回注输入图的高频细节，强度由 alpha 控制，输出裁剪到 [0,1]。"""
    blurred = cv2.GaussianBlur(input_img, (0, 0), sigmaX=3.0)
    hf = input_img - blurred
    return np.clip(diffused + alpha * hf, 0.0, 1.0)

def apply_knobs(stage1: np.ndarray, diffused: np.ndarray, input_img: np.ndarray, cfg: KnobConfig) -> np.ndarray:
    """按旋钮配置对 stage1/扩散结果/输入图施加 λ 混合与高频回注，返回最终融合图。"""
    out = blend(stage1, diffused, cfg.lam)
    return reinject_hf(out, input_img, cfg.alpha)

if __name__ == "__main__":
    # 使用示例：合成数组演示默认旋钮的融合输出范围
    rng = np.random.default_rng(0)
    s1 = rng.random((16, 16, 3), dtype=np.float32)
    d = rng.random((16, 16, 3), dtype=np.float32)
    inp = np.clip(s1 + 0.1, 0, 1)
    out = apply_knobs(s1, d, inp, KnobConfig())
    print(f"apply_knobs 输出范围: [{out.min():.2f}, {out.max():.2f}]")
