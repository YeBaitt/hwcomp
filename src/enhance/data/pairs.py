"""ImagePairs 真实双相机对解析与 2K/3.5K 同分辨率构造。"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

@dataclass
class Pair:
    """一对图像路径：lq_path 为退化图（如 X_ARC.png），hr_path 为参考原图（X_ARC_gt.png）。"""
    lq_path: Path
    hr_path: Path

def find_pairs(root: Path) -> list:
    """扫描 root 下所有 *.png，配对 *_ARC.png 与 *_ARC_gt.png，返回 Pair 列表。"""
    pairs = []
    for lq in sorted(root.rglob("*.png")):
        if lq.name.endswith("_gt.png"):
            continue
        gt = lq.with_name(lq.stem + "_gt.png")
        if gt.exists():
            pairs.append(Pair(lq_path=lq, hr_path=gt))
    return pairs

def load_lq_hr(pair: Pair) -> Tuple[np.ndarray, np.ndarray]:
    """读取一对图像为 RGB float32 数组，范围 [0,1]，返回 (lq, hr)。"""
    lq = cv2.imread(str(pair.lq_path))
    hr = cv2.imread(str(pair.hr_path))
    lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return lq, hr

def to_same_res(lq: np.ndarray, hr: np.ndarray, kind: str) -> Tuple[np.ndarray, np.ndarray]:
    """把 lq/hr 缩放到同一尺寸，返回 (input, target)：2k 取 lq 尺寸，35k 取 hr 尺寸。"""
    if kind == "2k":
        target = cv2.resize(hr, (lq.shape[1], lq.shape[0]), interpolation=cv2.INTER_CUBIC)
        return lq, target
    if kind == "35k":
        inp = cv2.resize(lq, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
        return inp, hr
    raise ValueError(f"未知 kind: {kind}")

if __name__ == "__main__":
    # 使用示例：构造两幅不同尺寸的假图，演示 2K / 3.5K 同分辨率构造
    lq = np.random.rand(32, 48, 3).astype(np.float32)
    hr = np.random.rand(64, 96, 3).astype(np.float32)
    inp, target = to_same_res(lq, hr, "2k")
    print(f"2k: input={inp.shape} target={target.shape}")
    inp, target = to_same_res(lq, hr, "35k")
    print(f"35k: input={inp.shape} target={target.shape}")
