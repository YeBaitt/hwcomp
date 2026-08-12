"""PatchDataset：2K/3.5K 混合采样 + 共享增强，返回 (input, target) 张量。"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .pairs import find_pairs, load_lq_hr, to_same_res

def _shared_augment(lq: np.ndarray, hr: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """对 lq/hr 施加同一随机几何变换（翻转/90° 旋转），返回连续数组。"""
    if rng.random() < 0.5:
        lq, hr = lq[:, ::-1], hr[:, ::-1]
    if rng.random() < 0.5:
        lq, hr = lq[::-1], hr[::-1]
    k = int(rng.integers(0, 4))
    if k:
        lq, hr = np.rot90(lq, k), np.rot90(hr, k)
    return np.ascontiguousarray(lq), np.ascontiguousarray(hr)

class EnhancementDataset(Dataset):
    """2K/3.5K 混合采样数据集：随机取一对图像，共享增强后同位置裁剪出 patch 张量。"""

    def __init__(self, pairs_root: Path, patch_size: int = 256, kind_2k_weight: float = 0.7, seed: int = 42):
        """扫描 pairs_root 下图像对，并记录采样参数与随机数生成器。"""
        self.pairs = find_pairs(pairs_root)
        self.patch_size = patch_size
        self.kind_2k_weight = kind_2k_weight
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        """返回数据集长度（至少 64，随图像对数量线性增长）。"""
        return max(64, len(self.pairs) * 64)

    def _crop(self, img: np.ndarray, y: int, x: int, ps: int) -> np.ndarray:
        """按给定裁剪位置取 ps×ps 子块（输入与目标必须用同一位置）。"""
        return img[y:y + ps, x:x + ps]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """取第 idx 个样本，返回 (input, target) 的 (3,H,W) float32 张量。"""
        pair = self.pairs[idx % len(self.pairs)]
        lq, hr = load_lq_hr(pair)
        kind = "2k" if self.rng.random() < self.kind_2k_weight else "35k"
        inp, target = to_same_res(lq, hr, kind)
        inp, target = _shared_augment(inp, target, self.rng)
        h, w = inp.shape[:2]
        ps = min(self.patch_size, h, w)
        y = int(self.rng.integers(0, h - ps + 1))
        x = int(self.rng.integers(0, w - ps + 1))
        inp, target = self._crop(inp, y, x, ps), self._crop(target, y, x, ps)
        inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).float()
        target_t = torch.from_numpy(target.transpose(2, 0, 1)).float()
        return inp_t, target_t

if __name__ == "__main__":
    # 使用示例：共享增强对输入/目标施加同一几何变换，保持 HR = LQ * 0.5 的线性关系
    lq = np.zeros((16, 20, 3), dtype=np.float32)
    lq[:5] = 1.0
    hr = lq * 0.5
    rng = np.random.default_rng(0)
    a, b = _shared_augment(lq.copy(), hr.copy(), rng)
    print(f"shared augment: input={a.shape} target={b.shape} keep={bool(np.allclose(a, b * 2.0, atol=1e-5))}")
