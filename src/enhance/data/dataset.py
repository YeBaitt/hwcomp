"""PatchDataset：2K/3.5K 混合采样 + 共享增强，返回 (input, target) 张量。"""
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .pairs import Pair, find_pairs, load_lq_hr, npz_cache_key, to_same_res

MAX_CACHED_PAIRS = 256

def load_pair_float(pair: Pair, cache_root: Optional[Path] = None,
                    pairs_root: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """加载一对图像为 float32 [0,1]：优先读 npz uint8 缓存，否则解码 PNG。"""
    if cache_root is not None:
        key = npz_cache_key(pair, pairs_root) if pairs_root is not None else pair.lq_path.stem
        cache_file = cache_root / f"{key}.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            lq = data["lq"].astype(np.float32) / 255.0
            hr = data["hr"].astype(np.float32) / 255.0
            return lq, hr
    return load_lq_hr(pair)

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

def _lru_get(cache: "OrderedDict", key) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """命中则移到队尾并返回，否则返回 None。"""
    if key not in cache:
        return None
    cache.move_to_end(key)
    return cache[key]

def _lru_put(cache: "OrderedDict", key, value, budget: int) -> None:
    """写入 LRU 缓存并在条目数超出 budget 时淘汰最久未使用项。"""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > budget:
        cache.popitem(last=False)

class EnhancementDataset(Dataset):
    """2K/3.5K 混合采样数据集：随机取一对图像，共享增强后同位置裁剪出 patch 张量。

    内置两级 LRU 有界缓存：_decode_cache 缓存解码后的 float32 原图，
    _res_cache 缓存 to_same_res 输出。缓存按“图像对数量”限制（cache_pairs），
    而非按字节数：每个条目是整幅 float32 图像（解码约 119MB 起、35k 重采样约 190MB），
    因此峰值内存约为 cache_pairs × 单对平均字节数（如 128 对可达数十 GB），
    但上界不随图像对总数增长而 OOM；也可通过 cache_root 读取预解压的 uint8 npz 加速解码。
    """

    def __init__(self, pairs_root: Path, patch_size: int = 256, kind_2k_weight: float = 0.7,
                 seed: int = 42, length_factor: int = 64, cache_pairs: int = MAX_CACHED_PAIRS,
                 cache_root: Optional[Path] = None, pairs: Optional[List[Pair]] = None):
        """扫描 pairs_root 下图像对，并记录采样参数与随机数生成器。

        length_factor 控制每对图像生成的样本数，默认 64（保证全测试兼容）。
        cache_pairs 限制内存中缓存的最大图像对数（按对数计，非字节数）；cache_root 指向预解压 npz 缓存目录。
        pairs 可显式指定子集（如训练/验证划分），默认扫描全部。
        """
        self.pairs_root = pairs_root
        self.pairs = pairs if pairs is not None else find_pairs(pairs_root)
        if not self.pairs:
            raise ValueError(f"目录下未找到图像对: {pairs_root}")
        self.patch_size = patch_size
        self.kind_2k_weight = kind_2k_weight
        self.rng = np.random.default_rng(seed)
        self._length_factor = length_factor
        self._cache_pairs = cache_pairs
        self.cache_root = cache_root
        self._decode_cache: "OrderedDict[int, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self._res_cache: "OrderedDict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

    def __len__(self):
        """返回数据集长度（至少 64，随图像对数量线性增长）。"""
        return max(64, len(self.pairs) * self._length_factor)

    def _crop(self, img: np.ndarray, y: int, x: int, ps: int) -> np.ndarray:
        """按给定裁剪位置取 ps×ps 子块（输入与目标必须用同一位置）。"""
        return img[y:y + ps, x:x + ps]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """取第 idx 个样本，返回 (input, target) 的 (3,H,W) float32 张量。"""
        pair_idx = idx % len(self.pairs)
        kind = "2k" if self.rng.random() < self.kind_2k_weight else "35k"

        # 一级缓存：解码后的 float32 原图（LRU 有界）
        decoded = _lru_get(self._decode_cache, pair_idx)
        if decoded is None:
            decoded = load_pair_float(self.pairs[pair_idx], self.cache_root, self.pairs_root)
            _lru_put(self._decode_cache, pair_idx, decoded, self._cache_pairs)
        lq, hr = decoded

        # 二级缓存：to_same_res 输出（2k/35k 两种分辨率，LRU 有界）
        cache_key = (pair_idx, kind)
        res = _lru_get(self._res_cache, cache_key)
        if res is None:
            res = to_same_res(lq, hr, kind)
            _lru_put(self._res_cache, cache_key, res, self._cache_pairs * 2)
        inp, target = res

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
