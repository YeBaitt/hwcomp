"""ImagePairs 预解压：把 LQ/HR 解码为 uint8 落盘 npz，供训练快速加载与色彩一致性快检。"""
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .dataset import load_pair_float
from .pairs import Pair, find_pairs, load_lq_hr
from .profile import estimate_profile

def _cache_one(args: Tuple[Pair, Path]) -> int:
    """解码一对图像并保存为 uint8 npz，返回新增对数（恒为 1）。"""
    pair, cache_root = args
    lq, hr = load_lq_hr(pair)
    lq_u8 = (np.clip(lq, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    hr_u8 = (np.clip(hr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    out = cache_root / f"{pair.lq_path.stem}.npz"
    np.savez(str(out), lq=lq_u8, hr=hr_u8)
    return 1

def build_pair_cache(pairs_root: Path, cache_root: Path, num_workers: int = 4) -> int:
    """为 pairs_root 下所有图像对构建 uint8 npz 缓存，返回新增缓存的对数。

    已存在的缓存文件会跳过（支持断点续跑）；用多进程并行解码加速。
    """
    pairs = find_pairs(pairs_root)
    if not pairs:
        return 0
    cache_root.mkdir(parents=True, exist_ok=True)
    todo = [(pair, cache_root) for pair in pairs
            if not (cache_root / f"{pair.lq_path.stem}.npz").exists()]
    if not todo:
        return 0
    if num_workers <= 1:
        return sum(_cache_one(t) for t in todo)
    with Pool(num_workers) as pool:
        return sum(pool.imap_unordered(_cache_one, todo))

def check_color_consistency(pairs_root: Path, cache_root: Path, sample_n: int = 50) -> None:
    """色彩一致性快检：抽样估计逐对增益/偏移，报告均值并标记可疑对。

    对每对图像用最小二乘估计 LQ→GT 逐通道增益/偏移（同 profile.estimate_profile），
    增益超出 [0.5, 2.0] 或偏移超出 [-0.2, 0.2] 的对标记为可疑。
    """
    pairs = find_pairs(pairs_root)
    if not pairs:
        print("[色彩快检] 未找到图像对")
        return
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pairs), size=min(sample_n, len(pairs)), replace=False)
    gain_abs = []
    off_abs = []
    suspicious = []
    for i in idx:
        pair = pairs[int(i)]
        lq, hr = load_pair_float(pair, cache_root)
        p = estimate_profile(lq, hr)
        gain_abs.append(np.abs(p.color_gain - 1.0))
        off_abs.append(np.abs(p.color_offset))
        bad = bool(np.any((p.color_gain < 0.5) | (p.color_gain > 2.0))
                   or np.any((p.color_offset < -0.2) | (p.color_offset > 0.2)))
        if bad:
            suspicious.append((pair.lq_path.stem, p.color_gain, p.color_offset))
    print(f"[色彩快检] 抽样 {len(idx)} 对，mean|gain-1|={float(np.mean(gain_abs)):.4f} "
          f"mean|offset|={float(np.mean(off_abs)):.4f}")
    if suspicious:
        print(f"[色彩快检] 可疑对 {len(suspicious)} 个:")
        for stem, g, o in suspicious[:10]:
            print(f"  {stem}: gain={g} offset={o}")
    else:
        print("[色彩快检] 未发现可疑对")

if __name__ == "__main__":
    # 使用示例：构造一对合成小图，演示缓存构建与色彩快检（不触碰真实 11k 数据）
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "pairs"
        cache = Path(tmp) / "cache"
        root.mkdir()
        yy, xx = np.mgrid[0:32, 0:48]
        base = ((np.sin(yy / 5) * np.cos(xx / 4) + 1) / 2).astype(np.float32)
        lq = np.stack([base, base * 0.9, base * 1.1], axis=-1)
        hr = np.clip(lq * 1.1 + 0.03, 0.0, 1.0)
        cv2.imwrite(str(root / "demo_ARC.png"), cv2.cvtColor((lq * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(root / "demo_ARC_gt.png"), cv2.cvtColor((hr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        n = build_pair_cache(root, cache, num_workers=1)
        print(f"build_pair_cache 新增缓存 {n} 对")
        check_color_consistency(root, cache, sample_n=1)
