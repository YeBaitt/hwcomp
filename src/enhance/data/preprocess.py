"""ImagePairs 预解压：把 LQ/HR 解码为 uint8 落盘 npz，供训练快速加载与色彩一致性快检。"""
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .dataset import load_pair_float
from .pairs import Pair, find_pairs, load_lq_hr, npz_cache_key
from .profile import estimate_profile

def _cache_one(args: Tuple[Pair, Path, Path]) -> int:
    """解码一对图像并保存为 uint8 npz，返回新增对数（损坏对跳过返回 0）。"""
    pair, pairs_root, cache_root = args
    try:
        lq, hr = load_lq_hr(pair)
    except Exception as err:  # 损坏/缺失图像对 → 跳过，避免拖垮整个缓存构建
        key = npz_cache_key(pair, pairs_root)
        print(f"[cache] 跳过损坏对 {key}: {err}")
        return 0
    lq_u8 = (np.clip(lq, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    hr_u8 = (np.clip(hr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    out = cache_root / f"{npz_cache_key(pair, pairs_root)}.npz"
    np.savez(str(out), lq=lq_u8, hr=hr_u8)
    return 1

def _load_bad_keys(pairs_root: Path) -> set:
    """读取 pairs_root 上级目录的 bad_pairs.txt 为键集合（不存在则返回空集）。"""
    bad_file = pairs_root.parent / "bad_pairs.txt"
    if not bad_file.exists():
        return set()
    return {line.strip() for line in bad_file.read_text(encoding="utf-8").splitlines() if line.strip()}

def build_pair_cache(pairs_root: Path, cache_root: Path, num_workers: int = 4) -> int:
    """为 pairs_root 下所有图像对构建 uint8 npz 缓存，返回新增缓存的对数。

    已存在的缓存文件会跳过（支持断点续跑）；bad_pairs.txt 中记录的坏对也会排除；
    用多进程并行解码加速。
    """
    pairs = find_pairs(pairs_root)
    if not pairs:
        return 0
    cache_root.mkdir(parents=True, exist_ok=True)
    bad_keys = _load_bad_keys(pairs_root)
    todo = [(pair, pairs_root, cache_root) for pair in pairs
            if npz_cache_key(pair, pairs_root) not in bad_keys
            and not (cache_root / f"{npz_cache_key(pair, pairs_root)}.npz").exists()]
    if not todo:
        return 0
    if num_workers <= 1:
        return sum(_cache_one(t) for t in todo)
    with Pool(num_workers) as pool:
        return sum(pool.imap_unordered(_cache_one, todo))

def _decode_ok(pair: Pair) -> bool:
    """尝试完整解码一对图像，任一损坏/缺失或解码异常则返回 False。"""
    try:
        lq, hr = load_lq_hr(pair)
    except Exception:
        return False
    return lq is not None and hr is not None

def verify_pairs(pairs_root: Path, num_workers: int = 4) -> list:
    """校验 pairs_root 下所有图像对是否可完整解码，返回坏对的缓存键列表。

    逐对尝试完整解码（load_lq_hr）；解码失败或返回 None 的对记为坏对，键写入
    pairs_root 上级目录的 bad_pairs.txt（UTF-8，每行一个，末尾换行；无坏对则写空文件）。
    """
    pairs = find_pairs(pairs_root)
    if num_workers <= 1:
        oks = [_decode_ok(p) for p in pairs]
    else:
        with Pool(num_workers) as pool:
            oks = pool.map(_decode_ok, pairs)
    bad_keys = [npz_cache_key(p, pairs_root) for p, ok in zip(pairs, oks) if not ok]
    bad_file = pairs_root.parent / "bad_pairs.txt"
    bad_file.write_text("\n".join(bad_keys) + ("\n" if bad_keys else ""), encoding="utf-8")
    print(f"[verify] 损坏/缺失 {len(bad_keys)}/{len(pairs)} 对")
    for k in bad_keys[:20]:
        print(f"  {k}")
    return bad_keys

def check_color_consistency(pairs_root: Path, cache_root: Path, sample_n: int = 50) -> None:
    """色彩一致性快检：抽样估计逐对增益/偏移，报告均值并标记可疑对。

    对每对图像先用双三次把 HR 缩到 LQ 分辨率，再用最小二乘估计 LQ→GT 逐通道
    增益/偏移（同 profile.estimate_profile）；增益超出 [0.5, 2.0] 或偏移超出
    [-0.2, 0.2] 的对标记为可疑。
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
        lq, hr = load_pair_float(pair, cache_root, pairs_root)
        hr = cv2.resize(hr, (lq.shape[1], lq.shape[0]))
        p = estimate_profile(lq, hr)
        gain_abs.append(np.abs(p.color_gain - 1.0))
        off_abs.append(np.abs(p.color_offset))
        bad = bool(np.any((p.color_gain < 0.5) | (p.color_gain > 2.0))
                   or np.any((p.color_offset < -0.2) | (p.color_offset > 0.2)))
        if bad:
            suspicious.append((npz_cache_key(pair, pairs_root), p.color_gain, p.color_offset))
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
        bad = verify_pairs(root, num_workers=1)
        assert not bad, f"合成对应无坏对: {bad}"
        n = build_pair_cache(root, cache, num_workers=1)
        print(f"build_pair_cache 新增缓存 {n} 对")
        check_color_consistency(root, cache, sample_n=1)
