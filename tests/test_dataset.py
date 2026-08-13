import cv2
import numpy as np
import pytest
import torch

from enhance.data.dataset import EnhancementDataset, _shared_augment

def _img(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return (np.stack([xx, yy, (xx + yy) / 2], axis=-1) / 255.0)

def test_shared_augment_preserves_geometry():
    # 构造确定性关系 HR = LQ * 0.5；共享增强必须对两图施加同一几何变换（线性变换保持该关系）
    lq = np.zeros((16, 20, 3), dtype=np.float32)
    lq[:5] = 1.0
    hr = lq * 0.5
    for seed in range(8):
        rng = np.random.default_rng(seed)
        a, b = _shared_augment(lq.copy(), hr.copy(), rng)
        assert a.shape == b.shape
        assert a.shape in {(16, 20, 3), (20, 16, 3)}  # rot90 会交换 H/W
        assert np.allclose(a, b * 2.0, atol=1e-5)

def test_dataset_crop_alignment(tmp_path):
    # 写入真实 PNG 对（hr = lq * 0.5，空间梯度图），验证 __getitem__ 返回张量且输入/目标裁剪窗口对应
    lq = _img(64, 64)
    hr = lq * 0.5
    cv2.imwrite(str(tmp_path / "X_ARC.png"), cv2.cvtColor((lq * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(tmp_path / "X_ARC_gt.png"), cv2.cvtColor((hr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    ds = EnhancementDataset(tmp_path, patch_size=32, seed=0)
    inp, tgt = ds[0]
    assert inp.shape == tgt.shape == (3, 32, 32)
    assert inp.dtype == tgt.dtype == torch.float32
    assert 0.0 <= float(inp.min()) and float(inp.max()) <= 1.0
    assert 0.0 <= float(tgt.min()) and float(tgt.max()) <= 1.0
    # 输入/目标同位置裁剪：HR = LQ * 0.5 关系必须保持（atol 容忍 uint8 量化 + 立方插值振铃）
    assert torch.allclose(inp, tgt * 2.0, atol=2e-2)

def test_dataset_empty_root_raises(tmp_path):
    with pytest.raises(ValueError):
        EnhancementDataset(tmp_path)

def test_dataset_loads_from_npz_cache(tmp_path):
    # PNG 写零（若走解码则 input 全 0），npz 缓存写非零常量以证明走缓存路径
    root = tmp_path / "pairs"
    cache = tmp_path / "cache"
    root.mkdir()
    cache.mkdir()
    cv2.imwrite(str(root / "X_ARC.png"), np.zeros((32, 32, 3), dtype=np.uint8))
    cv2.imwrite(str(root / "X_ARC_gt.png"), np.zeros((32, 32, 3), dtype=np.uint8))
    np.savez(str(cache / "X_ARC.npz"),
             lq=np.full((32, 32, 3), 128, dtype=np.uint8),
             hr=np.full((32, 32, 3), 64, dtype=np.uint8))
    ds = EnhancementDataset(root, patch_size=16, seed=0, cache_root=cache)
    inp, tgt = ds[0]
    assert inp.shape == tgt.shape == (3, 16, 16)
    assert torch.allclose(inp, torch.full_like(inp, 128.0 / 255.0), atol=1e-3)
    assert torch.allclose(tgt, torch.full_like(tgt, 64.0 / 255.0), atol=1e-3)

def test_cache_is_bounded(tmp_path):
    # 构造 10 对真实 PNG，cache_pairs=3，反复访问不同对，缓存长度不得超过预算
    root = tmp_path / "pairs"
    root.mkdir()
    for i in range(10):
        lq = _img(32, 32)
        hr = lq * 0.5
        cv2.imwrite(str(root / f"p{i}_ARC.png"), cv2.cvtColor((lq * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(root / f"p{i}_ARC_gt.png"), cv2.cvtColor((hr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    ds = EnhancementDataset(root, patch_size=8, seed=0, length_factor=1, cache_pairs=3)
    for i in range(10):
        _ = ds[i]
    assert len(ds._decode_cache) <= 3
    assert len(ds._res_cache) <= 6
