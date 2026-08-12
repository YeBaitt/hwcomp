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
