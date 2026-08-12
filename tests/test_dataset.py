import numpy as np
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
