import cv2
import numpy as np
import pytest

from enhance.data.pairs import find_pairs, to_same_res

def _img(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    return (((np.sin(yy / 5) * np.cos(xx / 4)) + 1) / 2).astype(np.float32)[..., None].repeat(3, axis=-1)

def test_to_same_res_2k_shape_and_value():
    lq = _img(32, 48)
    hr = _img(64, 96)
    inp, target = to_same_res(lq, hr, "2k")
    assert inp.shape == target.shape == (32, 48, 3)
    assert inp.dtype == target.dtype == np.float32
    # 2k 语义：target = hr 双三次缩到 lq 尺寸并裁剪到 [0,1]（锁定语义，防实现漂移）
    expected = np.clip(cv2.resize(hr, (48, 32), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)
    assert np.allclose(target, expected, atol=1e-6)
    assert 0.0 <= target.min() and target.max() <= 1.0
    assert 0.0 <= inp.min() and inp.max() <= 1.0

def test_to_same_res_35k_shape():
    lq = _img(32, 48)
    hr = _img(64, 96)
    inp, target = to_same_res(lq, hr, "35k")
    assert inp.shape == target.shape == (64, 96, 3)

def test_to_same_res_unknown_kind_raises():
    lq = _img(8, 8)
    hr = _img(16, 16)
    with pytest.raises(ValueError):
        to_same_res(lq, hr, "4k")

def test_find_pairs(tmp_path):
    for i in range(3):
        (tmp_path / f"a{i}_ARC.png").write_bytes(b"x")
        (tmp_path / f"a{i}_ARC_gt.png").write_bytes(b"x")
    (tmp_path / "lonely.png").write_bytes(b"x")
    pairs = find_pairs(tmp_path)
    assert len(pairs) == 3
