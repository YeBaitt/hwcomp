import numpy as np

from enhance.data.profile import apply_color_normalize, estimate_profile


def _ramp(h, w):
    yy, xx = np.mgrid[0:h, 0:w] / 255.0
    base = np.stack([xx, yy, (xx + yy) / 2], axis=-1).astype(np.float32)
    return base


def test_estimate_recovers_gain_offset():
    rng = np.random.default_rng(0)
    hr = _ramp(64, 64)
    gain = np.array([1.1, 0.9, 1.0], dtype=np.float32)
    off = np.array([0.05, -0.03, 0.0], dtype=np.float32)
    lq = hr * gain + off  # 构造退化：GT→LQ
    p = estimate_profile(lq, hr)
    # Profile 记录 LQ→GT 增益/偏移（设计文档 §4.2），故恢复的是 1/gain 与 -off/gain
    assert np.allclose(p.color_gain, 1.0 / gain, atol=1e-2)
    assert np.allclose(p.color_offset, -off / gain, atol=1e-2)


def test_apply_normalize_aligns_mean():
    rng = np.random.default_rng(1)
    hr = _ramp(48, 64)
    lq = hr * 1.15 + 0.06
    p = estimate_profile(lq, hr)
    out = apply_color_normalize(lq, p)
    assert abs(out.mean() - hr.mean()) < 1e-3
    assert out.min() >= 0.0 and out.max() <= 1.0
