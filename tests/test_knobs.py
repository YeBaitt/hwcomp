import numpy as np

from enhance.fusion.knobs import KnobConfig, apply_knobs, blend, reinject_hf

def test_blend_endpoints():
    a = np.zeros((8, 8, 3), dtype=np.float32)
    b = np.ones((8, 8, 3), dtype=np.float32)
    assert np.allclose(blend(a, b, 0.0), a)
    assert np.allclose(blend(a, b, 1.0), b)

def test_reinject_zero_alpha_identity():
    a = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32)
    assert np.allclose(reinject_hf(a, a, 0.0), a, atol=1e-6)

def test_apply_knobs_output_range():
    s1 = np.zeros((16, 16, 3), dtype=np.float32)
    d = np.ones((16, 16, 3), dtype=np.float32)
    inp = np.full((16, 16, 3), 0.5, dtype=np.float32)
    out = apply_knobs(s1, d, inp, KnobConfig(lam=0.5, alpha=0.5))
    assert out.min() >= 0.0 and out.max() <= 1.0
