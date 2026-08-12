import numpy as np

from enhance.evaluate.metrics import psnr, ssim

def test_psnr_identical_is_inf():
    a = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32)
    assert psnr(a, a) == float("inf")

def test_psnr_noisy_smaller():
    rng = np.random.default_rng(1)
    a = rng.random((32, 32, 3), dtype=np.float32)
    b = np.clip(a + 0.05, 0, 1)
    assert psnr(a, b) < psnr(a, a)
    assert 10 < psnr(a, b) < 40

def test_ssim_range():
    rng = np.random.default_rng(2)
    a = rng.random((32, 32, 3), dtype=np.float32)
    b = np.clip(a + 0.05, 0, 1)
    s = ssim(a, b)
    assert 0.0 <= s <= 1.0
    assert ssim(a, a) > ssim(a, b)
