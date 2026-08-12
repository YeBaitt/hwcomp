import numpy as np
import pytest

from enhance.model.stage2 import stage2_refine

@pytest.mark.gpu
def test_stage2_refine_shapes(tmp_path):
    img = np.random.default_rng(0).random((256, 256, 3), dtype=np.float32)
    out = stage2_refine(img, tmp_path / "out.png", steps=5, guidance=1.0, tile_size=256, stride=128)
    assert out.shape == img.shape
    assert 0.0 <= out.min() and out.max() <= 1.0
