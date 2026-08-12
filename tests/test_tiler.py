import numpy as np

from enhance.inference.tiler import accumulate_tile, finalize, tile_weights, tiles_for

def test_tiles_cover_every_pixel():
    tiles = tiles_for(64, 96, tile_size=32, overlap=8)
    covered = np.zeros((64, 96), dtype=bool)
    for y0, y1, x0, x1 in tiles:
        covered[y0:y1, x0:x1] = True
    assert covered.all()

def test_stitch_reconstructs_constant():
    # 常量图经任意网格缝合后逐像素还原 → partition-of-unity 成立（含图像边界与贴底块的大重叠区）
    img = np.full((64, 96, 3), 0.7, dtype=np.float32)
    canvas = np.zeros_like(img)
    ws = np.zeros((64, 96), dtype=np.float32)
    tiles = tiles_for(64, 96, 32, 8)
    for rect, w in zip(tiles, tile_weights(tiles, (64, 96))):
        y0, y1, x0, x1 = rect
        accumulate_tile(canvas, ws, rect, img[y0:y1, x0:x1], w)
    out = finalize(canvas, ws)
    assert np.abs(out - 0.7).max() < 1e-4
    assert np.isfinite(out).all()  # 无 NaN/Inf

def test_tile_weights_boundary_one():
    # 边界权重和恒为 1（partition-of-unity）→ 顶/底边逐像素无黑边。
    # 注意：单块在顶/底边的权重沿另一轴仍带接缝斜坡（如顶块 w[0,:] 右侧 1→0），
    # 因此校验整行权重和而非单块全行恒 1。
    tiles = tiles_for(64, 96, 32, 8)
    total = np.zeros((64, 96), dtype=np.float32)
    for rect, w in zip(tiles, tile_weights(tiles, (64, 96))):
        y0, y1, x0, x1 = rect
        total[y0:y1, x0:x1] += w
    assert np.allclose(total[0, :], 1.0)   # 顶边权重和为 1，避免黑边
    assert np.allclose(total[-1, :], 1.0)  # 底边权重和为 1

def test_tile_weights_internal_ramp_zero():
    tiles = tiles_for(64, 96, 32, 8)
    for rect, w in zip(tiles, tile_weights(tiles, (64, 96))):
        y0, y1, x0, x1 = rect
        if y0 > 0 and y1 < 64 and x0 > 0 and x1 < 96:
            assert w[0, 0] == 0.0 and w[-1, -1] == 0.0  # 纯内部块四角在接缝处权重为 0
