"""4K 分块与重叠加权融合（像素域，partition-of-unity）。"""
from typing import List, Tuple

import numpy as np

def tiles_for(height: int, width: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    """固定步长铺块，最后一块对齐图像边缘；返回 (y0,y1,x0,x1) 列表。"""
    def spans(L: int, T: int) -> List[Tuple[int, int]]:
        step = max(1, T - overlap)
        starts = list(range(0, max(1, L - T), step))
        if L - T not in starts:  # 最后一块对齐图像边缘（去重：L==T 时 range 已含 0）
            starts.append(L - T)
        return [(s, min(s + T, L)) for s in starts]

    th, tw = min(tile_size, height), min(tile_size, width)
    ys, xs = spans(height, th), spans(width, tw)
    return [(y0, y1, x0, x1) for (y0, y1) in ys for (x0, x1) in xs]

def tile_weights(tiles: List[Tuple[int, int, int, int]], image_shape: Tuple[int, int]) -> List[np.ndarray]:
    """为每个 tile 生成 partition-of-unity 权重块。

    相邻两块在重叠区线性互补（前块 1→0、后块 0→1），
    贴图像边界的一侧权重恒为 1，避免黑边。重叠宽度按实际相邻起点差自适应。
    """
    T = tiles[0][1] - tiles[0][0]
    ystarts = sorted({t[0] for t in tiles})
    xstarts = sorted({t[2] for t in tiles})
    return [_tile_axis_weight(y0, y1, ystarts, T)[:, None] * _tile_axis_weight(x0, x1, xstarts, T)[None, :]
            for (y0, y1, x0, x1) in tiles]

def _tile_axis_weight(s: int, e: int, starts: List[int], T: int) -> np.ndarray:
    """单个 tile 的一维权重：与上块重叠处 0→1、与下块重叠处 1→0，其余为 1。"""
    w = np.ones(e - s)
    idx = starts.index(s)
    if idx > 0:
        o = starts[idx - 1] + T - s
        w[:o] = np.linspace(0.0, 1.0, o)
    if idx < len(starts) - 1:
        o = s + T - starts[idx + 1]
        w[-o:] = np.linspace(1.0, 0.0, o)
    return w

def accumulate_tile(canvas: np.ndarray, weight_sum: np.ndarray, rect: Tuple[int, int, int, int],
                    tile: np.ndarray, weights: np.ndarray) -> None:
    """把单个 tile 乘权重后就地累加进 canvas 与 weight_sum。"""
    y0, y1, x0, x1 = rect
    canvas[y0:y1, x0:x1] += tile * weights[..., None]
    weight_sum[y0:y1, x0:x1] += weights

def finalize(canvas: np.ndarray, weight_sum: np.ndarray) -> np.ndarray:
    """按权重和归一化 canvas，返回缝合后的完整图像。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(canvas, weight_sum[..., None], out=canvas, where=weight_sum[..., None] > 0)
    return out

if __name__ == "__main__":
    # 使用示例：常量图铺块、加权累加并按权重和归一化还原
    img = np.full((64, 96, 3), 0.5, dtype=np.float32)
    canvas = np.zeros_like(img)
    ws = np.zeros((64, 96), dtype=np.float32)
    tiles = tiles_for(64, 96, 32, 8)
    for rect, w in zip(tiles, tile_weights(tiles, (64, 96))):
        y0, y1, x0, x1 = rect
        accumulate_tile(canvas, ws, rect, img[y0:y1, x0:x1], w)
    out = finalize(canvas, ws)
    print(f"tiles: {len(tiles)}，重建误差 max={np.abs(out - 0.5).max():.2e}")
