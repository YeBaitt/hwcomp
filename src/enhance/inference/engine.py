"""4K 增强推理引擎：stage1 → stage2 分块 → 缝合 → 旋钮融合。"""
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from enhance.config import Config
from enhance.fusion.knobs import KnobConfig, apply_knobs
from enhance.inference.tiler import accumulate_tile, finalize, tile_weights, tiles_for
from enhance.model.stage1 import load_nafnet
from enhance.model.stage2 import stage2_refine

class EnhancementEngine:
    """4K 增强推理引擎。Stage-1 模型惰性加载并缓存，避免重复载权。

    默认路径：整图一次 stage2_refine（内部走 DiffBIR cldm_tiled 4K 重叠平铺，进程内完成）。
    整图失败时回退到逐块 _stage2_tiled 缝合路径。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.knobs = KnobConfig(lam=cfg.lam, n=cfg.n, w=cfg.w, alpha=cfg.alpha,
                                beta=cfg.beta, sigma=cfg.sigma)
        self._stage1_model = None

    def _stage1_ckpt(self) -> Path:
        """优先使用训练产出的 stage1_best.pth，缺失时退回预训练权重。"""
        best = Path(self.cfg.ckpt_root) / "stage1_best.pth"
        if best.exists():
            return best
        return Path(self.cfg.stage1_pretrained)

    def _stage1(self, img: np.ndarray) -> np.ndarray:
        """惰性加载 stage-1 模型并对输入图分块推理，缝合后裁剪到 [0,1]。

        4K 全图一次前向会 OOM（NAFNet SimpleGate 内部 128 通道中间张量在 3072×4096
        分辨率需 ~3 GB/块），因此按 tile_size 分块推理、重叠加权缝合。
        """
        if self._stage1_model is None:
            self._stage1_model = load_nafnet(str(self._stage1_ckpt()), self.device)
        h, w = img.shape[:2]
        # 图像小于 tile_size 时直接全图推理
        if h <= self.cfg.tile_size and w <= self.cfg.tile_size:
            return self._stage1_full(img)
        canvas = np.zeros_like(img)
        ws = np.zeros((h, w), dtype=np.float32)
        tiles = tiles_for(h, w, self.cfg.tile_size, self.cfg.overlap)
        for rect, weight in zip(tiles, tile_weights(tiles, (h, w))):
            y0, y1, x0, x1 = rect
            tile = img[y0:y1, x0:x1]
            out = self._stage1_full(tile)
            accumulate_tile(canvas, ws, rect, out, weight)
        return finalize(canvas, ws)

    def _stage1_full(self, img: np.ndarray) -> np.ndarray:
        """对单块（或小图）做一次 NAFNet 前向推理，输出裁剪到 [0,1]。"""
        x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(self.device)
        with torch.no_grad():
            if self.device.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    y = self._stage1_model(x)
            else:
                y = self._stage1_model(x)
        # NAFNet 输出是残差（可越界），作为 stage2 控制图与旋钮输入前必须裁剪到 [0,1]
        return np.clip(y[0].permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0)

    def _stage2_tiled(self, img: np.ndarray, tmp: Path) -> np.ndarray:
        """逐块 stage2（保底）：分块 → 每块 stage2_refine → 分区缝合。整图路径失败时退回。"""
        h, w = img.shape[:2]
        canvas = np.zeros_like(img)
        ws = np.zeros((h, w), dtype=np.float32)
        tiles = tiles_for(h, w, self.cfg.tile_size, self.cfg.overlap)
        for rect, w in zip(tiles, tile_weights(tiles, (h, w))):
            y0, y1, x0, x1 = rect
            tile = img[y0:y1, x0:x1]
            out = stage2_refine(tile, tmp / f"{y0}_{x0}.png",
                                steps=self.cfg.steps, guidance=self.knobs.w,
                                tile_size=self.cfg.tile_size, stride=self.cfg.tile_size - self.cfg.overlap)
            accumulate_tile(canvas, ws, rect, out, w)
        return finalize(canvas, ws)

    def _stage2(self, img: np.ndarray, tmp: Optional[Path] = None) -> np.ndarray:
        """整图一次 stage2_refine（内部 cldm_tiled 4K 进程内平铺），失败时退回逐块缝合。

        供 Task 11 旋钮网格重用：缓存的 diffused 结果可在不同 (λ,α) 下廉价重组，
        无需重复运行扩散或退化到慢速逐块路径。tmp 为 None 时自动创建临时目录。
        """
        _cleanup = tmp is None
        if tmp is None:
            tmp = Path(tempfile.mkdtemp())
        try:
            try:
                return stage2_refine(img, tmp / "stage2.png", steps=self.cfg.steps, guidance=self.knobs.w,
                                     tile_size=self.cfg.tile_size, stride=self.cfg.tile_size - self.cfg.overlap)
            except Exception as exc:
                print(f"整图 stage2 失败（{type(exc).__name__}: {exc}），退回逐块路径")
                return self._stage2_tiled(img, tmp)
        finally:
            if _cleanup:
                shutil.rmtree(tmp, ignore_errors=True)

    def enhance(self, img: np.ndarray) -> np.ndarray:
        """执行完整的两阶段增强流水线，返回 RGB float32 [0,1] 增强图像。

        首先调用 stage-1 NAFNet 打底，然后将结果作为 control 图送入 stage-2 扩散细化，
        最后通过旋钮融合得到最终输出。
        """
        tmp = Path(tempfile.mkdtemp())
        try:
            s1 = self._stage1(img)
            d = self._stage2(s1, tmp)
            return apply_knobs(s1, d, img, self.knobs)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def enhance_path(self, in_path: Path, out_path: Path) -> None:
        """从路径读取图像、增强、写入输出路径（JPEG 质量 95）。"""
        bgr = cv2.imread(str(in_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = self.enhance(rgb)
        out_bgr = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

if __name__ == "__main__":
    # 使用示例：构造配置对合成小块图像执行增强流水线
    from enhance.config import Config

    cfg = Config.from_yaml(Path(__file__).resolve().parents[3] / "config.yaml")
    engine = EnhancementEngine(cfg)
    rng = np.random.default_rng(0)
    img = rng.random((64, 64, 3), dtype=np.float32)
    out = engine.enhance(img)
    print(f"合成输入 shape={img.shape}，增强输出 shape={out.shape}")
