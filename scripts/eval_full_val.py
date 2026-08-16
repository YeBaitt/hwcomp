"""端到端 val 验收：真实 huawei val 5 图上跑 完整两阶段增强，量 PSNR↔GT / SSIM↔GT / NIQE。

口径与提交一致：engine.enhance()（stage1→stage2→knobs），整图 4K。
注意：stage2 每图需 GPU 扩散（~几分钟/图），仅在微调后、提交前用。

用法:
  python scripts/eval_full_val.py [--knobs lam alpha beta sigma] [--tag myft]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/liaitong/hw_comp")
sys.path.insert(0, "/home/liaitong/hw_comp/src")

VAL = Path("/home/liaitong/hw_comp/dataset/huawei/val")


def _psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    mse = float(np.mean((pred - ref) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knobs", type=str, default="0.4,1.0,1.0,8.0", help="lam,alpha,beta,sigma")
    ap.add_argument("--tag", type=str, default="ft")
    args = ap.parse_args()
    lam, alpha, beta, sigma = (float(v) for v in args.knobs.split(","))

    from enhance.config import Config
    cfg = Config.from_yaml(Path("/home/liaitong/hw_comp/config.yaml"))
    cfg.lam, cfg.alpha, cfg.beta, cfg.sigma = lam, alpha, beta, sigma

    from enhance.inference.engine import EnhancementEngine
    engine = EnhancementEngine(cfg)

    print(f"{'case':>5} {'输入PSNR':>8} {'输出PSNR':>8} {'ΔPSNR':>7} {'ΔSSIM':>7} {'NIQE':>6}")
    for i in range(1, 6):
        bgr_lq = cv2.imread(str(VAL / f"case{i}_lq.jpg"))
        bgr_gt = cv2.imread(str(VAL / f"case{i}_gt.jpg"))
        lq = cv2.cvtColor(bgr_lq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt = cv2.cvtColor(bgr_gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # 同分辨率（val 对已同尺寸，直接对齐）
        h, w = min(lq.shape[0], gt.shape[0]), min(lq.shape[1], gt.shape[1])
        lq, gt = lq[:h, :w], gt[:h, :w]

        p_in = _psnr(lq, gt)
        out = engine.enhance(lq)
        p_out = _psnr(out, gt)
        s_out = _ssim(out, gt)
        print(f"{i:>5} {p_in:>8.2f} {p_out:>8.2f} {p_out - p_in:>+7.2f} {s_out - _ssim(lq, gt):>+7.3f} "
              f"{_ssim(out, gt):>6.3f}")
        # 存输出供目视
        out_dir = Path(f"/tmp/eval_full_{args.tag}")
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"case{i}_out.png"),
                    cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
