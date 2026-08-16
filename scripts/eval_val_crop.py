"""快速 val 检查：中心 1024² 裁剪跑全管线，量 PSNR/SSIM↔GT。微调迭代用（~30s/图 vs 全图 1.5hr）。

用法:
  python scripts/eval_val_crop.py --knobs 0.4,1.0,1.0,8.0 --tag ft1 [--crop 1024]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/liaitong/hw_comp")
sys.path.insert(0, "/home/liaitong/hw_comp/src")

VAL = Path("/home/liaitong/hw_comp/dataset/huawei/val")


def _psnr(p, r):
    mse = float(np.mean((p - r) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def _ssim(a, b):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knobs", type=str, default="0.4,1.0,1.0,8.0")
    ap.add_argument("--tag", type=str, default="ft")
    ap.add_argument("--crop", type=int, default=1024)
    args = ap.parse_args()
    lam, alpha, beta, sigma = (float(v) for v in args.knobs.split(","))

    from enhance.config import Config
    cfg = Config.from_yaml(Path("/home/liaitong/hw_comp/config.yaml"))
    cfg.lam, cfg.alpha, cfg.beta, cfg.sigma = lam, alpha, beta, sigma
    from enhance.inference.engine import EnhancementEngine
    engine = EnhancementEngine(cfg)

    from enhance.evaluate.metrics import niqe as _niqe  # 无参考感知代理（Gate C）

    print(f"{'case':>5} {'crop':>6} {'输入PSNR':>8} {'输出PSNR':>8} {'ΔPSNR':>7} {'ΔSSIM':>7} {'inNIQE':>7} {'outNIQE':>7}")
    for i in range(1, 6):
        lq = cv2.cvtColor(cv2.imread(str(VAL / f"case{i}_lq.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt = cv2.cvtColor(cv2.imread(str(VAL / f"case{i}_gt.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = lq.shape[:2]
        ps = min(args.crop, h, w)
        y, x = (h - ps) // 2, (w - ps) // 2
        lq_c, gt_c = lq[y:y + ps, x:x + ps], gt[y:y + ps, x:x + ps]
        p_in = _psnr(lq_c, gt_c)
        out = engine.enhance(lq_c)
        p_out, s_out = _psnr(out, gt_c), _ssim(out, gt_c)
        n_in, n_out = _niqe(lq_c, device="cpu"), _niqe(out, device="cpu")
        print(f"{i:>5} {ps:>6} {p_in:>8.2f} {p_out:>8.2f} {p_out - p_in:>+7.2f} {s_out - _ssim(lq_c, gt_c):>+7.3f} {n_in:>7.2f} {n_out:>7.2f}", flush=True)
        out_dir = Path(f"/tmp/evalcrop_{args.tag}"); out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"case{i}_out.png"), cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"case{i}_lq.png"), cv2.cvtColor((lq_c * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"case{i}_gt.png"), cv2.cvtColor((gt_c * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
