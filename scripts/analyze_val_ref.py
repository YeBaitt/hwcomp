"""val 代理参考指标：用缓存重建两份提交配置，算 PSNR/SSIM/LPIPS↔GT + NIQE/BRISQUE/MUSIQ。

动机：test 100 张无 GT，参考指标只能在 val 上算代理。此处用 knob 调优时缓存的
val stage1/stage2 输出（每张图的 lq/gt/s1/s2_w2.0 npy），按恒等式在纯 CPU 上重建
"my_work（λ0.2/α0.5/β0）" 与 "probe（λ0.4/α1.0/β1.0）" 两档输出，与提交逐字节同源，
无需再跑 GPU 扩散。

前置（一次性）：val 5 张图的 GPU 缓存 npy 在 --cache 目录，键名为
  case{N}_{lq,gt,s1,s2_w2.0}.npy
（由 knob 网格调优时先整图跑一遍 stage1+stage2 并落盘产生，分辨率通常 1024² 中心裁剪）。

用法:
  python scripts/analyze_val_ref.py --cache /tmp/knob_sweep --out /tmp/val_metrics.json
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from enhance.fusion.knobs import blend, lf_anchor
from skimage.metrics import structural_similarity

CACHE = Path("/tmp/knob_sweep")


def _psnr(p, r):
    mse = float(np.mean((p.astype(np.float64) - r.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def _ssim(p, r):
    return float(structural_similarity(p, r, channel_axis=2, data_range=1.0))


def _lpips(a, b):
    import lpips
    lp = lpips.LPIPS(net="alex")
    ta = torch.from_numpy(a.transpose(2, 0, 1))[None] * 2 - 1
    tb = torch.from_numpy(b.transpose(2, 0, 1))[None] * 2 - 1
    with torch.no_grad():
        return float(lp(ta, tb).item())


def _nr(name, img):
    import pyiqa
    m = pyiqa.create_metric(name, device="cuda")
    with torch.no_grad():
        return float(m(torch.from_numpy(img.transpose(2, 0, 1))[None].cuda()).mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=Path("/tmp/val_metrics.json"))
    ap.add_argument("--n", type=int, default=5, help="val 图数")
    args = ap.parse_args()

    import pyiqa
    niqe_m = pyiqa.create_metric("niqe", device="cuda")
    brisque_m = pyiqa.create_metric("brisque", device="cuda")
    musiq_m = pyiqa.create_metric("musiq", device="cuda")

    rows = []
    cols = ["case", "psnr_lq", "ssim_lq", "lpips_lq",
            "psnr_mywork", "ssim_mywork", "lpips_mywork",
            "psnr_probe", "ssim_probe", "lpips_probe",
            "niqe_lq", "brisque_lq", "musiq_lq",
            "niqe_mywork", "brisque_mywork", "musiq_mywork",
            "niqe_probe", "brisque_probe", "musiq_probe"]
    for i in range(1, args.n + 1):
        lq = np.load(args.cache / f"case{i}_lq.npy")
        gt = np.load(args.cache / f"case{i}_gt.npy")
        s1 = np.load(args.cache / f"case{i}_s1.npy")
        d = np.load(args.cache / f"case{i}_s2_w2.0.npy")
        hf = lq - cv2.GaussianBlur(lq, (0, 0), sigmaX=3.0)

        mywork = np.clip(blend(s1, d, 0.2) + 0.5 * hf, 0, 1)  # λ0.2 α0.5 β0
        probe = lf_anchor(np.clip(blend(s1, d, 0.4) + 1.0 * hf, 0, 1), lq, 1.0, 8.0)  # λ0.4 α1.0 β1.0

        def ref(img):
            return _psnr(img, gt), _ssim(img, gt), _lpips(img, gt)

        def nref(img):
            with torch.no_grad():
                t = torch.from_numpy(img.transpose(2, 0, 1))[None].cuda()
                return (float(niqe_m(t).mean().item()), float(brisque_m(t).mean().item()),
                        float(musiq_m(t).mean().item()))

        r = [i, *ref(lq), *ref(mywork), *ref(probe), *nref(lq), *nref(mywork), *nref(probe)]
        rows.append(r)
        print(f"case{i}: LQ PSNR {r[1]:.2f} SSIM {r[2]:.3f} LPIPS {r[3]:.4f} | "
              f"mywork {r[4]:.2f}/{r[5]:.3f}/{r[6]:.4f} | probe {r[7]:.2f}/{r[8]:.3f}/{r[9]:.4f} | "
              f"NIQE mywork {r[13]:.2f} probe {r[16]:.2f} | MUSIQ {r[14]:.1f}/{r[17]:.1f}", flush=True)

    arr = np.array(rows)
    out = {c: (float(arr[:, i].mean()), float(arr[:, i].std())) for i, c in enumerate(cols)}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n== 均值 ==")
    for c in cols:
        print(f"{c}: {out[c][0]:.4f} ± {out[c][1]:.4f}")


if __name__ == "__main__":
    main()
