"""在 5 对 val 上网格搜索 (λ, α)，输出 CSV。只在 val 上调参。"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from enhance.config import Config
from enhance.evaluate.metrics import niqe, psnr, ssim
from enhance.fusion.knobs import KnobConfig, apply_knobs
from enhance.inference.engine import EnhancementEngine

LAM_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ALPHA_GRID = [0.0, 0.3, 0.5]

def load_pair(vdir: Path, name: str):
    """读取 name_lq/name_gt 为 RGB float32 [0,1]，返回 (lq, gt)。"""
    lq = cv2.cvtColor(cv2.imread(str(vdir / f"{name}_lq.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gt = cv2.cvtColor(cv2.imread(str(vdir / f"{name}_gt.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return lq, gt

def main():
    cfg = Config.from_yaml(Path("config.yaml"))
    engine = EnhancementEngine(cfg)
    vdir = Path(cfg.val_dir)
    names = [f"case{i}" for i in range(1, 6)]
    imgs = {n: load_pair(vdir, n) for n in names}

    # 每张图只跑一次 stage1 + stage2（整图路径），缓存后网格内仅做廉价 apply_knobs 重组
    s1, base, stage2_secs, failed = {}, {}, {}, []
    for n in names:
        lq, _ = imgs[n]
        try:
            t0 = time.time()
            s1[n] = engine._stage1(lq)
            t1 = time.time()
            base[n] = engine._stage2(s1[n])
            stage2_secs[n] = time.time() - t1
            print(f"{n}: stage1={t1 - t0:.1f}s stage2={stage2_secs[n]:.1f}s")
        except Exception as exc:
            failed.append((n, type(exc).__name__, str(exc)))
            print(f"{n}: stage1/stage2 失败（{type(exc).__name__}: {exc}），跳过该图")

    # 网格搜索：对成功图像求 PSNR/SSIM/NIQE 均值（无参考 brisque/musiq 未纳入网格，按 brief 仅取 NIQE）
    ok = [n for n in names if n not in {f[0] for f in failed}]
    rows = []
    for lam in LAM_GRID:
        for alpha in ALPHA_GRID:
            ps, ss, nq = [], [], []
            for n in ok:
                lq, gt = imgs[n]
                out = apply_knobs(s1[n], base[n], lq, KnobConfig(lam=lam, alpha=alpha))
                ps.append(psnr(out, gt))
                ss.append(ssim(out, gt))
                nq.append(niqe(out))
            rows.append({"lam": lam, "alpha": alpha,
                         "psnr": round(float(np.mean(ps)), 3),
                         "ssim": round(float(np.mean(ss)), 3),
                         "niqe": round(float(np.mean(nq)), 3)})

    out_csv = Path(cfg.out_root) / "knob_grid.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lam", "alpha", "psnr", "ssim", "niqe"])
        w.writeheader()
        w.writerows(rows)
    print("写至", out_csv)
    print("最优 PSNR 行:", max(rows, key=lambda r: r["psnr"]))
    if failed:
        print("失败图像:", failed)

if __name__ == "__main__":
    main()
