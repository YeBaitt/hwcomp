"""对 val 5 对的 LQ 基线打印全指标汇总。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from enhance.config import Config
from enhance.evaluate.metrics import report


def main():
    cfg = Config.from_yaml(Path("config.yaml"))
    vdir = Path(cfg.val_dir)
    totals = {k: [] for k in ["psnr", "ssim", "niqe", "brisque", "musiq"]}
    for i in range(1, 6):
        lq = cv2.cvtColor(cv2.imread(str(vdir / f"case{i}_lq.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt = cv2.cvtColor(cv2.imread(str(vdir / f"case{i}_gt.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r = report(lq, gt)
        print(f"case{i} LQ:", {k: round(v, 3) for k, v in r.items()})
        for k in totals:
            totals[k].append(r[k])
    print("均值:", {k: round(float(np.mean(v)), 3) for k, v in totals.items()})


if __name__ == "__main__":
    main()
