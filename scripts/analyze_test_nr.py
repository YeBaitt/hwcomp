"""对两份提交（probe vs my_work）在 test 100 张上计算无参考感知指标。

用途：官方打分即基于 test 输出，此脚本直接在这些输出上算 NIQE/BRISQUE/PIQE/
ILNIQE/MUSIQ/MANIQA（+ 亮度/锐度/JPEG 体积等低层统计），用于判断官方评分
与哪个指标正相关。

前置：先解压提交 zip，得到含 100 张 caseN.jpg 的两个目录。
用法:
  python scripts/analyze_test_nr.py \
      --probe /tmp/sub_probe --mywork /tmp/sub_mywork \
      --input dataset/huawei/test --out /tmp/test_nr_metrics.csv
注：仅用本地已缓存权重的 pyiqa 指标，避免联网下载卡住。
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

LOWER_BETTER = ["niqe", "brisque", "piqe", "ilniqe"]
HIGHER_BETTER = ["musiq", "maniqa"]
ALL_M = LOWER_BETTER + HIGHER_BETTER


def _load(p: Path):
    bgr = cv2.imread(str(p))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _lapvar(img):
    g = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True, help="probe 提交解压目录")
    ap.add_argument("--mywork", type=Path, required=True, help="my_work 提交解压目录")
    ap.add_argument("--input", type=Path, default=Path("dataset/huawei/test"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/test_nr_metrics.csv"))
    args = ap.parse_args()

    import pyiqa
    metrics = {}
    for name in ALL_M:
        try:
            metrics[name] = pyiqa.create_metric(name, device="cuda")
            print(f"[init] {name} OK", flush=True)
        except Exception as e:
            print(f"[warn] {name} init 失败: {e}", flush=True)

    cases = sorted(p.name for p in args.input.glob("case*.jpg"))
    assert len(cases) == 100, len(cases)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "src"] + ALL_M + ["mean_luma", "std_luma", "lapvar", "jpeg_bytes"])
        f.flush()

    t0 = time.time()
    for k, name in enumerate(cases):
        img_in = _load(args.input / name)
        img_p = _load(args.probe / name)
        img_m = _load(args.mywork / name)
        # 输入基线只算 NIQE/BRISQUE/MUSIQ（其余指标慢，非核心对比）
        for src, img, full in (("input", img_in, False), ("probe", img_p, True), ("mywork", img_m, True)):
            row = [name, src]
            with torch.no_grad():
                t = torch.from_numpy(img.transpose(2, 0, 1))[None].cuda()
                for mn in ALL_M:
                    if mn not in metrics or (not full and mn not in ("niqe", "brisque", "musiq")):
                        row.append("")
                        continue
                    try:
                        row.append(f"{float(metrics[mn](t).mean().item()):.4f}")
                    except Exception as e:
                        row.append(f"ERR:{e}")
            row.append(f"{float(img.mean()):.4f}")
            row.append(f"{float(img.std()):.4f}")
            row.append(f"{_lapvar(img):.1f}")
            src_dir = {"probe": args.probe, "mywork": args.mywork, "input": args.input}[src]
            row.append(f"{Path(src_dir / name).stat().st_size}")
            with open(args.out, "a", newline="") as f:
                csv.writer(f).writerow(row)
                f.flush()
        if (k + 1) % 5 == 0:
            print(f"[{k + 1}/100] {time.time() - t0:.0f}s", flush=True)
    print(f"done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
