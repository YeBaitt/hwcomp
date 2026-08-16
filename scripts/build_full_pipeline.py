"""全管线提交构建：新 stage1 + stage2 → 100 张测试图 → output_dir/case*.jpg → zip。

GPU 大任务（stage2 扩散 ~分钟/图）。引擎内 stage2 pipeline 为单例，权重只载一次。
用法:
  python scripts/build_full_pipeline.py --knobs 0.4,1.0,1.0,8.0 --tag submit_new [--limit 10]
"""
import argparse
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/liaitong/hw_comp")
sys.path.insert(0, "/home/liaitong/hw_comp/src")

TEST = Path("/home/liaitong/hw_comp/dataset/huawei/test")
OUT = Path("/home/liaitong/hw_comp/output_dir")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knobs", type=str, default="0.4,1.0,1.0,8.0", help="lam,alpha,beta,sigma")
    ap.add_argument("--tag", type=str, default="submit_new")
    ap.add_argument("--limit", type=int, default=0, help="只构建前 N 张（0=全部），探针用")
    args = ap.parse_args()
    lam, alpha, beta, sigma = (float(v) for v in args.knobs.split(","))

    from enhance.config import Config
    cfg = Config.from_yaml(Path("/home/liaitong/hw_comp/config.yaml"))
    cfg.lam, cfg.alpha, cfg.beta, cfg.sigma = lam, alpha, beta, sigma

    from enhance.inference.engine import EnhancementEngine
    engine = EnhancementEngine(cfg)

    cases = sorted(TEST.glob("case*.jpg"))
    if args.limit > 0:
        cases = cases[:args.limit]
    print(f"[build] {len(cases)} 张测试图，knobs=({lam},{alpha},{beta},{sigma})", flush=True)
    t0 = time.time()
    means = []
    for i, p in enumerate(cases):
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = engine.enhance(rgb)
        means.append(float(np.mean(np.abs(out - rgb))))
        out_bgr = cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(OUT / p.name), out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        el = time.time() - t0
        print(f"[build] {i + 1}/{len(cases)} {p.name} {el / (i + 1):.1f}s/图 meanΔ={means[-1]:.3f}", flush=True)

    print(f"[build] 完成 {len(cases)} 张，总 {time.time() - t0:.0f}s，meanΔ 均值 {np.mean(means):.3f}")
    base = Path("/tmp") / f"pkg_{args.tag}"
    base.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(base / f"{args.tag}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.glob("*.jpg")):
            z.write(f, f"output_dir/{f.name}")
    print(f"[build] 提交包: {base}/{args.tag}.zip")


if __name__ == "__main__":
    main()
