"""合成匹配退化训练对生成器（val 标定参数，ImagePairs 风格命名，stage1 训练器可直接消费）。

退化配方（来自 /tmp/calibrate_deg.py 在 huawei val 上的标定）：
  blur σ ~ U(2.5,5.0)  +  高斯噪声 std ~ U(0.008,0.040)  +  JPEG q ~ U(60,95)
覆盖 val 典型的 blur 主导退化（26~35dB 范围），σ 上限拉到 5 兼顾重退化尾部。

用法:
  python scripts/synth_pairs.py --src dataset/ImagePairs/train --pattern "*_gt.png" --out dataset/syn4k --n 30
  python scripts/synth_pairs.py --src <DIV8K目录> --pattern "*.png" --out dataset/syn4k --n 500
输出 dataset/syn4k/train/synNNNNNN_ARC.png (lq) + _ARC_gt.png (gt, PNG 无损)。
"""
import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np


def degrade(clean: np.ndarray, rng: random.Random) -> tuple:
    """施加标定退化：blur + 高斯噪声 + JPEG 压缩，返回 (lq, 参数)。"""
    sigma = rng.uniform(2.5, 5.0)
    noise_std = rng.uniform(0.008, 0.040)
    jpeg_q = rng.randint(60, 95)
    lq = cv2.GaussianBlur(clean, (0, 0), sigmaX=sigma)
    rng_np = np.random.default_rng(rng.getrandbits(32))
    lq = lq + rng_np.normal(0.0, noise_std, lq.shape).astype(np.float32)
    lq = np.clip(lq, 0.0, 1.0)
    # JPEG 压缩模拟（编码后解码，像素级别复现手机压缩伪影）
    bgr = cv2.cvtColor((lq * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    lq = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return lq, sigma, noise_std, jpeg_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--pattern", type=str, default="*.png", help="干净源文件通配符，ImagePairs 用 *_gt.png")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--long", type=int, default=4096, help="目标长边")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(args.src.rglob(args.pattern))
    files = [f for f in files if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not files:
        raise SystemExit(f"{args.src} 下未找到 {args.pattern} 图片")
    rng = random.Random(args.seed)
    rng.shuffle(files)
    files = files[: args.n]
    print(f"[synth] 干净源 {len(files)} 张 → 目标长边 {args.long}，退化 blur+noise+jpeg")

    out_dir = args.out / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    t0 = time.time()
    for idx, f in enumerate(files):
        gt_bgr = cv2.imread(str(f))
        if gt_bgr is None:
            print(f"  [skip] {f.name} 读取失败"); continue
        gt = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # 长边对齐到目标分辨率
        h, w = gt.shape[:2]
        scale = args.long / max(h, w)
        if scale < 1.0:
            gt = cv2.resize(gt, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        lq, sigma, noise_std, jpeg_q = degrade(gt, rng)
        name = f"syn{idx:06d}"
        # lq 存为 PNG（内容含 JPEG 压缩像素），gt 存为 PNG 无损——与 find_pairs 的 *_ARC.png/_ARC_gt.png 约定一致
        cv2.imwrite(str(out_dir / f"{name}_ARC.png"),
                    cv2.cvtColor((lq * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{name}_ARC_gt.png"),
                    cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        mse = float(np.mean((lq - gt) ** 2))
        psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)
        log.append((name, round(sigma, 2), round(noise_std, 4), jpeg_q, round(psnr, 2)))
        if (idx + 1) % 10 == 0:
            print(f"[synth] {idx + 1}/{len(files)} {time.time() - t0:.0f}s", flush=True)

    with open(args.out / "degrade_params.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["name", "sigma", "noise_std", "jpeg_q", "syn_psnr"])
        w.writerows(log)
    psnrs = [r[4] for r in log]
    print(f"[synth] 完成 {len(log)} 对，生成 lq↔gt PSNR 均值 {np.mean(psnrs):.1f}（范围 {min(psnrs):.1f}~{max(psnrs):.1f}）")
    print(f"[synth] 输出: {out_dir}  参数表: {args.out / 'degrade_params.csv'}")


if __name__ == "__main__":
    main()
