"""预计算 stage2 微调条件：对每对合成数据运行 stage1(NAFNet) 得到条件图，存 _ARC_cond.png。

部署对齐：stage2 部署时条件 = stage1(lq)，故微调条件也必须 = stage1(lq)，不能是 swinir(lq)。

用法:
  python scripts/precompute_conds.py --pairs dataset/syn4k/train --cond dataset/syn4k/cond \
      --ckpt checkpoints/stage1_best.pth [--device cuda]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/liaitong/hw_comp")
sys.path.insert(0, "/home/liaitong/hw_comp/src")
_VENDOR = Path("/home/liaitong/hw_comp/vendor")
sys.path.insert(0, str(_VENDOR / "NAFNet"))
from enhance.inference.tiler import accumulate_tile, finalize, tile_weights, tiles_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("dataset/syn4k/train"))
    ap.add_argument("--cond", type=Path, default=Path("dataset/syn4k/cond"))
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/stage1_best.pth"))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    args = ap.parse_args()

    from basicsr.models.archs.NAFNet_arch import NAFNet
    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    state = torch.load(args.ckpt, map_location="cpu")
    state = state.get("state_dict") or state.get("params", state)
    model.load_state_dict(state, strict=False)
    model.eval().to(args.device)
    print(f"[conds] 加载 stage1 {args.ckpt} @ {args.device}")

    args.cond.mkdir(parents=True, exist_ok=True)
    lq_files = sorted(args.pairs.glob("*_ARC.png"))
    lq_files = [f for f in lq_files if not (args.cond / f.name.replace("_ARC.png", "_ARC_cond.png")).exists()]
    print(f"[conds] 待计算条件 {len(lq_files)} 张")
    t0 = time.time()
    for i, lq_path in enumerate(lq_files):
        bgr = cv2.imread(str(lq_path))
        if bgr is None:
            print(f"  [skip] {lq_path.name}"); continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = img.shape[:2]
        if h <= args.tile_size and w <= args.tile_size:
            x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(args.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                y = model(x)
            out = np.clip(y[0].permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0)
        else:
            canvas = np.zeros_like(img)
            ws = np.zeros((h, w), dtype=np.float32)
            for rect, wt in zip(tiles_for(h, w, args.tile_size, args.overlap),
                                tile_weights(tiles_for(h, w, args.tile_size, args.overlap), (h, w))):
                y0, y1, x0, x1 = rect
                tile = img[y0:y1, x0:x1]
                xt = torch.from_numpy(tile.transpose(2, 0, 1))[None].to(args.device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    yt = model(xt)
                accumulate_tile(canvas, ws, rect,
                                np.clip(yt[0].permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0), wt)
            out = finalize(canvas, ws)
        name = lq_path.name.replace("_ARC.png", "_ARC_cond.png")
        cv2.imwrite(str(args.cond / name),
                    cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        if (i + 1) % 20 == 0:
            print(f"[conds] {i + 1}/{len(lq_files)} {time.time() - t0:.0f}s", flush=True)
    print(f"[conds] 完成 {len(lq_files)} 张 → {args.cond}")


if __name__ == "__main__":
    main()
