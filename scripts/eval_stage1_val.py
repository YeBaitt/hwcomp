"""评估 stage1 checkpoint 在真实 huawei val 上的 PSNR↔GT，与"输入原图基线"同口径对比。

口径：中心 384 裁剪（与 train_stage1._eval_val 一致），bf16 autocast 前向，输出 clamp [0,1]。
用法：
  python scripts/eval_stage1_val.py                        # 只打印输入基线（无模型）
  python scripts/eval_stage1_val.py checkpoints/stage1_best.pth
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/liaitong/hw_comp")
sys.path.insert(0, "/home/liaitong/hw_comp/src")
_VENDOR = Path("/home/liaitong/hw_comp/vendor")
sys.path.insert(0, str(_VENDOR / "NAFNet"))

VAL = Path("/home/liaitong/hw_comp/dataset/huawei/val")
PATCH = 384


def _psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    mse = float(np.mean((pred - ref) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))


def load_center(path: str, patch: int = PATCH):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    ps = min(patch, h, w)
    y, x = (h - ps) // 2, (w - ps) // 2
    return img[y:y + ps, x:x + ps]


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    model = None
    if ckpt:
        from basicsr.models.archs.NAFNet_arch import NAFNet
        model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                       enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]).cuda().eval()
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("state_dict") or state.get("params", state)
        model.load_state_dict(state, strict=False)
        print(f"[eval] 加载 {ckpt}")

    inputs, outputs = [], []
    print(f"{'case':>5} {'输入基线':>8} {'模型输出':>8} {'增益':>7}")
    for i in range(1, 6):
        lq = load_center(str(VAL / f"case{i}_lq.jpg"))
        gt = load_center(str(VAL / f"case{i}_gt.jpg"))
        p_in = _psnr(lq, gt)
        inputs.append(p_in)
        if model:
            xt = torch.from_numpy(lq.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xt)
            out = out.float().clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0)
            p_out = _psnr(out, gt)
            outputs.append(p_out)
            print(f"{i:>5} {p_in:>8.2f} {p_out:>8.2f} {p_out - p_in:>+7.2f}")
        else:
            print(f"{i:>5} {p_in:>8.2f}       —")

    print(f"\n均值: 输入基线 {np.mean(inputs):.2f}" + (f"  /  模型 {np.mean(outputs):.2f}  增益 {np.mean(outputs) - np.mean(inputs):+.2f}dB" if outputs else ""))


if __name__ == "__main__":
    main()
