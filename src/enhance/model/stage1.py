"""Stage-1：NAFNet 保真打底，加载与推理。"""
import sys
from pathlib import Path
from typing import Tuple

import torch

_VENDOR = Path(__file__).resolve().parents[3] / "vendor"

def load_nafnet(weights_path: str, device: str = "cuda", width: int = 64,
                middle_blk_num: int = 12,
                enc_blk_nums: Tuple[int, ...] = (2, 2, 4, 8),
                dec_blk_nums: Tuple[int, ...] = (2, 2, 2, 2)) -> torch.nn.Module:
    """加载 NAFNet 模型并注入预训练权重，支持参数化架构（默认 width=64 SIDD 去噪配置）。

    Args:
        weights_path: 权重文件路径（.pth），支持 state_dict / params 键嵌套。
        device: 目标设备，默认 "cuda"。
        width: 特征通道宽度，默认 64。
        middle_blk_num: 中间块数量。
        enc_blk_nums: 编码器各阶段块数。
        dec_blk_nums: 解码器各阶段块数。

    Returns:
        eval 模式的 NAFNet 模型（已移到目标设备）。
    """
    sys.path.insert(0, str(_VENDOR / "NAFNet"))
    # 延迟导入：sys.path 须先指向 vendor/NAFNet
    from basicsr.models.archs.NAFNet_arch import NAFNet

    model = NAFNet(img_channel=3, width=width, middle_blk_num=middle_blk_num,
                   enc_blk_nums=list(enc_blk_nums), dec_blk_nums=list(dec_blk_nums))
    ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("state_dict") or ckpt.get("params", ckpt)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()

if __name__ == "__main__":
    # 使用示例：加载 SIDD 预训练权重，对合成小 patch 推理并打印输出形状
    weights = _VENDOR / "NAFNet" / "weights" / "NAFNet-SIDD-width64.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_nafnet(str(weights), device=device)
    x = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        y = model(x)
    print(f"输入形状={x.shape} 输出形状={y.shape} 范围=[{float(y.min()):.3f},{float(y.max()):.3f}]")
