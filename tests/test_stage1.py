"""Stage-1 NAFNet 加载器与训练流程的单元测试。"""
from pathlib import Path

import pytest
import torch

from enhance.model.stage1 import load_nafnet


def test_load_nafnet_tiny_arch(tmp_path):
    """用合成小架构权重测试 load_nafnet：构造、保存、加载、推理，验证输出形状与设备。"""
    tiny_weights = tmp_path / "tiny.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构造 tiny NAFNet 并保存其随机 state_dict
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "NAFNet"))
    # 延迟导入：sys.path 须先指向 vendor/NAFNet
    from basicsr.models.archs.NAFNet_arch import NAFNet

    ref = NAFNet(img_channel=3, width=4, middle_blk_num=1,
                 enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])
    ref = ref.to(device)
    torch.save({"state_dict": ref.state_dict()}, tiny_weights)

    # 通过 load_nafnet 加载
    model = load_nafnet(str(tiny_weights), device=device, width=4,
                        middle_blk_num=1, enc_blk_nums=(1, 1), dec_blk_nums=(1, 1))

    # 验证 eval 模式
    assert not model.training
    # 验证设备
    for p in model.parameters():
        assert p.device.type == device
        break

    # 验证推理形状
    x = torch.randn(1, 3, 64, 64).to(device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 64, 64)

    # 验证权重加载正确性：同一输入的输出应与参考模型一致
    with torch.no_grad():
        assert torch.allclose(model(x), ref(x), atol=1e-5)


def test_load_nafnet_params_key(tmp_path):
    """load_nafnet 应正确处理 params 键嵌套的 checkpoint（真实 NAFNet 保存格式）。"""
    weights = tmp_path / "params_key.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "NAFNet"))
    # 延迟导入：sys.path 须先指向 vendor/NAFNet
    from basicsr.models.archs.NAFNet_arch import NAFNet

    ref = NAFNet(img_channel=3, width=4, middle_blk_num=1,
                 enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])
    torch.save({"params": ref.state_dict()}, weights)

    model = load_nafnet(str(weights), device=device, width=4,
                        middle_blk_num=1, enc_blk_nums=(1, 1), dec_blk_nums=(1, 1))

    assert not model.training
    for p in model.parameters():
        assert p.device.type == device
        break

    x = torch.randn(1, 3, 32, 32).to(device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 32, 32)


def test_load_nafnet_flat_state_dict(tmp_path):
    """load_nafnet 应正确处理无嵌套的 flat state_dict。"""
    weights = tmp_path / "flat.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "NAFNet"))
    # 延迟导入：sys.path 须先指向 vendor/NAFNet
    from basicsr.models.archs.NAFNet_arch import NAFNet

    ref = NAFNet(img_channel=3, width=4, middle_blk_num=1,
                 enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])
    torch.save(ref.state_dict(), weights)

    model = load_nafnet(str(weights), device=device, width=4,
                        middle_blk_num=1, enc_blk_nums=(1, 1), dec_blk_nums=(1, 1))

    assert not model.training
    for p in model.parameters():
        assert p.device.type == device
        break

    x = torch.randn(1, 3, 32, 32).to(device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 32, 32)


def test_load_nafnet_default_params():
    """load_nafnet 默认参数（width=64）创建的模型应可前向传播。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "NAFNet"))

    # 用一个临时文件保存随机权重
    import tempfile
    # 延迟导入：sys.path 须先指向 vendor/NAFNet
    from basicsr.models.archs.NAFNet_arch import NAFNet

    ref = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                 enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save(ref.state_dict(), f.name)
        tmp_path = f.name

    try:
        model = load_nafnet(tmp_path, device=device)
        assert not model.training
        x = torch.randn(1, 3, 128, 128).to(device)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, 128, 128)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
