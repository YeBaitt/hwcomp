"""CPU 冒烟测试：验证增强引擎 core 逻辑（不依赖 GPU / 5GB 模型权重）。

通过 monkeypatch 将 stage1/stage2 替换为轻量模拟，确保流水线端到端跑通。
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enhance.config import Config
from enhance.inference.engine import EnhancementEngine


@pytest.fixture
def cfg():
    """构造一个最小可用配置对象，避免依赖 config.yaml 文件。"""
    return Config(
        data_root=Path("dataset"),
        image_pairs_train_dir=Path("dataset/ImagePairs/train"),
        val_dir=Path("dataset/huawei/val"),
        test_dir=Path("dataset/huawei/test"),
        out_root=Path("output"),
        ckpt_root=Path("checkpoints"),
        submit_dir=Path("output_dir"),
        zip_path=Path("my_work.zip"),
        patch_size=256,
        batch_size=2,
        num_workers=0,
        seed=42,
        kind_2k_weight=0.7,
        stage1_epochs=1,
        tile_size=64,
        overlap=16,
        steps=5,
        scheduler="dpm",
        batch_tiles=2,
        use_tta=False,
        stage1_pretrained="vendor/NAFNet/weights/NAFNet-SIDD-width64.pth",
        lam=0.3,
        n=0.15,
        w=2.0,
        alpha=0.4,
    )


def _mock_load_nafnet(weights_path, device="cpu", **kwargs):
    """返回一个轻量恒等模块，模拟 NAFNet 行为（输入→输出同形状）。"""
    return torch.nn.Identity().to(device)


def _mock_stage2_refine(control_image, out_path, steps=20, guidance=2.0,
                        tile_size=512, stride=256):
    """模拟 stage-2：返回 control 图副本，并写入输出文件。"""
    # 确保输出路径的父目录存在
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # 写入一个占位文件，模拟 stage2 的写盘行为
    with open(out_path, "wb") as f:
        f.write(b"mock")
    return control_image.copy()


def test_enhance_returns_same_shape(cfg, monkeypatch):
    """engine.enhance() 在 mock 模型下应返回与输入同形状的输出。"""
    monkeypatch.setattr(
        "enhance.inference.engine.load_nafnet", _mock_load_nafnet
    )
    monkeypatch.setattr(
        "enhance.inference.engine.stage2_refine", _mock_stage2_refine
    )
    engine = EnhancementEngine(cfg)
    img = np.random.default_rng(42).random((128, 128, 3), dtype=np.float32)
    out = engine.enhance(img)
    assert out.shape == img.shape, f"期望 shape={img.shape}，实际={out.shape}"
    assert out.dtype == np.float32
    # 输出应在 [0,1] 范围（旋钮融合后裁剪）
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_enhance_path_writes_file(cfg, monkeypatch, tmp_path):
    """engine.enhance_path() 应成功读取输入图像并写入输出文件。"""
    monkeypatch.setattr(
        "enhance.inference.engine.load_nafnet", _mock_load_nafnet
    )
    monkeypatch.setattr(
        "enhance.inference.engine.stage2_refine", _mock_stage2_refine
    )
    engine = EnhancementEngine(cfg)
    in_path = tmp_path / "test_in.jpg"
    out_path = tmp_path / "test_out.jpg"
    # 写一个最小的 JPEG 输入（1x1 像素）
    import cv2
    cv2.imwrite(str(in_path), np.zeros((1, 1, 3), dtype=np.uint8))
    engine.enhance_path(in_path, out_path)
    assert out_path.exists(), f"输出文件 {out_path} 未创建"


def test_temp_dir_cleaned(cfg, monkeypatch):
    """enhance() 调用后临时目录应已清理。"""
    monkeypatch.setattr(
        "enhance.inference.engine.load_nafnet", _mock_load_nafnet
    )
    monkeypatch.setattr(
        "enhance.inference.engine.stage2_refine", _mock_stage2_refine
    )
    engine = EnhancementEngine(cfg)
    img = np.random.default_rng(99).random((32, 32, 3), dtype=np.float32)
    # 记录 enhance 前后的 /tmp 条目
    import os
    before = set(os.listdir(tempfile.gettempdir()))
    _ = engine.enhance(img)
    after = set(os.listdir(tempfile.gettempdir()))
    new_entries = after - before
    assert len(new_entries) == 0, f"临时目录未被清理: {new_entries}"
