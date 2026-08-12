"""Stage-1 NAFNet 微调：L1 损失 + AMP + checkpoint。"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from enhance.config import Config
from enhance.data.dataset import EnhancementDataset

_VENDOR = Path(__file__).resolve().parents[3] / "vendor"
sys.path.insert(0, str(_VENDOR / "NAFNet"))


def _build_model() -> torch.nn.Module:
    """构造 NAFNet width=64 SIDD 架构并移到 GPU。"""
    # 延迟导入：sys.path 须先指向 vendor/NAFNet
    from basicsr.models.archs.NAFNet_arch import NAFNet

    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    return model.cuda()


def train_stage1(cfg: Config) -> Path:
    """运行 Stage-1 NAFNet 微调训练，返回最终 checkpoint 路径。

    若 cfg.stage1_pretrained 指向存在的文件，则以此为 warm-start；
    否则随机初始化从头训练。
    训练使用 L1 损失 + AMP + AdamW + CosineAnnealingLR。
    """
    # length_factor=2 使每对图像每 epoch 只出 2 个 patch，加速 epoch 迭代
    ds = EnhancementDataset(cfg.image_pairs_train_dir, patch_size=cfg.patch_size,
                            kind_2k_weight=cfg.kind_2k_weight, seed=cfg.seed,
                            length_factor=2)
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    print(f"[stage1] 数据集: {len(ds)} samples, {len(dl)} batches/epoch")
    model = _build_model()

    # warm-start：加载预训练权重
    pretrained_path = Path(cfg.stage1_pretrained)
    if pretrained_path.exists():
        ckpt = torch.load(pretrained_path, map_location="cpu")
        state = ckpt.get("state_dict") or ckpt.get("params", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[stage1] warm-start 成功: {pretrained_path}")
    else:
        print("[stage1] 未找到预训练权重，随机初始化")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.stage1_epochs)
    model.train()
    for epoch in range(cfg.stage1_epochs):
        total = 0.0
        i = -1
        for i, (x, y) in enumerate(dl):
            x, y = x.cuda(), y.cuda()
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = torch.nn.functional.l1_loss(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss)
            if i % 50 == 0:
                print(f"epoch {epoch} step {i} loss {float(loss):.4f}")
        sched.step()
        print(f"epoch {epoch} avg_loss {total / max(i + 1, 1):.4f}")

    # 保存最终 epoch 的 checkpoint（文件名 stage1_best.pth 为 Task 10 引擎约定名称）
    ckpt_path = cfg.ckpt_root / "stage1_best.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    print(f"[stage1] checkpoint 已保存: {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    # 使用示例：从 config.yaml 加载配置并运行完整训练
    cfg = Config.from_yaml(Path("config.yaml"))
    train_stage1(cfg)
