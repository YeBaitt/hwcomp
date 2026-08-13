"""Stage-1 NAFNet 微调：L1 损失 + AMP + warmup/cosine 调度 + 验证早停 + checkpoint。"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from enhance.config import Config
from enhance.data.dataset import EnhancementDataset, load_pair_float
from enhance.data.pairs import find_pairs

_VENDOR = Path(__file__).resolve().parents[3] / "vendor"
sys.path.insert(0, str(_VENDOR / "NAFNet"))

def _build_model() -> torch.nn.Module:
    """构造 NAFNet width=64 SIDD 架构并移到 GPU。"""
    # 延迟导入：sys.path 须先指向 vendor/NAFNet
    from basicsr.models.archs.NAFNet_arch import NAFNet

    model = NAFNet(img_channel=3, width=64, middle_blk_num=12,
                   enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2])
    return model.cuda()

def _psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    """计算两幅 [0,1] RGB 数组的 PSNR（dB）。"""
    mse = float(np.mean((pred - ref) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))

@torch.no_grad()
def _eval_val(model: torch.nn.Module, val_ds: EnhancementDataset, patch_size: int) -> float:
    """在验证集上计算中心裁剪 patch 的平均 PSNR（fp16 评估，确定性与可复现）。"""
    device = next(model.parameters()).device
    model.eval()
    scores = []
    for pair in val_ds.pairs:
        lq, hr = load_pair_float(pair, val_ds.cache_root)
        h, w = lq.shape[:2]
        ps = min(patch_size, h, w)
        y = (h - ps) // 2
        x = (w - ps) // 2
        inp = lq[y:y + ps, x:x + ps]
        tgt = hr[y:y + ps, x:x + ps]
        xt = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.cuda.amp.autocast():
            out = model(xt)
        out = out.float().clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0)
        scores.append(_psnr(out, tgt))
    model.train()
    return float(np.mean(scores))

def train_stage1(cfg: Config) -> Path:
    """运行 Stage-1 NAFNet 微调训练，返回最优 checkpoint 路径。

    若 cfg.stage1_pretrained 指向存在的文件则 warm-start，否则随机初始化。
    训练用 L1 损失 + AMP + AdamW + warmup/cosine 调度，验证集 PSNR 早停，
    保存最优 stage1_best.pth。
    """
    # 划分训练/验证：find_pairs 已排序，前 val_holdout_n 对作为验证集（确定性）
    all_pairs = find_pairs(cfg.image_pairs_train_dir)
    val_n = min(cfg.val_holdout_n, max(0, len(all_pairs) - 1))
    val_pairs = all_pairs[:val_n]
    train_pairs = all_pairs[val_n:]

    # length_factor=2 使每对图像每 epoch 只出 2 个 patch，加速 epoch 迭代
    ds = EnhancementDataset(cfg.image_pairs_train_dir, patch_size=cfg.patch_size,
                            kind_2k_weight=cfg.kind_2k_weight, seed=cfg.seed,
                            length_factor=2, cache_pairs=cfg.cache_pairs,
                            cache_root=cfg.cache_root, pairs=train_pairs)
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    val_ds = EnhancementDataset(cfg.image_pairs_train_dir, patch_size=cfg.patch_size,
                                kind_2k_weight=cfg.kind_2k_weight, seed=cfg.seed,
                                length_factor=1, cache_pairs=val_n,
                                cache_root=cfg.cache_root, pairs=val_pairs)
    print(f"[stage1] 训练集 {len(train_pairs)} 对 / 验证集 {len(val_pairs)} 对, {len(dl)} batches/epoch")
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
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05, total_iters=cfg.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.stage1_epochs - cfg.warmup_epochs))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine],
                                                  milestones=[cfg.warmup_epochs])

    ckpt_path = cfg.ckpt_root / "stage1_best.pth"
    best_psnr = -1.0
    patience = 0
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
        avg_loss = total / max(i + 1, 1)
        val_psnr = _eval_val(model, val_ds, cfg.patch_size)
        print(f"epoch {epoch} avg_loss {avg_loss:.4f} val_psnr {val_psnr:.3f}")

        # 早停：验证 PSNR 无提升则累计耐心，超阈值提前结束
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            patience = 0
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict()}, ckpt_path)
            print(f"[stage1] 新最优 val_psnr={val_psnr:.3f}，已保存 {ckpt_path}")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[stage1] 早停：连续 {cfg.early_stop_patience} 个 epoch 无提升")
                break

    return ckpt_path

if __name__ == "__main__":
    # 使用示例：从 config.yaml 加载配置并运行完整训练
    cfg = Config.from_yaml(Path("config.yaml"))
    train_stage1(cfg)
