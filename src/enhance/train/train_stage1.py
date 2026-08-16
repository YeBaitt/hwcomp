"""Stage-1 NAFNet 微调：L1 损失 + bf16 AMP + 输出 clamp + warmup/cosine 调度 + 验证早停 + checkpoint。

训练数值稳定性设计（实测支撑）：
- bf16 autocast 前向：4090 原生 bf16 提速；与 fp32 输出实测一致（max_out 1.105 vs 1.107），无精度退化。
- 输出 clamp(0,1) 后算 fp32 L1：个别输入（过饱和/高光/裁剪像素）会触发模型超范围输出，
  导致 loss 尖峰（曾实测达 41，>1 在 [0,1] 数据下不可能，只能是超范围输出造成）。
  clamp 使 loss<=1 恒成立、超范围元素梯度为 0，权重更新恒有界，尖峰不再污染权重。
- 非有限 loss 批次直接跳过（不进 backward），clip_grad_norm 兜底。
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from enhance.config import Config
from enhance.data.dataset import EnhancementDataset, load_pair_float
from enhance.data.pairs import find_pairs, npz_cache_key, to_same_res

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

def _make_scheduler(opt: torch.optim.Optimizer, warmup_epochs: int, stage1_epochs: int):
    """构造 warmup(LinearLR) + cosine(CosineAnnealingLR) 的 SequentialLR 调度器。"""
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.05, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, stage1_epochs - warmup_epochs))
    return torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine],
                                                 milestones=[warmup_epochs])

def _early_stop_step(val_psnr: float, best_psnr: float, patience: int, patience_limit: int) -> tuple:
    """根据本次验证 PSNR 更新早停状态，返回 (新最优值, 新耐心, 是否新最优, 是否应早停)。"""
    if val_psnr > best_psnr:
        return val_psnr, 0, True, False
    patience += 1
    return best_psnr, patience, False, patience >= patience_limit

@torch.no_grad()
def _eval_val(model: torch.nn.Module, val_ds: EnhancementDataset, patch_size: int) -> float:
    """在验证集上计算中心裁剪 patch 的平均 PSNR（fp16 评估，确定性与可复现）。"""
    device = next(model.parameters()).device
    model.eval()
    scores = []
    for pair in val_ds.pairs:
        lq, hr = load_pair_float(pair, val_ds.cache_root, val_ds.pairs_root)
        # 目标先对齐到 lq 分辨率，再同位置裁剪，保证输出与目标空间对应、PSNR 有意义
        inp, tgt = to_same_res(lq, hr, "2k")
        h, w = inp.shape[:2]
        ps = min(patch_size, h, w)
        y = (h - ps) // 2
        x = (w - ps) // 2
        inp = inp[y:y + ps, x:x + ps]
        tgt = tgt[y:y + ps, x:x + ps]
        xt = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(xt)
        out = out.float().clamp(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0)
        scores.append(_psnr(out, tgt))
    model.train()
    return float(np.mean(scores))

@torch.no_grad()
def _eval_real_val(model: torch.nn.Module, val_dir: str, patch_size: int) -> float:
    """在真实 huawei val（5 张 case{i}_lq/gt.jpg）上计算中心 384 PSNR。

    合成代理 val 与真实 val 退化分布有差异（合成代理含 JPEG/随机 σ，真实 val 为手机退化），
    实测合成代理持续上升而真实 val 在 ~epoch3 峰值后回落——故保存判据必须用真实 val。
    """
    import cv2
    device = next(model.parameters()).device
    model.eval()
    vdir = Path(val_dir)
    scores = []
    for i in range(1, 6):
        lq_jpg, gt_jpg = vdir / f"case{i}_lq.jpg", vdir / f"case{i}_gt.jpg"
        if not (lq_jpg.exists() and gt_jpg.exists()):
            continue
        lq = cv2.cvtColor(cv2.imread(str(lq_jpg)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tgt = cv2.cvtColor(cv2.imread(str(gt_jpg)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = lq.shape[:2]
        ps = min(patch_size, h, w)
        y, x = (h - ps) // 2, (w - ps) // 2
        lq = lq[y:y + ps, x:x + ps]
        tgt = tgt[y:y + ps, x:x + ps]
        xt = torch.from_numpy(lq.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
    pairs_root = Path(cfg.image_pairs_train_dir)
    bad_file = pairs_root.parent / "bad_pairs.txt"
    if bad_file.exists():
        bad_keys = {line.strip() for line in bad_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        before = len(all_pairs)
        all_pairs = [p for p in all_pairs if npz_cache_key(p, pairs_root) not in bad_keys]
        print(f"[stage1] 排除损坏对 {before - len(all_pairs)} 对（来自 bad_pairs.txt）")
    val_n = min(cfg.val_holdout_n, max(0, len(all_pairs) - 1))
    val_pairs = all_pairs[:val_n]
    train_pairs = all_pairs[val_n:]

    # length_factor 控制每对图像每 epoch 的采样数（全量数据下默认 1）
    ds = EnhancementDataset(cfg.image_pairs_train_dir, patch_size=cfg.patch_size,
                            kind_2k_weight=cfg.kind_2k_weight, seed=cfg.seed,
                            length_factor=cfg.length_factor, cache_pairs=cfg.cache_pairs,
                            cache_root=cfg.cache_root, pairs=train_pairs)
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    val_ds = EnhancementDataset(cfg.image_pairs_train_dir, patch_size=cfg.patch_size,
                                kind_2k_weight=cfg.kind_2k_weight, seed=cfg.seed,
                                length_factor=1, cache_pairs=val_n,
                                cache_root=cfg.cache_root, pairs=val_pairs)
    print(f"[stage1] 训练集 {len(train_pairs)} 对 / 验证集 {len(val_pairs)} 对, {len(dl)} batches/epoch")
    model = _build_model()

    # warm-start：加载预训练权重，并用其真实验证 PSNR 初始化 best_psnr。
    # 否则 best_psnr=-1 会让首个 epoch 的更差 val 覆盖掉更优的 warm-start checkpoint
    #（曾实测：warm-start 24.689 被首个 epoch 23.966 覆盖）。
    pretrained_path = Path(cfg.stage1_pretrained)
    warm_val = -1.0
    if pretrained_path.exists():
        ckpt = torch.load(pretrained_path, map_location="cpu")
        state = ckpt.get("state_dict") or ckpt.get("params", ckpt)
        model.load_state_dict(state, strict=False)
        # 保存判据用真实 val（合成代理不跟踪真实退化分布）
        warm_val = _eval_real_val(model, cfg.val_dir, cfg.patch_size)
        print(f"[stage1] warm-start 成功: {pretrained_path} (warm 点 real_val_psnr={warm_val:.3f})")
    else:
        print("[stage1] 未找到预训练权重，随机初始化")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.stage1_lr)
    sched = _make_scheduler(opt, cfg.warmup_epochs, cfg.stage1_epochs)
    accum = max(int(getattr(cfg, "grad_accum", 1)), 1)  # 梯度累积：patch 512 需 batch 2 + 累积到等效 batch 4

    ckpt_path = cfg.ckpt_root / "stage1_best.pth"
    best_psnr = warm_val
    patience = 0
    model.train()
    for epoch in range(cfg.stage1_epochs):
        total = 0.0
        n_valid = 0
        i = -1
        opt.zero_grad()
        for i, (x, y) in enumerate(dl):
            x, y = x.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            # 输出 clamp 到 [0,1] 后再算 fp32 L1：个别输入会触发模型超范围输出
            #（已实测 bf16/fp32 一致，非精度问题），clamp 保证 loss<=1 且超范围元素的
            # 梯度被阻断，权重更新恒有界，避免 loss 尖峰（曾见 41）污染权重。
            loss = torch.nn.functional.l1_loss(out.float().clamp(0.0, 1.0), y)
            if not torch.isfinite(loss):
                print(f"[stage1] WARNING 跳过非有限 loss 批次: epoch {epoch} step {i}")
                continue
            (loss / accum).backward()
            total += float(loss)
            n_valid += 1
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad()
            if i % 50 == 0:
                print(f"epoch {epoch} step {i} loss {float(loss):.4f}")
        if n_valid % accum != 0:  # 尾部不足一个累积窗时补一次更新
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()
        avg_loss = total / max(n_valid, 1)
        val_psnr = _eval_val(model, val_ds, cfg.patch_size)          # 合成代理（仅日志）
        real_val_psnr = _eval_real_val(model, cfg.val_dir, cfg.patch_size)  # 保存判据
        print(f"epoch {epoch} avg_loss {avg_loss:.4f} val_psnr {val_psnr:.3f} real_val {real_val_psnr:.3f}")

        # 早停：真实 val PSNR 无提升则累计耐心，超阈值提前结束
        best_psnr, patience, is_new_best, should_stop = _early_stop_step(
            real_val_psnr, best_psnr, patience, cfg.early_stop_patience)
        if is_new_best:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict()}, ckpt_path)
            print(f"[stage1] 新最优 val_psnr={val_psnr:.3f}，已保存 {ckpt_path}")
        if should_stop:
            print(f"[stage1] 早停：连续 {cfg.early_stop_patience} 个 epoch 无提升")
            break

    return ckpt_path

if __name__ == "__main__":
    # 使用示例：从 config.yaml 加载配置并运行完整训练
    cfg = Config.from_yaml(Path("config.yaml"))
    train_stage1(cfg)
