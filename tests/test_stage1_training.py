"""Stage-1 训练新逻辑（PSNR / 调度器 / 早停）的 CPU 单元测试。"""
import numpy as np
import pytest
import torch

from enhance.train.train_stage1 import _early_stop_step, _make_scheduler, _psnr

def test_psnr_identical_is_inf():
    """两幅完全相同的 [0,1] 数组 PSNR 应为 +inf。"""
    a = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    assert _psnr(a, a) == float("inf")

def test_psnr_known_shift():
    """常数偏移 0.1 → mse=0.01 → 10*log10(100)=20 dB，与 metrics.psnr(max_val=1) 约定一致。"""
    ref = np.full((8, 8, 3), 0.5, dtype=np.float32)
    pred = np.full((8, 8, 3), 0.6, dtype=np.float32)
    assert _psnr(pred, ref) == pytest.approx(20.0)

def test_make_scheduler_milestones_and_tmax():
    """warmup=3、epochs=60 时 SequentialLR 的里程碑为 [3]，cosine T_max=57。"""
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sched = _make_scheduler(opt, warmup_epochs=3, stage1_epochs=60)
    assert list(sched._milestones) == [3]
    assert sched._schedulers[1].T_max == 57  # 60 - 3

def test_make_scheduler_lr_ramp_then_decay():
    """学习率先 warmup 单调上升，随后 cosine 衰减到接近 0。"""
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    sched = _make_scheduler(opt, warmup_epochs=3, stage1_epochs=60)
    lrs = [sched.get_last_lr()[0]]
    for _ in range(3):
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    assert lrs[1] < lrs[2] < lrs[3]  # warmup 阶段单调上升
    assert lrs[3] == pytest.approx(1.0, abs=1e-6)  # 峰值回到 base_lr
    for _ in range(57):
        sched.step()
    assert sched.get_last_lr()[0] < 0.1  # cosine 周期末衰减到接近 0

def test_early_stop_step_improves_resets_patience():
    """验证 PSNR 提升时：更新最优、清零耐心、标记新最优且不早停。"""
    best, patience, is_new_best, should_stop = _early_stop_step(20.0, 19.0, 5, 8)
    assert best == 20.0 and patience == 0 and is_new_best and not should_stop

def test_early_stop_step_triggers_stop_after_patience():
    """连续无提升达到耐心阈值时：保持最优、耐心递增、触发早停。"""
    best, patience, is_new_best, should_stop = _early_stop_step(19.0, 20.0, 7, 8)
    assert best == 20.0 and patience == 8 and not is_new_best and should_stop

def test_early_stop_loop_saves_best_and_breaks():
    """用真实早停逻辑模拟一串验证 PSNR：应在提升 epoch 保存最优，连续无提升后提前 break。"""
    val_psnrs = [18.0, 19.0, 18.5, 18.6, 18.7, 18.8, 18.9, 18.95, 18.99]
    patience_limit = 6
    best, patience, saves, stopped_at = -1.0, 0, [], None
    for e, v in enumerate(val_psnrs):
        best, patience, is_new_best, should_stop = _early_stop_step(v, best, patience, patience_limit)
        if is_new_best:
            saves.append(e)
        if should_stop:
            stopped_at = e
            break
    assert saves == [0, 1]  # 仅前两个 epoch 提升，后续均低于 best=19.0
    assert stopped_at == 7  # 第 2~7 连续 6 次无提升后早停
