import cv2
import numpy as np

from enhance.data.preprocess import build_pair_cache, check_color_consistency

def _write_pair(root, name, gain=1.0, off=0.0):
    yy, xx = np.mgrid[0:32, 0:48]
    base = ((np.sin(yy / 5) * np.cos(xx / 4) + 1) / 2).astype(np.float32)
    hr = np.stack([base, base * 0.9, base * 1.1], axis=-1)
    lq = np.clip(hr * gain + off, 0.0, 1.0)
    cv2.imwrite(str(root / f"{name}_ARC.png"), cv2.cvtColor((lq * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(root / f"{name}_ARC_gt.png"), cv2.cvtColor((hr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

def _write_pair_res(root, name, lq_h, lq_w, hr_h, hr_w):
    yy, xx = np.mgrid[0:hr_h, 0:hr_w]
    base = ((np.sin(yy / 5) * np.cos(xx / 4) + 1) / 2).astype(np.float32)
    hr = np.stack([base, base * 0.9, base * 1.1], axis=-1)
    lq = cv2.resize(hr, (lq_w, lq_h))
    cv2.imwrite(str(root / f"{name}_ARC.png"), cv2.cvtColor((lq * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(root / f"{name}_ARC_gt.png"), cv2.cvtColor((hr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

def test_build_pair_cache(tmp_path):
    root = tmp_path / "pairs"
    cache = tmp_path / "cache"
    root.mkdir()
    _write_pair(root, "a")
    _write_pair(root, "b")
    assert build_pair_cache(root, cache, num_workers=1) == 2
    for stem in ("a_ARC", "b_ARC"):
        p = cache / f"{stem}.npz"
        assert p.exists()
        d = np.load(p)
        assert set(d.files) == {"lq", "hr"}
        assert d["lq"].dtype == np.uint8 and d["hr"].dtype == np.uint8
        assert d["lq"].shape == (32, 48, 3) and d["hr"].shape == (32, 48, 3)

def test_build_pair_cache_resume(tmp_path):
    root = tmp_path / "pairs"
    cache = tmp_path / "cache"
    root.mkdir()
    _write_pair(root, "a")
    assert build_pair_cache(root, cache, num_workers=1) == 1
    assert build_pair_cache(root, cache, num_workers=1) == 0  # 已存在 → 跳过
    _write_pair(root, "b")
    assert build_pair_cache(root, cache, num_workers=1) == 1

def test_check_color_consistency_runs(tmp_path, capsys):
    root = tmp_path / "pairs"
    cache = tmp_path / "cache"
    root.mkdir()
    _write_pair(root, "a", gain=1.1, off=0.05)
    _write_pair(root, "b", gain=0.9, off=-0.03)
    build_pair_cache(root, cache, num_workers=1)
    check_color_consistency(root, cache, sample_n=2)
    out = capsys.readouterr().out
    assert "mean|gain-1|" in out
    assert "mean|offset|" in out

def test_check_color_consistency_different_res(tmp_path, capsys):
    # 不同分辨率（lq 32x48、hr 64x96）：旧实现 lstsq 行数不匹配会抛 LinAlgError；
    # 新实现先把 HR 缩到 LQ 分辨率再估计，且增益/偏移≈单位，不应标记可疑
    root = tmp_path / "pairs"
    cache = tmp_path / "cache"
    root.mkdir()
    _write_pair_res(root, "a", 32, 48, 64, 96)
    build_pair_cache(root, cache, num_workers=1)
    check_color_consistency(root, cache, sample_n=1)
    out = capsys.readouterr().out
    assert "mean|gain-1|" in out
    assert "未发现可疑对" in out

def test_build_pair_cache_subdir_key(tmp_path):
    # 不同子目录下同名 lq 应生成不同缓存键（相对路径），避免 npz 覆盖
    root = tmp_path / "pairs"
    (root / "sub1").mkdir(parents=True)
    (root / "sub2").mkdir(parents=True)
    _write_pair(root / "sub1", "a")
    _write_pair(root / "sub2", "a")
    cache = tmp_path / "cache"
    assert build_pair_cache(root, cache, num_workers=1) == 2
    assert (cache / "sub1__a_ARC.npz").exists()
    assert (cache / "sub2__a_ARC.npz").exists()
