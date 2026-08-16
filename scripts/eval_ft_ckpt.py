"""微调 stage2 gate 评测 runner：baseline + 各 ckpt × 多个 knob 配置。

每个 (weight, knobs) 组合替换部署权重 → 跑 eval_val_crop → 恢复原权重。
结果追加写 /tmp/stage2_ft_eval_results.txt（含 NIQE 列）。
用法:
  python scripts/eval_ft_ckpt.py --baseline --steps 1000,2000,3000 \
      --knobs "0.4,1.0,1.0,8.0" "1.0,0.0,0.0,8.0" --crop 1024
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

_WEIGHTS = Path("/home/liaitong/hw_comp/vendor/DiffBIR/weights")
_ORIG = _WEIGHTS / "DiffBIR_v2.1_orig.pt"
_DEPLOY = _WEIGHTS / "DiffBIR_v2.1.pt"
_CKPT_DIR = Path("/home/liaitong/hw_comp/checkpoints/stage2_ft/checkpoints")
_RESULTS = Path("/tmp/stage2_ft_eval_results.txt")
_CROP = Path("/home/liaitong/hw_comp/scripts/eval_val_crop.py")

# 备份自 FT 前、crop 基线 (-1.9dB) 同权重的部署文件，即真·原始 v2.1 controlnet。
_ORIG_MD5 = "ea976b4ba586954aed9c2c3ac39ed5b9"


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=str, default="1000,2000,3000")
    ap.add_argument("--knobs", type=str, nargs="+", default=["0.4,1.0,1.0,8.0"])
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--baseline", action="store_true", help="先跑原权重 baseline")
    ap.add_argument("--ckpt-dir", type=str, default=str(_CKPT_DIR),
                    help="微调 checkpoint 目录（默认 ft1000 的 stage2_ft/checkpoints）")
    args = ap.parse_args()
    ckpt_dir = Path(args.ckpt_dir)

    assert _ORIG.exists(), f"缺原权重备份 {_ORIG}"
    assert _md5(_ORIG) == _ORIG_MD5, f"{_ORIG} md5 {_md5(_ORIG)} != 真值 {_ORIG_MD5}，拒绝运行"
    assert _md5(_DEPLOY) == _ORIG_MD5, "部署权重非原权重，先恢复再跑"

    steps = [int(s) for s in args.steps.split(",")] if args.steps else []
    for ckpt in [f"baseline"] + [f"{s:07d}.pt" for s in steps]:
        if ckpt == "baseline":
            if not args.baseline:
                continue
            # 原权重已在位，无需交换
            src = _ORIG
        else:
            src = _CKPT_DIR / ckpt
            assert src.exists(), f"缺 ckpt {src}"
        for kc in args.knobs:
            lam = kc.split(",")[0]
            tag = f"{ckpt.removesuffix('.pt')}_lam{lam}"
            try:
                shutil.copyfile(src, _DEPLOY)  # baseline 时 src==_ORIG，内容幂等
                print(f"[eval] 权重 {src.name}   knobs=({kc})", flush=True)
                r = subprocess.run(
                    [sys.executable, str(_CROP), "--knobs", kc, "--tag", tag,
                     "--crop", str(args.crop)],
                    capture_output=True, text=True,
                )
                with _RESULTS.open("a") as f:
                    f.write(f"=== {ckpt} knobs=({kc}) tag {tag} ===\n")
                    for line in r.stdout.splitlines():
                        s = line.strip()
                        if s and (s[0].isdigit() or (s.startswith("case") and "ΔPSNR" not in s)):
                            f.write(s + "\n")
                    f.write("\n")
                if r.returncode != 0:
                    print(f"[eval] {ckpt}/{kc} 失败 rc={r.returncode}\n{r.stderr[-2000:]}", flush=True)
            finally:
                shutil.copyfile(_ORIG, _DEPLOY)  # 无论成败恢复原权重
    assert _md5(_DEPLOY) == _ORIG_MD5, "结束后部署权重 != 原权重！"


if __name__ == "__main__":
    main()
