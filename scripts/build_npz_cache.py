"""为合成图像对生成 npz uint8 缓存，加速 stage1 训练解码。

load_pair_float 的 npz 缓存键 = lq 相对 pairs_root 的路径（去扩展名、分隔符转 __）。
本脚本并行解码 PNG → 写 npz（不压缩，读写最快；uint8 保持原始精度）。

用法:
  python scripts/build_npz_cache.py --pairs dataset/div8k_syn/train --cache dataset/div8k_syn/cache
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def _encode_one(args):
    pair_dir, cache_dir, lq_rel = args
    lq_path = Path(pair_dir) / lq_rel
    gt_path = lq_path.with_name(lq_path.stem + "_gt.png")
    lq = cv2.cvtColor(cv2.imread(str(lq_path)), cv2.COLOR_BGR2RGB)
    hr = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB)
    if lq is None or hr is None:
        return (lq_rel, False)
    key = Path(lq_rel).with_suffix("").as_posix().replace("/", "__")
    np.savez(Path(cache_dir) / f"{key}.npz", lq=lq, hr=hr)
    return (lq_rel, True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("dataset/div8k_syn/train"))
    ap.add_argument("--cache", type=Path, default=Path("dataset/div8k_syn/cache"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    lq_files = sorted(args.pairs.glob("*_ARC.png"))
    pending = [f.relative_to(args.pairs).as_posix() for f in lq_files
               if not (args.cache / (Path(f.relative_to(args.pairs)).with_suffix("").as_posix().replace("/", "__") + ".npz")).exists()]
    print(f"[npz] 待生成 {len(pending)}/{len(lq_files)} 张缓存 → {args.cache}")

    t0 = time.time()
    ok = 0
    fail = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rel, success in ex.map(_encode_one, [(str(args.pairs), str(args.cache), r) for r in pending]):
            if success:
                ok += 1
            else:
                fail.append(rel)
    print(f"[npz] 完成 {ok}/{len(pending)}，失败 {len(fail)}，耗时 {time.time() - t0:.0f}s")
    for f in fail[:10]:
        print(f"  FAIL {f}")
    if fail:
        (args.pairs.parent / "bad_pairs.txt").write_text("\n".join(fail) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
