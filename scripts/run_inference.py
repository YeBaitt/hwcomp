"""100 张 test 图推理 → output_dir/caseXX.jpg，支持断点续跑与自动打包。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enhance.config import Config
from enhance.inference.engine import EnhancementEngine
from enhance.submit.package import package, validate_names

def main():
    cfg = Config.from_yaml(Path("config.yaml"))
    engine = EnhancementEngine(cfg)
    test_dir = Path(cfg.test_dir)
    out_dir = Path(cfg.submit_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted(test_dir.glob("case*.jpg"))
    if not cases:
        raise SystemExit("test_dir 未匹配到 case*.jpg")
    for i, p in enumerate(cases):
        # 输出名 = 输入 stem 去掉 _lq 后缀（若存在），保证与官方 caseN.jpg 严格对应
        out_name = p.stem[:-3] if p.stem.endswith("_lq") else p.stem
        out = out_dir / f"{out_name}.jpg"
        if out.exists():
            print(f"[{i + 1}/{len(cases)}] 跳过已存在 {out.name}", flush=True)
            continue
        print(f"[{i + 1}/{len(cases)}] 处理 {p.name}", flush=True)
        try:
            tmp_out = out.with_suffix(".tmp.jpg")
            engine.enhance_path(p, tmp_out)
            os.replace(tmp_out, out)
        except Exception as exc:
            # 单张失败不中断整批：跳过并在命名校验阶段汇总，重跑时续跑补齐
            print(f"[{i + 1}/{len(cases)}] {p.name} 失败（{type(exc).__name__}: {exc}），跳过", flush=True)
    # 命名校验
    expected = [f"case{i}.jpg" for i in range(1, len(cases) + 1)]
    errs = validate_names(list(out_dir.glob("*.jpg")), expected)
    if errs:
        print("命名错误：", errs, flush=True)
        raise SystemExit(1)
    package(out_dir, Path(cfg.zip_path))
    print("打包完成：", cfg.zip_path, flush=True)

if __name__ == "__main__":
    main()
