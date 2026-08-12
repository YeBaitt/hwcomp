"""端到端冒烟：1 张 val 图走通 增强→指标，输出对比表。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from enhance.config import Config
from enhance.evaluate.metrics import report
from enhance.inference.engine import EnhancementEngine

def main():
    cfg = Config.from_yaml(Path("config.yaml"))
    engine = EnhancementEngine(cfg)
    val_dir = Path(cfg.val_dir)
    lq = cv2.cvtColor(cv2.imread(str(val_dir / "case1_lq.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gt = cv2.cvtColor(cv2.imread(str(val_dir / "case1_gt.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = engine.enhance(lq)
    base = report(lq, gt)
    enh = report(out, gt)
    print("LQ   :", {k: round(v, 3) for k, v in base.items()})
    print("增强后:", {k: round(v, 3) for k, v in enh.items()})

if __name__ == "__main__":
    main()
