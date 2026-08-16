"""汇总 analyze_test_nr.py 产出的 CSV：probe vs my_work 配对差、胜场数与均值表。

用法:
  python scripts/analyze_nr_summary.py --csv /tmp/test_nr_metrics.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

LOWER_BETTER = ["niqe", "brisque", "piqe", "ilniqe"]
HIGHER_BETTER = ["musiq", "maniqa", "nima", "topiq_nr", "clipiqa"]
ALL_M = LOWER_BETTER + HIGHER_BETTER
EXTRA = ["mean_luma", "std_luma", "lapvar", "jpeg_bytes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="/tmp/test_nr_metrics.csv")
    args = ap.parse_args()

    rows = defaultdict(dict)  # case -> src -> {metric: value}
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            case, src = r["case"], r["src"]
            rows[case][src] = {k: float(v) for k, v in r.items()
                               if k in ALL_M + EXTRA and v and "ERR" not in v}
    cases = sorted(rows)
    srcs = ["input", "probe", "mywork"]

    print(f"== 每指标均值（{len(cases)} 张，input=原图基线）==")
    print(f"{'metric':<12}" + "".join(f"{s:>12}" for s in srcs))
    means = {}
    for m in ALL_M + EXTRA:
        line = f"{m:<12}"
        for s in srcs:
            vals = [rows[c][s][m] for c in cases if m in rows[c][s]]
            means[(m, s)] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))
            line += f"{np.mean(vals):>12.4f}" if vals else f"{'—':>12}"
        print(line)

    print(f"\n== probe − my_work 配对差（{len(cases)} 张，正值=probe 更大）==")
    print(f"{'metric':<12}{'meanΔ':>9}{'stdΔ':>9}{'probe胜':>10}")
    summary = {}
    for m in ALL_M + EXTRA:
        d = [rows[c]["probe"][m] - rows[c]["mywork"][m] for c in cases
             if m in rows[c]["probe"] and m in rows[c]["mywork"]]
        if not d:
            continue
        md, sd = float(np.mean(d)), float(np.std(d))
        wins = (sum(1 for x in d if x < 0) if m in LOWER_BETTER else
                sum(1 for x in d if x > 0) if m in HIGHER_BETTER else 0)
        summary[m] = {"mean_delta": md, "std_delta": sd, "probe_wins": wins, "n": len(d)}
        better = "↑" if (m in HIGHER_BETTER and md > 0) or (m in LOWER_BETTER and md < 0) else "↓"
        print(f"{m:<12}{md:>9.4f}{sd:>9.4f}{wins:>8}/{len(d)}{better:>4}")

    print("\n（参考官方分：probe 3.188 > my_work 2.82，probe 胜 +0.368）")
    print("与官方一致（probe 更好）的指标：", [m for m in ALL_M if m in summary and
          ((summary[m]["mean_delta"] > 0 and m in HIGHER_BETTER) or
           (summary[m]["mean_delta"] < 0 and m in LOWER_BETTER))])


if __name__ == "__main__":
    main()
