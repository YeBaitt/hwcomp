#!/usr/bin/env python
"""Analyze fine-tuned IRControlNet checkpoints vs the original. CPU-only."""
import torch
import numpy as np
import json

ORIG = "/home/liaitong/hw_comp/vendor/DiffBIR/weights/DiffBIR_v2.1_orig.pt"
FTS = [
    "/home/liaitong/hw_comp/checkpoints/stage2_ft/checkpoints/0001000.pt",
    "/home/liaitong/hw_comp/checkpoints/stage2_ft/checkpoints/0002000.pt",
    "/home/liaitong/hw_comp/checkpoints/stage2_ft/checkpoints/0003000.pt",
]

def load(path):
    return torch.load(path, map_location="cpu")

def non_finite_scan(sd, label):
    bad = []
    n = 0
    for k, t in sd.items():
        if not torch.is_tensor(t):
            continue
        n += t.numel()
        if not torch.isfinite(t).all():
            bad.append(k)
    return bad, n

def count_nonfinite(sd):
    cnt = 0
    for k, t in sd.items():
        if torch.is_tensor(t):
            cnt += int((~torch.isfinite(t)).sum().item())
    return cnt

print("=" * 80)
print("Loading original ...")
base = load(ORIG)
print("Original dtype of params:", sorted(set(str(t.dtype) for t in base.values())))
print("num params in orig:", len(base))

# Per-parameter reference norms
base_l2 = {k: (t.float() ** 2).sum().item() for k, t in base.items() if torch.is_tensor(t)}

# NaNs in original
nan_base = count_nonfinite(base)
print(f"original non-finite count: {nan_base}")

eps = torch.finfo(torch.float32).eps  # 1.19e-7

report = {}
all_names = list(base.keys())
for path in FTS:
    step = path.rsplit("/", 1)[1].replace(".pt", "")
    sd = load(path)
    print("=" * 80)
    print(f"STEP {step}: dtype(s) = {sorted(set(str(t.dtype) for t in sd.values()))}")

    assert set(sd.keys()) == set(base.keys()), f"key mismatch at {step}"
    nf = count_nonfinite(sd)
    print(f"  non-finite count: {nf}")
    if nf:
        bad = [k for k, t in sd.items() if torch.is_tensor(t) and not torch.isfinite(t).all()]
        print("  non-finite param names:", bad[:20])

    # Per-param relative drift ||ft-base||_2 / ||base||_2
    rel = {}
    sqdiff = 0.0
    sqbase = 0.0
    n_ident = 0
    n_below_eps = 0
    n_total = 0
    for k in all_names:
        b = base[k].float()
        f = sd[k].float()
        d = f - b
        db = (d ** 2).sum().item()
        bb = base_l2[k]
        sqdiff += db
        sqbase += bb
        n_total += d.numel()
        if db == 0.0:
            n_ident += d.numel()
        # relative per-element scale of the change vs base scale
        rel[k] = np.sqrt(db) / np.sqrt(bb) if bb > 0 else (np.sqrt(db) if db > 0 else 0.0)
        if bb > 0 and rel[k] < eps:
            n_below_eps += d.numel()

    # overall relative drift (norm ratio)
    overall = np.sqrt(sqdiff) / np.sqrt(sqbase) if sqbase > 0 else 0.0

    # rank top-10
    top = sorted(rel.items(), key=lambda kv: -kv[1])[:10]
    print(f"  OVERALL relative L2 drift ||ft-base||_2/||base||_2 = {overall:.6e}")
    print(f"  #params identical (diff==0 exactly): {n_ident} / {n_total} ({100.0*n_ident/n_total:.4f}%)")
    print(f"  #params drift < float32 eps ({eps:.2e}): {n_below_eps} / {n_total} ({100.0*n_below_eps/n_total:.4f}%)")
    print(f"  #params drift < 1e-3: ", end="")
    n_low = sum(1 for k in all_names if base_l2[k] > 0 and rel[k] < 1e-3)
    print(f"{n_low}/{len(all_names)}")
    print("  TOP-10 drifted layers:")
    for k, v in top:
        print(f"    {v:.6e}  {k}")

    report[step] = {
        "overall_rel_drift": float(overall),
        "non_finite": int(nf),
        "n_identical": int(n_ident),
        "n_total": int(n_total),
        "below_eps_frac": float(n_below_eps / n_total),
        "top10": [(k, float(v)) for k, v in top],
        "dtype": sorted(set(str(t.dtype) for t in sd.values())),
    }
    del sd

# --- pairwise distinctness ---
print("=" * 80)
print("PAIRWISE DISTINCTNESS (fraction of params bit-identical)")
sds = {("orig"): load(ORIG)}
for path in FTS:
    step = path.rsplit("/", 1)[1].replace(".pt", "")
    sds[step] = load(path)

labels = list(sds.keys())
def frac_identical(a, b):
    tot = 0
    same = 0
    for k in all_names:
        ta, tb = a[k], b[k]
        n = ta.numel()
        tot += n
        # byte-level compare via to('cpu').contiguous().view(torch.uint8)
        ba = ta.to("cpu").contiguous().view(torch.uint8)
        bb = tb.to("cpu").contiguous().view(torch.uint8)
        same += int(torch.eq(ba, bb).sum().item())
    return same / tot

for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        f = frac_identical(sds[labels[i]], sds[labels[j]])
        print(f"  {labels[i]:>8} vs {labels[j]:<8}: {100.0*f:.4f}% bit-identical params")

# --- cross-step drift table + monotonicity ---
print("=" * 80)
print("CROSS-STEP DRIFT SUMMARY")
for step, r in report.items():
    print(f"  step {step}: overall={r['overall_rel_drift']:.6e} nonfinite={r['non_finite']} "
          f"identical_frac={r['n_identical']/r['n_total']:.2e} below_eps_frac={r['below_eps_frac']:.2e} dtype={r['dtype']}")

ov = [report[s]["overall_rel_drift"] for s in sorted(report)]
print("  overall drift sequence:", ["%.4e" % x for x in ov])
mono = all(ov[i] < ov[i + 1] for i in range(len(ov) - 1))
print("  monotonically increasing:", mono)
for s in sorted(report):
    o = report[s]["overall_rel_drift"]
    flag = "  <-- SUSPICIOUS (zero drift)" if o == 0 else ("  <-- LARGE (>1e-2)" if o > 1e-2 else "")
    print(f"    step {s}: {o:.6e}{flag}")

with open("/home/liaitong/hw_comp/analyze_ft_drift_report.json", "w") as fh:
    json.dump(report, fh, indent=2)
print("JSON report written to /home/liaitong/hw_comp/analyze_ft_drift_report.json")
