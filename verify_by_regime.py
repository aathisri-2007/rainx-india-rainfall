from pathlib import Path
import numpy as np
import pandas as pd

MODELS  = Path(__file__).parent / "models"
REPORTS = Path(__file__).parent / "reports"

df = pd.read_csv(MODELS / "test_predictions.csv", parse_dates=["date"])

def rmse(o, p): return float(np.sqrt(np.mean((o-p)**2)))
def bias(o, p): return float(np.mean(p-o))

def contingency(o, p, thr):
    oe = o >= thr; pe = p >= thr
    return (int((oe&pe).sum()), int((oe&~pe).sum()),
            int((~oe&pe).sum()), int((~oe&~pe).sum()))

def pod(h,m,f,c): return h/(h+m) if (h+m)>0 else float("nan")
def far(h,m,f,c): return f/(h+f) if (h+f)>0 else float("nan")
def csi(h,m,f,c):
    d = h+m+f
    return h/d if d>0 else float("nan")
def ets(h,m,f,c):
    d = h+m+f
    if d == 0: return float("nan")
    n = h+m+f+c
    h_rand = (h+m)*(h+f)/n
    den = d - h_rand
    return (h - h_rand)/den if den > 0 else float("nan")

rows = []
for regime in sorted(df["regime_label"].unique()):
    sub = df[df["regime_label"] == regime]
    o = sub["rainfall_mm"].values
    raw = sub["gefs_mm"].values
    # prefer pred_regime_feat (QM), fallback pred_global (RF)
    corr_col = "pred_regime_feat" if "pred_regime_feat" in sub.columns else "pred_global"
    corr = sub[corr_col].values

    row = {
        "regime": regime,
        "n": int(len(sub)),
        "rmse_raw": round(rmse(o, raw), 3),
        "rmse_corr": round(rmse(o, corr), 3),
        "bias_raw": round(bias(o, raw), 3),
        "bias_corr": round(bias(o, corr), 3),
    }

    for label, thr in [("light", 2.5), ("moderate", 20.0), ("heavy", 64.5)]:
        hr, mr, fr, cr = contingency(o, raw, thr)
        hc, mc, fc, cc = contingency(o, corr, thr)
        row[f"csi_{label}_raw"]  = round(csi(hr, mr, fr, cr), 3) if (hr+mr+fr)>0 else 0.0
        row[f"csi_{label}_corr"] = round(csi(hc, mc, fc, cc), 3) if (hc+mc+fc)>0 else 0.0
        row[f"pod_{label}_corr"] = round(pod(hc, mc, fc, cc), 3) if (hc+mc)>0 else 0.0

    rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(REPORTS / "verification_by_regime.csv", index=False)

print("="*70)
print("PER-REGIME VERIFICATION")
print("="*70)
print(out.to_string(index=False))
print(f"\nSaved: {REPORTS / 'verification_by_regime.csv'}")