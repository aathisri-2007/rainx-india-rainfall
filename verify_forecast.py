from pathlib import Path
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"
REPORTS    = Path(__file__).parent / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("VERIFICATION REPORT — Raw vs Corrected Forecast")
print("=" * 60)

df = pd.read_csv(MODELS_DIR / "test_predictions.csv", parse_dates=["date"])

OBS = df["rainfall_mm"].values
RAW = df["pred_raw"].values
CORR = df["pred_regime_feat"].values    # regime-aware corrected


def rmse(o, p):  return float(np.sqrt(np.mean((o - p)**2)))
def mae(o, p):   return float(np.mean(np.abs(o - p)))
def bias(o, p):  return float(np.mean(p - o))


def contingency(o, p, thr):
    """2x2 contingency: hits, misses, false alarms, correct negatives."""
    o_ev = o >= thr
    p_ev = p >= thr
    hits  = int(np.sum( o_ev &  p_ev))
    miss  = int(np.sum( o_ev & ~p_ev))
    fa    = int(np.sum(~o_ev &  p_ev))
    cn    = int(np.sum(~o_ev & ~p_ev))
    return hits, miss, fa, cn


def pod(h, m, f, c):   return h / (h + m) if (h + m) > 0 else float("nan")
def far(h, m, f, c):   return f / (h + f) if (h + f) > 0 else float("nan")
def csi(h, m, f, c):
    d = h + m + f
    return h / d if d > 0 else float("nan")
def ets(h, m, f, c):
    d = h + m + f
    if d == 0: return float("nan")
    n = h + m + f + c
    h_rand = (h + m) * (h + f) / n
    denom = d - h_rand
    return (h - h_rand) / denom if denom > 0 else float("nan")
def fss(o, p, thr, win=3):
    """Simple FSS using a 1D neighbourhood (row order)."""
    o_ev = (o >= thr).astype(float)
    p_ev = (p >= thr).astype(float)
    if o_ev.sum() == 0 or p_ev.sum() == 0:
        return float("nan")
    def smooth(a):
        pad = win // 2
        ap = np.pad(a, pad, mode="edge")
        return np.convolve(ap, np.ones(win)/win, mode="valid")[:len(a)]
    os_, ps = smooth(o_ev), smooth(p_ev)
    mse_f = np.mean((os_ - ps)**2)
    mse_ref = np.mean(os_**2 + ps**2)
    return float(1 - mse_f / mse_ref) if mse_ref > 0 else float("nan")


print("\n========== CONTINUOUS METRICS ==========")
print(f"{'Metric':<10} {'Raw GEFS':>12} {'Corrected':>12}")
print("-" * 36)
print(f"{'RMSE':<10} {rmse(OBS, RAW):>12.3f} {rmse(OBS, CORR):>12.3f}")
print(f"{'MAE':<10} {mae(OBS, RAW):>12.3f} {mae(OBS, CORR):>12.3f}")
print(f"{'Bias':<10} {bias(OBS, RAW):>12.3f} {bias(OBS, CORR):>12.3f}")

THRESHOLDS = [
    ("Light (>=2.5mm)",        2.5),
    ("Moderate (>=20mm)",      20.0),
    ("Heavy (>=64.5mm)",       64.5),
    ("Very heavy (>=115.6mm)", 115.6),
    ("Extr. heavy (>=204.5mm)",204.5),
]

print("\n========== EVENT METRICS ==========")
print(f"{'Threshold':<24} {'Model':<10} {'POD':>7} {'FAR':>7} {'CSI':>7} {'ETS':>7} {'FSS':>7}")
print("-" * 80)

report_rows = []
for label, thr in THRESHOLDS:
    for model_name, preds in [("Raw", RAW), ("Corrected", CORR)]:
        h, m, f, c = contingency(OBS, preds, thr)
        row = {
            "threshold": label,
            "threshold_mm": thr,
            "model": model_name,
            "hits": h, "misses": m, "false_alarms": f, "correct_negatives": c,
            "POD": pod(h, m, f, c),
            "FAR": far(h, m, f, c),
            "CSI": csi(h, m, f, c),
            "ETS": ets(h, m, f, c),
            "FSS": fss(OBS, preds, thr, win=3),
        }
        report_rows.append(row)
        print(f"{label:<24} {model_name:<10} "
              f"{row['POD']:>7.3f} {row['FAR']:>7.3f} {row['CSI']:>7.3f} "
              f"{row['ETS']:>7.3f} {row['FSS']:>7.3f}")

rep = pd.DataFrame(report_rows)
out_csv = REPORTS / "verification_report.csv"
rep.to_csv(out_csv, index=False)
print(f"\nSaved verification report: {out_csv}")

# --- Summary by threshold ---
print("\n========== SUMMARY: CORRECTED vs RAW improvement ==========")
for label, thr in THRESHOLDS:
    r_raw = rep[(rep.threshold == label) & (rep.model == "Raw")].iloc[0]
    r_cor = rep[(rep.threshold == label) & (rep.model == "Corrected")].iloc[0]
    delta_csi = r_cor["CSI"] - r_raw["CSI"]
    delta_ets = r_cor["ETS"] - r_raw["ETS"]
    print(f"  {label:<24}  dCSI={delta_csi:+.3f}   dETS={delta_ets:+.3f}")