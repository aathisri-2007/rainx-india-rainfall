from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from functools import lru_cache
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

BASE    = Path(__file__).parent
REPORTS = BASE / "reports"
MODELS  = BASE / "models"
ADMIN   = BASE / "data" / "admin"
STATIC  = BASE / "static"

REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
STATIC.mkdir(exist_ok=True)

# Data file paths — all relative to project root for cloud deployment
IMD_DISTRICTS = BASE / "data" / "district_daily_rainfall_2016_2025.csv"
REGIME_LABELS = BASE / "data" / "regimes" / "regime_labels_2016.csv"
REAN_DIR      = BASE / "data" / "reanalysis"

app = FastAPI(title="RainX — India Rainfall Forecast Post-Processing API")

# Serve static files (logo, etc.)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _s(v):
    if v is None: return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f): return None
        return f
    except Exception:
        return v


@app.get("/", response_class=HTMLResponse)
def root():
    return (BASE / "index.html").read_text(encoding="utf-8")


# ============================================================
# CACHES
# ============================================================
_IMD_CACHE = None

def _load_imd():
    global _IMD_CACHE
    if _IMD_CACHE is None:
        if not IMD_DISTRICTS.exists():
            print(f"WARNING: IMD file not found at {IMD_DISTRICTS}")
            return pd.DataFrame(columns=["date","district_id","rainfall_mm"])
        print(f"Loading IMD from {IMD_DISTRICTS}...")
        _IMD_CACHE = pd.read_csv(IMD_DISTRICTS,
                                 usecols=["date","district_id","rainfall_mm"],
                                 parse_dates=["date"])
        print(f"  IMD: {len(_IMD_CACHE):,} rows")
    return _IMD_CACHE


@lru_cache(maxsize=1)
def _load_district_meta():
    p = REPORTS / "district_forecast_latest.csv"
    if not p.exists(): return {}
    df = pd.read_csv(p, usecols=["district_id","district","state"])
    df["district_id"] = df["district_id"].astype(str)
    return dict(zip(df["district_id"], zip(df["district"], df["state"])))


@lru_cache(maxsize=1)
def _load_live_df():
    p = REPORTS / "realtime_forecast.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    df["district_id"] = df["district_id"].astype(int).astype(str)
    return df


@lru_cache(maxsize=1)
def _load_geojson_base():
    from shapely.geometry import shape, mapping
    # Prefer the simplified version (smaller, cloud-friendly)
    candidates = [
        ADMIN / "district_simplified.geojson",
        ADMIN / "district_nwic_wgs84.geojson",
    ]
    src = next((c for c in candidates if c.exists()), None)
    if src is None:
        return {"type":"FeatureCollection","features":[]}
    with open(src, encoding="utf-8") as f:
        gj = json.load(f)
    out = []
    for feat in gj["features"]:
        try:
            geom = shape(feat["geometry"]).simplify(0.02, preserve_topology=True)
            out.append({"type":"Feature",
                        "properties":feat["properties"],
                        "geometry":mapping(geom)})
        except Exception:
            out.append(feat)
    return {"type":"FeatureCollection","features":out}


@lru_cache(maxsize=1)
def _load_historical_predictions():
    p = MODELS / "test_predictions.csv"
    if not p.exists(): return None
    df = pd.read_csv(p, parse_dates=["date"])
    df["district_id"] = df["district_id"].astype(str)
    meta = _load_district_meta()
    df["district"] = df["district_id"].map(lambda x: meta.get(x, ("District "+x,""))[0])
    df["state"]    = df["district_id"].map(lambda x: meta.get(x, ("","Unknown"))[1])
    return df


def _invalidate_cache():
    _load_live_df.cache_clear()
    _load_geojson_base.cache_clear()
    _load_historical_predictions.cache_clear()
    _load_district_meta.cache_clear()


# ============================================================
# LIVE FORECAST
# ============================================================
@app.get("/api/live_summary")
def live_summary():
    df = _load_live_df()
    if df is None: return {"available": False}
    prob_cols = [c for c in ["P_heavy","P_very_heavy","P_extremely_heavy"] if c in df.columns]
    top = df.nlargest(20, "pred_regime_feat")[
        ["district_id","district","state","regime_label","gefs_mm","pred_regime_feat"] + prob_cols
    ].round(3).to_dict(orient="records")
    return {
        "available":        True,
        "date":             str(df["date"].iloc[0]),
        "regime":           str(df["regime_label"].mode().iloc[0]) if "regime_label" in df.columns else "unknown",
        "n_districts":      int(len(df)),
        "mean_raw":         round(float(df["gefs_mm"].mean()), 2),
        "mean_corrected":   round(float(df["pred_regime_feat"].mean()), 2),
        "max_corrected":    round(float(df["pred_regime_feat"].max()), 2),
        "heavy_count":      int((df["pred_regime_feat"] >= 64.5).sum()),
        "very_heavy_count": int((df["pred_regime_feat"] >= 115.6).sum()),
        "n_regimes":        int(df["regime_label"].nunique()) if "regime_label" in df.columns else 1,
        "top_districts":    top,
    }


@app.get("/api/live_districts")
def live_districts():
    df = _load_live_df()
    if df is None: return []
    cols = ["district_id","district","state","regime_label","gefs_mm","pred_regime_feat"]
    if "regime_confidence" in df.columns: cols.append("regime_confidence")
    prob_cols = [c for c in ["P_heavy","P_very_heavy","P_extremely_heavy"] if c in df.columns]
    cols += prob_cols
    return df[cols].round(4).to_dict(orient="records")


@app.get("/api/live_map_data")
def live_map_data():
    df = _load_live_df()
    gj = _load_geojson_base()
    if df is None:
        return {"type":"FeatureCollection","features":[]}
    fmap = dict(zip(df["district_id"], df["pred_regime_feat"]))
    rawmap = dict(zip(df["district_id"], df["gefs_mm"]))
    regmap = dict(zip(df["district_id"], df["regime_label"])) if "regime_label" in df.columns else {}
    confmap = dict(zip(df["district_id"], df["regime_confidence"])) if "regime_confidence" in df.columns else {}
    probmap = dict(zip(df["district_id"], df["P_heavy"])) if "P_heavy" in df.columns else {}

    out_features = []
    for feat in gj["features"]:
        p2 = feat["properties"]
        raw = str(p2.get("dtcode",""))
        norm = str(int(raw)) if raw.isdigit() else raw
        out_features.append({
            "type":"Feature",
            "properties":{
                "district": p2.get("district"),
                "state_name": p2.get("state_name"),
                "dtcode": p2.get("dtcode"),
                "rain_mm": _s(fmap.get(norm)),
                "raw_mm": _s(rawmap.get(norm)),
                "regime": regmap.get(norm,"-"),
                "regime_confidence": _s(confmap.get(norm)) if confmap else None,
                "p_heavy": _s(probmap.get(norm)) if probmap else None,
            },
            "geometry": feat["geometry"],
        })
    return {"type":"FeatureCollection","features":out_features}


@app.get("/api/live_district/{district_id}")
def live_district(district_id: str):
    df = _load_live_df()
    if df is None: return {"error":"no live forecast"}
    row = df[df["district_id"] == district_id]
    if len(row) == 0: return {"error":"not found"}
    r = row.iloc[0]
    return {
        "district_id":       str(r["district_id"]),
        "district":          str(r["district"]),
        "state":             str(r["state"]),
        "date":              str(r["date"]),
        "regime":            str(r.get("regime_label","-")),
        "regime_confidence": _s(r.get("regime_confidence", None)),
        "raw_mm":            round(float(r["gefs_mm"]), 2),
        "corrected_mm":      round(float(r["pred_regime_feat"]), 2),
        "correction_mm":     round(float(r["pred_regime_feat"]) - float(r["gefs_mm"]), 2),
        "P_heavy":           _s(r.get("P_heavy", None)),
        "P_very_heavy":      _s(r.get("P_very_heavy", None)),
        "P_extremely_heavy": _s(r.get("P_extremely_heavy", None)),
    }


@app.get("/api/district_features/{district_id}")
def district_features(district_id: str):
    df = _load_live_df()
    if df is None: return {"available": False, "message":"No live forecast"}
    row = df[df["district_id"] == district_id]
    if len(row) == 0: return {"available": False, "message":"Not found"}
    r = row.iloc[0]
    return {
        "available":         True,
        "district_id":       str(r["district_id"]),
        "district":          str(r["district"]),
        "state":             str(r["state"]),
        "regime":            str(r.get("regime_label","-")),
        "regime_confidence": _s(r.get("regime_confidence", None)),
        "raw_nwp_mm":        round(float(r["gefs_mm"]), 2),
        "corrected_mm":      round(float(r["pred_regime_feat"]), 2),
        "features": {
            "slp_anomaly":    _s(r.get("slp_anomaly", None)),
            "hgt500_anomaly": _s(r.get("hgt500_anomaly", None)),
            "wind_dir_from":  _s(r.get("wind_dir_from", None)),
            "wind_speed_ms":  _s(r.get("wind_speed_ms", None)),
            "rh_anomaly":     _s(r.get("rh_anomaly", None)),
            "monsoon_index":  _s(r.get("monsoon_index", None)),
        },
    }


@app.get("/api/time_series/{district_id}")
def time_series(district_id: str):
    p = MODELS / "test_predictions.csv"
    if not p.exists(): return {"available": False}
    df = pd.read_csv(p, parse_dates=["date"])
    df["district_id"] = df["district_id"].astype(str)
    sub = df[df["district_id"] == district_id].sort_values("date")
    if len(sub) == 0: return {"available": False, "message":"No historical data"}
    return {
        "available": True,
        "dates":     sub["date"].dt.strftime("%Y-%m-%d").tolist(),
        "observed":  sub["rainfall_mm"].round(2).tolist(),
        "raw_nwp":   sub["gefs_mm"].round(2).tolist(),
        "corrected": sub["pred_regime_feat"].round(2).tolist(),
    }


# ============================================================
# DATA STATUS
# ============================================================
@app.get("/api/data_status")
def data_status():
    checks = []
    imd_p = IMD_DISTRICTS
    checks.append({"name":"Observed Rainfall (IMD)","ok":imd_p.exists(),
                   "detail":"2016–2025" if imd_p.exists() else "Not found"})
    gefs_p = BASE / "data" / "gefs_india_2016_2019.csv"
    checks.append({"name":"NWP Forecast (GEFS)","ok":gefs_p.exists(),
                   "detail":"2016–2019" if gefs_p.exists() else "Not found"})
    nc_count = len(list(REAN_DIR.glob("*.nc"))) if REAN_DIR.exists() else 0
    checks.append({"name":"Atmospheric Features (NCEP)","ok":nc_count>0,
                   "detail":f"{nc_count} NetCDF files" if nc_count else "Cached in trained model"})
    gj = ADMIN / "district_simplified.geojson"
    if not gj.exists(): gj = ADMIN / "district_nwic_wgs84.geojson"
    checks.append({"name":"District Boundaries (SOI)","ok":gj.exists(),
                   "detail":"733 districts" if gj.exists() else "Not found"})
    for name, fname in [("Regime Classifier","regime_classifier.joblib"),
                        ("Bias Correction","bias_correction_models.joblib"),
                        ("Probability Model","heavy_rain_probability.joblib")]:
        f = MODELS / fname
        checks.append({"name":name,"ok":f.exists(),
                       "detail":f"{f.stat().st_size/1024:.0f} KB" if f.exists() else "Not loaded"})
    rt = REPORTS / "realtime_forecast.csv"
    checks.append({"name":"Live Forecast","ok":rt.exists(),
                   "detail":"Available" if rt.exists() else "Run fetch_realtime.py"})
    return {"checks":checks, "all_ok":all(c["ok"] for c in checks)}


# ============================================================
# AVAILABLE DATES + HISTORICAL PREDICTIONS
# ============================================================
@app.get("/api/available_dates")
def available_dates():
    dates = []
    rt = REPORTS / "realtime_forecast.csv"
    if rt.exists():
        dates.extend(pd.read_csv(rt, usecols=["date"])["date"].astype(str).str[:10].unique().tolist())
    hist = _load_historical_predictions()
    if hist is not None:
        dates.extend(hist["date"].dt.strftime("%Y-%m-%d").unique().tolist())
    dates = sorted(set(dates), reverse=True)
    return {"latest": dates[0] if dates else None, "all": dates, "count": len(dates)}


def _build_date_response(df, date, is_live):
    prob_cols = [c for c in ["P_heavy","P_very_heavy","P_extremely_heavy"] if c in df.columns]
    regime = df["regime_label"].iloc[0] if "regime_label" in df.columns else "unknown"
    top = df.nlargest(20, "pred_regime_feat")[
        ["district_id","district","state","regime_label","gefs_mm","pred_regime_feat"] + prob_cols
    ].round(3).to_dict(orient="records")
    all_cols = ["district_id","district","state","regime_label","gefs_mm","pred_regime_feat"] + prob_cols
    if "observed_mm" in df.columns: all_cols.append("observed_mm")
    return {
        "available": True, "date": date, "regime": str(regime),
        "n_districts": int(len(df)),
        "mean_raw": round(float(df["gefs_mm"].mean()), 2),
        "mean_corrected": round(float(df["pred_regime_feat"].mean()), 2),
        "max_corrected": round(float(df["pred_regime_feat"].max()), 2),
        "heavy_count": int((df["pred_regime_feat"] >= 64.5).sum()),
        "very_heavy_count": int((df["pred_regime_feat"] >= 115.6).sum()),
        "top_districts": top,
        "all_districts": df[all_cols].round(4).fillna("").to_dict(orient="records"),
        "is_live": is_live,
        "has_observed": "observed_mm" in df.columns,
    }


@app.get("/api/forecast_any")
def forecast_any(date: str = Query(...)):
    rt = REPORTS / "realtime_forecast.csv"
    if rt.exists():
        rtd = pd.read_csv(rt)
        rtd["date"] = rtd["date"].astype(str).str[:10]
        sel = rtd[rtd["date"] == date].copy()
        if len(sel) > 0:
            meta = _load_district_meta()
            if "district" not in sel.columns:
                sel["district"] = sel["district_id"].astype(str).map(lambda x: meta.get(x,("?","?"))[0])
                sel["state"]    = sel["district_id"].astype(str).map(lambda x: meta.get(x,("?","?"))[1])
            return _build_date_response(sel, date, is_live=True)

    hist = _load_historical_predictions()
    if hist is not None:
        hs = hist[hist["date"].dt.strftime("%Y-%m-%d") == date].copy()
        if len(hs) > 0:
            hs = hs.rename(columns={"rainfall_mm": "observed_mm"})
            return _build_date_response(hs, date, is_live=False)

    imd = _load_imd()
    imd_day = imd[imd["date"].dt.strftime("%Y-%m-%d") == date]
    if len(imd_day) > 0:
        meta = _load_district_meta()
        rows = []
        for _, r in imd_day.iterrows():
            did = str(int(r["district_id"]))
            name, state = meta.get(did, ("District " + did, ""))
            rows.append({"district_id":did,"district":name,"state":state,
                         "observed_mm":round(float(r["rainfall_mm"]),2)})
        return {"available": False, "date": date,
                "message":"NWP forecast not available for this date.",
                "observed_only": True, "n_observed": len(rows), "observed": rows}
    return {"available": False, "date": date, "message": f"No data for {date}"}


# ============================================================
# REGIME CLASSIFIER METADATA
# ============================================================
@lru_cache(maxsize=1)
def _regime_metadata():
    p = MODELS / "regime_predictions.csv"
    if not p.exists(): return {"available": False}
    df = pd.read_csv(p)
    bundle = joblib.load(MODELS / "regime_classifier.joblib")
    model = bundle["model"]; feats = bundle["features"]
    labels = sorted(df["regime"].dropna().unique().tolist())
    importances = sorted(
        [{"feature":f,"importance":float(i)} for f,i in zip(feats, model.feature_importances_)],
        key=lambda x:-x["importance"])

    y_all = df["regime"].astype(str)
    counts = y_all.value_counts(); rare = counts[counts<2].index.tolist()
    df_kept = df[~y_all.isin(rare)].reset_index(drop=True)
    y_kept = df_kept["regime"].astype(str)

    try:
        _, idx_test = train_test_split(df_kept.index, test_size=0.3,
                                       random_state=42, stratify=y_kept)
        test = df_kept.loc[idx_test]
        n_train = len(df_kept) - len(test); n_test = len(test)
    except Exception:
        n = len(df_kept); n_test = int(n*0.3); n_train = n - n_test
        test = df_kept.iloc[-n_test:]

    y_true = test["regime"].astype(str); y_pred = test["regime_pred"].astype(str)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class = []
    for lab in labels:
        if lab in report:
            per_class.append({"regime":lab,
                              "precision":round(report[lab]["precision"],3),
                              "recall":round(report[lab]["recall"],3),
                              "f1":round(report[lab]["f1-score"],3),
                              "support":int(report[lab]["support"])})
    return {"available":True,"n_train_days":n_train,"n_test_days":n_test,
            "accuracy":round(float((y_true==y_pred).mean()),4),
            "labels":labels,"features":feats,"importances":importances,
            "confusion_matrix":cm,"classification":per_class}


@app.get("/api/regime_metadata")
def regime_metadata(): return _regime_metadata()


# ============================================================
# MODEL COMPARISON + SCATTER + BIAS
# ============================================================
@lru_cache(maxsize=1)
def _model_comparison():
    p = MODELS / "test_predictions.csv"
    if not p.exists(): return {"available": False}
    df = pd.read_csv(p); o = df["rainfall_mm"].values
    def rmse(x): return float(np.sqrt(np.mean((o-x)**2)))
    def mae(x):  return float(np.mean(np.abs(o-x)))
    def bias(x): return float(np.mean(x-o))
    models = []
    if "gefs_mm" in df.columns:
        models.append({"name":"Raw GEFS","rmse":round(rmse(df["gefs_mm"].values),3),
                       "mae":round(mae(df["gefs_mm"].values),3),"bias":round(bias(df["gefs_mm"].values),3)})
    if "pred_global" in df.columns:
        models.append({"name":"RF (log)","rmse":round(rmse(df["pred_global"].values),3),
                       "mae":round(mae(df["pred_global"].values),3),"bias":round(bias(df["pred_global"].values),3)})
    if "pred_regime_feat" in df.columns:
        models.append({"name":"RF + Regime","rmse":round(rmse(df["pred_regime_feat"].values),3),
                       "mae":round(mae(df["pred_regime_feat"].values),3),"bias":round(bias(df["pred_regime_feat"].values),3)})
    return {"available":True,"models":models,"n":int(len(df))}


@app.get("/api/model_comparison")
def model_comparison(): return _model_comparison()


@app.get("/api/scatter_data")
def scatter_data(n: int = 800):
    p = MODELS / "test_predictions.csv"
    if not p.exists(): return {"raw":[],"corrected":[]}
    full = pd.read_csv(p)
    df = full.sample(n=min(n,len(full)), random_state=42)
    raw = df[["gefs_mm","rainfall_mm"]].round(2).values.tolist()
    corr = df[["pred_regime_feat","rainfall_mm"]].round(2).values.tolist() if "pred_regime_feat" in df.columns else []
    return {"raw":raw,"corrected":corr}


@app.get("/api/bias_by_range")
def bias_by_range():
    p = MODELS / "test_predictions.csv"
    if not p.exists(): return []
    df = pd.read_csv(p)
    bins = [0,2.5,20,64.5,115.6,204.5,5000]
    labels = ["0-2.5","2.5-20","20-64.5","64.5-115.6","115.6-204.5",">204.5"]
    df["bucket"] = pd.cut(df["rainfall_mm"], bins=bins, labels=labels, right=False)
    out = []
    for lab in labels:
        sub = df[df["bucket"] == lab]
        if len(sub) == 0: continue
        row = {"bucket":lab,"n":int(len(sub))}
        if "gefs_mm" in sub.columns:
            row["raw_bias"] = round(float((sub["gefs_mm"]-sub["rainfall_mm"]).mean()),2)
        if "pred_regime_feat" in sub.columns:
            row["corr_bias"] = round(float((sub["pred_regime_feat"]-sub["rainfall_mm"]).mean()),2)
        out.append(row)
    return out


# ============================================================
# VERIFICATION
# ============================================================
@app.get("/api/verification")
def verification():
    p = REPORTS / "verification_report.csv"
    if not p.exists(): return []
    df = pd.read_csv(p)
    return df.replace({np.nan: None}).round(4).to_dict(orient="records")


@app.get("/api/verification_by_regime")
def verification_by_regime():
    p = REPORTS / "verification_by_regime.csv"
    if not p.exists(): return []
    df = pd.read_csv(p)
    return df.replace({np.nan: None}).to_dict(orient="records")


@app.get("/api/analytics")
def analytics():
    p = REPORTS / "analytics.json"
    if not p.exists(): return {"available": False}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    data["available"] = True
    return data


@app.get("/api/model_metadata")
def model_metadata():
    p = MODELS / "regime_classifier.json"
    if not p.exists(): return {"available": False}
    with open(p, encoding="utf-8") as f:
        meta = json.load(f)
    meta["available"] = True
    return meta


@app.get("/api/districts")
def districts():
    p = REPORTS / "realtime_forecast.csv"
    if not p.exists():
        p = REPORTS / "district_forecast_latest.csv"
    if not p.exists(): return []
    df = pd.read_csv(p)
    df["district_id"] = df["district_id"].astype(int).astype(str)
    out = df[["district_id","district","state"]].drop_duplicates("district_id").sort_values(["state","district"])
    return out.to_dict(orient="records")


@app.post("/api/reload")
def reload_data():
    _invalidate_cache()
    _regime_metadata.cache_clear()
    _model_comparison.cache_clear()
    return {"reloaded": True}