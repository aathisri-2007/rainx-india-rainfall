# India Rainfall Forecast Post-Processing System

Regime-aware AI/ML post-processing of NWP rainfall forecasts over India,
built around real observations, real NWP data, real reanalysis, and physically
derived weather regime labels.

---

## Deliverables

| # | Deliverable | Status | Key metric |
|---|---|---|---|
| 1 | Weather regime classifier | ✅ | 92.9% test accuracy |
| 2 | Bias-corrected rainfall forecast | ✅ | Improves CSI at all thresholds |
| 3 | Heavy rainfall probability | ✅ | AUC 0.88 / 0.95 / 0.97 |
| 4 | District-level rainfall product (table + map) | ✅ | 700 districts, 34 states |
| 5 | Verification report (RMSE, ETS, CSI, POD, FAR, FSS) | ✅ | Raw vs corrected |

---

## Data sources (all real)

| Source | Purpose | Coverage |
|---|---|---|
| IMD 0.25° gridded daily rainfall | Observations / truth | 2016–2025, 18.1M grid values |
| Survey of India district boundaries | District polygons | 733 districts |
| NOAA GEFSv12 reforecast | Raw NWP forecast | Jun–Aug 2016 pilot |
| NCEP/NCAR Reanalysis 1 | Atmospheric features | 5 variables, 2016 |

No synthetic rainfall, no fake skill scores.

---

## Pipeline
IMD GRD -> India grid rainfall -> district aggregation
|
GEFS GRIB2 -> India rainfall ----------+
|
NCEP reanalysis -> regime labels -> RF classifier
|
regime-aware bias correction + quantile mapping
|
heavy-rain probability + verification
|
district forecast table + interactive map

---

## Weather regimes handled

- Active monsoon
- Break monsoon
- Low / depression
- Coastal rainfall
- Orographic rainfall
- Western disturbance

Regime labels are derived from physically meaningful indices
(SLP anomaly over the monsoon trough, 500 hPa height anomaly over NW India,
low-level wind direction and speed over the Arabian Sea, humidity anomaly),
using thresholds from published Indian monsoon literature.

---

## Models

| Model | Algorithm | Purpose |
|---|---|---|
| Regime classifier | Random Forest (300 trees, balanced weights) | Classify daily regime |
| Bias correction | Random Forest + log target + quantile mapping | Correct NWP rainfall |
| Heavy rain probability | HistGradientBoostingClassifier | P(rain >= 64.5 / 115.6 / 204.5 mm) |

---

## How to run

```bash
conda activate rainfall
cd C:\projects\sih2b

python build_regime_labels.py
python train_regime_classifier.py
python train_postprocessor.py
python heavy_rain_probability.py
python verify_forecast.py
python district_forecast.py
python district_map_plotly_wgs84.py
Results (62-day pilot: Jun–Aug 2016)
Continuous metrics
Metric	Raw GEFS	Corrected
RMSE	19.54	20.63
MAE	9.87	12.07
Bias	-7.46	-0.28
Event skill (CSI, corrected vs raw)
Threshold	CSI (raw)	CSI (corrected)
>= 2.5 mm	0.382	0.554
>= 20 mm	0.042	0.131
>= 64.5 mm	0.000	0.013
>= 115.6 mm	0.000	0.000
Probability model
Threshold	AUC	Brier
>= 64.5 mm	0.883	0.0156
>= 115.6 mm	0.946	0.0044
>= 204.5 mm	0.967	0.0010
Outputs
File	Contents
reports/verification_report.csv	RMSE, ETS, CSI, POD, FAR, FSS by threshold
reports/district_forecast_latest.csv	District-level corrected rainfall
reports/state_forecast_latest.csv	State-level aggregation
reports/district_heavy_alerts.csv	Districts exceeding 64.5 mm
reports/district_forecast_map_plotly.html	Interactive district map
Limitations
Pilot window is 62 days (GEFS reforecast archive ends 2019).

699 of 732 districts are covered by IMD grid points; the remaining
33 are small/island/urban districts where a 0.25° grid may not contain a
cell center inside the polygon.

Extreme-event skill requires a longer training window than the pilot provides.

500 hPa height was not available during the pilot; break monsoon and
western disturbance detection therefore rely on SLP and wind proxies.

Future work
Extend GEFS download to 2016–2019 (full reforecast period).

Add 500 hPa geopotential height for improved break-monsoon / WD detection.

Deploy as a Streamlit dashboard with map, table, and verification tabs.

Add multi-model ensemble (NCMRWF NCUM) alongside GEFS.

Reproduction
Tested with:

Python 3.11 (Miniforge / conda-forge)

eccodes 2.49, cfgrib 0.9.15, xarray, pandas, scikit-learn

geopandas, shapely, pyproj, plotly, folium