from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
from pyproj import Transformer
from shapely.geometry import shape, mapping
from shapely.ops import transform

REPORTS = Path(__file__).parent / "reports"
GEOJSON = Path(__file__).parent / "data" / "admin" / "district_nwic.GeoJSON"
FORECAST = REPORTS / "district_forecast_latest.csv"
OUT_GJ  = Path(__file__).parent / "data" / "admin" / "district_nwic_wgs84.geojson"

fc = pd.read_csv(FORECAST)
fc["district_id"] = fc["district_id"].astype(int).astype(str)
fmap = dict(zip(fc["district_id"], fc["pred_regime_feat"]))

print("Loading GeoJSON...")
with open(GEOJSON, "r", encoding="utf-8") as f:
    gj = json.load(f)

print("Reprojecting EPSG:7755 -> EPSG:4326 ...")
transformer = Transformer.from_crs("EPSG:7755", "EPSG:4326", always_xy=True)

new_features = []
ids, values, names = [], [], []

for i, feat in enumerate(gj["features"]):
    geom = shape(feat["geometry"])
    geom_ll = transform(transformer.transform, geom)

    key = f"d{i}"
    props = dict(feat["properties"])
    props["id_final"] = key

    raw = str(props.get("dtcode", ""))
    norm = str(int(raw)) if raw.isdigit() else raw
    val = fmap.get(norm)

    new_features.append({
        "type": "Feature",
        "properties": props,
        "geometry": mapping(geom_ll),
    })

    ids.append(key)
    values.append(val)
    names.append(f"{props.get('district')} ({props.get('state_name')})")

new_gj = {"type": "FeatureCollection", "features": new_features}

with open(OUT_GJ, "w", encoding="utf-8") as f:
    json.dump(new_gj, f)
print(f"Saved WGS84 GeoJSON: {OUT_GJ}")

print(f"Districts: {len(ids)}   values present: {sum(v is not None for v in values)}")

fig = go.Figure(go.Choropleth(
    geojson=new_gj,
    locations=ids,
    z=values,
    featureidkey="properties.id_final",
    colorscale=[
        [0.0, "#f7f7f7"],
        [0.1, "#ffeda0"],
        [0.3, "#fd8d3c"],
        [0.6, "#e31a1c"],
        [1.0, "#8b0000"],
    ],
    zmin=0,
    zmax=150,
    colorbar_title="mm",
    text=names,
    hovertemplate="<b>%{text}</b><br>Corrected: %{z:.1f} mm<extra></extra>",
    marker_line_width=0.3,
    marker_line_color="black",
))

fig.update_geos(
    visible=False,
    projection_type="mercator",
    lataxis_range=[6, 38],
    lonaxis_range=[68, 98],
    bgcolor="white",
    showcountries=False,
    showcoastlines=False,
    showland=False,
    showframe=False,
)
fig.update_layout(
    title="India District Rainfall Forecast — Corrected (mm)",
    height=800,
    margin=dict(l=0, r=0, t=40, b=0),
)

out = REPORTS / "district_forecast_map_plotly.html"
fig.write_html(str(out))
print(f"Saved map: {out}")