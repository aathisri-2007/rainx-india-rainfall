import pandas as pd

from india_rainfall_postprocessing.data_quality import validate_alignment


def test_validate_alignment_keeps_valid_overlap():
    observed = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "district_id": ["D1", "D2", "D1"],
            "rainfall_mm": [10.0, 20.0, 30.0],
        }
    )
    forecast = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-02"],
            "district_id": ["D1", "D1", "D2"],
            "forecast_rainfall_mm": [15.0, 35.0, 25.0],
        }
    )

    obs, fcst, issues = validate_alignment(
        observed,
        forecast,
        {"observed": ["date", "district_id", "rainfall_mm"], "forecast": ["date", "district_id", "forecast_rainfall_mm"]},
    )

    assert not issues
    assert set(obs["district_id"]) == {"D1", "D2"}
    assert set(fcst["district_id"]) == {"D1", "D2"}
