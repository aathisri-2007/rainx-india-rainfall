"""
Model versioning metadata — attached to every trained model.
"""
from pathlib import Path
from datetime import datetime
import json
import joblib


def build_metadata(model_name, algorithm, features, training_period,
                   validation_period, test_period, metrics, extra=None):
    meta = {
        "model_name":        model_name,
        "algorithm":         algorithm,
        "features":          features,
        "training_period":   training_period,
        "validation_period": validation_period,
        "test_period":       test_period,
        "trained_at":        datetime.utcnow().isoformat() + "Z",
        "metrics":           metrics,
        "version":           "1.0.0",
    }
    if extra:
        meta.update(extra)
    return meta


def save_with_metadata(models_dir: Path, filename: str, payload: dict, metadata: dict):
    """Save a joblib bundle with embedded metadata + a sidecar JSON."""
    models_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["__metadata__"] = metadata

    joblib.dump(payload, models_dir / filename)

    sidecar = models_dir / (filename.replace(".joblib", "") + ".json")
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return models_dir / filename, sidecar


def load_metadata(models_dir: Path, name_base: str):
    """Load metadata sidecar for a model."""
    p = models_dir / (name_base + ".json")
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)