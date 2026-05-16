from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DASH_ROOT = Path(__file__).resolve().parent
MODEL_DIR = DASH_ROOT / "models"
DATA_DEFAULT = Path(__file__).resolve().parent / "data" / "predictive_maintenance_dataset_processed.csv"
ORIGINAL_DATA_PATH = DATA_DEFAULT.parent / "predictive_maintenance_dataset.csv"
PHOTO_PATH = DASH_ROOT / "assets" / "dev.png"

# Колонки `norm_*` в модели ↔ исходные непрерывные признаки в CSV до MinMax (см. maintaince_dataset_eda.ipynb)
NORM_TO_PHYSICAL_SOURCE: dict[str, str] = {
    "norm_Air temperature [K]": "Air temperature [K]",
    "norm_Rotational speed [rpm]": "Rotational speed [rpm]",
    "norm_Torque [Nm]": "Torque [Nm]",
    "norm_Tool wear [min]": "Tool wear [min]",
}


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    p = MODEL_DIR / "metadata.json"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def denorm_temperature_k(norm: np.ndarray | float, meta: dict) -> np.ndarray:
    b = meta.get("denorm_process_temp_k", {})
    lo, hi = float(b.get("min", 305.7)), float(b.get("max", 313.8))
    arr = np.asarray(norm, dtype=float)
    return lo + arr * (hi - lo)


def resolve_physical_feature_bounds(meta: dict) -> dict[str, dict[str, float]]:
    stored = meta.get("physical_feature_bounds")
    if isinstance(stored, dict) and all(k in stored for k in NORM_TO_PHYSICAL_SOURCE):
        return stored
    if not ORIGINAL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Нет physical_feature_bounds в metadata и не найден {ORIGINAL_DATA_PATH}"
        )
    o = pd.read_csv(ORIGINAL_DATA_PATH)
    return {
        norm_key: {
            "min": float(o[src].min()),
            "max": float(o[src].max()),
        }
        for norm_key, src in NORM_TO_PHYSICAL_SOURCE.items()
    }


def physical_value_to_unit(raw: float, lo: float, hi: float) -> float:
    span = float(hi) - float(lo)
    if span <= 0:
        return 0.0
    return float(np.clip((float(raw) - float(lo)) / span, 0.0, 1.0))


def drop_leak_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in ("UDI", "Product ID") if c in df.columns]
    if not drop:
        return df
    return df.drop(columns=drop)


def ensure_type_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    if "Type" not in df.columns:
        need = {"Type_H", "Type_L", "Type_M"}
        if not need.issubset(df.columns):
            raise ValueError(
                f"Ожидается столбец `Type` или столбцы {sorted(need)}; не хватает {sorted(need - set(df.columns))}"
            )
        return df
    d = pd.get_dummies(df["Type"], prefix="Type", dtype=float)
    for c in ("Type_H", "Type_L", "Type_M"):
        if c not in d.columns:
            d[c] = 0.0
    return pd.concat([df.drop(columns=["Type"]), d], axis=1)


def normalized_frame_from_physical(df_phys: pd.DataFrame, meta: dict) -> pd.DataFrame:
    bounds = resolve_physical_feature_bounds(meta)
    out = pd.DataFrame(index=df_phys.index)
    for fc in meta["feature_columns"]:
        if fc in NORM_TO_PHYSICAL_SOURCE:
            src = NORM_TO_PHYSICAL_SOURCE[fc]
            if src not in df_phys.columns:
                raise ValueError(f"Нет столбца исходных данных «{src}»")
            b = bounds[fc]
            raw = pd.to_numeric(df_phys[src], errors="coerce").astype(float)
            lo_f, hi_f = float(b["min"]), float(b["max"])
            span = hi_f - lo_f
            if span <= 0:
                out[fc] = 0.0
            else:
                out[fc] = np.clip((raw - lo_f) / span, 0.0, 1.0)
        else:
            if fc not in df_phys.columns:
                raise ValueError(f"Нет столбца «{fc}»")
            out[fc] = pd.to_numeric(df_phys[fc], errors="coerce").fillna(0).astype(float)
    return out


def csv_to_model_feature_frame(raw: pd.DataFrame, meta: dict) -> pd.DataFrame:
    feats: list[str] = meta["feature_columns"]
    if all(c in raw.columns for c in feats):
        return align_features(raw, feats)
    work = ensure_type_one_hot(drop_leak_id_columns(raw.copy()))
    srcs = set(NORM_TO_PHYSICAL_SOURCE.values())
    if not srcs.issubset(work.columns):
        miss = sorted(srcs - set(work.columns))
        raise ValueError(
            "Файл не в формате модели (столбцы norm_*) и не содержит исходных непрерывных "
            f"столбцов: {miss}. Либо добавьте их, либо используйте те же имена, что в обучающем CSV."
        )
    for bcol in feats:
        if bcol in NORM_TO_PHYSICAL_SOURCE:
            continue
        if bcol not in work.columns:
            raise ValueError(f"Нет столбца «{bcol}» (бинарный / тип продукта).")
    return normalized_frame_from_physical(work, meta)


def load_processed_dataset() -> pd.DataFrame:
    meta = load_metadata()
    path = DATA_DEFAULT
    if meta.get("data_path"):
        path = Path(meta["data_path"])
    return pd.read_csv(path)


@lru_cache(maxsize=8)
def _load_ml1():
    with open(MODEL_DIR / "ml1_ridge_pipeline.pkl", "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=8)
def _load_ml2():
    with open(MODEL_DIR / "ml2_gradient_boosting.pkl", "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=8)
def _load_ml3():
    from catboost import CatBoostRegressor

    m = CatBoostRegressor()
    m.load_model(str(MODEL_DIR / "ml3_catboost.cbm"))
    return m


@lru_cache(maxsize=8)
def _load_ml4():
    with open(MODEL_DIR / "ml4_bagging.pkl", "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=8)
def _load_ml5():
    with open(MODEL_DIR / "ml5_stacking.pkl", "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=8)
def _load_ml6():
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_DIR / "ml6_fcnn.keras")


@lru_cache(maxsize=8)
def _load_ml6_scaler():
    with open(MODEL_DIR / "ml6_fcnn_scaler.pkl", "rb") as f:
        return pickle.load(f)


def align_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"В данных нет столбцов: {missing}")
    return df[feature_columns].copy()


def predict_batch(model_key: str, X: pd.DataFrame, meta: dict) -> np.ndarray:
    Xa = align_features(X, meta["feature_columns"])
    if model_key == "ML1":
        return np.asarray(_load_ml1().predict(Xa), dtype=float)
    if model_key == "ML2":
        return np.asarray(_load_ml2().predict(Xa), dtype=float)
    if model_key == "ML3":
        return np.asarray(_load_ml3().predict(Xa), dtype=float)
    if model_key == "ML4":
        return np.asarray(_load_ml4().predict(Xa), dtype=float)
    if model_key == "ML5":
        return np.asarray(_load_ml5().predict(Xa), dtype=float)
    if model_key == "ML6":
        sc = _load_ml6_scaler()
        Xs = sc.transform(Xa)
        return _load_ml6().predict(Xs, verbose=0).reshape(-1)
    raise ValueError(model_key)


MODEL_LABELS = {
    "ML1": "ML1 — Ridge",
    "ML2": "ML2 — Gradient Boosting (sklearn)",
    "ML3": "ML3 — CatBoost",
    "ML4": "ML4 — Bagging",
    "ML5": "ML5 — Stacking (RF + HistGB → Ridge)",
    "ML6": "ML6 — полносвязная нейросеть (Keras)",
}
