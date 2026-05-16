import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "predictive_maintenance_dataset_processed.csv"
ORIG_PATH = ROOT / "data" / "predictive_maintenance_dataset.csv"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42

TARGET = "norm_Process temperature [K]"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET]


def load_xy() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def denorm_bounds() -> tuple[float, float]:
    if ORIG_PATH.exists():
        o = pd.read_csv(ORIG_PATH, usecols=["Process temperature [K]"])
        col = o["Process temperature [K]"]
        return float(col.min()), float(col.max())
    return 305.7, 313.8


def physical_feature_bounds(orig: pd.DataFrame) -> dict[str, dict[str, float]]:
    pairs = (
        ("norm_Air temperature [K]", "Air temperature [K]"),
        ("norm_Rotational speed [rpm]", "Rotational speed [rpm]"),
        ("norm_Torque [Nm]", "Torque [Nm]"),
        ("norm_Tool wear [min]", "Tool wear [min]"),
    )
    out: dict[str, dict[str, float]] = {}
    for norm_col, src in pairs:
        s = orig[src].astype(float)
        out[norm_col] = {"min": float(s.min()), "max": float(s.max())}
    return out


def build_keras_model(input_dim: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_xy()
    feats = list(X.columns)
    t_min, t_max = denorm_bounds()
    phys_bounds: dict[str, dict[str, float]] = {}
    if ORIG_PATH.exists():
        full_orig = pd.read_csv(ORIG_PATH)
        phys_bounds = physical_feature_bounds(full_orig)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    metrics: dict[str, float] = {}

    ml1 = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]
    )
    ml1.fit(X_train, y_train)
    metrics["ML1_Ridge"] = float(r2_score(y_test, ml1.predict(X_test)))
    with open(MODEL_DIR / "ml1_ridge_pipeline.pkl", "wb") as f:
        pickle.dump(ml1, f, protocol=pickle.HIGHEST_PROTOCOL)

    ml2 = GradientBoostingRegressor(
        random_state=RANDOM_STATE,
        n_estimators=250,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
    )
    ml2.fit(X_train, y_train)
    metrics["ML2_GradientBoosting"] = float(r2_score(y_test, ml2.predict(X_test)))
    with open(MODEL_DIR / "ml2_gradient_boosting.pkl", "wb") as f:
        pickle.dump(ml2, f, protocol=pickle.HIGHEST_PROTOCOL)

    ml3 = CatBoostRegressor(
        depth=6,
        iterations=400,
        learning_rate=0.05,
        loss_function="RMSE",
        verbose=False,
        random_seed=RANDOM_STATE,
    )
    ml3.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
    metrics["ML3_CatBoost"] = float(r2_score(y_test, ml3.predict(X_test)))
    ml3.save_model(str(MODEL_DIR / "ml3_catboost.cbm"))

    ml4 = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        n_estimators=40,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    ml4.fit(X_train, y_train)
    metrics["ML4_Bagging"] = float(r2_score(y_test, ml4.predict(X_test)))
    with open(MODEL_DIR / "ml4_bagging.pkl", "wb") as f:
        pickle.dump(ml4, f, protocol=pickle.HIGHEST_PROTOCOL)

    ml5 = StackingRegressor(
        estimators=[
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=120, random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
            (
                "hgb",
                HistGradientBoostingRegressor(
                    max_iter=200, random_state=RANDOM_STATE
                ),
            ),
        ],
        final_estimator=Ridge(alpha=1.0),
        n_jobs=-1,
    )
    ml5.fit(X_train, y_train)
    metrics["ML5_Stacking"] = float(r2_score(y_test, ml5.predict(X_test)))
    with open(MODEL_DIR / "ml5_stacking.pkl", "wb") as f:
        pickle.dump(ml5, f, protocol=pickle.HIGHEST_PROTOCOL)

    scaler = StandardScaler().fit(X_train)
    X_tr = scaler.transform(X_train)
    X_te = scaler.transform(X_test)

    import tensorflow as tf

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    ml6 = build_keras_model(X_tr.shape[1])
    ml6.fit(
        X_tr,
        y_train.to_numpy(),
        validation_data=(X_te, y_test.to_numpy()),
        epochs=200,
        batch_size=64,
        callbacks=[early],
        verbose=0,
    )
    y_pred = ml6.predict(X_te, verbose=0).reshape(-1)
    metrics["ML6_FCNN_Keras"] = float(r2_score(y_test, y_pred))
    ml6.save(MODEL_DIR / "ml6_fcnn.keras")
    with open(MODEL_DIR / "ml6_fcnn_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta = {
        "target": TARGET,
        "feature_columns": feats,
        "denorm_process_temp_k": {"min": t_min, "max": t_max},
        "physical_feature_bounds": phys_bounds,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "metrics_r2_test": metrics,
        "model_files": {
            "ML1": "ml1_ridge_pipeline.pkl",
            "ML2": "ml2_gradient_boosting.pkl",
            "ML3": "ml3_catboost.cbm",
            "ML4": "ml4_bagging.pkl",
            "ML5": "ml5_stacking.pkl",
            "ML6": "ml6_fcnn.keras",
            "ML6_scaler": "ml6_fcnn_scaler.pkl",
        },
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved models to", MODEL_DIR)
    for k, v in sorted(metrics.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: R2_test={v:.4f}")


if __name__ == "__main__":
    main()
    sys.exit(0)
