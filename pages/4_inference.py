from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from _common import (
    MODEL_LABELS,
    NORM_TO_PHYSICAL_SOURCE,
    csv_to_model_feature_frame,
    denorm_temperature_k,
    load_metadata,
    predict_batch,
    physical_value_to_unit,
    resolve_physical_feature_bounds,
)

st.set_page_config(page_title="Инференс", layout="wide")

st.title("Страница 4. Инференс моделей")

meta = load_metadata()
if not meta or "feature_columns" not in meta:
    st.error(
        "Нет `models/metadata.json`. Сначала выполните из корня репозитория: "
        "`python scripts/train_regression_models.py`"
    )
    st.stop()

feat_cols: list[str] = meta["feature_columns"]

st.markdown(
    """
Выберите модель и способ задания признаков. Непрерывные поля можно вводить **в исходных единицах**
(K, об/мин, Н·м, мин — как в `predictive_maintenance_dataset.csv`); перед моделью они приводятся к **[0, 1]**
линейным **MinMax** по минимуму и максимуму столбца в этом исходном файле — по тому же принципу, что при EDA.

Второй режим ручного ввода и загрузки CSV — **уже готовые `norm_*`**, как в `predictive_maintenance_dataset_processed.csv`.

Результат: нормализованная цель на [0, 1] и **оценка температуры процесса, K**, через обратное MinMax целевого признака
(`denorm_process_temp_k` в `metadata.json`).
"""
)

if meta.get("metrics_r2_test"):
    with st.expander("R² на отложенной тестовой выборке (при обучении скриптом)"):
        st.table(
            pd.Series(meta["metrics_r2_test"], name="R² test")
            .sort_values(ascending=False)
            .to_frame()
        )

model_key = st.selectbox("Модель", list(MODEL_LABELS.keys()), format_func=lambda k: MODEL_LABELS[k])

tab_csv, tab_form = st.tabs(["Загрузка CSV", "Ручной ввод"])

with tab_csv:
    st.caption(
        "Поддерживаются столбцы **обработанной** таблицы (`norm_*`) либо **исходные** столбцы "
        "(в т. ч. категориальный `Type` → будет развёрнут в `Type_*`). Поля вроде `UDI`, `Product ID`, "
        "`Process temperature [K]` при сборке признаков не используются."
    )
    up = st.file_uploader("Файл *.csv со строками наблюдений", type=["csv"])
    if up is not None:
        raw = pd.read_csv(io.BytesIO(up.read()))
        try:
            X_in = csv_to_model_feature_frame(raw, meta)
        except ValueError as e:
            st.error(str(e))
        else:
            st.success(f"Распознано наблюдений: {len(X_in)}")
            st.dataframe(X_in.head(), use_container_width=True)
            if st.button("Предсказать для всех строк", key="pred_csv"):
                preds = predict_batch(model_key, X_in, meta)
                preds_k = denorm_temperature_k(preds, meta)
                export = pd.concat(
                    [
                        raw.reset_index(drop=True),
                        pd.DataFrame(
                            {
                                "pred_norm_Process_temperature": preds,
                                "pred_Process_temperature_K": preds_k,
                            }
                        ),
                    ],
                    axis=1,
                )
                st.dataframe(
                    pd.concat(
                        [
                            X_in.reset_index(drop=True),
                            pd.DataFrame(
                                {
                                    "pred_norm_Process_temperature": preds,
                                    "pred_Process_temperature_K": preds_k,
                                }
                            ),
                        ],
                        axis=1,
                    ).head(20),
                    use_container_width=True,
                )
                st.download_button(
                    "Скачать результаты CSV",
                    data=export.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

with tab_form:
    phys_bounds = resolve_physical_feature_bounds(meta)
    input_fmt = st.radio(
        "Формат числовых признаков",
        (
            "Исходные величины (K, об/мин, Н·м, мин)",
            "Нормализованные [0, 1]",
        ),
        horizontal=True,
    )
    use_physical = input_fmt.startswith("Исходные")

    PHYS_LABEL = {
        "Air temperature [K]": "Температура воздуха, K",
        "Rotational speed [rpm]": "Скорость вращения, об/мин",
        "Torque [Nm]": "Крутящий момент, Н·м",
        "Tool wear [min]": "Износ инструмента, мин",
    }

    cols = st.columns(3)
    buf: dict[str, float] = {}
    for i, c in enumerate(feat_cols):
        col = cols[i % 3]
        if c in NORM_TO_PHYSICAL_SOURCE:
            src = NORM_TO_PHYSICAL_SOURCE[c]
            b = phys_bounds[c]
            lo, hi = float(b["min"]), float(b["max"])
            if use_physical:
                mid = (lo + hi) / 2
                v = col.number_input(
                    PHYS_LABEL.get(src, src),
                    value=float(mid),
                    format="%.4f",
                    key=f"f_{c}",
                    help=f"Min…max в `predictive_maintenance_dataset.csv` для «{src}»: [{lo:g}, {hi:g}]",
                )
                buf[c] = physical_value_to_unit(v, lo, hi)
            else:
                v = col.number_input(
                    f"{c} [0–1]",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.001,
                    format="%.4f",
                    key=f"f_{c}",
                )
                buf[c] = float(v)
        elif c.startswith(("Type_", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF")):
            v = col.selectbox(c, options=[0, 1], index=0, key=f"f_{c}")
            buf[c] = float(v)

    if st.button("Получить прогноз", key="pred_form"):
        row = pd.DataFrame([buf])
        try:
            pred_norm = float(predict_batch(model_key, row, meta)[0])
        except Exception as e:
            st.error(str(e))
        else:
            pred_k = float(denorm_temperature_k(pred_norm, meta))
            st.metric("Прогноз цели (нормализованная температура процесса)", f"{pred_norm:.4f}")
            st.metric(
                "Интерпретация: температура процесса (оценка)",
                f"{pred_k:.2f} K",
                help="Обратное преобразование MinMax по исходному диапазону Process temperature [K].",
            )
            st.info(
                f"Модель: **{MODEL_LABELS[model_key]}**. "
                "Непрерывные признаки при необходимости приведены к [0, 1] по MinMax из исходного AI4I CSV."
            )
