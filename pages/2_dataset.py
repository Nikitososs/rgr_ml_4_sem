import streamlit as st

from _common import DATA_DEFAULT, load_metadata, load_processed_dataset

st.set_page_config(page_title="Датасет", layout="wide")

st.title("Страница 2. Набор данных")

st.markdown(
    """
### Предметная область

Датасет AI4I 2020 Predictive Maintenance  описывает
работу промышленного оборудования: режимы процесса, износ инструмента и индикаторы отказов.
В дашборде решается задача **регрессии**: по признакам процесса оценивается
**нормализованная температура процесса** `norm_Process temperature [K]` (значения в диапазоне [0, 1]
относительно исходных температур в Кельвинах).

Исходные и обработанные данные: `data/predictive_maintenance_dataset.csv` и
`data/predictive_maintenance_dataset_processed.csv`.
"""
)

meta = load_metadata()
if meta:
    st.json(
        {
            "Целевая переменная": meta.get("target"),
            "Число признаков": len(meta.get("feature_columns", [])),
            "Денормализация T (К)": meta.get("denorm_process_temp_k"),
        }
    )

st.subheader("Признаки (после one-hot и нормализации)")

st.markdown(
    """
| Признак | Смысл |
|---------|--------|
| `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Бинарные индикаторы отказов / конкретных типов отказов |
| `Type_H`, `Type_L`, `Type_M` | One-hot кодирование типа продукта (H / L / M) |
| `norm_Air temperature [K]` | Нормализованная температура воздуха (0–1) |
| `norm_Rotational speed [rpm]` | Нормализованная скорость вращения |
| `norm_Torque [Nm]` | Нормализованный крутящий момент |
| `norm_Tool wear [min]` | Нормализованный износ инструмента |
| **`norm_Process temperature [K]`** | **Цель регрессии** — нормализованная температура процесса |
"""
)

st.subheader("Предобработка и EDA")

st.markdown(
    """
Ниже — Фрагменты кода предобработки данных.
"""
)

with st.expander("1. Загрузка, удаление идентификаторов и one-hot кодирование `Type`", expanded=False):
    st.code(
        """\
original_df = pd.read_csv("data/predictive_maintenance_dataset.csv")
df = original_df.copy()

df = df.drop(columns=["Product ID", "UDI"])

df = pd.get_dummies(df, "Type", dtype=int)

df[["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF", "Type_H", "Type_L", "Type_M"]] = df[
    ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF", "Type_H", "Type_L", "Type_M"]
].astype("int8")
""",
        language="python",
    )

with st.expander("2. Инженерия признаков (под критерии отказов из описания датасета)", expanded=False):
    st.code(
        """\
df["Temp Abs"] = abs(df["Air temperature [K]"] - df["Process temperature [K]"])
df["rad/s"] = df["Rotational speed [rpm]"] * 2 * 3.14 / 60
df["Power"] = df["rad/s"] * df["Torque [Nm]"]
df["overload"] = df["Torque [Nm]"] * df["Tool wear [min]"]

df["Critical temp diff and low rpm"] = (
    (df["Temp Abs"] < 8.6) & (df["Rotational speed [rpm]"] < 1380)
).astype(int)
df["Critical power value"] = ((df["Power"] <= 3500) | (df["Power"] >= 9000)).astype(int)
df["Critical overload value"] = (
    ((df["Type_L"] == 1) & (df["overload"] > 11000))
    | ((df["Type_M"] == 1) & (df["overload"] > 12000))
    | ((df["Type_H"] == 1) & (df["overload"] > 13000))
).astype(int)
""",
        language="python",
    )

with st.expander("3. MinMax-нормализация и сохранение итогового датасета", expanded=False):
    st.code(
        """\
columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp Abs",
    "rad/s",
    "Power",
    "overload",
]


def minmax_norm(col):
    return (col - col.min()) / (col.max() - col.min())


for col in columns:
    df[f"norm_{col}"] = minmax_norm(df[col])

norm_columns = [f"norm_{col}" for col in columns]

df = df.drop(
    columns=columns
    + [
        "Temp Abs",
        "rad/s",
        "Power",
        "overload",
        "Critical temp diff and low rpm",
        "Critical power value",
        "Critical overload value",
        "norm_Temp Abs",
        "norm_rad/s",
        "norm_Power",
        "norm_overload",
    ]
)

df.to_csv("data/predictive_maintenance_dataset_processed.csv", index=False)
""",
        language="python",
    )

st.markdown(
    """
На этапе EDA в ноутбуке дополнительно строятся распределения, корреляции, точечные диаграммы признаков
и визуализации по индикаторам «критических» режимов.
"""
)

st.markdown("Ниже — краткий просмотр обработанного файла (первые строки).")

df = load_processed_dataset()
st.dataframe(df.head(12), use_container_width=True)
st.caption(f"Файл: `{DATA_DEFAULT}` — строк: {len(df)}, столбцов: {len(df.columns)}.")
