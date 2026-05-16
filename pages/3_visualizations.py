import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from _common import load_processed_dataset

st.set_page_config(page_title="Визуализации", layout="wide")

st.title("Страница 3. Визуализации зависимостей")

df = load_processed_dataset()

target = "norm_Process temperature [K]"
feat_cols = [c for c in df.columns if c != target]

sample = df.sample(min(2500, len(df)), random_state=1)

type_lab = df[["Type_H", "Type_L", "Type_M"]].idxmax(axis=1)
plot_df = df.assign(Тип_продукта=type_lab.str.replace("Type_", "", regex=False))

st.subheader("1. Матрица scatter-графиков (scatter_matrix)")
dims = [
    "norm_Air temperature [K]",
    target,
    "norm_Rotational speed [rpm]",
    "norm_Torque [Nm]",
]
sub = df[dims].sample(min(1200, len(df)), random_state=42)
fig1 = px.scatter_matrix(
    sub,
    dimensions=dims,
    height=900,
    title="Парные зависимости (подвыборка строк)",
    opacity=0.35,
)
fig1.update_traces(marker=dict(size=5, line=dict(width=0)))
fig1.update_layout(dragmode="pan", margin=dict(l=40, r=40, t=60, b=40))
st.plotly_chart(fig1, use_container_width=True)

st.subheader("2. Корреляционная матрица (heatmap)")
corr = df[feat_cols + [target]].corr()
fig2 = px.imshow(
    corr,
    text_auto=".2f" if len(corr) <= 10 else False,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Pearson correlation",
    labels=dict(x="Признак", y="Признак", color="r"),
)
fig2.update_layout(height=max(480, 28 * len(corr)))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. Распределение цели по типу продукта (boxplot)")
fig3 = px.box(
    plot_df,
    x="Тип_продукта",
    y=target,
    color="Тип_продукта",
    points="outliers",
    title="Нормализованная температура процесса по типу продукта",
)
fig3.update_layout(showlegend=False, height=480, xaxis_title="Тип продукта")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("4. Распределение целевой переменной")
fig4 = px.histogram(
    df,
    x=target,
    nbins=45,
    opacity=0.75,
    title="Нормализованная температура процесса",
    marginal="violin",
)
fig4.update_layout(height=500, bargap=0.05)
st.plotly_chart(fig4, use_container_width=True)

st.subheader("5. Температура воздуха — температура процесса")
fig5 = px.scatter(
    sample,
    x="norm_Air temperature [K]",
    y=target,
    trendline="ols",
    opacity=0.35,
    title="Линейная регрессия",
    labels={
           "norm_Air temperature [K]": "Норм. температура воздуха",
        target: target,
    },
)
fig5.update_traces(marker=dict(size=8, line=dict(width=0)))
fig5.update_layout(height=520)

st.plotly_chart(fig5, use_container_width=True)

st.subheader("6. Доли типов продукта")
vc = plot_df["Тип_продукта"].value_counts().reset_index()
vc.columns = ["Тип", "Количество"]
fig6 = px.pie(
    vc,
    names="Тип",
    values="Количество",
    title="Распределение записей по типу продукта",
    hole=0.35,
)
fig6.update_layout(height=420, margin=dict(t=50, b=30))
st.plotly_chart(fig6, use_container_width=True)

st.subheader("7. Корреляции признаков")
s = corr[target].drop(labels=[target], errors="ignore")
s = s.iloc[np.argsort(np.abs(s.values))]
fig8_corr = go.Figure(
    go.Bar(
        x=s.values,
        y=s.index,
        orientation="h",
        marker_color=np.where(s.values >= 0, "#059669", "#dc2626"),
    )
)
fig8_corr.update_layout(
    title="Корреляция признаков с целевой переменной",
    height=max(400, 22 * len(s)),
    xaxis_title="Корреляция",
    yaxis_title="",
    margin=dict(l=120, r=30, t=50, b=40),
)
st.plotly_chart(fig8_corr, use_container_width=True)

st.subheader("8. Сетка распределений всех числовых признаков")
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
ncols = 4
nrows = int(np.ceil(len(numeric_cols) / ncols))
pad = nrows * ncols - len(numeric_cols)


def _short_title(name: str) -> str:
    t = name.replace("norm_", "норм. ")
    return t if len(t) <= 32 else t[:29] + "…"


_titles = [_short_title(c) for c in numeric_cols] + [""] * pad
fig_grid = make_subplots(
    rows=nrows,
    cols=ncols,
    subplot_titles=_titles,
    vertical_spacing=0.08,
    horizontal_spacing=0.05,
)
for i, col in enumerate(numeric_cols):
    r = i // ncols + 1
    c = i % ncols + 1
    nu = int(df[col].nunique())
    if nu <= 3:
        fig_grid.add_trace(
            go.Histogram(
                x=df[col],
                xbins=dict(start=-0.5, end=max(1.5, float(df[col].max()) + 0.5), size=1),
                showlegend=False,
                marker_line_width=1,
                marker_line_color="white",
                marker_color="#0d9488",
            ),
            row=r,
            col=c,
        )
    else:
        fig_grid.add_trace(
            go.Histogram(
                x=df[col],
                nbinsx=min(50, max(15, nu // 2)),
                showlegend=False,
                marker_color="#0d9488",
                marker_line_width=0.5,
                marker_line_color="white",
            ),
            row=r,
            col=c,
        )
fig_grid.update_layout(
    title_text="Гистограммы распределений",
    height=min(320 * nrows, 2200),
    showlegend=False,
    margin=dict(t=80, b=40, l=40, r=30),
    template="plotly_white",
)
fig_grid.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#e5e7eb")
fig_grid.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#e5e7eb")
st.plotly_chart(fig_grid, use_container_width=True)
