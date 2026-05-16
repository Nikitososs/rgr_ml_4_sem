import streamlit as st

from _common import PHOTO_PATH

st.set_page_config(
    page_title="РГР — ML Inference",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Расчётно-графическая работа")
st.subheader(
    "Тема: разработка веб-приложения (дашборда) для инференса моделей ML и анализа данных"
)

st.markdown("### Страница 1. Сведения о разработчике")

col1, col2 = st.columns([1, 2])
with col1:
    st.image(str(PHOTO_PATH), width=280)
with col2:
    st.markdown(
        """
            **ФИО:** Смирнов Никита Михайлович  
            **Учебная группа:** ФИТ_242

            **Тема РГР:** разработка веб-приложения (дашборда) для инференса моделей машинного обучения
            и представления результатов анализа данных (регрессия: датасет predictive maintenance).

            Используются шесть сериализованных моделей (Ridge, Gradient Boosting, CatBoost, Bagging, Stacking, FCNN),
            обученных на `predictive_maintenance_dataset_processed.csv`.
        """
    )

st.divider()
st.caption("Перейдите к разделам в боковой панели: набор данных, визуализации, инференс.")
