"""
app.py
------
Dashboard interactivo (Streamlit) que integra las 3 capas del proyecto:
datos -> forecast -> agente LLM.

Correr localmente con:
    streamlit run app.py

Deploy gratis en: https://share.streamlit.io (Streamlit Community Cloud)
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from generate_data import generate_sales_history, generate_inventory_snapshot
from forecast import build_reorder_report, forecast_sku_demand
from agent import generate_executive_summary

st.set_page_config(page_title="Agente de Reabastecimiento | NorthPeak Supply Co.", layout="wide")

st.title("📦 Agente de Pronóstico y Reabastecimiento")
st.caption(
    "Proyecto de portafolio — datos 100% sintéticos. "
    "Simula un pipeline de Supply Chain Analytics: forecasting estadístico + agente LLM."
)

# ---------------------------------------------------------------
# Carga / generación de datos (cacheado para no regenerar en cada click)
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    ventas = generate_sales_history()
    inventario = generate_inventory_snapshot()
    return ventas, inventario


@st.cache_data
def load_report(ventas, inventario):
    return build_reorder_report(ventas, inventario)


ventas_df, inventario_df = load_data()
reporte_df = load_report(ventas_df, inventario_df)

# ---------------------------------------------------------------
# Sidebar: configuración del agente
# ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Groq API Key (opcional)", type="password",
                             help="Gratis en console.groq.com. Sin key, se usa un resumen basado en reglas.")
    st.markdown("---")
    st.markdown("**Sobre el proyecto**")
    st.markdown(
        "- Datos: sintéticos (1 año de historial)\n"
        "- Forecast: Holt-Winters (statsmodels)\n"
        "- Agente: Groq / Llama 3.3 70B\n"
        "- [Ver código en GitHub](https://github.com/sebastiainq)"
    )

# ---------------------------------------------------------------
# KPIs generales
# ---------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("SKUs monitoreados", len(reporte_df))
col2.metric("Riesgo ALTO", len(reporte_df[reporte_df["riesgo_stockout"] == "ALTO"]))
col3.metric("Riesgo MEDIO", len(reporte_df[reporte_df["riesgo_stockout"] == "MEDIO"]))
col4.metric("Unidades sugeridas a pedir", int(reporte_df["unidades_sugeridas_pedir"].sum()))

st.markdown("---")

# ---------------------------------------------------------------
# Tabla de riesgo por SKU
# ---------------------------------------------------------------
st.subheader("📊 Reporte de riesgo de stockout por SKU")


def highlight_risk(val):
    colors = {"ALTO": "background-color: #ffcccc", "MEDIO": "background-color: #fff3cd", "BAJO": "background-color: #d4edda"}
    return colors.get(val, "")


st.dataframe(
    reporte_df.style.map(highlight_risk, subset=["riesgo_stockout"]),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------
# Gráfico de forecast por SKU seleccionado
# ---------------------------------------------------------------
st.subheader("📈 Forecast de demanda (30 días)")
sku_seleccionado = st.selectbox("Selecciona un SKU", reporte_df["sku"].tolist())

historico = ventas_df[ventas_df["sku"] == sku_seleccionado].tail(60).copy()
forecast = forecast_sku_demand(ventas_df, sku_seleccionado)

fig = px.line(historico, x="fecha", y="unidades_vendidas", title=f"Histórico + Forecast — {sku_seleccionado}")
fig.add_scatter(
    x=pd.date_range(start=pd.to_datetime(historico["fecha"]).max() + pd.Timedelta(days=1), periods=30),
    y=forecast.values,
    mode="lines",
    name="Forecast (30d)",
    line=dict(dash="dash", color="orange"),
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# Agente: resumen ejecutivo
# ---------------------------------------------------------------
st.markdown("---")
st.subheader("🤖 Resumen ejecutivo generado por el agente")

if st.button("Generar resumen ejecutivo"):
    with st.spinner("El agente está analizando el reporte..."):
        resumen = generate_executive_summary(reporte_df, api_key=api_key or None)
    st.markdown(resumen)
else:
    st.info("Presiona el botón para que el agente redacte el resumen ejecutivo basado en el reporte de arriba.")
