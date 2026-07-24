"""
forecast.py
-----------
Capa analitica del proyecto.

Que hace:
1. Para cada SKU, entrena un modelo de suavizado exponencial
   (Holt-Winters) sobre el historico de ventas para proyectar
   la demanda de los proximos N dias.
2. Calcula el Punto de Reorden (ROP) y el riesgo de quiebre de stock
   (stockout) usando formulas estandar de gestion de inventarios:

       Demanda diaria promedio (DDA) = demanda proyectada / dias
       ROP = (DDA * Lead Time) + Stock de Seguridad
       Dias de cobertura = Stock actual / DDA
       Riesgo = "ALTO" si dias_cobertura < lead_time_days
                "MEDIO" si dias_cobertura < lead_time_days * 1.5
                "BAJO" en otro caso

Por que Holt-Winters y no un modelo de ML mas complejo (ej. XGBoost,
LSTM): para series de tiempo cortas y con estacionalidad clara como
esta, los modelos de suavizado exponencial son el estandar de la
industria (usados en SAP, Oracle SCM, etc.), son explicables ante un
stakeholder de negocio, y no requieren GPU ni entrenamiento pesado.
Esto es justamente lo que se espera poder argumentar en una
entrevista tecnica: elegir la herramienta correcta para el problema,
no la mas compleja.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

FORECAST_HORIZON_DAYS = 30


def forecast_sku_demand(ventas_df: pd.DataFrame, sku: str, horizon: int = FORECAST_HORIZON_DAYS):
    """Entrena Holt-Winters para un SKU y devuelve la demanda proyectada diaria."""
    serie = ventas_df[ventas_df["sku"] == sku].sort_values("fecha")["unidades_vendidas"]
    serie = serie.reset_index(drop=True)

    # Holt-Winters con estacionalidad semanal (period=7)
    modelo = ExponentialSmoothing(
        serie,
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated",
    ).fit()

    forecast = modelo.forecast(horizon)
    forecast = forecast.clip(lower=0)  # no tiene sentido demanda negativa
    return forecast


def build_reorder_report(ventas_df: pd.DataFrame, inventario_df: pd.DataFrame) -> pd.DataFrame:
    """Genera el reporte consolidado de riesgo de stockout por SKU."""
    resultados = []

    for _, row in inventario_df.iterrows():
        sku = row["sku"]
        forecast = forecast_sku_demand(ventas_df, sku)

        demanda_diaria_promedio = round(forecast.mean(), 1)
        demanda_proyectada_30d = round(forecast.sum(), 0)

        lead_time = row["lead_time_days"]
        safety_stock = row["safety_stock"]
        stock_actual = row["stock_actual"]

        rop = round((demanda_diaria_promedio * lead_time) + safety_stock, 0)
        dias_cobertura = round(stock_actual / demanda_diaria_promedio, 1) if demanda_diaria_promedio > 0 else 999

        if dias_cobertura < lead_time:
            riesgo = "ALTO"
        elif dias_cobertura < lead_time * 1.5:
            riesgo = "MEDIO"
        else:
            riesgo = "BAJO"

        unidades_a_pedir = max(0, round(rop - stock_actual, 0))

        resultados.append({
            "sku": sku,
            "nombre": row["nombre"],
            "categoria": row["categoria"],
            "stock_actual": stock_actual,
            "demanda_diaria_prom": demanda_diaria_promedio,
            "demanda_proyectada_30d": demanda_proyectada_30d,
            "lead_time_days": lead_time,
            "punto_reorden": rop,
            "dias_cobertura": dias_cobertura,
            "riesgo_stockout": riesgo,
            "unidades_sugeridas_pedir": unidades_a_pedir,
        })

    return pd.DataFrame(resultados).sort_values(
        by="riesgo_stockout",
        key=lambda col: col.map({"ALTO": 0, "MEDIO": 1, "BAJO": 2}),
    )


if __name__ == "__main__":
    ventas = pd.read_csv("ventas_historicas.csv")
    inventario = pd.read_csv("inventario_actual.csv")
    reporte = build_reorder_report(ventas, inventario)
    print(reporte.to_string(index=False))
    reporte.to_csv("reporte_reorden.csv", index=False)
