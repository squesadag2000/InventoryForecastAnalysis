"""
generate_data.py
-----------------
Genera un dataset SINTETICO de ventas historicas e inventario para
simular el problema de forecasting + reabastecimiento de una cadena
de retail ficticia ("NorthPeak Supply Co.").

Por que datos sinteticos y no reales:
- Permite mostrar el proyecto en un portafolio publico sin problemas
  de confidencialidad.
- Se controla la estacionalidad, tendencia y ruido para que el modelo
  de forecasting tenga algo interesante que predecir (esto es clave:
  datos totalmente aleatorios no sirven para demostrar valor de un
  modelo de forecast).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ---------------------------------------------------------------
# 1. Definimos un catalogo ficticio de SKUs con distinto perfil
# ---------------------------------------------------------------
SKUS = [
    {"sku": "SKU-001", "nombre": "Auriculares Bluetooth", "categoria": "Electronica",
     "base_demand": 40, "trend": 0.05, "seasonality": 15, "lead_time_days": 12, "safety_stock": 60},
    {"sku": "SKU-002", "nombre": "Cafetera Electrica", "categoria": "Hogar",
     "base_demand": 25, "trend": 0.02, "seasonality": 8, "lead_time_days": 20, "safety_stock": 40},
    {"sku": "SKU-003", "nombre": "Mochila Laptop", "categoria": "Accesorios",
     "base_demand": 18, "trend": -0.01, "seasonality": 5, "lead_time_days": 15, "safety_stock": 25},
    {"sku": "SKU-004", "nombre": "Lampara LED Escritorio", "categoria": "Hogar",
     "base_demand": 30, "trend": 0.03, "seasonality": 10, "lead_time_days": 10, "safety_stock": 45},
    {"sku": "SKU-005", "nombre": "Cargador Portatil 20000mAh", "categoria": "Electronica",
     "base_demand": 55, "trend": 0.08, "seasonality": 20, "lead_time_days": 18, "safety_stock": 80},
    {"sku": "SKU-006", "nombre": "Silla Ergonomica Oficina", "categoria": "Muebles",
     "base_demand": 12, "trend": 0.04, "seasonality": 4, "lead_time_days": 30, "safety_stock": 15},
]

DAYS_HISTORY = 365  # un anio de historico


def generate_sales_history():
    """Genera ventas diarias por SKU con tendencia + estacionalidad + ruido."""
    start_date = datetime.today() - timedelta(days=DAYS_HISTORY)
    dates = [start_date + timedelta(days=i) for i in range(DAYS_HISTORY)]

    rows = []
    for sku in SKUS:
        for i, date in enumerate(dates):
            trend_component = sku["trend"] * i
            # estacionalidad semanal (mas ventas fin de semana) + anual (Q4 alto)
            weekly = 1 + 0.3 * np.sin(2 * np.pi * date.weekday() / 7)
            annual = 1 + 0.4 * np.sin(2 * np.pi * (date.timetuple().tm_yday - 60) / 365)
            noise = np.random.normal(0, sku["seasonality"] * 0.3)

            demand = (sku["base_demand"] + trend_component) * weekly * annual + noise
            demand = max(0, round(demand))

            rows.append({
                "fecha": date.strftime("%Y-%m-%d"),
                "sku": sku["sku"],
                "nombre": sku["nombre"],
                "categoria": sku["categoria"],
                "unidades_vendidas": demand,
            })

    return pd.DataFrame(rows)


def generate_inventory_snapshot():
    """Genera el estado actual de inventario (stock disponible) por SKU."""
    rows = []
    for sku in SKUS:
        # stock actual simulado con algo de aleatoriedad realista
        stock_actual = int(sku["base_demand"] * np.random.uniform(3, 15))
        rows.append({
            "sku": sku["sku"],
            "nombre": sku["nombre"],
            "categoria": sku["categoria"],
            "stock_actual": stock_actual,
            "lead_time_days": sku["lead_time_days"],
            "safety_stock": sku["safety_stock"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ventas = generate_sales_history()
    inventario = generate_inventory_snapshot()

    ventas.to_csv("ventas_historicas.csv", index=False)
    inventario.to_csv("inventario_actual.csv", index=False)

    print(f"Generadas {len(ventas)} filas de ventas historicas -> ventas_historicas.csv")
    print(f"Generado snapshot de inventario para {len(inventario)} SKUs -> inventario_actual.csv")
