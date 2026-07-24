#  Agente de Pronóstico y Reabastecimiento — XTR Supply Co.

Proyecto de portafolio que simula un pipeline real de **Supply Chain Analytics**:
predicción de demanda + agente de IA que traduce los resultados en decisiones
de negocio accionables.

>  **Nota:** "NorthPeak Supply Co." es una empresa ficticia y todos los datos
> son sintéticos, generados con estacionalidad y tendencia controladas para
> simular un escenario realista de retail/distribución.

##  Problema de negocio simulado

Un equipo de Supply Chain necesita saber, para cada SKU de su catálogo:
1. ¿Cuál va a ser la demanda de los próximos 30 días?
2. ¿Qué SKUs están en riesgo de quedarse sin stock (stockout)?
3. ¿Cuánto y cuándo pedir a proveedores?
4. ¿Cómo comunicar esto de forma priorizada a un gerente que no tiene tiempo
   de leer una tabla de 20 columnas?

Este es exactamente el tipo de problema que resuelven equipos como
**Amazon SCOT (Supply Chain Optimization Technologies)**.

##  Arquitectura

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   1. DATOS       │ --> │  2. FORECASTING (ML)  │ --> │  3. AGENTE (LLM)      │
│ Ventas históricas│     │  Holt-Winters +        │     │  Interpreta el        │
│ + inventario     │     │  Punto de Reorden      │     │  reporte y redacta    │
│ (sintéticos)     │     │  (statsmodels)         │     │  resumen ejecutivo    │
└─────────────────┘     └──────────────────────┘     └─────────────────────┘
```

**Decisión de diseño clave:** el LLM nunca calcula cifras — solo interpreta
resultados que ya produjo un modelo estadístico determinista. Esto evita que
el agente "alucine" números y es el patrón recomendado para arquitecturas
agénticas en contextos de negocio donde la precisión importa.

##  Stack

| Capa | Herramienta | Por qué |
|---|---|---|
| Datos | Python + Pandas + NumPy | Generación de series de tiempo sintéticas |
| Forecast | `statsmodels` (Holt-Winters) | Estándar de industria para forecasting con estacionalidad, explicable ante negocio |
| Agente | [Groq API](https://console.groq.com) (Llama 3.3 70B) | Free tier generoso, inferencia muy rápida, ideal para demos |
| Dashboard | Streamlit + Plotly | Deploy gratuito en Streamlit Community Cloud |

## Estructura del proyecto

```
inventory-forecast-agent/
├── generate_data.py   # Genera ventas históricas + inventario (sintético)
├── forecast.py         # Modelo Holt-Winters + cálculo de punto de reorden
├── agent.py             # Agente LLM que redacta el resumen ejecutivo
├── app.py                # Dashboard Streamlit (integra todo)
├── requirements.txt
└── README.md
```

## Cómo correrlo localmente

```bash
git clone https://github.com/sebastiainq/inventory-forecast-agent.git
cd inventory-forecast-agent
pip install -r requirements.txt

# Opcional: obtener una API key gratis en https://console.groq.com
export GROQ_API_KEY="tu_api_key_aqui"

streamlit run app.py
```

Si no configuras una API key, el proyecto sigue funcionando: el agente usa
un resumen basado en reglas (`_fallback_summary` en `agent.py`) para que la
demo nunca dependa de un servicio externo.

## Qué muestra este proyecto

- **Ingeniería de datos:** generación de series de tiempo con tendencia,
  estacionalidad semanal/anual y ruido controlado.
- **Analítica predictiva:** forecasting con modelos de series de tiempo
  (Holt-Winters), no solo promedios simples.
- **Lógica de negocio aplicada:** fórmulas reales de gestión de inventarios
  (punto de reorden, stock de seguridad, días de cobertura).
- **IA agéntica aplicada correctamente:** separación entre cálculo
  determinista (el modelo) e interpretación (el LLM), un patrón de diseño
  importante en sistemas de IA usados en producción.
- **Producto terminado:** dashboard interactivo, no solo un notebook.

##  Posibles extensiones futuras

- Reemplazar Holt-Winters por un modelo multivariado (Prophet con regresores externos, o LightGBM) para comparar performance.
- Agregar backtesting con métricas MAPE/WAPE para validar la calidad del forecast.
- Conectar el agente a un canal de Slack/Email para envío automático del resumen ejecutivo (vía n8n, por ejemplo).

---
*Proyecto desarrollado por Sebastián — [GitHub](https://github.com/sebastiainq)*
