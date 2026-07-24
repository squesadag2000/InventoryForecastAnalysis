"""
agent.py
--------
Capa de agente (razonamiento con LLM).

Importante para la entrevista: este agente NO calcula el forecast ni
el punto de reorden -- eso ya lo hizo forecast.py con un modelo
estadistico. El LLM recibe esos numeros YA CALCULADOS y los traduce
en un analisis ejecutivo priorizado, como lo haria un analista
senior armando el resumen para su gerente. Esta separacion
(determinismo del modelo + interpretacion del LLM) evita que el
agente "alucine" cifras -- un problema comun cuando se le pide a un
LLM que haga matematica directamente.

Usa Groq (https://console.groq.com) porque:
- Tiene un free tier generoso (suficiente para un proyecto de
  portafolio o demo en entrevista).
- Es muy rapido (LPU inference), ideal para demos en vivo.
- Es compatible con el SDK de OpenAI, facil de intercambiar por
  otro proveedor (Gemini, OpenAI, etc.) si se desea.
"""

import os
import pandas as pd

# Se usa el cliente de Groq (instalar con: pip install groq)
# Alternativa 100% gratuita sin API key: se puede sustituir por un
# modelo local via Ollama (ver README) cambiando solo esta funcion.
try:
    from groq import Groq
except ImportError:
    Groq = None


SYSTEM_PROMPT = """Eres un Analista de Supply Chain Senior en NorthPeak Supply Co.
Tu trabajo es leer el reporte de riesgo de stockout (ya calculado por un modelo
estadistico) y redactar un resumen ejecutivo para el Gerente de Operaciones.

Reglas:
- NO inventes ni recalcules numeros; usa solo los que se te dan.
- Prioriza los SKUs de riesgo ALTO primero.
- Se conciso, profesional, y orientado a accion (que se debe hacer y para cuando).
- Cierra con una recomendacion general de 1-2 lineas.
- Responde en espanol, formato de reporte ejecutivo con vinetas.
"""


def generate_executive_summary(reporte_df: pd.DataFrame, api_key: str | None = None) -> str:
    """Genera el resumen ejecutivo a partir del reporte de reorden."""
    api_key = api_key or os.environ.get("GROQ_API_KEY")

    tabla_texto = reporte_df.to_string(index=False)
    user_prompt = f"""Aqui esta el reporte de riesgo de inventario de esta semana:

{tabla_texto}

Redacta el resumen ejecutivo."""

    if not api_key or Groq is None:
        # Fallback sin API key: resumen basado en reglas simples,
        # para que el proyecto funcione igual sin conexion/credenciales.
        return _fallback_summary(reporte_df)

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return completion.choices[0].message.content


def _fallback_summary(reporte_df: pd.DataFrame) -> str:
    """Resumen generado sin LLM, por si no hay API key configurada."""
    alto = reporte_df[reporte_df["riesgo_stockout"] == "ALTO"]
    medio = reporte_df[reporte_df["riesgo_stockout"] == "MEDIO"]

    lineas = ["**Resumen Ejecutivo (modo local, sin API key configurada)**\n"]
    if len(alto) > 0:
        lineas.append(f"⚠️ {len(alto)} SKU(s) en riesgo ALTO de quiebre de stock:")
        for _, r in alto.iterrows():
            lineas.append(
                f"  - {r['nombre']} ({r['sku']}): cobertura de {r['dias_cobertura']} dias, "
                f"pedir {int(r['unidades_sugeridas_pedir'])} unidades de inmediato."
            )
    if len(medio) > 0:
        lineas.append(f"\n🟡 {len(medio)} SKU(s) en riesgo MEDIO, monitorear esta semana.")

    lineas.append("\nRecomendacion: priorizar ordenes de compra para los SKUs de riesgo ALTO "
                   "antes del cierre de semana para evitar quiebre de stock.")
    return "\n".join(lineas)


if __name__ == "__main__":
    reporte = pd.read_csv("reporte_reorden.csv")
    resumen = generate_executive_summary(reporte)
    print(resumen)
