"""Ahorro con los datos de cualquier PYME: pronóstico + reserva + evaluación en sus últimas semanas.

Hace con los datos y parámetros de la empresa lo mismo que el artículo hizo con su caso de estudio:
1. Divide su historial: las últimas semanas son la PRUEBA, las anteriores la VALIDACIÓN y el resto el
   ENTRENAMIENTO (src/forecast.periodos_automaticos).
2. Entrena y compara los modelos de pronóstico y elige el de menor error en validación.
3. Convierte el pronóstico en unidades a reservar cada día de la prueba (src/integracion).
4. Resuelve los cuatro escenarios en cada día de la prueba con sus pedidos reales (src/scenarios).
El resultado tiene la misma forma que los resultados guardados del caso de estudio, para que la pestaña
💰 Ahorro lo muestre igual.
"""

import pandas as pd

from src.forecast import evaluar, periodos_automaticos
from src.integracion import planear_reservas
from src.scenarios import escenarios_rango


def dias_de_prueba(datos, semanas_prueba, semanas_validacion=4):
    """Días operativos que se evaluarían (para estimar cuánto tardará el cálculo)."""
    periodos = periodos_automaticos(datos["pedidos"]["date"], semanas_prueba, semanas_validacion)
    inicio, fin = periodos["prueba"]
    return [f for f in sorted(datos["paradas"]["date"].unique()) if inicio <= f <= fin]


def calcular_ahorro(datos, config, semanas_prueba=4, semanas_validacion=4, segundos=2, al_avanzar=None):
    """Todo el cálculo. Lanza ValueError con un mensaje en español si los datos no alcanzan."""
    columnas = [c for c in ("date", "time", "client_id", "boxes") if c in datos["pedidos"]]
    pedidos = datos["pedidos"][columnas]
    clientes = datos["clientes"]
    periodos = periodos_automaticos(pedidos["date"], semanas_prueba, semanas_validacion)

    pronostico = evaluar(pedidos, clientes, None, periodos)
    reserva, detalle_reserva, parametros = planear_reservas(pedidos, clientes, config, pronostico)

    # E3 usa la reserva que acaba de calcular el modelo (en lugar de un reserva.csv)
    con_reserva = {**datos, "reserva": reserva}
    inicio, fin = periodos["prueba"]
    fechas = [f for f in sorted(datos["paradas"]["date"].unique()) if inicio <= f <= fin]
    detalle, _resumen, avisos = escenarios_rango(con_reserva, fechas, config, segundos, al_avanzar)
    detalle["date"] = pd.to_datetime(detalle["date"])

    sesgo = pronostico["sesgo_mensual"]
    return {
        "metricas": pronostico["metricas"],
        "sesgo": pd.DataFrame({"mes": sesgo.index, "sesgo": sesgo.to_numpy()}),
        "elegido": parametros["modelo"],
        "parametros": pd.Series({"modelo": parametros["modelo"], "alfa": parametros["alfa"],
                                 "q_alfa": parametros["q_alfa"]}),
        "detalle": detalle,
        "reserva": detalle_reserva,
        "periodos": periodos,
        "avisos": avisos,
    }
