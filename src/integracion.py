"""Integración del artículo: del pronóstico de pedidos a las unidades que hay que reservar.

El día anterior a la operación:
1. Los pedidos pronosticados de cada zona y franja se convierten en cajas con el promedio histórico de
   cajas por pedido de cada zona (periodo de entrenamiento), y se suman: cajas previstas del día, Ĉd.
2. Reservar una unidad de más cuesta su tarifa (c_res); que falte una cuesta el sobrecosto de contratarla
   urgente (c_urg − c_res). El nivel de servicio que equilibra los dos riesgos es el fractil crítico
   (ec. 2 del artículo, "problema del vendedor de periódicos"):
        α = (c_urg − c_res) / c_urg        → con las tarifas de config.yaml: 1,700 / 3,200 = 0.531
3. q_α = el cuantil α de la razón (cajas reales / cajas pronosticadas) en el periodo de VALIDACIÓN:
   un margen calibrado con los errores que el modelo ya tuvo, no un número fijo.
4. Ec. 3:  K = ⌈ Ĉd · q_α / (Q · u_max) ⌉   y   reservar = max(0, K − propias).

El resultado se escribe como reserva.csv (date, reserved_extra), el archivo que la app ya sabe leer
para el escenario E3.

Uso:  .venv/bin/python -m src.integracion data/caso_estudio
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecast import ENTRENAMIENTO, MODELOS, NOMBRES, evaluar, guardar
from src.load import cargar_config


def cajas_por_pedido(pedidos, clientes, rango=ENTRENAMIENTO):
    """Promedio de cajas por pedido de cada zona en el periodo dado."""
    p = pedidos.merge(clientes[["client_id", "zone"]], on="client_id")
    fechas = pd.to_datetime(p["date"]).dt.date
    p = p[(fechas >= rango[0]) & (fechas <= rango[1])]
    return p.groupby("zone")["boxes"].mean()


def cajas_por_dia(tabla, columna, cajas_zona):
    """Suma por día de (pedidos de cada zona y franja × cajas por pedido de la zona)."""
    cajas = tabla[columna] * tabla["zone"].map(cajas_zona)
    return cajas.groupby(tabla["date"]).sum()


def cajas_reales_por_dia(pedidos):
    return pedidos.assign(date=pd.to_datetime(pedidos["date"]).dt.date).groupby("date")["boxes"].sum()


def fractil_critico(config):
    """Ec. 2: α = (tarifa urgente − tarifa reservada) / tarifa urgente."""
    c = config["costos"]
    return (c["tarifa_urgente"] - c["tarifa_reservada"]) / c["tarifa_urgente"]


def calibrar_q_alfa(reales, previstas, alfa):
    """q_α = cuantil α de (real / pronosticado) en los días de validación."""
    razon = (reales.reindex(previstas.index) / previstas).dropna()
    return float(np.quantile(razon, alfa))


def unidades_a_reservar(cajas_previstas, q_alfa, config):
    """Ec. 3 del artículo para un día."""
    flota = config["flota"]
    k = math.ceil(cajas_previstas * q_alfa / (flota["capacidad_cajas"] * config["reserva"]["u_max"]))
    return max(0, k - flota["camionetas_propias"])


def modelo_a_usar(config, elegido_en_validacion):
    """El de config.yaml (pronostico.modelo), o el que ganó la validación si dice "auto"."""
    pedido = config.get("pronostico", {}).get("modelo", "auto")
    if pedido == "auto":
        return elegido_en_validacion
    if pedido not in MODELOS:
        raise ValueError(f"pronostico.modelo = '{pedido}' no existe. Opciones: auto, {', '.join(MODELOS)}.")
    return pedido


def planear_reservas(pedidos, clientes, config, resultado):
    """Unidades a reservar cada día del periodo de prueba, con el modelo elegido.

    resultado: lo que devuelve forecast.evaluar.
    """
    modelo = modelo_a_usar(config, resultado["elegido"])
    cajas_zona = cajas_por_pedido(pedidos, clientes)
    reales = cajas_reales_por_dia(pedidos)
    alfa = fractil_critico(config)
    q_alfa = calibrar_q_alfa(reales, cajas_por_dia(resultado["validacion"], modelo, cajas_zona), alfa)

    previstas = cajas_por_dia(resultado["prueba"], modelo, cajas_zona)
    reserva = pd.DataFrame({
        "date": previstas.index,
        "reserved_extra": [unidades_a_reservar(c, q_alfa, config) for c in previstas],
    })
    detalle = pd.DataFrame({
        "date": previstas.index,
        "cajas_previstas": previstas.to_numpy(),
        "cajas_reales": reales.reindex(previstas.index).to_numpy(),
        "reserved_extra": reserva["reserved_extra"].to_numpy(),
    })
    parametros = {"modelo": modelo, "alfa": alfa, "q_alfa": q_alfa,
                  "cajas_por_pedido": cajas_zona.round(3).to_dict()}
    return reserva, detalle, parametros


if __name__ == "__main__":
    origen = Path(sys.argv[1] if len(sys.argv) > 1 else "data/caso_estudio")
    raiz = Path(__file__).resolve().parent.parent
    config = cargar_config(raiz / "config.yaml")
    pedidos = pd.read_csv(origen / "pedidos.csv", dtype={"time": str})
    clientes = pd.read_csv(origen / "clientes.csv")
    tasa = pd.read_csv(origen / "tasa_real.csv") if (origen / "tasa_real.csv").exists() else None

    resultado = evaluar(pedidos, clientes, tasa)
    guardar(resultado, origen / "resultados")
    reserva, detalle, parametros = planear_reservas(pedidos, clientes, config, resultado)
    reserva.to_csv(origen / "reserva.csv", index=False)
    detalle.round(2).to_csv(origen / "resultados" / "reserva_detalle.csv", index=False)
    pd.Series({k: v for k, v in parametros.items() if k != "cajas_por_pedido"}).to_csv(
        origen / "resultados" / "parametros_reserva.csv", header=["valor"])

    print(f"Modelo: {NOMBRES[parametros['modelo']]} | α = {parametros['alfa']:.3f} | q_α = {parametros['q_alfa']:.3f}")
    print(f"{len(reserva)} días con reserva; total reservadas: {reserva['reserved_extra'].sum()} "
          f"| días con al menos una: {(reserva['reserved_extra'] > 0).sum()}")
