"""Pruebas del paso 4 del plan del cartel: indicadores de la evaluación (src/evaluacion.py)."""

from pathlib import Path

import pandas as pd
import pytest

from src.evaluacion import analizar, leer, tabla_escenarios
from src.load import cargar_config

RAIZ = Path(__file__).resolve().parent.parent
CONFIG = cargar_config(RAIZ / "config.yaml")


def detalle_a_mano():
    """Dos días (jueves y viernes) con números redondos.

    Jueves: E1 100 km / $5,000; E2 80 km / $4,800; E3 = E2; E4 = E2.
    Viernes: E1 150 km / $9,000 con 2 urgentes; E2 120 km / $8,500 con 2 urgentes;
             E3 120 km / $6,000 con 0 urgentes; E4 120 km / $5,500.
    """
    filas = []

    def fila(fecha, esc, km, costo, urgentes=0, reservadas=0, unidades=4):
        filas.append({"date": pd.Timestamp(fecha), "escenario": esc, "km": km, "costo": costo,
                      "urgentes": urgentes, "reservadas": reservadas, "propias": 4, "cajas": 400,
                      "capacidad": 110 * unidades, "no_atendidas": 0})

    for esc, km, costo in (("E1", 100, 5000), ("E2", 80, 4800), ("E3", 80, 4800), ("E4", 80, 4800)):
        fila("2025-11-13", esc, km, costo)
    fila("2025-11-14", "E1", 150, 9000, urgentes=2, unidades=6)
    fila("2025-11-14", "E2", 120, 8500, urgentes=2, unidades=6)
    fila("2025-11-14", "E3", 120, 6000, reservadas=2, unidades=6)
    fila("2025-11-14", "E4", 120, 5500, reservadas=2, unidades=6)
    return pd.DataFrame(filas)


def test_tabla_7():
    t = tabla_escenarios(detalle_a_mano(), CONFIG)
    assert t.loc["E1", "km_dia"] == 125 and t.loc["E1", "costo_dia"] == 7000
    assert t.loc["E1", "unidades_dia"] == 5  # (4 + 6) / 2
    assert t.loc["E3", "costo_caja"] == pytest.approx(10800 / 800)
    assert t.loc["E1", "utilizacion"] == pytest.approx(800 / (110 * 10))


def test_ahorros_y_brecha():
    a = analizar(detalle_a_mano(), CONFIG)
    assert a["km_E2_vs_E1"] == pytest.approx((200 / 250 - 1) * 100)          # −20 %
    assert a["costo_E3_vs_E1"] == pytest.approx((10800 / 14000 - 1) * 100)   # −22.9 %
    assert a["ahorro_E3_vs_E1_mxn"] == pytest.approx(3200)
    # Ahorro posible E2 → E4 = 13,300 − 10,300 = 3,000; logrado E2 → E3 = 2,500 → 83 %
    assert a["brecha_cerrada"] == pytest.approx(2500 / 3000 * 100)


def test_dias_y_viernes():
    a = analizar(detalle_a_mano(), CONFIG)
    assert (a["dias_E3_menor"], a["dias_E3_igual"], a["dias_E3_mayor"]) == (1, 1, 0)
    v = a["viernes"]
    assert v["dias"] == 1 and v["costo_dia_E1"] == 9000 and v["costo_dia_E3"] == 6000
    assert v["ahorro_E3_vs_E1"] == pytest.approx(100 / 3)
    assert (v["urgentes_E1"], v["urgentes_E3"]) == (2, 0)


def test_dias_incompletos_no_cuentan():
    d = pd.concat([detalle_a_mano(), pd.DataFrame([{
        "date": pd.Timestamp("2025-11-15"), "escenario": "E1", "km": 999, "costo": 99999, "urgentes": 9,
        "reservadas": 0, "propias": 4, "cajas": 400, "capacidad": 440, "no_atendidas": 0}])])
    assert analizar(d, CONFIG)["dias"] == 2


# --- Con el resultado guardado del caso de estudio (si ya se calculó) ---

@pytest.fixture(scope="module")
def guardado():
    detalle = leer(RAIZ / "data" / "caso_estudio")
    if detalle is None:
        pytest.skip("Falta correr: .venv/bin/python -m src.evaluacion data/caso_estudio")
    return analizar(detalle, CONFIG)


def test_caso_cubre_los_103_dias(guardado):
    assert guardado["dias"] == 103
    assert guardado["tabla"]["no_atendidas"].sum() == 0


def test_caso_el_ruteo_baja_km_y_el_pronostico_baja_urgentes(guardado):
    t = guardado["tabla"]
    assert guardado["km_E2_vs_E1"] < 0
    assert t.loc["E3", "urgentes"] < t.loc["E2", "urgentes"]
    assert t.loc["E4", "costo_total"] <= t.loc["E3", "costo_total"]  # E4 es el piso
