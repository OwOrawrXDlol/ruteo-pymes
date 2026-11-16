"""Pruebas del paso 3 del plan del cartel: del pronóstico a las unidades a reservar (src/integracion.py)."""

import copy
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.integracion import (cajas_por_dia, calibrar_q_alfa, fractil_critico, modelo_a_usar,
                             unidades_a_reservar)
from src.load import cargar_carpeta, cargar_config
from src.scenarios import escenarios_dia

RAIZ = Path(__file__).resolve().parent.parent
CASO = RAIZ / "data" / "caso_estudio"
CONFIG = cargar_config(RAIZ / "config.yaml")


def test_fractil_critico_del_articulo():
    # (3,200 − 1,500) / 3,200 = 0.531
    assert fractil_critico(CONFIG) == pytest.approx(0.531, abs=0.001)


def test_ecuacion_3():
    # 500 cajas × 1.0 / (110 × 0.95 = 104.5) = 4.78 → K = 5 → reservar 5 − 4 propias = 1
    assert unidades_a_reservar(500, 1.0, CONFIG) == 1
    # Con margen 1.1: 550 / 104.5 = 5.26 → K = 6 → 2
    assert unidades_a_reservar(500, 1.1, CONFIG) == 2
    # Un día flojo no reserva nada (nunca negativo)
    assert unidades_a_reservar(200, 1.05, CONFIG) == 0


def test_q_alfa_es_el_cuantil_de_real_entre_previsto():
    dias = [date(2025, 7, d) for d in range(1, 6)]
    previstas = pd.Series([100.0] * 5, index=dias)
    reales = pd.Series([90.0, 95, 100, 105, 110], index=dias)
    assert calibrar_q_alfa(reales, previstas, 0.5) == pytest.approx(1.0)
    assert calibrar_q_alfa(reales, previstas, 1.0) == pytest.approx(1.10)


def test_cajas_por_dia_usa_las_cajas_de_cada_zona():
    tabla = pd.DataFrame({"date": [date(2025, 9, 1)] * 2, "zone": ["Z1", "Z2"], "m": [10.0, 4.0]})
    assert cajas_por_dia(tabla, "m", pd.Series({"Z1": 6.0, "Z2": 2.5}))[date(2025, 9, 1)] == pytest.approx(70.0)


def test_modelo_auto_o_fijo():
    assert modelo_a_usar(CONFIG, "ridge") == "ridge"  # config.yaml dice "auto"
    fijo = copy.deepcopy(CONFIG)
    fijo["pronostico"]["modelo"] = "bosque"
    assert modelo_a_usar(fijo, "ridge") == "bosque"
    fijo["pronostico"]["modelo"] = "magia"
    with pytest.raises(ValueError, match="no existe"):
        modelo_a_usar(fijo, "ridge")


# --- El reserva.csv generado para el caso de estudio ---

def test_reserva_csv_del_caso():
    reserva = pd.read_csv(CASO / "reserva.csv", parse_dates=["date"])
    assert len(reserva) == 103
    assert reserva["date"].min().date() == date(2025, 9, 1) and reserva["date"].max().date() == date(2025, 12, 31)
    assert (reserva["reserved_extra"] >= 0).all()
    # Los viernes concentran la demanda: se reserva más que en un martes
    dia = reserva["date"].dt.day_of_week
    assert reserva.loc[dia == 4, "reserved_extra"].mean() > reserva.loc[dia == 1, "reserved_extra"].mean()


def test_parametros_guardados():
    p = pd.read_csv(CASO / "resultados" / "parametros_reserva.csv", index_col=0)["valor"]
    assert p["modelo"] == (CASO / "resultados" / "modelo_elegido.txt").read_text().strip()
    assert float(p["alfa"]) == pytest.approx(0.531, abs=0.001)
    assert 0.9 < float(p["q_alfa"]) < 1.3


def test_la_app_usa_la_reserva_en_e3():
    datos = cargar_carpeta(CASO, CONFIG)
    assert datos["reserva"] is not None
    reserva = pd.read_csv(CASO / "reserva.csv")
    fecha = date(2025, 11, 14)
    resultados, _ = escenarios_dia(datos, fecha, CONFIG, segundos=1)
    esperado = int(reserva.loc[reserva["date"] == "2025-11-14", "reserved_extra"].iloc[0])
    assert resultados["E3"]["reservadas"] == esperado and "reserva.csv" in resultados["E3"]["fuente"]
