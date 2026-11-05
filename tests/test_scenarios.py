"""Pruebas del paso 4: escenarios E1 a E4."""

import copy
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.load import cargar_carpeta, cargar_config
from src.scenarios import (
    brecha_cerrada, costo_dia, escenarios_dia, escenarios_rango, orden_empirico,
    kpis_pronostico, pronostico_sustituto, reserva_del_dia, simular_e1, tabla_kpis, unidades_a_reservar,
)
from tests.test_vrp import ALMACEN, CONFIG as CONFIG_VRP, GRADOS_POR_KM, al_norte

RAIZ = Path(__file__).resolve().parent.parent
LUNES = date(2025, 9, 29)

CONFIG = copy.deepcopy(CONFIG_VRP)
CONFIG["reserva"] = {"q_alfa": 1.025, "u_max": 0.95, "semanas_historia": 4}


def con_propias(n):
    c = copy.deepcopy(CONFIG)
    c["flota"]["camionetas_propias"] = n
    return c


def datos_de(paradas, fecha=LUNES, pedidos=None, reserva=None, operacion_real=None):
    """Arma el diccionario que devuelve load.cargar_carpeta a partir de unas paradas de un solo día."""
    paradas = paradas.assign(date=fecha, client_id=[f"C{i}" for i in range(len(paradas))], zone="")
    if pedidos is None:
        pedidos = paradas[["date", "boxes"]]
    return {"paradas": paradas, "pedidos": pedidos, "reserva": reserva, "operacion_real": operacion_real}


def en_angulos(*grados_y_cajas):
    """Paradas a 1 km del almacén en los ángulos dados (0° = este, 90° = norte)."""
    km_lon = GRADOS_POR_KM / np.cos(np.radians(ALMACEN[0]))
    return pd.DataFrame([{
        "lat": ALMACEN[0] + np.sin(np.radians(g)) * GRADOS_POR_KM,
        "lon": ALMACEN[1] + np.cos(np.radians(g)) * km_lon,
        "boxes": b, "zone": "",
    } for g, b in grados_y_cajas])


# --- Costo y reserva ---

def test_costo_del_dia():
    # 100 km·4.20 + 4·1,100 + 1·1,500 + 1·3,200 = 420 + 4,400 + 1,500 + 3,200
    assert costo_dia(100, 4, 1, 1, CONFIG) == pytest.approx(9520)


@pytest.mark.parametrize("cajas, reservar", [
    (430, 1),  # 430·1.025 / (110·0.95) = 4.22 → K = 5 → reservar 5 − 4 = 1
    (400, 0),  # 3.92 → K = 4 → 0
    (0, 0),
    (650, 3),  # 6.38 → K = 7 → 3
])
def test_ecuacion_3(cajas, reservar):
    assert unidades_a_reservar(cajas, CONFIG) == reservar


def test_pronostico_sustituto_promedia_4_semanas_previas():
    # Lunes previos con 400, 420, 440 y 460 cajas → 430; el martes no debe contar
    fechas = [LUNES - timedelta(weeks=k) for k in (4, 3, 2, 1)]
    pedidos = pd.DataFrame({"date": fechas + [LUNES - timedelta(days=6)], "boxes": [400, 420, 440, 460, 999]})
    assert pronostico_sustituto(pedidos, LUNES, CONFIG) == pytest.approx(430)


def test_sin_historia_no_hay_pronostico():
    pedidos = pd.DataFrame({"date": [LUNES - timedelta(weeks=2), LUNES], "boxes": [100, 100]})
    assert pronostico_sustituto(pedidos, LUNES, CONFIG) is None


def test_reserva_csv_tiene_prioridad():
    datos = datos_de(al_norte((1, 10)), reserva=pd.DataFrame({"date": [LUNES], "reserved_extra": [2]}))
    assert reserva_del_dia(datos, LUNES, CONFIG) == (2, "reserva.csv")
    assert reserva_del_dia(datos, LUNES + timedelta(days=1), CONFIG)[0] is None


# --- E1 simulado ---

def test_orden_empirico_gira_alrededor_del_almacen():
    paradas = en_angulos((90, 1), (270, 1), (0, 1), (180, 1))  # norte, sur, este, oeste
    assert orden_empirico(paradas, CONFIG) == [2, 0, 3, 1]  # este, norte, oeste, sur


def test_despachador_llena_en_orden_y_usa_mas_camionetas_que_el_vrp():
    # En orden angular vienen 70, 70, 40, 40 cajas. El despachador llena en ese orden:
    # [70] (70+70 no cabe), [70, 40] y [40] = 3 camionetas. Lo óptimo son 2: [70, 40] y [70, 40].
    paradas = en_angulos((10, 70), (20, 70), (30, 40), (40, 40))
    rutas, fuera = simular_e1(paradas, CONFIG)
    assert [r["cajas"] for r in rutas] == [70, 110, 40]
    assert fuera == []
    resultados, _ = escenarios_dia(datos_de(paradas), LUNES, CONFIG, segundos=0.5)
    assert resultados["E1"]["propias"] == 3
    assert resultados["E2"]["propias"] == 2


def test_despachador_contrata_urgente_si_no_alcanza_la_flota():
    rutas, _ = simular_e1(en_angulos((10, 100), (20, 100)), con_propias(1))
    assert [r["tipo"] for r in rutas] == ["propia", "urgente"]


# --- Escenarios de un día ---

def test_cuatro_escenarios_con_una_propia_y_dos_viajes():
    # 1 propia, 2 clientes de 100 cajas, reserva.csv dice 1:
    #   E2: 1 propia + 1 urgente;  E3: 1 propia + 1 reservada;  E4: igual que E3 pero con las rutas de E2
    reserva = pd.DataFrame({"date": [LUNES], "reserved_extra": [1]})
    datos = datos_de(al_norte((1, 100), (2, 100)), reserva=reserva)
    r, avisos = escenarios_dia(datos, LUNES, con_propias(1), segundos=0.5)
    assert (r["E2"]["propias"], r["E2"]["reservadas"], r["E2"]["urgentes"]) == (1, 0, 1)
    assert (r["E3"]["propias"], r["E3"]["reservadas"], r["E3"]["urgentes"]) == (1, 1, 0)
    assert (r["E4"]["propias"], r["E4"]["reservadas"], r["E4"]["urgentes"]) == (1, 1, 0)
    km = r["E2"]["km"]
    assert r["E2"]["costo"] == pytest.approx(4.20 * km + 1100 + 3200)
    assert r["E4"]["costo"] == pytest.approx(4.20 * km + 1100 + 1500)
    assert r["E2"]["utilizacion"] == pytest.approx(200 / 220)
    assert avisos == []


def test_reserva_que_no_se_usa_igual_se_paga():
    reserva = pd.DataFrame({"date": [LUNES], "reserved_extra": [2]})
    r, _ = escenarios_dia(datos_de(al_norte((1, 10)), reserva=reserva), LUNES, CONFIG, segundos=0.5)
    e3 = r["E3"]
    # Sale 1 camioneta (una reservada, porque ya está pagada) pero se pagan las 2
    assert e3["reservadas"] == 2 and e3["propias"] == 0
    assert e3["costo"] == pytest.approx(4.20 * e3["km"] + 2 * 1500)
    assert e3["utilizacion"] == pytest.approx(10 / 110)


def test_e1_real_se_lee_de_operacion_real():
    operacion = pd.DataFrame({"date": [LUNES], "km": [80.0], "vehicles": [4],
                              "urgent_hires": [1], "reserved_hires": [0]})
    r, _ = escenarios_dia(datos_de(al_norte((1, 10)), operacion_real=operacion), LUNES, CONFIG, segundos=0.5)
    assert r["E1"]["fuente"].startswith("real")
    assert r["E1"]["costo"] == pytest.approx(80 * 4.20 + 4 * 1100 + 3200)


def test_sin_historia_e3_no_disponible_y_se_avisa():
    r, avisos = escenarios_dia(datos_de(al_norte((1, 10))), LUNES, CONFIG, segundos=0.5)
    assert r["E3"] is None
    assert any("E3 no disponible" in a for a in avisos)


# --- Rango con los datos de ejemplo ---

def test_rango_con_datos_de_ejemplo():
    config = cargar_config(RAIZ / "config.yaml")
    datos = cargar_carpeta(RAIZ / "data" / "ejemplo", config)
    fechas = [date(2025, 9, 26), date(2025, 10, 9), date(2025, 10, 10)]  # el 26 sep no tiene historia
    detalle, resumen, avisos = escenarios_rango(datos, fechas, config, segundos=0.5)
    assert len(detalle) == 3 * 4 - 1
    assert list(resumen["dias"]) == [2, 2, 2, 2]
    assert any("usa 2 de 3 días" in a for a in avisos)
    assert resumen.loc["E4", "costo"] <= resumen.loc["E2", "costo"]
    assert resumen.loc["E2", "km"] < resumen.loc["E1", "km"]
    assert resumen.loc["E1", "costo_vs_E1_%"] == pytest.approx(0)
    assert brecha_cerrada(resumen) is not None


# --- KPI ---

def test_kpis_de_la_reserva():
    # 3 días: hacía falta 1, 0 y 2 unidades extra (urgentes de E2); E3 reservó 1, 1 y 1
    d = [LUNES + timedelta(days=k) for k in range(3)]
    detalle = pd.DataFrame(
        [{"date": f, "escenario": "E2", "urgentes": u, "reservadas": 0} for f, u in zip(d, [1, 0, 2])]
        + [{"date": f, "escenario": "E3", "urgentes": u, "reservadas": 1} for f, u in zip(d, [0, 0, 1])]
    )
    datos = {"reserva": pd.DataFrame({"date": d, "reserved_extra": [1, 1, 1]})}
    kp = kpis_pronostico(datos, detalle, CONFIG)
    assert (kp["dias_exactos"], kp["dias_de_mas"], kp["dias_de_menos"]) == (1, 1, 1)
    assert kp["reservadas_sin_usar"] == 1
    assert kp["urgentes_evitados"] == 3 - 1
    assert kp["wape"] is None  # con reserva.csv no hay cajas previstas


def test_wape_del_sustituto():
    # 4 lunes previos con 100 cajas → previsto 100; real 80 → WAPE = 20/80 = 25%, sesgo +25%
    fechas = [LUNES - timedelta(weeks=k) for k in (4, 3, 2, 1)]
    pedidos = pd.DataFrame({"date": fechas + [LUNES], "boxes": [100, 100, 100, 100, 80]})
    detalle = pd.DataFrame([{"date": LUNES, "escenario": e, "urgentes": 0, "reservadas": 0} for e in ("E2", "E3")])
    kp = kpis_pronostico({"reserva": None, "pedidos": pedidos}, detalle, CONFIG)
    assert kp["wape"] == pytest.approx(0.25)
    assert kp["sesgo"] == pytest.approx(0.25)


def test_tabla_kpis_con_datos_de_ejemplo():
    config = cargar_config(RAIZ / "config.yaml")
    datos = cargar_carpeta(RAIZ / "data" / "ejemplo", config)
    _, resumen, _ = escenarios_rango(datos, [date(2025, 10, 10)], config, segundos=0.5)
    tabla = tabla_kpis(resumen)
    assert list(tabla.columns) == ["E1", "E2", "E3", "E4"]
    assert tabla.loc["Fletes urgentes", "E4"] == 0
    assert tabla.loc["Costo vs E1", "E1"] == pytest.approx(0)
    assert tabla.loc["Costo por caja entregada (MXN)", "E2"] == pytest.approx(
        resumen.loc["E2", "costo"] / resumen.loc["E2", "cajas"])
