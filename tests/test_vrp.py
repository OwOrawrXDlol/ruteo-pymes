"""Pruebas del paso 3: distancias y ruteo.

Se usan de 1 a 8 clientes colocados a distancias conocidas al norte o al este
del almacén, para que la respuesta correcta se pueda sacar a mano.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from src.distances import haversine_km, matriz_distancias, matriz_tiempos_min
from src.vrp import resolver_dia

ALMACEN = (20.0, -98.0)
GRADOS_POR_KM = 1 / 111.195  # 1 km hacia el norte, en grados de latitud

CONFIG = {
    "almacen": {"lat": ALMACEN[0], "lon": ALMACEN[1]},
    "flota": {"camionetas_propias": 4, "capacidad_cajas": 110, "max_urgentes": 5},
    "tiempos": {"jornada_h": 9, "servicio_min": 6, "velocidad_kmh": 22},
    "distancias": {"factor_circuidad": 1.30},
    "costos": {"costo_km": 4.20, "costo_propia": 1100, "tarifa_reservada": 1500, "tarifa_urgente": 3200},
    "solver": {"segundos_por_dia": 1},
}


def config(**cambios):
    """Copia de CONFIG con algunos valores cambiados, p. ej. config(flota={"camionetas_propias": 1})."""
    c = copy.deepcopy(CONFIG)
    for seccion, valores in cambios.items():
        c[seccion].update(valores)
    return c


def al_norte(*km_y_cajas):
    """Paradas en línea recta al norte del almacén: al_norte((1, 10), (2, 20)) = 1 km con 10 cajas, ..."""
    return pd.DataFrame([
        {"lat": ALMACEN[0] + k * GRADOS_POR_KM, "lon": ALMACEN[1], "boxes": b} for k, b in km_y_cajas
    ])


# --- Distancias ---

def test_un_grado_de_latitud_mide_unos_111_km():
    assert haversine_km(20, -98, 21, -98) == pytest.approx(111.195, abs=0.01)


def test_matriz_aplica_circuidad_y_es_simetrica():
    m = matriz_distancias([20.0, 20.0 + GRADOS_POR_KM], [-98.0, -98.0], 1.30)
    assert m[0][1] == pytest.approx(1.30, abs=0.001)
    assert np.allclose(m, m.T) and m[0][0] == 0


def test_22_km_se_recorren_en_una_hora():
    assert matriz_tiempos_min(np.array([[22.0]]), 22)[0][0] == pytest.approx(60)


# --- Ruteo ---

def test_dia_sin_paradas():
    r = resolver_dia(al_norte(), CONFIG)
    assert r["rutas"] == [] and r["km"] == 0


def test_un_cliente_ida_y_vuelta():
    # 2 km al norte: ida y vuelta = 4 km × 1.30 = 5.2 km
    r = resolver_dia(al_norte((2, 10)), CONFIG)
    assert len(r["rutas"]) == 1
    assert r["km"] == pytest.approx(5.2, abs=0.01)
    # 5.2 km a 22 km/h = 14.2 min, más 6 min de descarga
    assert r["rutas"][0]["horas"] == pytest.approx((5.2 / 22 * 60 + 6) / 60, abs=0.001)


def test_clientes_en_linea_se_visitan_en_orden_con_una_camioneta():
    # 5 clientes a 1..5 km, 20 cajas cada uno (100 en total, caben en una):
    # lo óptimo es una sola camioneta que va hasta el km 5 y regresa = 10 km × 1.30 = 13 km
    paradas = al_norte((3, 20), (1, 20), (5, 20), (2, 20), (4, 20))
    r = resolver_dia(paradas, CONFIG)
    assert len(r["rutas"]) == 1 and r["propias"] == 1
    assert r["km"] == pytest.approx(13.0, abs=0.01)
    distancias_en_orden = [paradas.lat[i] for i in r["rutas"][0]["paradas"]]
    assert distancias_en_orden in (sorted(distancias_en_orden), sorted(distancias_en_orden, reverse=True))


def test_capacidad_obliga_a_usar_dos_camionetas():
    # 3 clientes de 50 cajas = 150 > 110: hacen falta 2 camionetas, ninguna con más de 110
    r = resolver_dia(al_norte((1, 50), (2, 50), (3, 50)), CONFIG)
    assert r["propias"] == 2 and r["urgentes"] == 0
    assert all(ruta["cajas"] <= 110 for ruta in r["rutas"])
    assert r["cajas"] == 150


def test_urgente_solo_si_no_alcanza_la_flota_propia():
    # 1 camioneta propia y 2 clientes de 100 cajas: la segunda tiene que ser urgente
    c = config(flota={"camionetas_propias": 1})
    r = resolver_dia(al_norte((1, 100), (2, 100)), c)
    assert (r["propias"], r["urgentes"]) == (1, 1)


def test_con_reserva_no_se_contrata_urgente():
    c = config(flota={"camionetas_propias": 1})
    r = resolver_dia(al_norte((1, 100), (2, 100)), c, reservadas=1)
    assert (r["propias"], r["reservadas_usadas"], r["urgentes"]) == (1, 1, 0)


def test_reservada_se_usa_antes_que_propia_porque_ya_esta_pagada():
    r = resolver_dia(al_norte((1, 10)), CONFIG, reservadas=1)
    assert (r["propias"], r["reservadas_usadas"]) == (0, 1)


def test_jornada_obliga_a_usar_dos_camionetas():
    # 2 clientes juntos, poca carga, pero 5 h de descarga cada uno: no caben en una jornada de 9 h
    c = config(tiempos={"servicio_min": 300})
    r = resolver_dia(al_norte((1, 5), (1.1, 5)), c)
    assert len(r["rutas"]) == 2
    assert all(ruta["horas"] <= 9 for ruta in r["rutas"])


def test_cliente_imposible_de_alcanzar_se_reporta():
    # Un cliente a 150 km: ida y vuelta = 390 km = casi 18 h, más que la jornada de 9 h
    r = resolver_dia(al_norte((1, 10), (150, 10)), CONFIG)
    assert r["no_atendidas"] == [1]
    assert len(r["rutas"]) == 1 and r["cajas"] == 10


def test_ocho_clientes_en_dos_direcciones():
    # 4 clientes al norte y 4 al este, 25 cajas cada uno (100 por lado):
    # lo óptimo es una camioneta por lado; cada una va hasta el km 4 y regresa (8 km × 1.30)
    norte = al_norte((1, 25), (2, 25), (3, 25), (4, 25))
    km_lon = GRADOS_POR_KM / np.cos(np.radians(ALMACEN[0]))
    este = pd.DataFrame([{"lat": ALMACEN[0], "lon": ALMACEN[1] + k * km_lon, "boxes": 25} for k in (1, 2, 3, 4)])
    r = resolver_dia(pd.concat([norte, este], ignore_index=True), CONFIG)
    assert r["propias"] == 2
    assert r["km"] == pytest.approx(2 * 8 * 1.30, abs=0.05)
    grupos = sorted(sorted(ruta["paradas"]) for ruta in r["rutas"])
    assert grupos == [[0, 1, 2, 3], [4, 5, 6, 7]]
