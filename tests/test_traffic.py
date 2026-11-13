"""Pruebas del paso 1 de TRAFICO.md: tráfico simulado (modo B).

Puntos colocados a distancias conocidas al norte y al este de un origen,
para que se pueda revisar a mano qué tramo toca cada incidente.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.distances import matriz_distancias, matriz_tiempos_min
from src.traffic import (Incidente, Trafico, cargar_incidentes, distancia_punto_segmento_m,
                         magnitud_simulada, trafico_simulado)

RAIZ = Path(__file__).resolve().parent.parent
ORIGEN = (20.0, -98.0)
GRADOS_POR_KM_LAT = 1 / 111.195                       # 1 km al norte
GRADOS_POR_KM_LON = 1 / (111.195 * np.cos(np.radians(20.0)))  # 1 km al este (a latitud 20°)


def punto(norte_km=0.0, este_km=0.0):
    return ORIGEN[0] + norte_km * GRADOS_POR_KM_LAT, ORIGEN[1] + este_km * GRADOS_POR_KM_LON


def tres_puntos():
    """Punto 0 en el origen, punto 1 a 4 km al norte, punto 2 a 4 km al este.

    Tramos: 0-1 sube por el eje norte, 0-2 va por el eje este, 1-2 es la diagonal.
    """
    puntos = [punto(), punto(norte_km=4), punto(este_km=4)]
    lats = [p[0] for p in puntos]
    lons = [p[1] for p in puntos]
    minutos = matriz_tiempos_min(matriz_distancias(lats, lons, 1.30), 22)
    return lats, lons, minutos


# --- Distancia punto-segmento ---

def test_punto_sobre_el_segmento_esta_a_cero():
    lat, lon = punto(norte_km=2)
    a, b = punto(), punto(norte_km=4)
    assert distancia_punto_segmento_m(lat, lon, *a, *b) == pytest.approx(0, abs=1)


def test_punto_a_un_lado_del_segmento():
    # 1 km al este de la mitad del segmento norte: la perpendicular mide 1000 m
    lat, lon = punto(norte_km=2, este_km=1)
    a, b = punto(), punto(norte_km=4)
    assert distancia_punto_segmento_m(lat, lon, *a, *b) == pytest.approx(1000, rel=0.01)


def test_punto_mas_alla_del_extremo_mide_hasta_el_extremo():
    # 3 km al norte del final del segmento (que termina en el km 4): 3000 m al extremo B
    lat, lon = punto(norte_km=7)
    a, b = punto(), punto(norte_km=4)
    assert distancia_punto_segmento_m(lat, lon, *a, *b) == pytest.approx(3000, rel=0.01)


# --- Incidentes simulados sobre la matriz ---

def test_incidente_sobre_un_tramo_cambia_su_tiempo():
    lats, lons, base = tres_puntos()
    # A mitad del tramo 0-1 (2 km al norte), corrido 100 m al oeste para que no toque los otros tramos
    lat, lon = punto(norte_km=2, este_km=-0.1)
    t = trafico_simulado(lats, lons, base, [Incidente(lat, lon, radio_m=300, factor=2.5)])
    assert t.minutos[0][1] == pytest.approx(base[0][1] * 2.5)
    assert t.minutos[1][0] == pytest.approx(base[1][0] * 2.5)  # el regreso también
    assert t.afectados[0][1] and t.afectados[1][0]


def test_incidente_lejos_no_cambia_nada():
    lats, lons, base = tres_puntos()
    lejos = Incidente(*punto(norte_km=20, este_km=20), radio_m=300, factor=2.5)
    t = trafico_simulado(lats, lons, base, [lejos])
    assert np.array_equal(t.minutos, base)
    assert not t.afectados.any()


def test_incidente_solo_afecta_los_tramos_que_toca():
    lats, lons, base = tres_puntos()
    lat, lon = punto(norte_km=2, este_km=-0.1)
    t = trafico_simulado(lats, lons, base, [Incidente(lat, lon, radio_m=300, factor=2.5)])
    # 0-2 (eje este) y 1-2 (diagonal) pasan a más de 1 km del incidente
    assert t.minutos[0][2] == base[0][2] and t.minutos[1][2] == base[1][2]
    assert t.afectados.sum() == 2  # solo 0→1 y 1→0


def test_el_radio_decide_si_toca():
    lats, lons, base = tres_puntos()
    # A 250 m al oeste del tramo 0-1
    lat, lon = punto(norte_km=2, este_km=-0.25)
    cerca = trafico_simulado(lats, lons, base, [Incidente(lat, lon, radio_m=300, factor=2)])
    lejos = trafico_simulado(lats, lons, base, [Incidente(lat, lon, radio_m=200, factor=2)])
    assert cerca.afectados[0][1]
    assert not lejos.afectados[0][1]


def test_tramo_cerrado_cuesta_mas_que_la_jornada():
    lats, lons, base = tres_puntos()
    lat, lon = punto(norte_km=2, este_km=-0.1)
    t = trafico_simulado(lats, lons, base, [Incidente(lat, lon, factor=1, cerrado=True)],
                         minutos_cerrado=1000)
    assert t.minutos[0][1] == 1000
    assert t.minutos[0][1] > 9 * 60


def test_dos_incidentes_en_el_mismo_tramo_usan_el_factor_mayor():
    lats, lons, base = tres_puntos()
    a = Incidente(*punto(norte_km=1, este_km=-0.1), factor=2.0)
    b = Incidente(*punto(norte_km=3, este_km=-0.1), factor=3.0)
    t = trafico_simulado(lats, lons, base, [a, b])
    assert t.minutos[0][1] == pytest.approx(base[0][1] * 3.0)


def test_no_modifica_la_matriz_original_ni_la_diagonal():
    lats, lons, base = tres_puntos()
    copia = base.copy()
    # Justo encima del punto 0: toca todos los tramos que salen de él, pero no "de 0 a 0"
    t = trafico_simulado(lats, lons, base, [Incidente(*punto(), factor=2.0)])
    assert np.array_equal(base, copia)
    assert np.all(np.diag(t.minutos) == 0)


def test_sin_incidentes_queda_igual():
    lats, lons, base = tres_puntos()
    t = trafico_simulado(lats, lons, base, [])
    assert isinstance(t, Trafico) and t.fuente == "simulado"
    assert np.array_equal(t.minutos, base)


def test_matriz_de_tamano_equivocado_da_error_claro():
    lats, lons, base = tres_puntos()
    with pytest.raises(ValueError, match="matriz"):
        trafico_simulado(lats[:2], lons[:2], base, [])


# --- Incidentes: validación y archivo de ejemplo ---

def test_factor_menor_que_uno_no_se_acepta():
    with pytest.raises(ValueError, match="factor"):
        Incidente(20.0, -98.0, factor=0.5)


def test_magnitud_simulada_en_escala_tomtom():
    assert magnitud_simulada(1.2, False) == 1
    assert magnitud_simulada(2.0, False) == 2
    assert magnitud_simulada(2.5, False) == 3
    assert magnitud_simulada(1.0, True) == 4


def test_archivo_de_ejemplo_se_carga():
    incidentes = cargar_incidentes(RAIZ / "data" / "incidentes_ejemplo.csv")
    assert len(incidentes) >= 5
    assert any(i.cerrado for i in incidentes)
    assert all(i.fuente == "simulado" and i.descripcion for i in incidentes)
    # Todos los puntos están cerca de Pachuca (el último es el de control, a propósito más lejos)
    assert all(19.9 < i.lat < 20.3 and -98.9 < i.lon < -98.5 for i in incidentes)


def test_archivo_sin_columnas_obligatorias(tmp_path):
    ruta = tmp_path / "malo.csv"
    ruta.write_text("nombre,lat\nx,20.1\n")
    with pytest.raises(ValueError, match="lon"):
        cargar_incidentes(ruta)


def test_archivo_usa_valores_por_omision(tmp_path):
    ruta = tmp_path / "minimo.csv"
    ruta.write_text("lat,lon\n20.1,-98.7\n")
    [inc] = cargar_incidentes(ruta, radio_m=500, factor=3.0)
    assert inc.radio_m == 500 and inc.factor == 3.0 and not inc.cerrado


def test_config_tiene_valores_del_modo_simulado():
    config = yaml.safe_load((RAIZ / "config.yaml").read_text())
    simulado = config["trafico"]["simulado"]
    assert simulado["radio_m"] > 0 and simulado["factor"] >= 1
    assert simulado["minutos_cerrado"] > config["tiempos"]["jornada_h"] * 60
