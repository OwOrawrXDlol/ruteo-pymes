"""Pruebas de los pasos 2 y 3 de TRAFICO.md: criterios de detección y aceptación,
y reoptimización desde la "hora actual".

Instancias pequeñas armadas a mano, con clientes a distancias conocidas al norte (N) y
al este (E) del almacén. Recordatorio de tiempos: 1 km en línea recta = 1.3 km por calle
= 3.55 min a 22 km/h. Cada parada suma 6 min de descarga.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from src.replan import (CAMBIAR, MANTENER, URGENTE, EstadoCamioneta, agrupar_para_intercambio,
                        cumple_retraso, detectar, estados_a_la_hora, horario, puntos_del_dia,
                        replanear, texto_hora)
from src.traffic import Incidente, trafico_simulado

ALMACEN = (20.0, -98.0)
KM_LAT = 1 / 111.195
KM_LON = 1 / (111.195 * np.cos(np.radians(20.0)))
MIN_POR_KM = 1.3 / 22 * 60  # minutos de manejo por km en línea recta

CONFIG = {
    "almacen": {"lat": ALMACEN[0], "lon": ALMACEN[1]},
    "flota": {"camionetas_propias": 4, "capacidad_cajas": 110, "max_urgentes": 5},
    "tiempos": {"jornada_h": 9, "servicio_min": 6, "velocidad_kmh": 22, "hora_salida": "08:00"},
    "distancias": {"factor_circuidad": 1.30},
    "costos": {"costo_km": 4.20, "costo_propia": 1100, "tarifa_reservada": 1500, "tarifa_urgente": 3200},
    "solver": {"segundos_por_dia": 1},
    "trafico": {"umbral_retraso_pct": 15, "umbral_retraso_min": 10, "magnitud_minima": 3,
                "radio_incidente_m": 300, "mejora_minima_min": 5,
                "segundos_reoptimizar": 1},
}


def config(**cambios):
    c = copy.deepcopy(CONFIG)
    for seccion, valores in cambios.items():
        c[seccion].update(valores)
    return c


def lugar(n=0.0, e=0.0):
    """Coordenadas de un punto a n km al norte y e km al este del almacén."""
    return ALMACEN[0] + n * KM_LAT, ALMACEN[1] + e * KM_LON


def paradas(*puntos, cajas=10):
    return pd.DataFrame([{"lat": lugar(n, e)[0], "lon": lugar(n, e)[1], "boxes": cajas} for n, e in puntos])


def plan(*rutas):
    """plan([0, 1], [2]) = Propia 1 visita las paradas 0 y 1; Propia 2 la 2."""
    return {"rutas": [{"vehiculo": f"Propia {i + 1}", "tipo": "propia", "paradas": list(r)}
                      for i, r in enumerate(rutas)]}


def con_incidentes(tabla, cfg, *incidentes):
    lats, lons, _, minutos = puntos_del_dia(tabla, cfg)
    return trafico_simulado(lats, lons, minutos, list(incidentes))


def fila(resultado, vehiculo="Propia 1"):
    t = resultado["tabla"]
    return t[t["vehiculo"] == vehiculo].iloc[0]


# --- Criterio de retraso: se piden los dos umbrales ---

def test_retraso_necesita_porcentaje_y_minutos():
    c = CONFIG["trafico"]
    assert cumple_retraso(20, 100, c)        # 20 % y 20 min: sí
    assert not cumple_retraso(12, 100, c)    # 12 %: porcentaje chico en ruta larga
    assert not cumple_retraso(5, 20, c)      # 25 % pero 5 min: retraso chico en ruta corta
    assert not cumple_retraso(0, 0, c)       # sin paradas pendientes


# --- Caso que debe cambiar ---

# Cuadro de 4 km: A al norte, B al noreste, C al este. Plan: almacén → A → B → C → almacén.
# Un choque grave a la mitad del lado A-B (factor 5) hace que convenga A → C → B (cruzar por
# la diagonal), que evita ese lado.
CUADRO = [(4, 0), (4, 4), (0, 4)]
CHOQUE_AB = Incidente(*lugar(4, 2), radio_m=300, factor=5.0)


def test_caso_que_debe_cambiar():
    tabla = paradas(*CUADRO)
    r = replanear(plan([0, 1, 2]), tabla, con_incidentes(tabla, CONFIG, CHOQUE_AB), CONFIG)
    f = fila(r)
    assert f["afectada"]
    assert "retraso" in f["criterios"] and "incidente" in f["criterios"]
    assert f["decision"] == CAMBIAR
    # Plan actual con tráfico: 4 + 4×5 + 4 + 4 = 32 km-equivalentes; nuevo: 4 + 5.66 + 4 + 5.66
    lado, diagonal = 4 * MIN_POR_KM, 4 * np.sqrt(2) * MIN_POR_KM
    assert f["min_con_trafico"] == pytest.approx(lado * 8 + 18, abs=0.1)
    assert f["min_plan_nuevo"] == pytest.approx(2 * lado + 2 * diagonal + 18, abs=0.1)
    assert f["ahorro_min"] == pytest.approx(f["min_con_trafico"] - f["min_plan_nuevo"])
    # La ruta adoptada ya no pasa de A a B ni de B a A
    nueva = r["rutas"][0]["pendientes_adoptadas"]
    assert sorted(nueva) == [0, 1, 2]
    assert nueva in ([0, 2, 1], [1, 2, 0])
    assert r["rutas"][0]["cambia"] and r["no_atendidas"] == []


def test_dos_afectadas_en_el_almacen_se_pasan_paradas():
    # Plan cruzado: Propia 1 = A (4 N) y C (4 E); Propia 2 = B (4.5 N) y D (4.5 E).
    # Un embotellamiento entre las dos diagonales afecta a las dos. Lo mejor es que una
    # se quede con el norte (A, B) y la otra con el este (C, D).
    tabla = paradas((4, 0), (4.5, 0), (0, 4), (0, 4.5))
    trafico = con_incidentes(tabla, CONFIG, Incidente(*lugar(2.18, 2.18), radio_m=300, factor=5.0))
    r = replanear(plan([0, 2], [1, 3]), tabla, trafico, CONFIG)
    assert list(r["tabla"]["decision"]) == [CAMBIAR, CAMBIAR]
    assert "reoptimizadas juntas" in fila(r)["nota"]
    grupos = sorted(sorted(x["pendientes_adoptadas"]) for x in r["rutas"])
    assert grupos == [[0, 1], [2, 3]]


# --- Casos que no superan la mejora mínima ---

def test_no_supera_la_mejora_minima():
    # El mismo cuadro, pero pidiendo al menos 60 min de ahorro: el nuevo ahorra ~45, se mantiene
    cfg = config(trafico={"mejora_minima_min": 60})
    tabla = paradas(*CUADRO)
    r = replanear(plan([0, 1, 2]), tabla, con_incidentes(tabla, cfg, CHOQUE_AB), cfg)
    f = fila(r)
    assert f["afectada"] and f["decision"] == MANTENER
    assert 40 < f["ahorro_min"] < 60
    assert "menos que el mínimo" in f["nota"]
    assert r["rutas"][0]["pendientes_adoptadas"] == [0, 1, 2] and not r["rutas"][0]["cambia"]


def test_afectada_sin_alternativa_se_mantiene():
    # Un solo cliente a 4 km con una obra en medio del camino: está afectada,
    # pero no hay otra forma de llegar (en este modelo), así que el ahorro es 0
    tabla = paradas((4, 0))
    trafico = con_incidentes(tabla, CONFIG, Incidente(*lugar(2, -0.1), factor=2.0))
    f = fila(replanear(plan([0]), tabla, trafico, CONFIG))
    assert f["afectada"] and f["decision"] == MANTENER
    assert f["ahorro_min"] == pytest.approx(0, abs=0.01)


# --- Caso que excede la jornada ---

def test_excede_jornada_y_requiere_urgente():
    # Jornada de 1 h. Cliente a 6 km: sin tráfico 42.5 min + 6 = 48.5 min (cabe);
    # con factor 2 en el camino son 91 min. Ni reoptimizando cabe: requiere urgente.
    # Propia 2 va a 2 km al sur, lejos del incidente: no se toca.
    cfg = config(tiempos={"jornada_h": 1})
    tabla = paradas((6, 0), (-2, 0))
    trafico = con_incidentes(tabla, cfg, Incidente(*lugar(3, -0.1), factor=2.0))
    r = replanear(plan([0], [1]), tabla, trafico, cfg)
    f = fila(r)
    assert f["min_sin_trafico"] == pytest.approx(12 * MIN_POR_KM + 6, abs=0.1)
    assert f["min_con_trafico"] > 60
    assert any("excede jornada" in x for x in f["criterios"].split("; "))
    assert f["decision"] == URGENTE
    assert r["no_atendidas"] == [0]
    otra = fila(r, "Propia 2")
    assert not otra["afectada"] and otra["decision"] == MANTENER and otra["nota"] == "no afectada"
    assert r["rutas"][1]["pendientes_adoptadas"] == [1]


def test_excede_jornada_pero_reordenar_lo_arregla():
    # Jornada de 90 min y el cuadro con el choque: con el plan actual (~132 min) no cabe,
    # con el nuevo (~87 min) sí. Se cambia aunque se exija un ahorro mínimo enorme (500 min),
    # porque la jornada es una regla y no una preferencia.
    tabla = paradas(*CUADRO)
    cfg = config(tiempos={"jornada_h": 1.5}, trafico={"mejora_minima_min": 500})
    f = fila(replanear(plan([0, 1, 2]), tabla, con_incidentes(tabla, cfg, CHOQUE_AB), cfg))
    assert "excede jornada" in f["criterios"]
    assert f["decision"] == CAMBIAR and "excede la jornada" in f["nota"]


# --- Criterio de incidente ---

def test_incidente_mayor_cerca_marca_aunque_no_haya_retraso():
    tabla = paradas((4, 0))
    lats, lons, _, minutos = puntos_del_dia(tabla, CONFIG)
    inc = Incidente(*lugar(2, -0.2), factor=1.0, magnitud=3)  # mayor, a 200 m, sin retraso medido
    trafico = trafico_simulado(lats, lons, minutos, [inc])
    d = detectar(EstadoCamioneta("Propia 1", "propia", 0, [1]), trafico, lats, lons, CONFIG)
    assert d["retraso_min"] == 0
    assert d["afectada"] and d["razones"] == ["incidente magnitud 3 (accidente) a 200 m"]


def test_incidente_menor_o_lejano_no_marca():
    tabla = paradas((4, 0))
    lats, lons, _, minutos = puntos_del_dia(tabla, CONFIG)
    estado = EstadoCamioneta("Propia 1", "propia", 0, [1])
    menor = Incidente(*lugar(2, -0.2), factor=1.0, magnitud=2)
    lejano = Incidente(*lugar(2, -0.5), factor=1.0, magnitud=4)  # a 500 m
    for inc in (menor, lejano):
        d = detectar(estado, trafico_simulado(lats, lons, minutos, [inc]), lats, lons, CONFIG)
        assert not d["afectada"]


def test_sin_trafico_nadie_cambia():
    tabla = paradas(*CUADRO)
    r = replanear(plan([0, 1, 2]), tabla, con_incidentes(tabla, CONFIG), CONFIG)
    assert list(r["tabla"]["decision"]) == [MANTENER]
    assert fila(r)["nota"] == "no afectada"


# --- Quién puede pasarse paradas ---

def test_solo_se_pasan_paradas_antes_de_salir():
    a = EstadoCamioneta("A", "propia", 0, [1])
    b = EstadoCamioneta("B", "propia", 0, [2])
    c = EstadoCamioneta("C", "propia", 3, [4], visitadas=[3], minutos_transcurridos=20, en_ruta=True)
    d = EstadoCamioneta("D", "propia", 0, [5], en_ruta=True)  # salió pero aún no entrega nada
    grupos = agrupar_para_intercambio([a, b, c, d])
    assert [[e.vehiculo for e in g] for g in grupos] == [["A", "B"], ["C"], ["D"]]


def test_matriz_de_otro_dia_da_error_claro():
    tabla = paradas(*CUADRO)
    trafico = con_incidentes(paradas((1, 0)), CONFIG)
    with pytest.raises(ValueError, match="no corresponde"):
        replanear(plan([0, 1, 2]), tabla, trafico, CONFIG)


# --- Paso 3: hora actual ---
# Línea al norte: A a 2 km, B a 4 km, C a 6 km, visitadas en ese orden.
# Cada tramo de 2 km = 7.09 min + 6 de descarga: A termina en 13.09, B en 26.18, C en 39.27.
TRAMO_2KM = 2 * MIN_POR_KM + 6


def test_hora_de_salida_y_fin_de_jornada():
    assert horario(CONFIG) == (480, 1020)
    assert texto_hora(510) == "08:30"


def test_paradas_visitadas_segun_la_hora():
    tabla = paradas((2, 0), (4, 0), (6, 0))
    _, _, _, minutos = puntos_del_dia(tabla, CONFIG)
    p = plan([0, 1, 2])

    [e] = estados_a_la_hora(p, minutos, 6, 0)  # antes de salir
    assert e.inicio == 0 and e.pendientes == [1, 2, 3] and not e.en_ruta

    [e] = estados_a_la_hora(p, minutos, 6, TRAMO_2KM - 0.5)  # descargando en A, aún no termina
    assert e.visitadas == [] and e.inicio == 0 and e.en_ruta

    [e] = estados_a_la_hora(p, minutos, 6, 20)  # terminó A, va hacia B
    assert e.visitadas == [1] and e.inicio == 1 and e.pendientes == [2, 3]
    assert e.minutos_transcurridos == pytest.approx(TRAMO_2KM)

    [e] = estados_a_la_hora(p, minutos, 6, 300)  # ya entregó todo
    assert e.pendientes == [] and e.inicio == 3


def test_reordena_las_pendientes_desde_la_ultima_visitada():
    # Cuadro con choque en el lado A-B. A los 25 min ya visitó A y le faltan B y C:
    # desde A conviene ir primero a C (diagonal) y luego a B, sin pasar por el lado A-B.
    tabla = paradas(*CUADRO)
    _, _, _, minutos = puntos_del_dia(tabla, CONFIG)
    p = plan([0, 1, 2])
    estados = estados_a_la_hora(p, minutos, 6, 25)
    r = replanear(p, tabla, con_incidentes(tabla, CONFIG, CHOQUE_AB), CONFIG, estados=estados)
    f = fila(r)
    lado, diagonal = 4 * MIN_POR_KM, 4 * np.sqrt(2) * MIN_POR_KM
    assert f["pendientes"] == 2
    assert f["min_con_trafico"] == pytest.approx(lado * 7 + 12, abs=0.1)   # A→B×5, B→C, C→almacén
    assert f["min_plan_nuevo"] == pytest.approx(2 * diagonal + lado + 12, abs=0.1)  # A→C, C→B, B→almacén
    assert f["decision"] == CAMBIAR
    ruta = r["rutas"][0]
    assert ruta["visitadas"] == [0] and ruta["pendientes_adoptadas"] == [2, 1]


def test_lo_ya_recorrido_no_cuenta():
    # Choque en el camino almacén-A. Antes de salir afecta; cuando A ya se visitó, ya no.
    tabla = paradas(*CUADRO)
    _, _, _, minutos = puntos_del_dia(tabla, CONFIG)
    p = plan([0, 1, 2])
    trafico = con_incidentes(tabla, CONFIG, Incidente(*lugar(2, -0.1), factor=5.0))
    antes = replanear(p, tabla, trafico, CONFIG)
    despues = replanear(p, tabla, trafico, CONFIG, estados=estados_a_la_hora(p, minutos, 6, 25))
    assert fila(antes)["afectada"]
    assert not fila(despues)["afectada"] and fila(despues)["retraso_min"] == 0


def test_en_ruta_no_se_pasan_paradas():
    # El plan cruzado de antes: antes de salir se reparten norte/este, pero ya en ruta
    # (minuto 1, cajas cargadas) cada camioneta se queda con sus propias paradas.
    tabla = paradas((4, 0), (4.5, 0), (0, 4), (0, 4.5))
    _, _, _, minutos = puntos_del_dia(tabla, CONFIG)
    trafico = con_incidentes(tabla, CONFIG, Incidente(*lugar(2.18, 2.18), radio_m=300, factor=5.0))
    p = plan([0, 2], [1, 3])
    r = replanear(p, tabla, trafico, CONFIG, estados=estados_a_la_hora(p, minutos, 6, 1))
    assert sorted(r["rutas"][0]["pendientes_adoptadas"]) == [0, 2]
    assert sorted(r["rutas"][1]["pendientes_adoptadas"]) == [1, 3]
    assert "reoptimizadas juntas" not in fila(r)["nota"]


def test_la_jornada_cuenta_lo_ya_trabajado():
    # Jornada de 1 h. A a 2 km y B a 4 km al norte. Sin tráfico: 7.1 + 7.1 + 14.2 + 12 = 40.4 min.
    # Obra a 3 km (factor 2.2): toca A→B y B→almacén, no almacén→A. A los 15 min ya visitó A
    # (terminó en el minuto 13.1). Le falta: A→B 15.6 + B→almacén 31.2 + 6 = 52.8 min, que
    # solos caben en 1 h, pero sumados a los 13.1 ya trabajados dan 65.9 > 60. Con una sola
    # parada pendiente no hay orden alternativo: requiere urgente.
    cfg = config(tiempos={"jornada_h": 1})
    tabla = paradas((2, 0), (4, 0))
    _, _, _, minutos = puntos_del_dia(tabla, cfg)
    p = plan([0, 1])
    trafico = con_incidentes(tabla, cfg, Incidente(*lugar(3, -0.1), factor=2.2))
    estados = estados_a_la_hora(p, minutos, 6, 15)
    f = fila(replanear(p, tabla, trafico, cfg, estados=estados))
    assert f["min_con_trafico"] == pytest.approx(6 * 2.2 * MIN_POR_KM + 6, abs=0.1)
    assert f["min_con_trafico"] < 60 < f["min_con_trafico"] + TRAMO_2KM  # solo no cabe por lo ya trabajado
    assert "excede jornada" in f["criterios"]
    assert f["decision"] == URGENTE
