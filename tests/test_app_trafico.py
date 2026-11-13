"""Pruebas del paso 4 de TRAFICO.md: botón, mapa, tabla de decisión y resumen de costo."""

from datetime import time
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.replan import estados_a_la_hora, puntos_del_dia, replanear, resumen_costos
from src.scenarios import costo_dia
from src.traffic import Incidente, trafico_simulado
from tests.test_replan import CHOQUE_AB, CONFIG, CUADRO, config, lugar, paradas, plan
from vista_trafico import dibujar_mapa, tabla_decision

RAIZ = Path(__file__).resolve().parent.parent


def boton(app, etiqueta):
    return next(b for b in app.button if b.label == etiqueta)


def escenario_de(p, tabla, cfg):
    """Un 'resultado de escenario' mínimo para el plan dado (solo propias)."""
    lats, lons, km, _ = puntos_del_dia(tabla, cfg)
    total = 0.0
    for r in p["rutas"]:
        nodos = [0] + [i + 1 for i in r["paradas"]] + [0]
        total += sum(km[a][b] for a, b in zip(nodos, nodos[1:]))
    propias = len(p["rutas"])
    return {"km": total, "propias": propias, "reservadas": 0, "urgentes": 0,
            "costo": costo_dia(total, propias, 0, 0, cfg), "rutas": p["rutas"]}


# --- Resumen de km y costo ---

def test_resumen_sin_cambios_es_igual_antes_y_despues():
    tabla = paradas(*CUADRO)
    p = plan([0, 1, 2])
    lats, lons, _, minutos = puntos_del_dia(tabla, CONFIG)
    r = replanear(p, tabla, trafico_simulado(lats, lons, minutos, []), CONFIG)
    costos = resumen_costos(escenario_de(p, tabla, CONFIG), r, tabla, CONFIG)
    assert costos["despues"] == pytest.approx(costos["antes"])


def test_resumen_cuenta_los_km_del_plan_nuevo():
    # Cuadro de 4 km: plan A→B→C = 16 km en línea recta; nuevo A→C→B = 8 + 2 diagonales (11.3)
    tabla = paradas(*CUADRO)
    p = plan([0, 1, 2])
    lats, lons, _, minutos = puntos_del_dia(tabla, CONFIG)
    r = replanear(p, tabla, trafico_simulado(lats, lons, minutos, [CHOQUE_AB]), CONFIG)
    costos = resumen_costos(escenario_de(p, tabla, CONFIG), r, tabla, CONFIG)
    assert costos["antes"]["km"] == pytest.approx(16 * 1.3, abs=0.05)
    assert costos["despues"]["km"] == pytest.approx((8 + 8 * 2 ** 0.5) * 1.3, abs=0.05)
    assert costos["despues"]["costo"] - costos["antes"]["costo"] == pytest.approx(
        (costos["despues"]["km"] - costos["antes"]["km"]) * CONFIG["costos"]["costo_km"])


def test_resumen_suma_la_urgente_que_hace_falta():
    cfg = config(tiempos={"jornada_h": 1})
    tabla = paradas((6, 0), (-2, 0))
    p = plan([0], [1])
    lats, lons, _, minutos = puntos_del_dia(tabla, cfg)
    trafico = trafico_simulado(lats, lons, minutos, [Incidente(*lugar(3, -0.1), factor=2.0)])
    r = replanear(p, tabla, trafico, cfg)
    costos = resumen_costos(escenario_de(p, tabla, cfg), r, tabla, cfg)
    assert costos["urgentes_extra"] == 1
    # La Propia 1 se queda sin paradas antes de salir: ya no sale (no se paga), y entra una urgente
    assert costos["despues"]["propias"] == 1 and costos["despues"]["urgentes"] == 1
    assert costos["despues"]["costo"] > costos["antes"]["costo"]


# --- Tabla y mapa ---

def test_tabla_de_decision_y_mapa():
    tabla = paradas(*CUADRO)
    p = plan([0, 1, 2])
    lats, lons, _, minutos = puntos_del_dia(tabla, CONFIG)
    trafico = trafico_simulado(lats, lons, minutos, [CHOQUE_AB])
    tabla["client_id"] = ["A", "B", "C"]
    estados = estados_a_la_hora(p, minutos, 6, 25)  # ya visitó A
    r = replanear(p, tabla, trafico, CONFIG, estados=estados)

    t = tabla_decision(r)
    assert list(t["Decisión"]) == ["✅ cambiar"]
    assert t["¿Afectada?"].iloc[0] == "sí" and "incidente" in t["Criterio"].iloc[0]

    html = dibujar_mapa(tabla, CONFIG, p, estados, trafico, r, ["#1f77b4"]).get_root().render()
    assert "8 8" in html                     # plan actual punteado
    assert "#9e9e9e" in html                 # lo ya recorrido en gris
    assert "#e00000" in html                 # tramo afectado en rojo
    assert "posición simulada" in html and "💥" in html


# --- La app ---

def test_pestana_trafico_recalcula_con_incidentes_de_ejemplo():
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=300)
    app.run()
    app.sidebar.slider[0].set_value(1).run()
    assert not app.exception
    assert any("Prueba de concepto" in w.value for w in app.warning)

    # Agregar todos los incidentes de ejemplo y recalcular a las 8:30
    app.multiselect[0].set_value(app.multiselect[0].options).run()
    boton(app, "Agregar seleccionados").click().run()
    next(s for s in app.slider if s.label.startswith("Hora actual")).set_value(time(8, 30)).run()
    boton(app, "🚦 Recalcular con tráfico").click().run()
    assert not app.exception
    etiquetas = [m.label for m in app.metric]
    assert "Cambian de ruta" in etiquetas and "Costo del día" in etiquetas
    decision = next(d for d in app.dataframe if "Decisión" in d.value.columns).value
    assert len(decision) >= 1 and decision["Decisión"].str.contains("cambiar|mantener|urgente").all()

    # Si cambia la hora, el resultado anterior ya no se muestra como vigente
    next(s for s in app.slider if s.label.startswith("Hora actual")).set_value(time(9, 0)).run()
    assert any("presiona **Recalcular con tráfico**" in i.value for i in app.info)


# --- Modo TomTom en la app (sin internet: respuestas falsas, caché en carpeta temporal) ---

def test_sin_clave_tomtom_queda_deshabilitado(monkeypatch):
    monkeypatch.delenv("TOMTOM_API_KEY", raising=False)
    monkeypatch.setattr("src.tomtom.load_dotenv", lambda *a, **k: False)  # no leer un .env real
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=300)
    app.run()
    radio = next(r for r in app.sidebar.radio if r.label == "Fuente de tráfico")
    assert radio.options == ["Incidente simulado (sin internet)"]
    assert any("TomTom en vivo está deshabilitado" in c.value for c in app.sidebar.caption)


def test_modo_tomtom_consulta_y_muestra_el_contador(monkeypatch, tmp_path):
    from src.tomtom import ClienteTomTom
    from tests.test_tomtom import TomTomFalso

    falso = TomTomFalso()
    monkeypatch.setenv("TOMTOM_API_KEY", "clave-de-prueba")
    monkeypatch.setattr("src.tomtom.load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(ClienteTomTom, "_abrir_http", lambda self, url, cuerpo: falso(url, cuerpo))
    original = ClienteTomTom.__init__
    monkeypatch.setattr(ClienteTomTom, "__init__",
                        lambda self, clave, carpeta, *a, **k: original(self, clave, tmp_path / "cache", *a, **k))

    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=300)
    app.run()
    app.sidebar.slider[0].set_value(1).run()
    next(r for r in app.sidebar.radio if r.label == "Fuente de tráfico").set_value("tomtom").run()
    assert not app.exception
    contador = next(d for d in app.dataframe if "Gratis al mes" in d.value.columns).value
    assert list(contador["Hoy"]) == [0, 0, 0]

    boton(app, "🚦 Recalcular con tráfico").click().run()
    assert not app.exception
    assert any("TomTom en vivo" in w.value and "cobertura" in w.value for w in app.warning)
    assert any(ruta.endswith("incidentDetails") for ruta, _ in falso.llamadas)
    assert sum("calculateRoute" in ruta for ruta, _ in falso.llamadas) >= 1
    contador = next(d for d in app.dataframe if "Gratis al mes" in d.value.columns).value
    assert contador["Hoy"].iloc[0] >= 1  # Routing ya se contó
