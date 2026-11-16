"""Pruebas del paso 5 del plan del cartel: pestaña 💰 Ahorro."""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import app_de_ejemplo
from vista_ahorro import cargar_resultados, patrones_demanda

RAIZ = Path(__file__).resolve().parent.parent


def test_sin_resultados_guardados_no_hay_nada_que_mostrar(tmp_path):
    assert cargar_resultados(None) is None
    assert cargar_resultados(tmp_path) is None


def test_patrones_de_viernes_y_manana():
    # Lunes 2 pedidos (mañana), viernes 4 pedidos (1 mañana, 3 tarde), una sola zona
    pedidos = pd.DataFrame({
        "date": ["2025-09-01"] * 2 + ["2025-09-05"] * 4,
        "time": ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"],
        "zone": ["Z1"] * 6,
    })
    tabla, manana, manana_viernes = patrones_demanda(pedidos)
    assert tabla.loc["Z1", "Viernes"] == pytest.approx(2.0)  # 4 pedidos frente a 2
    assert manana == 1.0 and manana_viernes == 0.25


def test_patrones_sin_hora_no_fallan():
    pedidos = pd.DataFrame({"date": ["2025-09-01", "2025-09-05"], "time": ["", ""], "zone": ["Z1", "Z1"]})
    _, manana, _ = patrones_demanda(pedidos)
    assert manana is None


def test_pestana_con_el_caso_de_estudio():
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=300)
    app.run()
    assert app.sidebar.radio[0].value == "Caso de estudio (demostración)"
    assert not app.exception
    assert app.tabs[0].label == "💰 Ahorro"
    etiquetas = {m.label: m for m in app.tabs[0].metric}
    assert etiquetas["Ahorro en 103 días"].value.startswith("$")
    assert "→" in etiquetas["Fletes urgentes"].value
    # Avisos de honestidad: demostración con simulación y resultados con los parámetros del artículo
    captions = [c.value for c in app.caption]
    assert any("caso de estudio" in c and "simulación" in c for c in captions)
    assert any("parámetros del artículo" in c for c in captions)
    # Una gráfica de costo por escenario y el mapa antes/después, a la vista; lo demás plegado en "Ver detalle"
    assert any(m.value.startswith("#### Antes y después") for m in app.tabs[0].markdown)
    assert sum(e.label.startswith("Ver detalle") for e in app.tabs[0].expander) >= 5
    # Tabla de modelos de pronóstico con el elegido marcado
    modelos = next(d.value for d in app.dataframe if "WAPE validación" in d.value.columns)
    assert modelos["Modelo"].str.contains("✔").sum() == 1


def test_pestana_con_datos_de_ejemplo_explica_que_falta_historia():
    # Los datos de ejemplo tienen 6 semanas: no alcanzan para entrenar, validar y probar
    app = app_de_ejemplo(300)
    assert not app.exception
    assert any("Hace falta un historial" in w.value for w in app.warning)
