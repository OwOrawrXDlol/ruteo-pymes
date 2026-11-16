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
    assert app.sidebar.radio[0].value == "Caso de estudio simulado (artículo)"
    assert not app.exception
    assert app.tabs[0].label == "💰 Ahorro"
    etiquetas = {m.label: m for m in app.metric}
    assert etiquetas["Ahorro frente a la práctica actual"].value.startswith("$")
    assert "→" in etiquetas["Contrataciones urgentes"].value
    assert any("caso de estudio simulado" in w.value and "no de una empresa" in w.value for w in app.warning)
    # Tabla de modelos de pronóstico con el elegido marcado
    modelos = next(d.value for d in app.dataframe if "WAPE validación" in d.value.columns)
    assert modelos["Modelo"].str.contains("✔").sum() == 1


def test_pestana_con_datos_de_ejemplo_explica_que_falta():
    app = app_de_ejemplo(300)
    assert not app.exception
    assert any("Caso de estudio simulado" in i.value for i in app.info)
