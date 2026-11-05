"""Pruebas del paso 5: la app abre sin errores con los datos de ejemplo."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.load import valores_supuestos

RAIZ = Path(__file__).resolve().parent.parent


def test_lista_de_supuestos_del_config():
    supuestos = dict((p, v) for p, v, _ in valores_supuestos(RAIZ / "config.yaml"))
    assert supuestos["flota.capacidad_cajas"] == "110"
    assert supuestos["costos.tarifa_urgente"] == "3200"


def test_app_abre_con_datos_de_ejemplo():
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120)
    app.run()
    app.sidebar.slider[0].set_value(1).run()
    assert not app.exception
    assert any("datos de ejemplo inventados" in w.value for w in app.warning)
    assert len(app.metric) >= 6
    assert len(app.dataframe) >= 2  # tabla de camionetas y comparación del día


def test_app_calcula_un_rango():
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=300)
    app.run()
    app.sidebar.slider[0].set_value(1).run()
    app.date_input[0].set_value(("2025-10-09", "2025-10-11")).run()
    app.button[0].click().run()
    assert not app.exception
    assert any("brecha" in m.label for m in app.metric)


def test_pestana_kpi_muestra_rango_y_pronostico():
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=300)
    app.run()
    app.sidebar.slider[0].set_value(1).run()
    app.date_input[0].set_value(("2025-10-09", "2025-10-11")).run()
    app.button[0].click().run()
    assert not app.exception
    etiquetas = [m.label for m in app.metric]
    assert "WAPE del pronóstico (cajas/día)" in etiquetas
    assert "Ahorro E3 vs práctica actual" in etiquetas
