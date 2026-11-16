"""Ayudas comunes de las pruebas."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parent.parent


def app_de_ejemplo(timeout=120):
    """Abre la app eligiendo los datos de ejemplo.

    Si hay archivos en data/ la app abre por defecto con esos (los de la empresa), y las
    pruebas no deben depender de ellos: así siguen funcionando cuando esos datos cambian.
    """
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=timeout)
    app.run()
    radio = app.sidebar.radio[0]
    if radio.value != "Datos de ejemplo (inventados)":
        radio.set_value("Datos de ejemplo (inventados)").run()
    return app
