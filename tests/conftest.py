"""Ayudas comunes de las pruebas."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parent.parent
EJEMPLO = RAIZ / "data" / "ejemplo"


def app_de_ejemplo(timeout=120):
    """Abre la app y sube los datos de ejemplo (data/ejemplo) como "Mis datos".

    Los datos de ejemplo ya no se ofrecen en la interfaz; las pruebas los suben como lo haría una PYME.
    Así tampoco dependen de los archivos que haya en data/ (los de la empresa), que pueden cambiar.
    """
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=timeout)
    app.run()
    app.sidebar.radio(key="origen_datos").set_value("Mis datos").run()
    for nombre in ("pedidos.csv", "clientes.csv"):
        app.file_uploader(key=nombre).upload(nombre, (EJEMPLO / nombre).read_bytes(), "text/csv")
    app.run()
    return app


def abrir(app, pestana):
    """Abre una pestaña ("🗺️ Rutas de un día", "🚦 Tráfico (demo)"...). Las pestañas son perezosas: solo
    se dibuja la abierta, así que hay que abrirla antes de buscar lo que contiene."""
    app.session_state["pestana"] = pestana
    return app.run()
