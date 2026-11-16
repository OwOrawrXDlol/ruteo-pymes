"""Pruebas de la pestaña 📂 Mis datos: el flujo guiado de 4 pasos."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.conftest import EJEMPLO, abrir, app_de_ejemplo

RAIZ = Path(__file__).resolve().parent.parent
CASO = RAIZ / "data" / "caso_estudio"


def boton(app, etiqueta):
    return next(b for b in app.button if b.label == etiqueta)


def test_cuatro_acciones_para_ver_mi_ahorro():
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=600)
    app.run()
    # Preparación (no cuenta como acción del usuario): una semana y 1 s para que la prueba sea corta
    app.sidebar.number_input(key="semanas_prueba").set_value(1)
    app.sidebar.number_input(key="segundos_ahorro").set_value(1)
    app.run()

    abrir(app, "📂 Mis datos")                                                                    # 1
    app.file_uploader(key="pedidos.csv").upload("pedidos.csv", (CASO / "pedidos.csv").read_bytes()).run()   # 2
    app.file_uploader(key="clientes.csv").upload("clientes.csv", (CASO / "clientes.csv").read_bytes()).run()  # 3
    assert not app.exception
    assert any("tarda unos" in c.value for c in app.caption)  # tiempo estimado junto al botón
    boton(app, "💰 Calcular mi ahorro").click().run()                                             # 4
    assert not app.exception
    # El botón cambia a "Mis datos", abre 💰 Ahorro y calcula ahí
    assert app.session_state["origen_datos"] == "Mis datos"
    assert app.session_state["pestana"] == "💰 Ahorro"
    assert any("tus datos y tus parámetros" in c.value for c in app.caption)
    assert any(m.label.startswith("Ahorro en") for m in app.tabs[0].metric)


def test_columnas_con_otros_nombres_se_emparejan():
    app = app_de_ejemplo(300)
    # Pedidos con los nombres de columna de otra empresa: la app los reconoce sola
    texto = (EJEMPLO / "pedidos.csv").read_text().splitlines()
    texto[0] = texto[0].replace("order_id", "Folio").replace("date", "Fecha").replace("client_id", "Cliente") \
                       .replace("boxes", "Cajas").replace("time", "Hora")
    app.file_uploader(key="pedidos.csv").upload("ventas.csv", "\n".join(texto).encode()).run()
    assert not app.exception
    assert any(c.value.startswith("✅ Pedidos:") for c in app.caption)
    # Como tuvo que adivinar, el emparejamiento se muestra abierto para revisarlo
    assert next(e for e in app.expander if "«ventas.csv»" in e.label).proto.expanded
    # Con los nombres de la plantilla no hay nada que revisar: queda plegado
    assert not next(e for e in app.expander if "«clientes.csv»" in e.label).proto.expanded


def test_falta_una_columna_dice_que_corregir():
    app = app_de_ejemplo(300)
    columnas = (EJEMPLO / "pedidos.csv").read_text().splitlines()[0].split(",")
    sin = columnas.index("client_id")
    texto = "\n".join(",".join(c for i, c in enumerate(linea.split(",")) if i != sin)  # sin la columna del cliente
                       for linea in (EJEMPLO / "pedidos.csv").read_text().splitlines())
    app.file_uploader(key="pedidos.csv").upload("pedidos.csv", texto.encode()).run()
    assert not app.exception
    errores = [e.value for e in app.error]
    assert any("corrige esto" in e and "Falta indicar qué columna es" in e for e in errores)
    # Sin pedidos válidos no hay botón de calcular todavía
    assert not any(b.label == "💰 Calcular mi ahorro" for b in app.button)


def test_clientes_que_faltan_se_explican_al_juntar():
    app = app_de_ejemplo(300)
    clientes = (EJEMPLO / "clientes.csv").read_text().splitlines()
    app.file_uploader(key="clientes.csv").upload("clientes.csv", "\n".join(clientes[:-3]).encode()).run()
    assert not app.exception
    assert any("Al juntar tus archivos" in e.value for e in app.error)
    assert any("Todavía no hay datos tuyos" in i.value for i in app.info)  # las demás pestañas esperan
