"""Pruebas del paso 5: la app abre sin errores con los datos de ejemplo."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from tests.conftest import abrir, app_de_ejemplo

from src.load import valores_supuestos

RAIZ = Path(__file__).resolve().parent.parent


def boton(app, etiqueta):
    """El botón con esa etiqueta (por posición cambiaría cada vez que se agrega uno)."""
    return next(b for b in app.button if b.label == etiqueta)


def test_lista_de_supuestos_del_config():
    supuestos = dict((p, v) for p, v, _ in valores_supuestos(RAIZ / "config.yaml"))
    assert supuestos["flota.capacidad_cajas"] == "110"
    assert supuestos["costos.tarifa_urgente"] == "3200"


def test_app_abre_con_datos_de_ejemplo():
    app = app_de_ejemplo(120)
    app.sidebar.slider[0].set_value(1).run()
    assert not app.exception
    # El aviso único sigue diciendo que los costos son supuestos (los de ejemplo ya no se ofrecen en la interfaz)
    assert any("supuestos" in c.value for c in app.caption)
    assert app.sidebar.radio(key="origen_datos").options == ["Caso de estudio (demostración)", "Mis datos"]
    abrir(app, "🗺️ Rutas de un día")
    assert not app.exception
    rutas = app.tabs[1]
    assert [m.label for m in rutas.metric][:4] == ["Costo del día", "Kilómetros", "Fletes urgentes", "Unidades que salen"]
    assert "→" in rutas.metric[0].value  # hoy → propuesta
    columnas = [list(d.value.columns) for d in rutas.dataframe]
    assert any("Orden de visita" in c for c in columnas)  # camionetas de la propuesta
    assert any("Como se hace hoy (E1)" in c for c in columnas)  # los 4 escenarios del día (antes 📊 y 📈)
    assert [t.label for t in app.tabs] == ["💰 Ahorro", "🗺️ Rutas de un día", "📂 Mis datos", "🚦 Tráfico (demo)",
                                             "🐄 Rancho Ramada"]


def test_app_calcula_un_rango():
    app = app_de_ejemplo(300)
    app.sidebar.slider[0].set_value(1).run()
    app.date_input[0].set_value(("2025-10-09", "2025-10-11")).run()
    boton(app, "Calcular periodo").click().run()
    assert not app.exception
    assert any(m.label == "Del ahorro máximo posible" for m in app.tabs[0].metric)


def test_el_periodo_muestra_indicadores_y_pronostico():
    app = app_de_ejemplo(300)
    app.sidebar.slider[0].set_value(1).run()
    app.date_input[0].set_value(("2025-10-09", "2025-10-11")).run()
    boton(app, "Calcular periodo").click().run()
    assert not app.exception
    etiquetas = [m.label for m in app.tabs[0].metric]
    assert "WAPE del pronóstico (cajas/día)" in etiquetas
    # La tabla de indicadores del periodo está en 💰 Ahorro (antes en la pestaña KPI)
    kpis = next(d.value for d in app.tabs[0].dataframe if "Costo logístico (MXN)" in d.value.index)
    assert list(kpis.columns) == ["Como se hace hoy (E1)", "Solo rutas optimizadas (E2)",
                                  "Pronóstico + rutas (propuesta) (E3)", "Ideal: saber el futuro (E4)"]


def test_trafico_usa_el_dia_elegido_en_rutas():
    app = abrir(app_de_ejemplo(300), "🗺️ Rutas de un día")
    app.sidebar.slider[0].set_value(1).run()
    dia = app.tabs[1].selectbox[0]
    otro = dia.options[0]  # el primer día (por omisión se abre el último)
    dia.select_index(0).run()
    abrir(app, "🚦 Tráfico (demo)")
    assert not app.exception
    assert any(otro[4:14] in s.value for s in app.tabs[3].subheader)  # "lun 06/10/2025 · ..." → "06/10/2025"
    # Al volver a rutas, el día sigue siendo el elegido
    abrir(app, "🗺️ Rutas de un día")
    assert app.tabs[1].selectbox[0].index == 0
