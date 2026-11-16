"""Pruebas del paso 1 del plan "cualquier PYME": parámetros de la empresa y perfiles (src/perfiles.py)."""

import copy
from pathlib import Path

import pytest

from src.load import cargar_config, valores_supuestos
from src.perfiles import (CAMPOS, POR_OMISION, aplicar_valores, cargar_perfiles, config_del_perfil, diferencias,
                          fusionar, leer, problemas)
from tests.conftest import abrir, app_de_ejemplo

RAIZ = Path(__file__).resolve().parent.parent
BASE = cargar_config(RAIZ / "config.yaml")
PERFILES = cargar_perfiles(RAIZ / "perfiles")
GANADO = "Transporte de ganado (ejemplo: Rancho Ramada)"


def test_el_perfil_de_ganado_cambia_el_negocio():
    assert GANADO in PERFILES
    g = config_del_perfil(BASE, PERFILES, GANADO)
    assert g["negocio"]["unidad"] == "cabezas" and g["flota"]["camionetas_propias"] == 3
    assert g["tiempos"]["velocidad_kmh"] == 100
    assert 19 < g["almacen"]["lat"] < 20 and -97 < g["almacen"]["lon"] < -96  # Cardel, Veracruz
    # Lo que el perfil no menciona queda como en config.yaml
    assert g["solver"] == BASE["solver"] and g["reserva"]["u_max"] == BASE["reserva"]["u_max"]


def test_los_perfiles_no_modifican_la_base():
    copia = copy.deepcopy(BASE)
    config_del_perfil(BASE, PERFILES, GANADO)
    aplicar_valores(BASE, {"flota.capacidad_cajas": 5})
    assert BASE == copia


def test_por_omision_es_config_yaml():
    assert config_del_perfil(BASE, PERFILES, POR_OMISION) == BASE
    assert diferencias(BASE, BASE) == []


def test_aplicar_valores_y_diferencias():
    nueva = aplicar_valores(BASE, {"flota.capacidad_cajas": 90, "negocio.unidad": "kilos", "costos.costo_km": 5.5})
    assert leer(nueva, "flota.capacidad_cajas") == 90 and leer(nueva, "costos.costo_km") == 5.5
    assert set(diferencias(BASE, nueva)) == {"flota.capacidad_cajas", "negocio.unidad", "costos.costo_km"}


def test_fusionar_por_seccion():
    r = fusionar({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 9}})
    assert r == {"a": {"x": 1, "y": 9}, "b": 3}


def test_cada_campo_editable_existe_en_config():
    for ruta, *_ in CAMPOS:
        leer(BASE, ruta)  # no lanza KeyError


def test_avisos_de_valores_sin_sentido():
    assert problemas(BASE) == []
    mal = aplicar_valores(BASE, {"costos.tarifa_urgente": 1000, "costos.tarifa_reservada": 1500})
    assert any("urgente" in a for a in problemas(mal))
    assert any("coordenadas" in a for a in problemas(aplicar_valores(BASE, {"almacen.lat": 95.0})))


def test_perfil_sin_nombre_da_error_claro(tmp_path):
    (tmp_path / "malo.yaml").write_text("config: {}\n")
    with pytest.raises(ValueError, match="nombre"):
        cargar_perfiles(tmp_path)


def test_los_valores_del_perfil_dicen_si_son_reales_o_supuestos():
    for linea in (RAIZ / "perfiles" / "ganado_rancho_ramada.yaml").read_text().splitlines():
        valor = ":" in linea and not linea.lstrip().startswith("#") and linea.split(":", 1)[1].split("#")[0].strip()
        if valor and linea.startswith("    "):
            assert "real" in linea or "supuesto" in linea, f"falta etiqueta: {linea}"


def test_la_unidad_esta_en_config_yaml():
    assert dict((p, v) for p, v, _ in valores_supuestos(RAIZ / "config.yaml"))["negocio.unidad"] == "cajas"


# --- En la app ---

def test_cambiar_la_unidad_se_ve_en_pantalla():
    app = app_de_ejemplo(300)
    app.sidebar.text_input(key=f"{POR_OMISION}|negocio.unidad").set_value("cabezas").run()
    abrir(app, "🗺️ Rutas de un día")
    assert not app.exception
    tablas = [d.value for d in app.dataframe if "Orden de visita" in d.value.columns]
    assert tablas and "Cabezas" in tablas[0].columns and "Cajas" not in tablas[0].columns
    assert any("parámetros de tu empresa" in c.value for c in app.caption)


def test_elegir_un_perfil_cambia_los_valores():
    app = app_de_ejemplo(300)
    app.sidebar.selectbox(key="perfil_elegido").set_value(GANADO).run()
    assert not app.exception
    assert app.sidebar.number_input(key=f"{GANADO}|flota.camionetas_propias").value == 3
    assert any("Transporte de ganado" in c.value for c in app.caption)


def test_la_pestana_ahorro_avisa_si_cambiaron_los_parametros():
    app = app_de_ejemplo(300)
    app.sidebar.radio[0].set_value("Caso de estudio (demostración)").run()
    app.sidebar.number_input(key=f"{POR_OMISION}|costos.costo_km").set_value(9.0).run()
    assert not app.exception
    # Sigue mostrando los guardados, pero avisa que son con los parámetros del artículo y ofrece recalcular
    assert any("parámetros del artículo" in w.value and "no con los tuyos" in w.value for w in app.warning)
    assert any(b.label == "🔄 Recalcular con mis parámetros" for b in app.button)
    assert any(m.label == "Ahorro en 103 días" for m in app.tabs[0].metric)


# --- Barra lateral mínima ---

def test_supuestos_vigentes_respetan_el_perfil_y_lo_escrito_a_mano():
    from vista_config import supuestos_vigentes
    ruta = RAIZ / "config.yaml"
    todos = dict((p, v) for p, v, _ in supuestos_vigentes(ruta, RAIZ / "perfiles", POR_OMISION, []))
    assert todos == dict((p, v) for p, v, _ in valores_supuestos(ruta))
    # Con el perfil de ganado: la capacidad es un supuesto del perfil (80); la unidad la dio la empresa (real)
    ganado = dict((p, v) for p, v, _ in supuestos_vigentes(ruta, RAIZ / "perfiles", GANADO, []))
    assert ganado["flota.capacidad_cajas"] == "80" and "negocio.unidad" not in ganado
    # Lo que la persona escribe a mano ya no cuenta como supuesto
    a_mano = dict((p, v) for p, v, _ in supuestos_vigentes(ruta, RAIZ / "perfiles", GANADO, ["flota.capacidad_cajas"]))
    assert "flota.capacidad_cajas" not in a_mano


def test_la_barra_lateral_muestra_pocos_controles():
    app = app_de_ejemplo(300)
    assert not app.exception
    # A la vista: el origen de los datos y dos plegables (Mi empresa y Opciones avanzadas)
    assert [r.label for r in app.sidebar.radio] == ["Datos"]
    assert [e.label for e in app.sidebar.expander] == ["🏢 Mi empresa", "⚙️ Opciones avanzadas"]
    # Lo esencial está en "Mi empresa"; el resto (incluido el perfil y los segundos) en "Opciones avanzadas"
    empresa, avanzadas = app.sidebar.expander
    assert len(empresa.number_input) + len(empresa.text_input) == 7
    assert avanzadas.selectbox(key="perfil_elegido") and avanzadas.slider(key="segundos")
