"""Pruebas del paso 2 del plan "cualquier PYME": adaptador de archivos propios (src/convert.py)."""

import io
import zipfile
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import pytest

from src.convert import (a_fecha, a_hora, a_numero, convertir, faltantes, leer_tabla, normalizar,
                         plantillas_zip, sugerir_mapeo)
from src.load import cargar_config, cargar_fuentes
from src.perfiles import cargar_perfiles, config_del_perfil
from tests.conftest import app_de_ejemplo

RAIZ = Path(__file__).resolve().parent.parent
GANADO = RAIZ / "data" / "ganado_pendiente"


# --- Leer ---

def test_lee_csv_con_punto_y_coma_y_acentos_latin1():
    contenido = "Fecha;Cliente;Cantidad\n06/10/2025;Tortillería;5\n".encode("latin-1")
    t = leer_tabla(contenido, "pedidos.csv")
    assert list(t.columns) == ["Fecha", "Cliente", "Cantidad"] and t.iloc[0]["Cliente"] == "Tortillería"


def test_lee_excel():
    memoria = io.BytesIO()
    pd.DataFrame({"fecha": [datetime(2025, 10, 6)], "cliente": ["C1"], "cajas": [3]}).to_excel(memoria, index=False)
    t = leer_tabla(memoria.getvalue(), "pedidos.xlsx")
    salida, problemas, _ = convertir(t, "pedidos.csv", sugerir_mapeo(t.columns, "pedidos.csv"))
    assert problemas == [] and salida.iloc[0]["date"] == "2025-10-06" and salida.iloc[0]["boxes"] == 3


def test_archivo_vacio_da_error_claro():
    with pytest.raises(ValueError, match="no tiene renglones"):
        leer_tabla(b"fecha,cliente\n", "pedidos.csv")


# --- Adivinar columnas ---

def test_normalizar_nombres():
    assert normalizar(" Fecha de Entrega ") == "fecha_de_entrega" and normalizar("Latitúd") == "latitud"


def test_sugiere_columnas_en_espanol():
    m = sugerir_mapeo(["Folio", "Fecha", "Hora", "Cliente", "Cajas"], "pedidos.csv")
    assert m == {"order_id": "Folio", "date": "Fecha", "time": "Hora", "client_id": "Cliente", "boxes": "Cajas"}
    m = sugerir_mapeo(["clave_cliente", "Latitud", "Longitud", "Zona"], "clientes.csv")
    assert m == {"client_id": "clave_cliente", "lat": "Latitud", "lon": "Longitud", "zone": "Zona"}


def test_detecta_lo_que_falta():
    m = sugerir_mapeo(["fecha", "algo"], "pedidos.csv")
    assert set(faltantes(m, "pedidos.csv")) == {"Cliente (o punto de entrega/carga)", "Cantidad (cajas, cabezas, kilos...)"}


# --- Valores ---

def test_fechas_iso_dia_primero_y_excel():
    f = a_fecha(pd.Series(["2025-10-06", "06/10/2025", "6/10/2025", "2025-10-06 00:00:00", "mañana"]))
    assert list(f[:4].dt.strftime("%Y-%m-%d")) == ["2025-10-06"] * 4
    assert pd.isna(f[4])


def test_horas():
    h = a_hora(pd.Series(["9:30", "09:30:15", "3:05 pm", "12:00 a.m.", "", time(14, 7), "25:00"]))
    assert list(h[:6]) == ["09:30", "09:30", "15:05", "00:00", "", "14:07"]
    assert pd.isna(h[6])  # 25:00 no es una hora: queda vacía y se reporta como problema


def test_numeros_con_coma():
    n = a_numero(pd.Series(["19,37", "-98.76", "1,200", "  45 ", "abc"]))
    assert list(n[:4]) == [19.37, -98.76, 1200.0, 45.0] and pd.isna(n[4])


# --- Convertir ---

def tabla(**columnas):
    return pd.DataFrame({k: list(v) for k, v in columnas.items()})


def test_genera_numero_de_pedido_si_no_viene():
    t = tabla(fecha=["2025-10-06"] * 2, cliente=["A", "B"], cajas=["3", "4"])
    salida, problemas, avisos = convertir(t, "pedidos.csv", sugerir_mapeo(t.columns, "pedidos.csv"))
    assert problemas == [] and list(salida["order_id"]) == ["P000001", "P000002"]
    assert any("numeraron" in a for a in avisos)


def test_problemas_se_reportan_y_no_se_inventan():
    t = tabla(fecha=["2025-10-06", "32/13/2025"], cliente=["A", "B"], cajas=["3", "x"])
    salida, problemas, _ = convertir(t, "pedidos.csv", sugerir_mapeo(t.columns, "pedidos.csv"))
    assert salida is None
    assert any("fechas no se entienden" in p and "32/13/2025" in p for p in problemas)
    assert any("no son números" in p for p in problemas)


def test_decimales_solo_se_redondean_si_se_pide():
    t = tabla(fecha=["2025-10-06"], cliente=["A"], kilos=["12.3"])
    m = sugerir_mapeo(t.columns, "pedidos.csv")
    _, problemas, _ = convertir(t, "pedidos.csv", m)
    assert any("decimales" in p for p in problemas)
    salida, problemas, _ = convertir(t, "pedidos.csv", m, redondear=True)
    assert problemas == [] and salida.iloc[0]["boxes"] == 13


def test_avisa_si_latitud_y_longitud_parecen_invertidas():
    t = tabla(cliente=["A"], latitud=["-98.75"], longitud=["20.10"])
    _, problemas, avisos = convertir(t, "clientes.csv", sugerir_mapeo(t.columns, "clientes.csv"))
    assert problemas == [] and any("invertidas" in a for a in avisos)


def test_falta_columna_obligatoria():
    t = tabla(cliente=["A"], latitud=["20.1"])
    salida, problemas, _ = convertir(t, "clientes.csv", sugerir_mapeo(t.columns, "clientes.csv"))
    assert salida is None and "Longitud" in problemas[0]


# --- Plantillas ---

def test_plantillas_se_cargan_en_la_app():
    z = zipfile.ZipFile(io.BytesIO(plantillas_zip()))
    assert {"pedidos.csv", "clientes.csv", "operacion_real.csv", "reserva.csv", "LEEME.txt"} <= set(z.namelist())
    datos = cargar_fuentes(io.BytesIO(z.read("pedidos.csv")), io.BytesIO(z.read("clientes.csv")),
                           cargar_config(RAIZ / "config.yaml"), io.BytesIO(z.read("operacion_real.csv")),
                           io.BytesIO(z.read("reserva.csv")))
    assert len(datos["paradas"]) == 2 and datos["reserva"] is not None


# --- Caso completo: los archivos de ganado tal como llegaron ---

def test_archivos_de_ganado_se_convierten_y_cargan():
    convertidos = {}
    for archivo in ("pedidos.csv", "clientes.csv", "operacion_real.csv"):
        t = leer_tabla((GANADO / archivo).read_bytes(), archivo)
        salida, problemas, _ = convertir(t, archivo, sugerir_mapeo(t.columns, archivo),
                                         vehiculos_incluyen_fletes=archivo == "operacion_real.csv")
        assert problemas == [], problemas
        convertidos[archivo] = io.BytesIO(salida.to_csv(index=False).encode())
    base = cargar_config(RAIZ / "config.yaml")
    config = config_del_perfil(base, cargar_perfiles(RAIZ / "perfiles"), "Transporte de ganado (ejemplo: Rancho Ramada)")
    datos = cargar_fuentes(convertidos["pedidos.csv"], convertidos["clientes.csv"], config,
                           convertidos["operacion_real.csv"])
    assert len(datos["clientes"]) == 10 and datos["paradas"]["boxes"].max() <= 80
    # El 30-nov: 4 jaulas, 1 urgente → 3 propias (la empresa tiene 3)
    op = datos["operacion_real"].set_index("date")
    assert op.loc[pd.Timestamp("2026-11-30").date(), "vehicles"] == 3


def test_la_opcion_mis_datos_no_falla_sin_archivos():
    app = app_de_ejemplo(300)
    for nombre in ("pedidos.csv", "clientes.csv"):
        app.file_uploader(key=nombre).set_value(None)  # quitar los archivos subidos
    app.run()
    assert not app.exception
    assert any("Todavía no hay datos tuyos" in i.value for i in app.info)
