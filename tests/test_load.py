"""Pruebas del paso 2: carga y validación de datos."""

from datetime import date
from io import StringIO
from pathlib import Path

import pytest

from src.load import (
    ErrorDatos, armar_paradas, cargar_carpeta, cargar_clientes, cargar_config,
    cargar_operacion_real, cargar_pedidos, cargar_reserva, paradas_del_dia, unir_pedidos_clientes,
)

RAIZ = Path(__file__).resolve().parent.parent

CLIENTES = """client_id,lat,lon,zone
A,20.10,-98.75,Z1
B,20.11,-98.76,Z1
C,20.12,-98.77,Z2
"""

PEDIDOS = """order_id,date,time,client_id,boxes
1,2025-09-01,09:00,A,10
2,2025-09-01,11:30,A,5
3,2025-09-01,,B,130
4,2025-09-02,08:15,C,7
"""


def csv(texto):
    return StringIO(texto)


def cargar_ejemplo_chico():
    clientes = cargar_clientes(csv(CLIENTES))
    return unir_pedidos_clientes(cargar_pedidos(csv(PEDIDOS)), clientes)


# --- Casos correctos ---

def test_pedidos_validos_se_convierten_a_fecha_y_entero():
    pedidos = cargar_pedidos(csv(PEDIDOS))
    assert pedidos["date"].iloc[0] == date(2025, 9, 1)
    assert pedidos["boxes"].sum() == 152


def test_union_pone_coordenadas_y_zona():
    datos = cargar_ejemplo_chico()
    fila = datos[datos["order_id"] == "4"].iloc[0]
    assert (fila.lat, fila.lon, fila.zone) == (20.12, -98.77, "Z2")


def test_pedidos_del_mismo_cliente_y_dia_se_suman():
    paradas = armar_paradas(cargar_ejemplo_chico(), capacidad=110)
    a = paradas[(paradas["client_id"] == "A")]
    assert len(a) == 1
    assert a.iloc[0].boxes == 15 and a.iloc[0].pedidos == 2


def test_pedido_mayor_a_una_camioneta_se_parte():
    paradas = armar_paradas(cargar_ejemplo_chico(), capacidad=110)
    b = paradas[paradas["client_id"] == "B"]
    assert list(b["boxes"]) == [110, 20]
    assert list(b["parte"]) == [1, 2]


def test_pedido_exacto_de_dos_camionetas_no_deja_parada_vacia():
    datos = cargar_ejemplo_chico()
    datos.loc[datos["client_id"] == "B", "boxes"] = 220
    b = armar_paradas(datos, capacidad=110).query("client_id == 'B'")
    assert list(b["boxes"]) == [110, 110]


def test_paradas_del_dia_filtra_por_fecha():
    paradas = armar_paradas(cargar_ejemplo_chico(), capacidad=110)
    dia = paradas_del_dia(paradas, date(2025, 9, 2))
    assert list(dia["client_id"]) == ["C"]
    assert list(dia.index) == [0]


def test_zona_es_opcional():
    clientes = cargar_clientes(csv("client_id,lat,lon\nA,20.1,-98.7\n"))
    assert clientes["zone"].iloc[0] == ""


def test_hora_es_opcional():
    pedidos = cargar_pedidos(csv("order_id,date,client_id,boxes\n1,2025-09-01,A,3\n"))
    assert len(pedidos) == 1


# --- Errores claros ---

def test_falta_columna():
    with pytest.raises(ErrorDatos, match="faltan columnas: boxes"):
        cargar_pedidos(csv("order_id,date,client_id\n1,2025-09-01,A\n"))


@pytest.mark.parametrize("cajas", ["0", "-3", "2.5", "muchas", ""])
def test_cajas_no_validas(cajas):
    with pytest.raises(ErrorDatos, match="'boxes' no son un entero mayor que 0"):
        cargar_pedidos(csv(f"order_id,date,client_id,boxes\n1,2025-09-01,A,{cajas}\n"))


def test_fecha_con_otro_formato():
    with pytest.raises(ErrorDatos, match="YYYY-MM-DD"):
        cargar_pedidos(csv("order_id,date,client_id,boxes\n1,01/09/2025,A,3\n"))


def test_hora_con_otro_formato():
    with pytest.raises(ErrorDatos, match="HH:MM"):
        cargar_pedidos(csv("order_id,date,time,client_id,boxes\n1,2025-09-01,9am,A,3\n"))


def test_pedido_repetido():
    with pytest.raises(ErrorDatos, match="'order_id' están repetidos: 1"):
        cargar_pedidos(csv("order_id,date,client_id,boxes\n1,2025-09-01,A,3\n1,2025-09-01,B,4\n"))


def test_cliente_sin_coordenadas():
    with pytest.raises(ErrorDatos, match="no tienen coordenadas.*B"):
        cargar_clientes(csv("client_id,lat,lon\nA,20.1,-98.7\nB,,-98.7\n"))


def test_coordenadas_imposibles():
    with pytest.raises(ErrorDatos, match="coordenadas imposibles"):
        cargar_clientes(csv("client_id,lat,lon\nA,200,-98.7\n"))


def test_pedido_de_cliente_desconocido():
    clientes = cargar_clientes(csv(CLIENTES))
    pedidos = cargar_pedidos(csv("order_id,date,client_id,boxes\n1,2025-09-01,ZZZ,3\n"))
    with pytest.raises(ErrorDatos, match="no están en clientes.csv: ZZZ"):
        unir_pedidos_clientes(pedidos, clientes)


def test_junta_varios_problemas_en_un_solo_error():
    texto = "order_id,date,client_id,boxes\n1,ayer,A,0\n"
    with pytest.raises(ErrorDatos) as error:
        cargar_pedidos(csv(texto))
    assert len(error.value.problemas) == 2


# --- Opcionales ---

def test_opcionales_ausentes_devuelven_none(tmp_path):
    assert cargar_operacion_real(tmp_path / "operacion_real.csv") is None
    assert cargar_reserva(None) is None


def test_reserva_valida():
    reserva = cargar_reserva(csv("date,reserved_extra\n2025-09-01,0\n2025-09-05,2\n"))
    assert list(reserva["reserved_extra"]) == [0, 2]


def test_operacion_real_con_km_malos():
    texto = "date,km,vehicles,urgent_hires,reserved_hires\n2025-09-01,abc,4,0,0\n"
    with pytest.raises(ErrorDatos, match="'km'"):
        cargar_operacion_real(csv(texto))


# --- Datos de ejemplo completos ---

def test_carga_carpeta_de_ejemplo():
    config = cargar_config(RAIZ / "config.yaml")
    datos = cargar_carpeta(RAIZ / "data" / "ejemplo", config)
    assert datos["paradas"]["boxes"].sum() == datos["pedidos"]["boxes"].sum()
    assert datos["paradas"]["boxes"].max() <= 110
    assert datos["operacion_real"] is None and datos["reserva"] is None
    assert any("130" in a or "2 viajes" in a for a in datos["avisos"])


def test_carpeta_sin_pedidos(tmp_path):
    config = cargar_config(RAIZ / "config.yaml")
    with pytest.raises(ErrorDatos, match="no existe"):
        cargar_carpeta(tmp_path, config)
