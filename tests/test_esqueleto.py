"""Pruebas del paso 1: que el config y los datos de ejemplo existan y estén bien formados."""

from pathlib import Path

import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parent.parent
EJEMPLO = RAIZ / "data" / "ejemplo"


def test_config_tiene_valores_de_tabla_5():
    config = yaml.safe_load((RAIZ / "config.yaml").read_text())
    assert config["flota"]["camionetas_propias"] == 4
    assert config["flota"]["capacidad_cajas"] == 110
    assert config["tiempos"]["jornada_h"] == 9
    assert config["costos"]["tarifa_urgente"] == 3200


def test_cada_valor_de_config_dice_si_es_real_o_supuesto():
    for linea in (RAIZ / "config.yaml").read_text().splitlines():
        es_valor = ":" in linea and not linea.strip().startswith("#") and linea.split(":", 1)[1].split("#")[0].strip()
        if es_valor:
            assert "real" in linea or "supuesto" in linea, f"falta etiqueta: {linea}"


def test_datos_de_ejemplo_tienen_columnas_del_contrato():
    clientes = pd.read_csv(EJEMPLO / "clientes.csv")
    pedidos = pd.read_csv(EJEMPLO / "pedidos.csv")
    assert {"client_id", "lat", "lon", "zone"} <= set(clientes.columns)
    assert {"order_id", "date", "time", "client_id", "boxes"} <= set(pedidos.columns)
    assert len(clientes) == 60
    assert pedidos["client_id"].isin(clientes["client_id"]).all()
    assert (pedidos["boxes"] > 0).all()
