"""Pruebas del paso 2 del plan del cartel: modelo predictivo de demanda (src/forecast.py)."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecast import (MODELOS, banderas_temporada, evaluar, franja_de, metricas, panel, variables)

RAIZ = Path(__file__).resolve().parent.parent
CASO = RAIZ / "data" / "caso_estudio"


def pedidos_chicos():
    """Dos zonas, tres días operativos (lunes, martes y el lunes siguiente)."""
    clientes = pd.DataFrame({"client_id": ["a", "b"], "zone": ["Z1", "Z2"]})
    pedidos = pd.DataFrame({
        "date": ["2025-03-03", "2025-03-03", "2025-03-03", "2025-03-04", "2025-03-10"],
        "time": ["09:00", "15:30", "12:59", "13:00", "10:00"],
        "client_id": ["a", "a", "b", "a", "a"],
    })
    return pedidos, clientes


# --- Tabla y variables ---

def test_franjas_mananas_antes_de_las_13():
    assert list(franja_de(pd.Series(["08:00", "12:59", "13:00", "18:45"]))) == ["manana", "manana", "tarde", "tarde"]


def test_panel_cuenta_y_rellena_con_ceros():
    t = panel(*pedidos_chicos())
    assert len(t) == 3 * 2 * 2  # días × zonas × franjas
    y = t.set_index(["date", "zone", "franja"])["y"]
    assert y[(date(2025, 3, 3), "Z1", "manana")] == 1 and y[(date(2025, 3, 3), "Z1", "tarde")] == 1
    assert y[(date(2025, 3, 3), "Z2", "manana")] == 1 and y[(date(2025, 3, 4), "Z2", "manana")] == 0


def test_rezagos_solo_usan_el_pasado():
    t = variables(panel(*pedidos_chicos())).set_index(["date", "zone", "franja"])
    # Martes Z1 tarde: el día anterior (lunes) hubo 1 pedido por la tarde
    assert t.loc[(date(2025, 3, 4), "Z1", "tarde"), "rezago_1"] == 1
    # Lunes 10: hace 7 días (lunes 3) hubo 1 pedido por la mañana en Z1
    assert t.loc[(date(2025, 3, 10), "Z1", "manana"), "rezago_7"] == 1
    # El primer día no tiene pasado
    assert np.isnan(t.loc[(date(2025, 3, 3), "Z1", "manana"), "rezago_1"])


def test_pedidos_sin_hora_o_sin_zona_dan_error_claro():
    pedidos, clientes = pedidos_chicos()
    with pytest.raises(ValueError, match="hora"):
        panel(pedidos.assign(time=""), clientes)
    with pytest.raises(ValueError, match="zona"):
        panel(pedidos, clientes.assign(zone=""))


def test_temporadas():
    assert banderas_temporada(date(2025, 11, 14))["buen_fin"] == 1
    assert banderas_temporada(date(2025, 11, 19))["post_buen_fin"] == 1
    assert banderas_temporada(date(2025, 12, 10))["diciembre"] == 1
    assert banderas_temporada(date(2025, 12, 27))["fin_anio"] == 1
    assert banderas_temporada(date(2025, 4, 16))["semana_santa"] == 1
    assert banderas_temporada(date(2025, 6, 10)) == dict.fromkeys(banderas_temporada(date(2025, 6, 10)), 0)


# --- Métricas ---

def test_wape_y_sesgo_como_en_el_articulo():
    # Real 10 y 10, previsto 8 y 14: errores −2 y +4 → WAPE = 6/20 = 30 %, sesgo = +2/20 = +10 %
    m = metricas([10, 10], [8, 14])
    assert m["wape"] == pytest.approx(0.30) and m["sesgo"] == pytest.approx(0.10)
    assert m["mae"] == pytest.approx(3.0) and m["rmse"] == pytest.approx(np.sqrt(10))


# --- Con el caso de estudio completo ---

@pytest.fixture(scope="module")
def resultado():
    pedidos = pd.read_csv(CASO / "pedidos.csv", dtype={"time": str})
    clientes = pd.read_csv(CASO / "clientes.csv")
    return evaluar(pedidos, clientes, pd.read_csv(CASO / "tasa_real.csv"))


def test_prueba_cubre_103_dias_sin_huecos(resultado):
    prueba = resultado["prueba"]
    assert prueba["date"].nunique() == 103 and len(prueba) == 103 * 8 * 2
    assert prueba[MODELOS + ["oraculo"]].notna().all().all()


def test_los_modelos_superan_a_las_referencias(resultado):
    w = resultado["metricas"].set_index("modelo")["wape"]
    mejor = min(w[m] for m in ("ridge", "bosque", "gboost"))
    assert mejor < w["prom4"] < w["ingenuo"]
    # Artículo: el mejor modelo baja ~27 % el error del ingenuo
    assert 1 - mejor / w["ingenuo"] > 0.20


def test_nadie_le_gana_al_oraculo(resultado):
    w = resultado["metricas"].set_index("modelo")["wape"]
    assert all(w[m] > w["oraculo"] for m in MODELOS)
    assert w["oraculo"] == pytest.approx(0.406, abs=0.03)


def test_el_error_por_dia_es_mucho_menor_que_por_celda(resultado):
    m = resultado["metricas"].set_index("modelo")
    elegido = resultado["elegido"]
    assert m.loc[elegido, "wape_diario"] < m.loc[elegido, "wape"] / 2  # artículo: 15.5 % frente a 41.8 %


def test_se_elige_por_validacion(resultado):
    m = resultado["metricas"].set_index("modelo")
    assert resultado["elegido"] == m.loc[MODELOS, "wape_validacion"].idxmin()
