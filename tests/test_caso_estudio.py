"""Pruebas del paso 1 del plan del cartel: el caso de estudio SIMULADO se parece al del artículo."""

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.load import cargar_carpeta, cargar_config

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "data" / "caso_estudio"

_spec = importlib.util.spec_from_file_location("generar_caso", CARPETA / "generar_caso.py")
generar_caso = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generar_caso)


@pytest.fixture(scope="module")
def datos():
    pedidos = pd.read_csv(CARPETA / "pedidos.csv", parse_dates=["date"])
    clientes = pd.read_csv(CARPETA / "clientes.csv")
    return pedidos, clientes


def test_tamano_del_caso(datos):
    pedidos, clientes = datos
    assert len(clientes) == 560 and clientes["zone"].nunique() == 8
    assert clientes["client_id"].is_unique and clientes["client_id"].str.startswith("SIM-").all()
    # Artículo: 48,948 pedidos. La reconstrucción no es idéntica: se acepta ±4 %
    assert abs(len(pedidos) - 48948) / 48948 < 0.04
    assert pedidos["client_id"].isin(clientes["client_id"]).all() and (pedidos["boxes"] > 0).all()


def test_dias_operativos_como_el_articulo(datos):
    pedidos, _ = datos
    dias = generar_caso.dias_operativos()
    assert len(dias) == 621
    assert sum(generar_caso.PRUEBA[0] <= d <= generar_caso.PRUEBA[1] for d in dias) == 103
    assert pedidos["date"].min().date() == date(2024, 1, 2) and pedidos["date"].max().date() == date(2025, 12, 31)
    assert (pedidos["date"].dt.day_of_week < 6).all()  # sin domingos
    assert date(2025, 12, 25) not in set(pedidos["date"].dt.date)  # sin festivos


def test_promedios_de_entrenamiento(datos):
    pedidos, clientes = datos
    ent = pedidos[(pedidos["date"] >= "2024-02-05") & (pedidos["date"] <= "2025-08-31")]
    dias = ent["date"].nunique()
    assert len(ent) / dias == pytest.approx(76.4, rel=0.04)       # pedidos por día
    assert ent["boxes"].sum() / dias == pytest.approx(401, rel=0.06)  # cajas por día
    por_zona = ent.merge(clientes, on="client_id").groupby("zone")["boxes"].mean()
    for zona, _t, _km, _ped, cajas, *_ in generar_caso.ZONAS:
        assert por_zona[zona] == pytest.approx(cajas, rel=0.12)


def test_patrones_de_viernes_y_sabado(datos):
    pedidos, clientes = datos
    ent = pedidos[(pedidos["date"] >= "2024-02-05") & (pedidos["date"] <= "2025-08-31")]
    m = ent.assign(dow=ent["date"].dt.day_of_week).merge(clientes, on="client_id").groupby(["zone", "dow"]).size().unstack()
    for zona, _t, _km, _ped, _caj, f_vie, f_sab, _pct in generar_caso.ZONAS:
        base = m.loc[zona, [0, 1, 2, 3]].mean()
        assert m.loc[zona, 4] / base == pytest.approx(f_vie, abs=0.15)
        assert m.loc[zona, 5] / base == pytest.approx(f_sab, abs=0.15)


def test_horas_de_recepcion(datos):
    pedidos, _ = datos
    hora = pedidos["time"].str[:2].astype(int)
    assert hora.between(8, 18).all()
    viernes = pedidos["date"].dt.day_of_week == 4
    assert (hora[~viernes] < 13).mean() == pytest.approx(0.439, abs=0.03)  # artículo: 43.9 %
    assert (hora[viernes] < 13).mean() == pytest.approx(0.331, abs=0.03)   # artículo: 33.1 %


def test_oraculo_cercano_al_del_articulo():
    celda, dia = generar_caso.calibrar()
    assert celda == pytest.approx(0.406, abs=0.03)  # artículo: 40.6 %
    assert dia == pytest.approx(0.125, abs=0.025)   # artículo: 12.5 %


def test_tasa_real_cubre_cada_zona_dia_franja(datos):
    tasa = pd.read_csv(CARPETA / "tasa_real.csv")
    assert len(tasa) == 621 * 8 * 2 and set(tasa["franja"]) == {"manana", "tarde"}
    assert (tasa["tasa"] > 0).all()


def test_se_regenera_igual_con_la_misma_semilla(tmp_path, datos):
    pedidos, _ = datos
    _, nuevos, _ = generar_caso.generar(tmp_path)
    assert len(nuevos) == len(pedidos)
    assert (tmp_path / "pedidos.csv").read_text() == (CARPETA / "pedidos.csv").read_text()


def test_la_app_puede_cargarlo():
    cargado = cargar_carpeta(CARPETA, cargar_config(RAIZ / "config.yaml"))
    assert len(cargado["clientes"]) == 560 and len(cargado["paradas"]) > 40000
    # Sin operacion_real (E1 se simula); la reserva la produce el modelo de pronóstico (src/integracion.py)
    assert cargado["operacion_real"] is None and cargado["reserva"] is not None


def test_esta_marcado_como_simulado():
    texto = (CARPETA / "LEEME.md").read_text()
    assert "SIMULADO" in texto and "NO SON DATOS DE UNA EMPRESA" in texto
