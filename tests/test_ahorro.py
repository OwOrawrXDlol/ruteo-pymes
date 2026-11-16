"""Pruebas del paso 3 del plan "cualquier PYME": calcular el ahorro con los datos de la empresa (src/ahorro.py)."""

from pathlib import Path

import pytest

from src.ahorro import calcular_ahorro, dias_de_prueba
from src.evaluacion import analizar
from src.load import cargar_carpeta, cargar_config
from src.perfiles import aplicar_valores
from tests.conftest import app_de_ejemplo

RAIZ = Path(__file__).resolve().parent.parent
CONFIG = cargar_config(RAIZ / "config.yaml")


@pytest.fixture(scope="module")
def caso():
    return cargar_carpeta(RAIZ / "data" / "caso_estudio", CONFIG)


def test_poca_historia_da_un_mensaje_claro():
    with pytest.raises(ValueError, match="al menos 15 semanas"):
        dias_de_prueba(cargar_carpeta(RAIZ / "data" / "ejemplo", CONFIG), 1, 1)


def test_una_semana_del_caso_con_otros_costos(caso):
    # Los parámetros de la "empresa" cambian: tarifa urgente más cara
    config = aplicar_valores(CONFIG, {"costos.tarifa_urgente": 4000.0})
    r = calcular_ahorro(caso, config, semanas_prueba=1, semanas_validacion=4, segundos=1)
    dias = r["detalle"]["date"].dt.date.unique()
    assert len(dias) == 5 and set(r["detalle"]["escenario"]) == {"E1", "E2", "E3", "E4"}
    assert "oraculo" not in set(r["metricas"]["modelo"])  # una empresa real no tiene tasa verdadera
    # α cambia con las tarifas de la empresa: (4,000 − 1,500) / 4,000
    assert float(r["parametros"]["alfa"]) == pytest.approx(0.625)
    a = analizar(r["detalle"], config)
    assert a["dias"] == 5 and a["tabla"].loc["E4", "costo_total"] <= a["tabla"].loc["E3", "costo_total"]


def test_en_la_app_se_calcula_con_un_boton():
    app = app_de_ejemplo(400)
    app.sidebar.radio[0].set_value("Caso de estudio (demostración)").run()
    app.sidebar.number_input(key="Distribuidora urbana del artículo (por omisión)|costos.costo_km").set_value(5.0).run()
    next(n for n in app.number_input if n.label == "Semanas a evaluar").set_value(1).run()
    next(n for n in app.number_input if n.label == "Segundos de búsqueda por ruteo").set_value(1).run()
    next(b for b in app.button if b.label == "🔄 Recalcular con mis parámetros").click().run()
    assert not app.exception
    assert any("Resultados con **tus parámetros**" in c.value for c in app.caption)
    assert any(m.label.startswith("Ahorro en") and m.label != "Ahorro en 103 días" for m in app.tabs[0].metric)
