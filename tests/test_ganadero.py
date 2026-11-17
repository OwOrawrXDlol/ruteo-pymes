"""Pruebas del ejemplo ganadero (src/ganadero.py). Casos chicos que se pueden revisar a mano."""

from datetime import date, timedelta

import pandas as pd
import pytest

from conftest import RAIZ
from src import ganadero as g

LUNES = date(2025, 11, 3)   # un lunes


def oc(n, embarque=LUNES, aviso=10, cabezas=80, origen="Villahermosa"):
    """n OCs iguales, para casos a mano."""
    return pd.DataFrame([{
        "oc_id": f"OC-{k:04d}", "proveedor": "VHS-01", "origen": origen, "cabezas": cabezas,
        "fecha_aviso": embarque - timedelta(days=aviso), "fecha_embarque": embarque,
        "dias_aviso": aviso, "flete_mxn": g.FLETE[origen]} for k in range(1, n + 1)])


def dias(d):
    return LUNES + timedelta(days=d)


# --- Parte 1: datos ---

def test_dia_habil_salta_el_fin_de_semana():
    assert g.habil(date(2025, 11, 7)) == date(2025, 11, 7)    # viernes
    assert g.habil(date(2025, 11, 8)) == date(2025, 11, 10)   # sábado → lunes
    assert g.habil(date(2025, 11, 9)) == date(2025, 11, 10)   # domingo → lunes


def test_ordenes_respetan_los_rangos_de_la_empresa():
    o = g.generar_ordenes()
    assert o["cabezas"].between(30, 87).all()
    assert (o["cabezas"] >= 70).mean() > 0.75                  # la mayoría llena la jaula
    assert set(o["origen"]) == set(g.FLETE)
    assert (o["flete_mxn"] == o["origen"].map(g.FLETE)).all()
    assert o["dias_aviso"].between(1, 14).all()
    assert all(d.weekday() < 5 for d in o["fecha_embarque"])  # embarques de lunes a viernes
    por_semana = o.groupby(o["fecha_embarque"].map(lambda d: d.isocalendar()[:2])).agg(
        jaulas=("oc_id", "size"), proveedores=("proveedor", "nunique"))
    assert len(por_semana) == 104
    assert por_semana["jaulas"].between(4, 10).all()
    assert por_semana["proveedores"].between(2, 5).all()
    assert 5.5 < por_semana["jaulas"].mean() < 7


def test_semilla_fija_y_aviso_solo_cambia_el_aviso():
    a, b = g.generar_ordenes(), g.generar_ordenes()
    pd.testing.assert_frame_equal(a, b)
    c = g.generar_ordenes(pct_aviso_largo=0.9)
    mismas = ["oc_id", "proveedor", "origen", "cabezas", "fecha_embarque"]
    pd.testing.assert_frame_equal(a[mismas], c[mismas])
    assert (c["dias_aviso"] >= 7).mean() > (a["dias_aviso"] >= 7).mean()


def test_lotes_chicos_solo_cambia_las_cabezas():
    a, c = g.generar_ordenes(), g.generar_ordenes(pct_parcial=0.4)
    pd.testing.assert_frame_equal(a.drop(columns="cabezas"), c.drop(columns="cabezas"))
    assert (c["cabezas"] < 70).mean() > (a["cabezas"] < 70).mean()


def test_el_csv_es_el_del_generador():
    leido = g.leer_ordenes(RAIZ / "data" / "ejemplo_ganadero" / "ordenes.csv")
    pd.testing.assert_frame_equal(leido, g.generar_ordenes(), check_dtype=False)


# --- Parte 2: escenario A (despacho reactivo) ---

def test_una_oc_sale_a_tiempo_en_jaula_propia():
    r, viajes, sin_usar = g.simular(oc(1), g.apartar_reactivo)
    assert r.loc[0, "fecha_carga"] == LUNES and r.loc[0, "medio"] == "Propia 1"
    assert r.loc[0, "retraso"] == 0 and sin_usar == []
    assert viajes.loc[0, "fin"] == dias(2)                     # 3 días de viaje: lunes a miércoles


def test_cuatro_confirmadas_el_mismo_dia_aparta_una_fletera():
    # 3 jaulas propias y 4 OCs avisadas con 10 días: se aparta 1 fletera una semana antes y nadie espera
    r, _, sin_usar = g.simular(oc(4, aviso=10), g.apartar_reactivo, p_fletera=0)
    assert (r["retraso"] == 0).all()
    assert (r["medio"] == "Fletera apartada").sum() == 1
    assert r["medio"].str.startswith("Propia").sum() == 3
    assert sin_usar == []


def test_cuatro_de_ultimo_momento_sin_fletera_esperan_a_que_regrese_una_jaula():
    # Avisadas 1 día antes: no hubo tiempo de apartar. Sin fletera (p = 0), la 4.ª espera 3 días
    r, _, _ = g.simular(oc(4, aviso=1), g.apartar_reactivo, p_fletera=0)
    assert sorted(r["retraso"]) == [0, 0, 0, 3]
    assert r.loc[r["retraso"] == 3, "fecha_carga"].iloc[0] == dias(3)   # jueves, cuando regresa la jaula
    # Con fletera segura (p = 1) sale el mismo día como fletera de último momento
    r, _, _ = g.simular(oc(4, aviso=1), g.apartar_reactivo, p_fletera=1)
    assert (r["retraso"] == 0).all() and (r["medio"] == "Fletera de último momento").sum() == 1


def test_permiso_vencido_se_re_tramita():
    # 7 OCs de último momento el lunes, sin fletera: 3 salen el lunes, 3 el jueves y la 7.ª ya no alcanza.
    # Su permiso vence el viernes; el sábado se nota, el trámite es el lunes y el nuevo sale el martes.
    r, _, _ = g.simular(oc(7, aviso=1), g.apartar_reactivo, p_fletera=0)
    ultima = r.sort_values("retraso").iloc[-1]
    assert ultima["permisos_vencidos"] == 1
    assert ultima["fecha_carga"] == dias(8)                   # martes de la semana siguiente
    assert ultima["retraso"] == 8
    assert r["permisos_vencidos"].sum() == 1


def test_indicadores_a_mano():
    r, _, sin_usar = g.simular(oc(4, aviso=1, cabezas=80), g.apartar_reactivo, p_fletera=0)
    i = g.indicadores(r, sin_usar, inicio=LUNES, semanas=1)
    # Una OC de 80 cabezas con 3 días de retraso: 80 × 250 kg × 1 % × 3 = 600 kg; × $55 = $33,000
    assert i["Días de retraso"] == 3 and i["OCs con retraso"] == 1
    assert i["Merma (kg)"] == pytest.approx(600)
    assert i["Merma ($)"] == pytest.approx(33_000)
    assert i["Cabezas con retraso"] == 80
    assert i["Fletes contratados"] == 0
    assert i["Costo de fletes ($)"] == 4 * 16_000              # cada viaje cuesta igual, propio o contratado
    # Jaulas propias: 3 viajes del lunes (3 días c/u) + 1 del jueves (jue a sáb) = 12 de 21 días-jaula
    assert i["Utilización de jaulas propias"] == pytest.approx(12 / 21)


def test_escenario_a_calibrado_con_las_fleteras_de_la_empresa():
    # Dato de la empresa: contratan fletera unas 5 veces al mes
    o = g.del_periodo(g.generar_ordenes())
    r, _, sin_usar = g.simular(o, g.apartar_reactivo)
    i = g.indicadores(r, sin_usar)
    assert 3.5 <= i["Fletes contratados"] / 12 <= 8
    assert r["fecha_carga"].notna().all()


# --- Parte 3: pronóstico y escenario B ---

def test_cuantil_poisson_a_mano():
    assert g.cuantil_poisson(0.9, 0) == 0                     # si no se espera nada, no se cubre nada
    # Poisson(1): P(0) = 0.368, P(≤1) = 0.736, P(≤2) = 0.920
    assert g.cuantil_poisson(0.30, 1) == 0
    assert g.cuantil_poisson(0.50, 1) == 1
    assert g.cuantil_poisson(0.90, 1) == 2
    assert g.cuantil_poisson(1.0, 1) < 20                     # con α = 1 no se queda pensando


def test_alfa_a_mano():
    # Una jaula de 80 cabezas: un día de espera = 80 × 250 × 1 % × $55 = $11,000.
    # Con p = 0.5 se espera 1 día en promedio → falta = $11,000; sobra = 50 % × $16,000 = $8,000
    o = oc(1)
    assert g.alfa_apartar(g.BASE, o) == pytest.approx(11_000 / 19_000)
    # Sin pago por fletera sin usar, siempre conviene apartar
    assert g.alfa_apartar({**g.BASE, "costo_sin_usar": 0}, o) == 1
    # Con sobrecosto de último momento (otra PYME), conviene apartar más
    assert g.alfa_apartar({**g.BASE, "sobrecosto": 5_000}, o) > g.alfa_apartar(g.BASE, o)


def test_ideal_aparta_aunque_avisen_tarde():
    # 4 OCs avisadas 1 día antes, sin fletera de último momento: A deja una esperando 3 días;
    # el ideal, que ya las conocía, apartó 1 fletera y nadie espera
    a, _, _ = g.simular(oc(4, aviso=1), g.apartar_reactivo, p_fletera=0)
    ideal, _, _ = g.simular(oc(4, aviso=1), g.apartar_ideal, p_fletera=0)
    assert a["retraso"].sum() == 3 and ideal["retraso"].sum() == 0


def test_pronostico_cubre_las_sorpresas():
    # Se pronostican 20 jaulas esa semana, todas de aviso corto: 4 por día hábil. Con α = 0.9 se cubren
    # 7 (cuantil 0.9 de Poisson(4)); sobran 3 jaulas propias → se apartan 4 fleteras para el lunes.
    # Llegan 2 OCs de último momento sin fletera disponible: salen a tiempo y quedan 2 fleteras sin usar.
    regla = g.apartar_con_pronostico({LUNES: 20}, frac_corto=1.0, alfa=0.9)
    b, _, sin_usar = g.simular(oc(2, aviso=1), regla, p_fletera=0)
    assert b["retraso"].sum() == 0
    assert (b["medio"] == "Fletera apartada").sum() == 2
    assert len(sin_usar) == 2                                  # cubrir de más deja fleteras sin usar


def test_rezagos_no_ven_el_futuro():
    # Al pronosticar la semana W solo se conoce hasta W-2 (se aparta una semana antes)
    t = g.variables_semanales(g.tabla_semanal(g.generar_ordenes()))
    bal = t[t["origen"] == "Balancán"].reset_index(drop=True)
    assert bal.loc[5, "rezago_2"] == bal.loc[3, "y"]
    assert bal.loc[5, "prom_4sem"] == pytest.approx(bal.loc[0:3, "y"].mean())


def test_comparacion_completa():
    o = g.generar_ordenes()
    res, pron = g.comparar(o)
    assert pron["elegido"] in g.MODELOS_SEMANA
    assert 0.2 < pron["frac_corto"] < 0.6
    costo = {e: g.costo_total(res[e]["indicadores"]) for e in res}
    assert costo["I"] <= costo["B"] <= costo["A"] + 1          # B nunca peor que A en este ejemplo
    # Sin juntar lotes, los fletes cuestan lo mismo: cada OC es un viaje del mismo precio
    assert len({res[e]["indicadores"]["Costo de fletes ($)"] for e in "ABI"}) == 1


# --- Parte 5: ruteo (juntar dos lotes chicos) ---

def dos_lotes(origen_a, origen_b, cab_a=40, cab_b=40, dias_entre=0):
    a = oc(1, cabezas=cab_a, origen=origen_a)
    b = oc(1, embarque=LUNES + timedelta(days=dias_entre), cabezas=cab_b, origen=origen_b)
    b["oc_id"], b["proveedor"] = "OC-0002", "OTRO-01"
    return pd.concat([a, b], ignore_index=True)


def test_junta_dos_lotes_chicos_cercanos():
    km = g.cargar_km()
    viajes, pares = g.consolidar(dos_lotes("Balancán", "Tenosique"), km)
    assert len(viajes) == 1 and len(pares) == 1
    v = viajes.iloc[0]
    assert v["cabezas"] == 80 and v["lotes"] == 2
    # Engorda → Balancán (el más lejano) → Tenosique → engorda: el desvío son los km extra
    extra = km[(g.ENGORDA, "Balancán")] + km[("Balancán", "Tenosique")] + km[(g.ENGORDA, "Tenosique")] \
        - 2 * km[(g.ENGORDA, "Balancán")]
    assert v["flete_mxn"] == pytest.approx(45_000 + extra * g.COSTO_KM)
    assert v["origen"] == "Balancán + Tenosique"
    # Se ahorra el flete de Tenosique ($18,000) menos el desvío
    assert pares.loc[0, "ahorro"] == pytest.approx(18_000 - extra * g.COSTO_KM)


def test_no_junta_si_no_caben_estan_lejos_o_esperan_mucho():
    km = g.cargar_km()
    assert len(g.consolidar(dos_lotes("Balancán", "Tenosique", 50, 50), km)[0]) == 2   # 100 > 87 cabezas
    assert len(g.consolidar(dos_lotes("Balancán", "Medellín"), km)[0]) == 2            # 690 km entre ellos
    assert len(g.consolidar(dos_lotes("Balancán", "Tenosique", dias_entre=3), km)[0]) == 2   # 3 días de espera


def test_la_espera_del_primer_lote_cuenta_como_merma():
    km = g.cargar_km()
    viajes, pares = g.consolidar(dos_lotes("Balancán", "Tenosique", dias_entre=1), km)
    assert len(viajes) == 1
    assert viajes.iloc[0]["cab_dias_espera"] == 40                 # 40 cabezas esperan 1 día
    assert viajes.iloc[0]["fecha_embarque"] == dias(1)             # sale cuando está listo el segundo
    r, _, sin_usar = g.simular(viajes, g.apartar_reactivo)
    i = g.indicadores(r, sin_usar, inicio=LUNES, semanas=1)
    assert i["Merma (kg)"] == pytest.approx(40 * 250 * 0.01)       # 100 kg
    assert i["OCs"] == 2 and i["Viajes"] == 1


def test_comparacion_con_ruteo():
    res, _ = g.comparar(g.generar_ordenes())
    assert res["C"]["indicadores"]["Viajes"] < res["A"]["indicadores"]["Viajes"]
    assert res["C"]["indicadores"]["OCs"] == res["A"]["indicadores"]["OCs"]
    assert res["C"]["indicadores"]["Costo de fletes ($)"] < res["A"]["indicadores"]["Costo de fletes ($)"]


# --- Pantalla ---

def test_pestana_rancho_ramada():
    from streamlit.testing.v1 import AppTest
    from conftest import abrir
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=180)
    app.run()
    abrir(app, "🐄 Rancho Ramada")
    assert not app.exception
    pestana = app.tabs[4]
    etiquetas = [m.label for m in pestana.metric]
    assert etiquetas[:5] == ["Costo del año", "Viajes", "Días de retraso", "Merma", "Fletes contratados"]
    assert any("Rancho Ramada" in c.value for c in pestana.caption)
    textos = [c.value for c in pestana.caption] + [m.value for m in pestana.markdown]
    assert not any("simulad" in t.lower() for t in textos)
    # Con los valores base no aparece la línea de parámetros ajustados
    assert not any(c.value.startswith("Parámetros ajustados") for c in pestana.caption)
    # La barra de merma (al final) cambia la merma de arriba y aparece la línea de ajustados
    antes = next(m.value for m in pestana.metric if m.label == "Merma")
    pestana.slider(key="g_merma_dia").set_value(2.0).run()
    pestana = app.tabs[4]
    assert next(m.value for m in pestana.metric if m.label == "Merma") != antes
    assert any("Merma de peso por día de espera 2.0 %" in c.value for c in pestana.caption)
    # Más lotes chicos y más desvío permitido: el ruteo junta más viajes
    viajes = next(m.value for m in pestana.metric if m.label == "Viajes")
    pestana.slider(key="g_pct_parcial").set_value(40).run()
    app.tabs[4].slider(key="g_max_desvio_km").set_value(300).run()
    assert next(m.value for m in app.tabs[4].metric if m.label == "Viajes") != viajes
    # Volver a los valores base
    app.tabs[4].button(key="g_restablecer").click().run()
    assert not any(c.value.startswith("Parámetros ajustados") for c in app.tabs[4].caption)
