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
    assert costo["D"] <= costo["B"] <= costo["A"] + 1          # B nunca peor que A en este ejemplo
    # Sin juntar lotes, los fletes cuestan lo mismo: cada OC es un viaje del mismo precio
    assert len({res[e]["indicadores"]["Costo de fletes ($)"] for e in "ABD"}) == 1


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


# --- Mejoras (docs/Mejoras_programa_Rancho_Ramada.pdf) ---

def abrir_rancho():
    from streamlit.testing.v1 import AppTest
    from conftest import abrir
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=240)
    app.run()
    abrir(app, "🐄 Rancho Ramada")
    assert not app.exception
    return app


def test_textos_con_pesos_no_se_vuelven_formula():
    # Cambio 1: dos $ en un mismo texto de Markdown se leen como LaTeX; deben ir escapados (\$)
    app = abrir_rancho()
    t = app.tabs[4]
    textos = [c.value for c in t.caption] + [m.value for m in t.markdown] + [i.value for i in t.info]
    for texto in textos:
        sin_escapar = texto.replace(r"\$", "")
        assert "$" not in sin_escapar, texto


def test_escenario_d_y_costos_en_la_tabla():
    # Cambios 3 y 6: D · Información perfecta aparece, y la tabla trae los costos y el ahorro contra A
    app = abrir_rancho()
    tabla = next(d.value for d in app.tabs[4].dataframe if "Costo total" in d.value.index)
    assert list(tabla.columns) == ["A · Actual", "B · Pronóstico", "C · Pronóstico + ruteo", "D · Información perfecta"]
    for fila in ("Costo total", "Ahorro contra A", "Costo de fletes", "Costo de merma"):
        assert fila in tabla.index
    assert tabla.loc["Ahorro contra A", "A · Actual"] == "—"


def test_aviso_cuando_b_es_igual_a_a():
    # Cambio 4: con los valores base B = A y se explica; con sobrecosto de urgencia B ya actúa y no hay aviso
    app = abrir_rancho()
    avisos = [i.value for i in app.tabs[4].info]
    assert any("sale igual que hoy" in a for a in avisos)
    app.tabs[4].slider(key="g_sobrecosto").set_value(17_000).run()
    assert not any("sale igual que hoy" in i.value for i in app.tabs[4].info)


def test_merma_con_delta_y_colores():
    # Cambio 7: Merma con su delta en pesos; retraso y merma en rojo cuando suben (delta_color inverse)
    app = abrir_rancho()
    metricas = {m.label: m for m in app.tabs[4].metric}
    assert metricas["Merma"].delta.startswith(("+$", "-$"))
    for nombre in ("Días de retraso", "Merma"):        # los dos suben con los valores base: deben ir en rojo
        assert metricas[nombre].delta.startswith("+")
        assert "GREEN" not in str(metricas[nombre].proto)     # el rojo es el valor 0 del proto y no se imprime
    assert "GREEN" in str(metricas["Viajes"].proto)           # bajan los viajes: verde


def test_tabla_de_modelos_en_porcentaje():
    # Cambio 8: la tabla del pronóstico ya no trae "WAPE por origen" y los valores van × 100
    app = abrir_rancho()
    t = next(d.value for d in app.tabs[4].dataframe if "Modelo" in d.value.columns)
    assert "WAPE por origen" not in t.columns
    assert t["WAPE semanal (prueba)"].between(1, 100).all()


def test_combinaciones_del_mapa():
    # Cambio 9: los viajes con dos lotes se agrupan por combinación de orígenes
    import vista_ganadero as v
    res = v.calcular(tuple(sorted(g.BASE.items())), None)
    juntas = v.combinaciones(res)
    assert sum(juntas.values()) == res["C"]["indicadores"]["Viajes con 2 puntos de carga"]
    assert any(n > 1 for n in juntas.values())          # hay combinaciones repetidas


def test_barra_lateral_del_rancho_y_se_conservan_los_controles():
    # Cambio 10: en la pestaña del rancho la barra lateral muestra al rancho; al salir, los controles vuelven
    from streamlit.testing.v1 import AppTest
    from conftest import abrir
    app = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=240)
    app.run()
    app.sidebar.radio(key="origen_datos").set_value("Mis datos").run()
    abrir(app, "🐄 Rancho Ramada")
    textos = " ".join(c.value for c in app.sidebar.caption)
    assert "3 jaulas propias" in textos and "camionetas" not in textos
    abrir(app, "🗺️ Rutas de un día")
    assert app.sidebar.radio(key="origen_datos").value == "Mis datos"


def test_validar_ordenes_de_la_empresa():
    # Cambio 11: el archivo de la empresa se revisa y se deja como el de ejemplo
    buena = pd.DataFrame({"OC_ID": ["A1", "A2"], "Origen": ["balancan", "Palenque"], "Cabezas": [80, 32],
                          "Fecha_embarque": ["2026-10-26", "28/10/2026"], "Días aviso": [8, 2]})
    o, problemas = g.validar_ordenes(buena)
    assert problemas == []
    assert list(o["origen"]) == ["Balancán", "Palenque/Ocosingo"]
    assert list(o["flete_mxn"]) == [45_000, 18_000]
    assert o.loc[1, "fecha_aviso"] == date(2026, 10, 26)
    mala = pd.DataFrame({"oc_id": ["A1", "A1"], "origen": ["Monterrey", "Balancán"], "cabezas": [120, 80],
                         "fecha_embarque": ["2026-10-26", "2026-10-27"], "dias_aviso": [3, 3]})
    o, problemas = g.validar_ordenes(mala)
    assert o is None
    texto = " ".join(problemas)
    assert "Monterrey" in texto and "cabezas" in texto and "repetidos" in texto
    assert g.validar_ordenes(pd.DataFrame({"oc_id": [1]}))[1][0].startswith("Faltan columnas")


def test_pocas_semanas_usan_el_pronostico_sustituto():
    # Con las 4 semanas de órdenes que tiene hoy el sistema del rancho no se pueden entrenar los modelos
    o = g.del_periodo(g.generar_ordenes(), date(2026, 10, 5), 4)
    inicio, semanas = g.periodo_de(o)
    assert (inicio, semanas) == (date(2026, 10, 19), 2)
    res, pron = g.comparar(o)
    assert pron["sustituto"] and pron["elegido"] == "prom_hist"
    assert res["A"]["indicadores"]["OCs"] == len(g.del_periodo(o, inicio, semanas))
    assert g.periodo_de(g.generar_ordenes()) == (g.INICIO_EVALUACION, 52)


def test_subir_ordenes_de_la_empresa_en_la_pantalla():
    app = abrir_rancho()
    o = g.del_periodo(g.generar_ordenes(), date(2026, 10, 5), 4)
    csv = o.drop(columns=["fecha_aviso", "flete_mxn", "proveedor"]).to_csv(index=False).encode()
    app.tabs[4].file_uploader(key="g_archivo").upload("ordenes.csv", csv, "text/csv")
    app.run()
    assert not app.exception
    t = app.tabs[4]
    assert any(c.value.startswith(f"Con las órdenes de la empresa ({len(o)} órdenes)") for c in t.caption)
    assert next(m.label for m in t.metric) == "Costo del periodo"
    # La merma medida reemplaza a la barra
    t.number_input(key="g_merma_medida").set_value(1.5).run()
    assert any("Merma medida por la empresa 1.50 %" in c.value for c in app.tabs[4].caption)


def test_descargas():
    # Cambio 12: tabla, mapa (PNG y HTML), logo y foto
    import vista_ganadero as v
    res = v.calcular(tuple(sorted(g.BASE.items())), None)
    assert v.mapa_png(res)[:8] == b"\x89PNG\r\n\x1a\n"
    app = abrir_rancho()
    etiquetas = [b.label for b in app.tabs[4].get("download_button")]
    assert etiquetas == ["Escenarios (CSV)", "Mapa de rutas (PNG)", "Mapa interactivo (HTML)", "Logo", "Foto del ganado"]


def test_calendario_en_espanol():
    # Cambio 2: el eje de días del calendario se arma en español (Vega pondría "Mon 03")
    app = abrir_rancho()
    specs = [str(c.proto.spec) for c in app.tabs[4].get("vega_lite_chart")]
    calendarios = [s for s in specs if "recurso" in s]
    assert calendarios and all("'lun'" in s and "day(datum.value)" in s for s in calendarios)
