"""Pestaña "💰 Ahorro" de la app: los resultados del cartel, calculados con el caso de estudio.

Lee lo que ya se calculó y guardó en <carpeta de datos>/resultados/ (pronóstico, reserva y evaluación
de los 103 días de prueba), así que se muestra al instante. Para recalcular:
    .venv/bin/python -m src.integracion data/caso_estudio
    .venv/bin/python -m src.evaluacion data/caso_estudio --recalcular
"""

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.evaluacion import analizar, leer
from src.forecast import NOMBRES as NOMBRES_MODELO

DIA_BUEN_FIN = date(2025, 11, 14)

# Cifras publicadas en el artículo (Tablas 6 y 7), solo para la comparación honesta
ARTICULO = [
    ("Km: modelo de rutas frente a práctica empírica", "−21.1 %", "km_E2_vs_E1", "{:+.1f} %"),
    ("Costo: pronóstico + rutas frente a práctica empírica", "−15.7 %", "costo_E3_vs_E1", "{:+.1f} %"),
    ("Costo: pronóstico + rutas frente a rutas sin pronóstico", "−12.6 %", "costo_E3_vs_E2", "{:+.1f} %"),
    ("Parte del ahorro máximo que se logra", "61 %", "brecha_cerrada", "{:.0f} %"),
]


@st.cache_data(show_spinner=False)
def _leer_archivos(carpeta, _marca):
    """Lee los resultados guardados. _marca (fecha de modificación) hace que se relean si cambian."""
    r = Path(carpeta) / "resultados"
    return {
        "metricas": pd.read_csv(r / "metricas_pronostico.csv"),
        "sesgo": pd.read_csv(r / "sesgo_mensual.csv"),
        "elegido": (r / "modelo_elegido.txt").read_text().strip(),
        "parametros": pd.read_csv(r / "parametros_reserva.csv", index_col=0)["valor"],
        "detalle": leer(carpeta),
    }


def cargar_resultados(carpeta):
    """Los resultados guardados de esa carpeta de datos, o None si no existen."""
    if carpeta is None:
        return None
    r = Path(carpeta) / "resultados"
    necesarios = ["metricas_pronostico.csv", "sesgo_mensual.csv", "modelo_elegido.txt",
                  "parametros_reserva.csv", "escenarios_prueba.csv"]
    if not all((r / n).exists() for n in necesarios):
        return None
    marca = max((r / n).stat().st_mtime for n in necesarios)
    return _leer_archivos(str(carpeta), marca)


def dinero(x):
    return f"${x:,.0f}"


def patrones_demanda(pedidos):
    """Factor viernes y sábado por zona (frente a lunes a jueves) y % de pedidos por la mañana."""
    p = pedidos.assign(dow=pd.to_datetime(pedidos["date"]).dt.day_of_week)
    dias = p.groupby("dow")["date"].nunique().reindex(range(6))  # lunes a sábado (NaN si no hubo)
    por_dia = p.groupby(["zone", "dow"]).size().unstack(fill_value=0).reindex(columns=range(6)) / dias
    base = por_dia[[0, 1, 2, 3]].mean(axis=1)
    tabla = pd.DataFrame({"Viernes": por_dia[4] / base, "Sábado": por_dia[5] / base}).round(2)
    con_hora = p[p.get("time", pd.Series("", index=p.index)).astype(str).str.match(r"^\d{1,2}:\d{2}$")]
    if con_hora.empty:
        return tabla, None, None
    manana = con_hora["time"].str.split(":").str[0].astype(int) < 13
    viernes = con_hora["dow"] == 4
    return tabla, float(manana[~viernes].mean()), float(manana[viernes].mean())


def mostrar(resultados, config, datos, calcular_dia, dibujar_mapa, nombres_escenario):
    """Dibuja la pestaña. calcular_dia(fecha) devuelve (resultados del día, avisos)."""
    if resultados is None:
        st.info("Esta pestaña muestra la evaluación del **caso de estudio simulado** del artículo. "
                "Elige \"Caso de estudio simulado (artículo)\" en la barra lateral.")
        return

    a = analizar(resultados["detalle"], config)
    t = a["tabla"]
    st.warning("Resultados del **caso de estudio simulado** (no de una empresa): 560 clientes, 8 zonas, "
               f"evaluados en los {a['dias']} días de prueba (1-sep a 31-dic-2025) con los mismos pedidos "
               "en los cuatro escenarios.")

    # --- Lo principal: el ahorro ---
    st.subheader("Ahorro del sistema integrado (pronóstico + VRP)")
    c = st.columns(4)
    c[0].metric("Ahorro frente a la práctica actual", dinero(a["ahorro_E3_vs_E1_mxn"]),
                f"{a['costo_E3_vs_E1']:+.1f} % de costo logístico", delta_color="inverse",
                help=f"En {a['dias']} días: E1 {dinero(t.loc['E1', 'costo_total'])} → E3 "
                     f"{dinero(t.loc['E3', 'costo_total'])}")
    c[1].metric("Kilómetros recorridos", f"{t.loc['E2', 'km_total']:,.0f}",
                f"{a['km_E2_vs_E1']:+.1f} % (VRP frente a empírico)", delta_color="inverse")
    c[2].metric("Contrataciones urgentes", f"{int(t.loc['E1', 'urgentes'])} → {int(t.loc['E3', 'urgentes'])}",
                f"{int(t.loc['E3', 'urgentes'] - t.loc['E1', 'urgentes'])} con unidades reservadas un día antes",
                delta_color="inverse")
    c[3].metric("Del ahorro máximo posible", f"{a['brecha_cerrada']:.0f} %",
                help="(costo E2 − costo E3) ÷ (costo E2 − costo E4). 100 % = tan bueno como saber el futuro.")

    st.caption(f"El modelo de rutas recorta la distancia ({a['km_E2_vs_E1']:+.1f} % km, {a['costo_E2_vs_E1']:+.1f} % "
               f"de costo) y el pronóstico recorta el costo de la flota ({a['costo_E3_vs_E2']:+.1f} % adicional "
               "frente al VRP sin pronóstico), porque cambia fletes urgentes por unidades reservadas a tiempo.")

    # --- Costo por escenario ---
    st.markdown("#### Costo logístico diario promedio (MXN)")
    filas = {"Todo el periodo": {nombres_escenario[e]: t.loc[e, "costo_dia"] for e in t.index}}
    for nombre, s in a["subperiodos"].items():
        filas[nombre] = {nombres_escenario[e]: s[f"costo_dia_{e}"] for e in t.index}
    filas["viernes"] = {nombres_escenario["E1"]: a["viernes"]["costo_dia_E1"],
                        nombres_escenario["E3"]: a["viernes"]["costo_dia_E3"]}
    grafica = pd.DataFrame(filas).T
    st.bar_chart(grafica, stack=False, y_label="MXN por día", height=330)
    st.caption("E4 supone saber la demanda del día siguiente sin error: no es alcanzable, es el piso de costo.")

    tabla = pd.DataFrame({
        "Escenario": [nombres_escenario[e] for e in t.index],
        "Km por día": t["km_dia"], "Km totales": t["km_total"], "Unidades por día": t["unidades_dia"],
        "Utilización": t["utilizacion"], "Urgentes": t["urgentes"].astype(int),
        "Reservadas": t["reservadas"].astype(int), "Costo por día": t["costo_dia"],
        "Costo total": t["costo_total"], "Costo por caja": t["costo_caja"],
    })
    st.dataframe(tabla, hide_index=True, column_config={
        "Km por día": st.column_config.NumberColumn(format="%.1f"),
        "Km totales": st.column_config.NumberColumn(format="%.0f"),
        "Unidades por día": st.column_config.NumberColumn(format="%.2f"),
        "Utilización": st.column_config.NumberColumn(format="percent"),
        "Costo por día": st.column_config.NumberColumn(format="dollar"),
        "Costo total": st.column_config.NumberColumn(format="dollar"),
        "Costo por caja": st.column_config.NumberColumn(format="$%.2f"),
    })

    c = st.columns(3)
    v = a["viernes"]
    c[0].metric("Viernes: costo por día", dinero(v["costo_dia_E3"]), f"{-v['ahorro_E3_vs_E1']:+.1f} % frente a E1",
                delta_color="inverse", help=f"{v['dias']} viernes; E1: {dinero(v['costo_dia_E1'])} por día")
    c[1].metric("Viernes: urgentes", f"{v['urgentes_E1']} → {v['urgentes_E3']}")
    c[2].metric("Viernes: utilización E3", f"{v['utilizacion_E3']:.1%}")

    with st.expander("¿El pronóstico gana siempre? (días y prueba estadística)"):
        st.markdown(
            f"- E3 costó **menos** que E2 en **{a['dias_E3_menor']}** días, **igual** en {a['dias_E3_igual']} y "
            f"**más** en {a['dias_E3_mayor']}. En esos días el pronóstico se pasó y se pagaron reservas que no "
            f"hacían falta (sobrecosto acumulado {dinero(a['sobrecosto_E3_mayor'])}).\n"
            f"- El VRP recorrió menos km que la práctica empírica en **{a['dias_km_E2_menor']} de {a['dias']}** días.\n"
            f"- Prueba de rangos con signo de Wilcoxon (diferencias diarias pareadas): km E1 vs E2, "
            f"p = {a['wilcoxon_km_E1_E2']:.1g}; costo E2 vs E3, p = {a['wilcoxon_costo_E2_E3']:.1g}. "
            "Un p menor que 0.05 indica que la diferencia es consistente día con día y no casualidad.")

    # --- Pronóstico ---
    st.divider()
    st.subheader("Modelo predictivo de demanda")
    m = resultados["metricas"].set_index("modelo")
    elegido = resultados["elegido"]
    p = resultados["parametros"]
    st.markdown(
        f"Pronostica los pedidos de cada **zona, día y franja** (mañana 8-13 h, tarde 13-19 h) con un día de "
        f"anticipación. Modelo elegido por menor error en validación (jul-ago 2025): "
        f"**{NOMBRES_MODELO[elegido]}**. En la prueba quedó a "
        f"**{(m.loc[elegido, 'wape'] - m.loc['oraculo', 'wape']) * 100:.1f} puntos** del error irreducible "
        f"y **{(1 - m.loc[elegido, 'wape'] / m.loc['ingenuo', 'wape']) * 100:.0f} % por debajo** del pronóstico "
        f"ingenuo; agregado por día, su error fue de **{m.loc[elegido, 'wape_diario']:.1%}**."
        if "oraculo" in m.index else "")
    st.markdown("#### Error de predicción por modelo (WAPE, %)")
    wape = (m["wape"] * 100).rename(index=lambda k: NOMBRES_MODELO.get(k, k)).to_frame("WAPE (%)")
    st.bar_chart(wape, horizontal=True, height=280, x_label="WAPE por zona, día y franja (%)")
    st.dataframe(pd.DataFrame({
        "Modelo": [NOMBRES_MODELO.get(k, k) + (" ✔" if k == elegido else "") for k in m.index],
        "MAE (pedidos)": m["mae"], "RMSE (pedidos)": m["rmse"], "WAPE": m["wape"], "Sesgo": m["sesgo"],
        "WAPE diario": m["wape_diario"], "WAPE viernes": m["wape_viernes"], "WAPE validación": m["wape_validacion"],
    }), hide_index=True, column_config={
        c: st.column_config.NumberColumn(format="percent") for c in
        ("WAPE", "Sesgo", "WAPE diario", "WAPE viernes", "WAPE validación")
    } | {c: st.column_config.NumberColumn(format="%.2f") for c in ("MAE (pedidos)", "RMSE (pedidos)")})
    st.caption("WAPE = suma de |pronóstico − real| ÷ suma de lo real (ec. 1 del artículo). El oráculo usa la tasa "
               "verdadera del simulador: no es un modelo disponible en la práctica; mide el error que nadie puede "
               "evitar porque una zona con 5 pedidos promedio varía por azar.")
    sesgo = resultados["sesgo"]
    st.caption("Sesgo mensual del modelo elegido (negativo = se quedó corto): " +
               " · ".join(f"{mes}: {s:+.1%}" for mes, s in zip(sesgo["mes"], sesgo["sesgo"])))
    st.markdown(f"**Del pronóstico a la reserva:** α = (tarifa urgente − tarifa reservada) ÷ tarifa urgente = "
                f"**{float(p['alfa']):.3f}**; margen calibrado en validación q<sub>α</sub> = **{float(p['q_alfa']):.3f}**; "
                f"unidades a reservar = ⌈cajas previstas × q<sub>α</sub> ÷ (110 × 0.95)⌉ − camionetas propias.",
                unsafe_allow_html=True)

    tabla_z, manana, manana_vie = patrones_demanda(datos["pedidos"])
    with st.expander("Patrones de la demanda en los datos"):
        if manana is not None:
            st.markdown(f"Pedidos por la mañana: **{manana:.1%}** en un día normal y **{manana_vie:.1%}** los "
                        "viernes (los clientes comerciales piden por la tarde para surtirse antes del fin de semana).")
        st.dataframe(tabla_z.rename_axis("Zona").reset_index(), hide_index=True)
        st.caption("Pedidos promedio de ese día ÷ promedio de lunes a jueves.")

    # --- Viernes del Buen Fin ---
    st.divider()
    st.subheader(f"Rutas del viernes {DIA_BUEN_FIN:%d/%m/%Y} (Buen Fin)")
    fila = resultados["detalle"]
    fila = fila[fila["date"].dt.date == DIA_BUEN_FIN].set_index("escenario")
    if fila.empty:
        st.caption("Ese día no está en la evaluación guardada.")
        return
    e1, e3 = fila.loc["E1"], fila.loc["E3"]
    st.markdown(f"Mismas {int(e1['capacidad'] / config['flota']['capacidad_cajas'])} unidades y {int(e1['cajas'])} "
                f"cajas: **{(e3['km'] / e1['km'] - 1) * 100:+.1f} % de kilómetros** ({e1['km']:.1f} → {e3['km']:.1f}) "
                f"y urgentes {int(e1['urgentes'])} → {int(e3['urgentes'])}; costo del día "
                f"{dinero(e1['costo'])} → {dinero(e3['costo'])}.")
    if st.toggle("Ver los mapas de ese día (tarda unos 15 s la primera vez)", key="mapas_buen_fin"):
        dia, _ = calcular_dia(DIA_BUEN_FIN)
        paradas = datos["paradas"][datos["paradas"]["date"] == DIA_BUEN_FIN].reset_index(drop=True)
        izquierda, derecha = st.columns(2)
        for columna, e in ((izquierda, "E1"), (derecha, "E3")):
            with columna:
                st.markdown(f"**{nombres_escenario[e]}** · {dia[e]['km']:.1f} km")
                st_folium(dibujar_mapa(paradas, dia[e], config), height=420, use_container_width=True,
                          returned_objects=[], key=f"buen-fin-{e}")
        st.caption("Cada color es una unidad de reparto; los mapas se recalculan con la búsqueda de rutas, así que "
                   "los km pueden diferir unas décimas de la evaluación guardada.")

    with st.expander("Comparación con las cifras publicadas en el artículo"):
        st.markdown("El caso de estudio es una **reconstrucción** de la simulación del artículo (su código original "
                    "no está disponible), así que las cifras salen parecidas, no idénticas:")
        st.dataframe(pd.DataFrame([
            {"Indicador": nombre, "Artículo": publicado, "Esta simulación": formato.format(a[clave])}
            for nombre, publicado, clave, formato in ARTICULO
        ] + [{"Indicador": "Contrataciones urgentes (E1 → E3)", "Artículo": "105 → 31",
              "Esta simulación": f"{int(t.loc['E1', 'urgentes'])} → {int(t.loc['E3', 'urgentes'])}"}]),
            hide_index=True)
