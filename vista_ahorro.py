"""Pestaña "💰 Ahorro": la pantalla de inicio. Cuenta la historia en pocos segundos (cifras, una gráfica y
un mapa antes/después) y deja todo lo demás en secciones plegables "Ver detalle".

Con el caso de estudio lee lo que ya se calculó y guardó en <carpeta de datos>/resultados/ (pronóstico,
reserva y evaluación de los 103 días de prueba), así que se muestra al instante. Para recalcular:
    .venv/bin/python -m src.integracion data/caso_estudio
    .venv/bin/python -m src.evaluacion data/caso_estudio --recalcular
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import interfaz as ui
from interfaz import NOMBRES, dinero, km, pct
from src.ahorro import calcular_ahorro, dias_de_prueba
from src.evaluacion import analizar, leer
from src.forecast import NOMBRES as NOMBRES_MODELO
from src.perfiles import diferencias
from src.scenarios import ESCENARIOS, brecha_cerrada, escenarios_rango, kpis_pronostico, reserva_del_dia

DIA_BUEN_FIN = date(2025, 11, 14)
CLAVE_PROPIOS = "ahorro_con_mis_datos"
CLAVE_RANGO = "rango"
DIAS_SEMANA = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]

# Cifras publicadas en el artículo (Tablas 6 y 7), solo para la comparación honesta
ARTICULO = [
    ("Km: rutas optimizadas frente a como se hace hoy", "−21.1 %", "km_E2_vs_E1", "{:+.1f} %"),
    ("Costo: propuesta frente a como se hace hoy", "−15.7 %", "costo_E3_vs_E1", "{:+.1f} %"),
    ("Costo: propuesta frente a solo rutas optimizadas", "−12.6 %", "costo_E3_vs_E2", "{:+.1f} %"),
    ("Parte del ahorro máximo que se logra", "61 %", "brecha_cerrada", "{:.0f} %"),
]


# --- Resultados guardados ---

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


def _p(valor):
    return "no aplica (muy pocos días distintos)" if valor is None else f"{valor:.1g}"


def _periodo(detalle):
    fechas = detalle["date"]
    return f"{fechas.min():%d/%m/%Y} a {fechas.max():%d/%m/%Y}"


# --- Calcular el ahorro con los datos o parámetros propios ---
# Las opciones (semanas a evaluar, semanas para elegir el modelo, segundos) están en ⚙️ Opciones avanzadas.

def estimar(datos, ajustes):
    """(días que se evaluarían, minutos aproximados). Lanza ValueError si el historial no alcanza."""
    dias = dias_de_prueba(datos, ajustes["semanas_prueba"], ajustes["semanas_validacion"])
    return dias, max(1, round(len(dias) * (2 * ajustes["segundos_ahorro"] + 1) / 60))


def _clave(clave_datos, config, ajustes):
    """Identifica un cálculo: mismos datos, parámetros y opciones = mismo resultado."""
    opciones = (ajustes["semanas_prueba"], ajustes["semanas_validacion"], ajustes["segundos_ahorro"])
    return (clave_datos, json.dumps(config, sort_keys=True, default=str), opciones)


def _calcular(datos, config, clave, ajustes):
    """Corre el cálculo completo con una barra de avance y lo guarda en la sesión."""
    barra = st.progress(0.0, text="Entrenando el pronóstico...")
    try:
        resultados = calcular_ahorro(datos, config, ajustes["semanas_prueba"], ajustes["semanas_validacion"],
                                     ajustes["segundos_ahorro"],
                                     al_avanzar=lambda i, n: barra.progress(i / n, text=f"Día {i} de {n}"))
        st.session_state[CLAVE_PROPIOS] = (clave, resultados)
    except ValueError as error:
        st.error(str(error))
    barra.empty()


def resultados_propios(datos, config, clave_datos, ajustes, calcular_ya=False, etiqueta_boton=None):
    """Los resultados con los datos y parámetros actuales, si ya se calcularon (o None).

    calcular_ya: calcular en este momento (lo pide el botón del paso ④ de 📂 Mis datos).
    etiqueta_boton: si se da, muestra ese botón con el tiempo estimado (para recalcular aquí mismo).
    """
    clave = _clave(clave_datos, config, ajustes)
    if etiqueta_boton:
        try:
            dias, minutos = estimar(datos, ajustes)
        except ValueError as error:
            st.warning(str(error))
            return None
        c = st.columns([1, 2], vertical_alignment="center")
        calcular_ya = c[0].button(etiqueta_boton, type="primary") or calcular_ya
        c[1].caption(f"Evalúa {len(dias)} días ({dias[0]:%d/%m/%Y} a {dias[-1]:%d/%m/%Y}); tarda unos {minutos} min. "
                     "Opciones en ⚙️ Opciones avanzadas.")
    if calcular_ya:
        _calcular(datos, config, clave, ajustes)
    guardado = st.session_state.get(CLAVE_PROPIOS)
    if guardado and guardado[0] == clave:
        return guardado[1]
    if guardado and etiqueta_boton:
        st.caption("Cambiaste los datos, los parámetros o las opciones: vuelve a presionar el botón.")
    return None


# --- Lo que se ve a la primera ---

def cifras_principales(a, t):
    """Las 4 cifras grandes: ahorro, km, urgentes y parte del ahorro máximo."""
    c = st.columns(4)
    c[0].metric(f"Ahorro en {a['dias']} días", dinero(a["ahorro_E3_vs_E1_mxn"]),
                f"{pct(a['costo_E3_vs_E1'])} del costo", delta_color="inverse",
                help=f"{NOMBRES['E1']}: {dinero(t.loc['E1', 'costo_total'])} → {NOMBRES['E3']}: "
                     f"{dinero(t.loc['E3', 'costo_total'])}")
    c[1].metric("Kilómetros recorridos", km(t.loc["E2", "km_total"]),
                f"{pct(a['km_E2_vs_E1'])} con rutas optimizadas", delta_color="inverse",
                help=f"{NOMBRES['E1']}: {km(t.loc['E1', 'km_total'])}. La propuesta recorre casi lo mismo que "
                     f"solo optimizar rutas ({km(t.loc['E3', 'km_total'])}).")
    urgentes_e1, urgentes_e3 = int(t.loc["E1", "urgentes"]), int(t.loc["E3", "urgentes"])
    c[2].metric("Fletes urgentes", f"{urgentes_e1} → {urgentes_e3}",
                f"{urgentes_e3 - urgentes_e1:+d} al reservar un día antes", delta_color="inverse",
                help="Fletes contratados el mismo día, la tarifa más cara.")
    c[3].metric("Del ahorro máximo posible", "—" if a["brecha_cerrada"] is None else pct(a["brecha_cerrada"], False),
                help=f"Qué parte del ahorro que daría «{NOMBRES['E4']}» logra el pronóstico. 100 % = tan bueno como "
                     "saber el futuro. \"—\" = no hubo nada que reservar (la flota propia alcanzó).")


def mapa_antes_despues(resultados, config, datos, calcular_dia, dibujar_mapa, es_caso):
    """Rutas de un día representativo: como se hace hoy (izquierda) contra la propuesta (derecha)."""
    detalle = resultados["detalle"]
    if es_caso and (detalle["date"].dt.date == DIA_BUEN_FIN).any():
        dia_mapa, titulo = DIA_BUEN_FIN, f"Antes y después: viernes {DIA_BUEN_FIN:%d/%m/%Y} (Buen Fin)"
    else:
        e1_dias = detalle[detalle["escenario"] == "E1"]
        dia_mapa = e1_dias.loc[e1_dias["cajas"].idxmax(), "date"].date()
        titulo = f"Antes y después: el día de más demanda ({dia_mapa:%d/%m/%Y})"
    st.markdown(f"#### {titulo}")
    fila = detalle[detalle["date"].dt.date == dia_mapa].set_index("escenario")
    e1, e3 = fila.loc["E1"], fila.loc["E3"]
    st.caption(f"{int(e1['cajas'])} {config['negocio']['unidad']} · km {e1['km']:,.1f} → {e3['km']:,.1f} "
               f"({pct((e3['km'] / e1['km'] - 1) * 100)}) · urgentes {int(e1['urgentes'])} → {int(e3['urgentes'])} · "
               f"costo del día {dinero(e1['costo'])} → {dinero(e3['costo'])}")

    with st.spinner("Dibujando las rutas de ese día (unos 15 s la primera vez)..."):
        dia, _ = calcular_dia(dia_mapa, config)
    paradas = datos["paradas"][datos["paradas"]["date"] == dia_mapa].reset_index(drop=True)
    for columna, e in zip(st.columns(2), ("E1", "E3")):
        with columna:
            r = dia.get(e)
            ui.titulo_escenario(e, km(r["km"], 1) if r else "")
            if r is None or not r["rutas"]:
                st.caption("No hay rutas para dibujar ese día.")
                continue
            st_folium(dibujar_mapa(paradas, r, config, ui.tonos(e, len(r["rutas"]))), height=420,
                      use_container_width=True, returned_objects=[], key=f"mapa-ahorro-{dia_mapa}-{e}")
    st.caption("Cada tono es una camioneta. Los mapas se recalculan con la búsqueda de rutas, así que los km pueden "
               "diferir unas décimas de la evaluación.")


# --- Ver detalle ---

def detalle_numeros(a, t, u):
    with st.expander("Ver detalle: números completos por escenario"):
        st.markdown(f"Las rutas optimizadas recortan la distancia ({pct(a['km_E2_vs_E1'])} km, "
                    f"{pct(a['costo_E2_vs_E1'])} de costo) y el pronóstico recorta el costo de la flota "
                    f"({pct(a['costo_E3_vs_E2'])} adicional frente a solo rutas), porque cambia fletes urgentes por "
                    "unidades reservadas a tiempo.")
        st.dataframe(pd.DataFrame({
            "Escenario": [ui.etiqueta(e) for e in t.index],
            "Km por día": t["km_dia"], "Km totales": t["km_total"], "Unidades por día": t["unidades_dia"],
            "Utilización": t["utilizacion"], "Urgentes": t["urgentes"].astype(int),
            "Reservadas": t["reservadas"].astype(int), "Costo por día": t["costo_dia"],
            "Costo total": t["costo_total"], f"Costo por {u}": t["costo_caja"],
        }), hide_index=True, column_config={
            "Km por día": st.column_config.NumberColumn(format="%.1f"),
            "Km totales": st.column_config.NumberColumn(format="%.0f"),
            "Unidades por día": st.column_config.NumberColumn(format="%.2f"),
            "Utilización": st.column_config.NumberColumn(format="percent"),
            "Costo por día": st.column_config.NumberColumn(format="dollar"),
            "Costo total": st.column_config.NumberColumn(format="dollar"),
            f"Costo por {u}": st.column_config.NumberColumn(format="$%.2f"),
        })

        st.markdown("**Costo por día, por periodo (MXN)**")
        filas = {"Todo el periodo": {ui.etiqueta(e): t.loc[e, "costo_dia"] for e in t.index}}
        for nombre, s in a["subperiodos"].items():
            filas[nombre] = {ui.etiqueta(e): s[f"costo_dia_{e}"] for e in t.index}
        v = a["viernes"]
        if v:
            filas["viernes"] = {ui.etiqueta("E1"): v["costo_dia_E1"], ui.etiqueta("E3"): v["costo_dia_E3"]}
        tabla = pd.DataFrame(filas).T[[ui.etiqueta(e) for e in t.index]]
        st.bar_chart(tabla, stack=False, y_label="MXN por día", height=300, color=[ui.COLORES[e] for e in t.index])

        if v:
            c = st.columns(3)
            c[0].metric("Viernes: costo por día", dinero(v["costo_dia_E3"]),
                        f"{pct(-v['ahorro_E3_vs_E1'])} frente a hoy", delta_color="inverse",
                        help=f"{v['dias']} viernes; {NOMBRES['E1']}: {dinero(v['costo_dia_E1'])} por día")
            c[1].metric("Viernes: urgentes", f"{v['urgentes_E1']} → {v['urgentes_E3']}")
            c[2].metric("Viernes: utilización de la propuesta", f"{v['utilizacion_E3']:.1%}")


def detalle_gana_siempre(a):
    with st.expander("Ver detalle: ¿el pronóstico gana siempre? (días y prueba estadística)"):
        st.markdown(
            f"- La propuesta costó **menos** que solo optimizar rutas en **{a['dias_E3_menor']}** días, **igual** en "
            f"{a['dias_E3_igual']} y **más** en {a['dias_E3_mayor']}. En esos días el pronóstico se pasó y se pagaron "
            f"reservas que no hacían falta (sobrecosto acumulado {dinero(a['sobrecosto_E3_mayor'])}).\n"
            f"- Las rutas optimizadas recorrieron menos km que como se hace hoy en **{a['dias_km_E2_menor']} de "
            f"{a['dias']}** días.\n"
            f"- Prueba de Wilcoxon (compara las diferencias día con día; un p menor que 0.05 indica que la diferencia "
            f"es consistente y no casualidad): km hoy contra rutas optimizadas, p = {_p(a['wilcoxon_km_E1_E2'])}; "
            f"costo rutas optimizadas contra propuesta, p = {_p(a['wilcoxon_costo_E2_E3'])}.")


def detalle_pronostico(resultados, config):
    with st.expander("Ver detalle: el modelo de pronóstico"):
        m = resultados["metricas"].set_index("modelo")
        elegido = resultados["elegido"]
        p = resultados["parametros"]
        st.markdown("**WAPE** = qué tanto se equivoca un pronóstico, en %: suma de |pronóstico − real| ÷ suma de lo "
                    "real (ec. 1 del artículo). Menos es mejor.")
        texto = (f"Pronostica los pedidos de cada zona y día con un día de anticipación (por franja de mañana y tarde "
                 f"cuando los pedidos traen hora). Modelo elegido por menor error en validación: "
                 f"**{NOMBRES_MODELO[elegido]}**, **{(1 - m.loc[elegido, 'wape'] / m.loc['ingenuo', 'wape']) * 100:.0f} % "
                 f"por debajo** del pronóstico ingenuo; agregado por día, su error fue de "
                 f"**{m.loc[elegido, 'wape_diario']:.1%}**.")
        if "oraculo" in m.index:
            texto += (f" Quedó a **{(m.loc[elegido, 'wape'] - m.loc['oraculo', 'wape']) * 100:.1f} puntos** del error "
                      "irreducible (oráculo).")
        st.markdown(texto)
        wape = (m["wape"] * 100).rename(index=lambda k: NOMBRES_MODELO.get(k, k)).to_frame("WAPE (%)")
        st.bar_chart(wape, horizontal=True, height=260, x_label="WAPE por zona, día y franja (%)")
        st.dataframe(pd.DataFrame({
            "Modelo": [NOMBRES_MODELO.get(k, k) + (" ✔" if k == elegido else "") for k in m.index],
            "MAE (pedidos)": m["mae"], "RMSE (pedidos)": m["rmse"], "WAPE": m["wape"], "Sesgo": m["sesgo"],
            "WAPE diario": m["wape_diario"], "WAPE viernes": m["wape_viernes"], "WAPE validación": m["wape_validacion"],
        }), hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="percent") for c in
            ("WAPE", "Sesgo", "WAPE diario", "WAPE viernes", "WAPE validación")
        } | {c: st.column_config.NumberColumn(format="%.2f") for c in ("MAE (pedidos)", "RMSE (pedidos)")})
        if "oraculo" in m.index:
            st.caption("El oráculo usa la tasa verdadera con que se generaron los pedidos: no es un modelo disponible "
                       "en la práctica; mide el error que nadie puede evitar porque una zona con 5 pedidos promedio "
                       "varía por azar.")
        sesgo = resultados["sesgo"]
        st.caption("Sesgo mensual del modelo elegido (negativo = se quedó corto): " +
                   " · ".join(f"{mes}: {s:+.1%}" for mes, s in zip(sesgo["mes"], sesgo["sesgo"])))
        st.markdown(f"**Del pronóstico a la reserva.** α = (tarifa urgente − tarifa reservada) ÷ tarifa urgente = "
                    f"**{float(p['alfa']):.3f}** (qué tan caro es quedarse corto frente a pasarse); q<sub>α</sub> = "
                    f"**{float(p['q_alfa']):.3f}** (margen que se suma al pronóstico para cubrir su error, calibrado en "
                    f"validación). Unidades a reservar = ⌈{config['negocio']['unidad']} previstas × q<sub>α</sub> ÷ "
                    f"({config['flota']['capacidad_cajas']} × {config['reserva']['u_max']})⌉ − camionetas propias.",
                    unsafe_allow_html=True)


def detalle_patrones(datos):
    tabla_z, manana, manana_vie = patrones_demanda(datos["pedidos"])
    with st.expander("Ver detalle: patrones de la demanda"):
        if manana is not None:
            st.markdown(f"Pedidos por la mañana: **{manana:.1%}** en un día normal y **{manana_vie:.1%}** los viernes.")
        st.dataframe(tabla_z.rename_axis("Zona").reset_index(), hide_index=True)
        st.caption("Pedidos promedio de ese día ÷ promedio de lunes a jueves.")


def detalle_articulo(a, t):
    with st.expander("Ver detalle: comparación con las cifras del artículo"):
        st.markdown("El caso de estudio es una **reconstrucción** de la simulación del artículo (su código original "
                    "no está disponible), así que las cifras salen parecidas, no idénticas:")
        st.dataframe(pd.DataFrame([
            {"Indicador": nombre, "Artículo": publicado, "Esta reconstrucción": formato.format(a[clave])}
            for nombre, publicado, clave, formato in ARTICULO
        ] + [{"Indicador": "Fletes urgentes (hoy → propuesta)", "Artículo": "105 → 31",
              "Esta reconstrucción": f"{int(t.loc['E1', 'urgentes'])} → {int(t.loc['E3', 'urgentes'])}"}]),
            hide_index=True)


def calcular_periodo(datos, clave_datos, config, segundos, calcular_dia):
    """Ver detalle: los cuatro escenarios calculados en vivo para las fechas que se elijan.

    calcular_dia(fecha, config) devuelve (resultados del día, avisos) y recuerda lo ya calculado.
    """
    U, u = config["negocio"]["unidad"], config["negocio"]["unidad_singular"]
    with st.expander("Ver detalle: calcular otro periodo"):
        st.caption("Calcula los cuatro escenarios en vivo para las fechas que elijas, con la reserva de reserva.csv "
                   "o, si no hay, con un pronóstico sustituto (promedio de las semanas previas).")
        fechas = sorted(datos["paradas"]["date"].unique())
        # Por omisión, desde el primer día en que la propuesta tiene con qué calcular la reserva
        con_reserva = [f for f in fechas if reserva_del_dia(datos, f, config)[0] is not None]
        desde = con_reserva[0] if con_reserva else fechas[0]
        rango = st.date_input("Desde / hasta", (desde, fechas[-1]), min_value=fechas[0], max_value=fechas[-1])
        if len(rango) != 2:
            return
        dias = [f for f in fechas if rango[0] <= f <= rango[1]]
        clave = (clave_datos, json.dumps(config, sort_keys=True, default=str), tuple(dias), segundos)
        c = st.columns([1, 2], vertical_alignment="center")
        pulsado = c[0].button("Calcular periodo", type="primary")
        c[1].caption(f"{len(dias)} días con pedidos; cada día nuevo tarda hasta {2 * segundos} s.")
        if pulsado:
            barra = st.progress(0.0, text="Calculando...")
            st.session_state[CLAVE_RANGO] = (clave, escenarios_rango(
                datos, dias, config, segundos,
                al_avanzar=lambda i, n: barra.progress(i / n, text=f"Día {i} de {n}"),
                calcular_dia=lambda d, f, cfg, s: calcular_dia(f, cfg),
            ))
            barra.empty()

        guardado = st.session_state.get(CLAVE_RANGO)
        if not guardado or guardado[0] != clave:
            return
        detalle, resumen, avisos = guardado[1]
        for aviso in avisos:
            if "no disponible" not in aviso:  # los de cada día se ven en la pestaña de rutas
                st.caption(f"ℹ️ {aviso}")
        if resumen.empty:
            st.warning("No hay días con los cuatro escenarios disponibles en este periodo.")
            return

        brecha = brecha_cerrada(resumen)
        st.metric("Del ahorro máximo posible", "—" if brecha is None else pct(brecha, False),
                  help=f"Qué parte del ahorro que daría «{NOMBRES['E4']}» logra el pronóstico.")
        st.caption(f"{int(resumen['dias'].iloc[0])} días comparados con los mismos pedidos.")
        ui.mostrar_kpis(resumen, U, u)
        ui.grafica_costos(resumen["costo"].to_dict(), "Costo total del periodo (MXN)")

        st.markdown("**Costo por día**")
        por_dia = detalle.pivot(index="date", columns="escenario", values="costo")
        por_dia = por_dia[[e for e in ESCENARIOS if e in por_dia.columns]]
        colores = [ui.COLORES[e] for e in por_dia.columns]
        por_dia.columns = [ui.etiqueta(e) for e in por_dia.columns]
        # Formato año-mes-día para que la gráfica los ordene bien
        por_dia.index = [f"{f:%Y-%m-%d} {DIAS_SEMANA[f.weekday()]}" for f in por_dia.index]
        st.bar_chart(por_dia, stack=False, y_label="MXN", height=300, color=colores)

        st.markdown("**Pronóstico y reserva de la propuesta**")
        kp = kpis_pronostico(datos, detalle, config)
        if kp is None:
            st.caption("La propuesta no está disponible en este periodo.")
        else:
            c = st.columns(4)
            c[0].metric(f"WAPE del pronóstico ({U}/día)", "—" if kp["wape"] is None else f"{kp['wape']:.1%}",
                        help="Qué tanto se equivoca el pronóstico, en %.")
            c[1].metric("Sesgo", "—" if kp["sesgo"] is None else f"{kp['sesgo']:+.1%}",
                        help="Positivo = se pasa; negativo = se queda corto.")
            c[2].metric("Urgentes evitados", kp["urgentes_evitados"])
            c[3].metric("Reservas sin usar", kp["reservadas_sin_usar"])
            st.caption(f"Reserva **exacta** {kp['dias_exactos']} días, **de más** {kp['dias_de_mas']}, **de menos** "
                       f"{kp['dias_de_menos']} (de {kp['dias']}). Fuente: {kp['fuente']}. " +
                       (f"El WAPE no se puede medir con reserva.csv porque trae unidades, no {U} previstas."
                        if kp["wape"] is None else
                        "WAPE y sesgo son del pronóstico **sustituto** (promedio de semanas previas), no del modelo "
                        "del equipo."))
        st.download_button("Descargar detalle por día (CSV)", detalle.to_csv(index=False).encode("utf-8"),
                           "escenarios_por_dia.csv")


# --- La pantalla completa ---

def mostrar(resultados, config, datos, calcular_dia, dibujar_mapa, es_caso):
    """Dibuja unos resultados (guardados o recién calculados).

    config: los parámetros con los que se calcularon esos resultados.
    es_caso: True para los resultados guardados del caso de estudio del artículo.
    """
    a = analizar(resultados["detalle"], config)
    t = a["tabla"]
    for aviso in resultados.get("avisos", []):
        if "no disponible" not in aviso:
            st.caption(f"ℹ️ {aviso}")

    cifras_principales(a, t)
    if a["costo_E3_vs_E1"] >= 0:
        st.info("Con estos datos **no hubo ahorro**. Suele pasar cuando cada viaje ya va lleno (no hay nada que juntar) "
                "o cuando la flota propia alcanza casi siempre (no hay urgentes que evitar).")

    st.markdown("#### Costo logístico por día, en promedio")
    ui.grafica_costos({e: t.loc[e, "costo_dia"] for e in t.index}, "MXN por día")
    st.caption(f"«{NOMBRES['E4']}» supone conocer la demanda de mañana sin error: es el piso de costo, no algo "
               "alcanzable.")

    # El mapa tarda: se le guarda su lugar y se dibuja al final, para que lo demás aparezca sin esperar
    lugar_mapa = st.container()

    detalle_numeros(a, t, config["negocio"]["unidad_singular"])
    detalle_gana_siempre(a)
    if "metricas" in resultados:
        detalle_pronostico(resultados, config)
    detalle_patrones(datos)
    if es_caso:
        detalle_articulo(a, t)
    else:
        st.download_button("Descargar el detalle por día y escenario (CSV)",
                           resultados["detalle"].to_csv(index=False).encode("utf-8"), "ahorro_por_dia.csv")
    return lugar_mapa, lambda: mapa_antes_despues(resultados, config, datos, calcular_dia, dibujar_mapa, es_caso)


def pestana(guardados, config_base, config, datos, clave_datos, calcular_dia, dibujar_mapa, ajustes,
            calcular_ya=False):
    """La pestaña completa.

    Caso de estudio: los resultados guardados (calculados con los parámetros del artículo), y un botón para
    recalcular si se cambiaron parámetros. Mis datos: los resultados del cálculo que se lanza desde el
    paso ④ de 📂 Mis datos (calcular_ya=True justo después de presionarlo).
    calcular_dia(fecha, config) devuelve (resultados del día, avisos).
    """
    U, u = config["negocio"]["unidad"], config["negocio"]["unidad_singular"]
    cambiados = [r for r in diferencias(config_base, config) if not r.startswith("negocio.")]
    mapa = None
    if guardados is not None:
        n, periodo = guardados["detalle"]["date"].nunique(), _periodo(guardados["detalle"])
        propios = None
        if cambiados:
            st.warning("Estos resultados son con los **parámetros del artículo**, no con los tuyos (cambiaste " +
                       ", ".join(cambiados[:4]) + (" y otros" if len(cambiados) > 4 else "") + ").")
            propios = resultados_propios(datos, config, clave_datos, ajustes, calcular_ya,
                                         "🔄 Recalcular con mis parámetros")
        if propios is not None:
            st.caption(f"✅ Resultados con **tus parámetros**, evaluados en {propios['detalle']['date'].nunique()} días "
                       f"({_periodo(propios['detalle'])}).")
            mapa = mostrar(propios, config, datos, calcular_dia, dibujar_mapa, es_caso=False)
        else:
            if not cambiados:
                st.caption(f"Resultados guardados de los {n} días de prueba ({periodo}), con los mismos pedidos en los "
                           "cuatro escenarios y los parámetros del artículo.")
            # Los guardados se analizan con los parámetros con que se calcularon (solo la unidad es la tuya)
            mapa = mostrar(guardados, {**config_base, "negocio": config["negocio"]}, datos, calcular_dia,
                           dibujar_mapa, es_caso=True)
    else:
        propios = resultados_propios(datos, config, clave_datos, ajustes, calcular_ya)
        if propios is not None:
            st.caption(f"✅ Resultados con **tus datos y tus parámetros**, evaluados en "
                       f"{propios['detalle']['date'].nunique()} días ({_periodo(propios['detalle'])}) con tus pedidos "
                       "reales en los cuatro escenarios.")
            mapa = mostrar(propios, config, datos, calcular_dia, dibujar_mapa, es_caso=False)
        else:
            st.info("Aún no calculas tu ahorro con tus datos: ve a **📂 Mis datos** y presiona **💰 Calcular mi "
                    "ahorro** (paso ④).")

    calcular_periodo(datos, clave_datos, config, ajustes["segundos"], calcular_dia)
    ui.glosario(U, u)
    if mapa is not None:  # al final, porque tarda
        lugar, dibujar = mapa
        with lugar:
            dibujar()
