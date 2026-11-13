"""Pestaña "🚦 Tráfico" de la app: botón "Recalcular con tráfico" (prueba de concepto).

No hay GPS de las camionetas ni app para el chofer: la posición de cada camioneta a la
"hora actual" se simula suponiendo que todo iba según el plan sin tráfico.
"""

from datetime import time, timedelta

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.replan import (CAMBIAR, MANTENER, URGENTE, estados_a_la_hora, horario, puntos_del_dia,
                        replanear, replanear_con_tomtom, resumen_costos, texto_hora, tramos)
from src.tomtom import APIS, ErrorTomTom
from src.traffic import TIPOS_INCIDENTE, Incidente, cargar_incidentes, trafico_simulado

ICONOS = {"accidente": "💥", "obra": "🚧", "cierre": "⛔", "embotellamiento": "🚗",
          "inundación": "🌊", "otro": "⚠️", "carril cerrado": "🚧", "niebla": "🌫️", "lluvia": "🌧️",
          "hielo": "🧊", "viento": "💨", "vehículo descompuesto": "🚙", "condiciones peligrosas": "⚠️"}
MAGNITUDES = {0: "desconocida", 1: "menor", 2: "moderada", 3: "mayor", 4: "cierre"}
DECISIONES = {CAMBIAR: "✅ cambiar", MANTENER: "⏸️ mantener", URGENTE: "🚨 requiere urgente"}
GRIS = "#9e9e9e"
ROJO = "#e00000"
CLAVE_INCIDENTES = "trafico_incidentes"
CLAVE_RESULTADO = "trafico_resultado"
CLAVE_CLIC = "trafico_ultimo_clic"


# --- Incidentes guardados en la sesión (como diccionarios, para poder compararlos) ---

def _incidentes():
    return st.session_state.setdefault(CLAVE_INCIDENTES, [])


def _a_diccionario(inc):
    return {"lat": round(inc.lat, 6), "lon": round(inc.lon, 6), "tipo": inc.tipo, "radio_m": inc.radio_m,
            "factor": inc.factor, "cerrado": inc.cerrado, "descripcion": inc.descripcion}


def _objetos(lista):
    return [Incidente(**d) for d in lista]


def panel_incidentes(config, ruta_ejemplo):
    """Controles para armar la lista de incidentes. Devuelve los valores para el próximo clic."""
    st.markdown("**Incidentes simulados**")
    st.caption("Haz clic en el mapa para poner un incidente con estos valores:")
    simulado = config["trafico"]["simulado"]
    tipo = st.selectbox("Tipo", TIPOS_INCIDENTE, format_func=lambda t: f"{ICONOS[t]} {t}")
    c = st.columns(2)
    radio = c[0].number_input("Radio (m)", 50, 3000, int(simulado["radio_m"]), step=50,
                              help="Un tramo que pasa a menos de esta distancia queda afectado.")
    factor = c[1].number_input("Factor de lentitud", 1.0, 10.0, float(simulado["factor"]), step=0.5,
                               help="2.5 = un tramo afectado tarda 2.5 veces más.")
    cerrado = st.checkbox("Cerrado (el tramo no se puede usar)", value=(tipo == "cierre"))

    with st.expander("Agregar de la lista de ejemplo"):
        if not ruta_ejemplo.exists():
            st.caption(f"No se encontró {ruta_ejemplo.name}.")
        else:
            ejemplos = cargar_incidentes(ruta_ejemplo, simulado["radio_m"], simulado["factor"])
            nombres = st.multiselect("Puntos predefinidos (ubicaciones ilustrativas)",
                                     [e.descripcion for e in ejemplos])
            if st.button("Agregar seleccionados", disabled=not nombres):
                ya = {(d["lat"], d["lon"]) for d in _incidentes()}
                for e in ejemplos:
                    d = _a_diccionario(e)
                    if e.descripcion in nombres and (d["lat"], d["lon"]) not in ya:
                        _incidentes().append(d)

    lista = _incidentes()
    if lista:
        st.dataframe(pd.DataFrame([{
            "": ICONOS.get(d["tipo"], "⚠️"),
            "Incidente": d["descripcion"] or d["tipo"],
            "Radio (m)": d["radio_m"],
            "Factor": "cerrado" if d["cerrado"] else f"×{d['factor']:g}",
        } for d in lista]), hide_index=True)
        c = st.columns(2)
        if c[0].button("Quitar el último"):
            lista.pop()
            st.rerun()
        if c[1].button("Quitar todos"):
            lista.clear()
            st.rerun()
    else:
        st.caption("Todavía no hay incidentes.")
    return {"tipo": tipo, "radio_m": float(radio), "factor": float(factor), "cerrado": cerrado}


# --- Mapa ---

def _tooltip_incidente(inc):
    if inc.fuente == "tomtom":
        retraso = "sin dato" if inc.retraso_seg is None else f"{inc.retraso_seg / 60:.0f} min"
        efecto = f"retraso estimado: {retraso}"
    elif inc.cerrado:
        efecto = f"tramos a menos de {inc.radio_m:.0f} m: cerrados"
    else:
        efecto = f"tramos a menos de {inc.radio_m:.0f} m tardan ×{inc.factor:g}"
    titulo = inc.descripcion or inc.tipo
    return (f"<b>{ICONOS.get(inc.tipo, '⚠️')} {titulo}</b><br>tipo: {inc.tipo}<br>"
            f"magnitud: {inc.magnitud} ({MAGNITUDES.get(inc.magnitud, '?')})<br>{efecto}<br>"
            f"fuente: {'TomTom' if inc.fuente == 'tomtom' else 'simulado'}")


def dibujar_mapa(paradas, config, plan, estados, trafico, resultado, colores):
    """Rutas actuales punteadas, nuevas sólidas, tramos afectados en rojo, visitado en gris."""
    lats, lons, _, _ = puntos_del_dia(paradas, config)

    def punto(n):
        return (lats[n], lons[n])

    mapa = folium.Map(location=punto(0), zoom_start=13, tiles="OpenStreetMap")
    folium.Marker(punto(0), tooltip="Almacén", icon=folium.Icon(color="black", icon="home")).add_to(mapa)

    for k, (ruta, estado) in enumerate(zip(plan["rutas"], estados)):
        color = colores[k % len(colores)]
        nombre = ruta["vehiculo"]
        actual = estado.pendientes
        cambia = bool(resultado) and resultado["rutas"][k]["cambia"]
        nueva = [n + 1 for n in resultado["rutas"][k]["pendientes_adoptadas"]] if resultado else actual

        # Lo ya recorrido (simulado), en gris
        if estado.visitadas:
            folium.PolyLine([punto(n) for n in [0] + estado.visitadas], color=GRIS, weight=3, opacity=0.7,
                            tooltip=f"{nombre}: ya recorrido (simulado)").add_to(mapa)
        # Plan actual (punteado). Si la camioneta no cambia, la línea sólida queda encima.
        if actual and (not resultado or cambia):
            folium.PolyLine([punto(n) for n in [estado.inicio] + actual + [0]], color=color, weight=3,
                            opacity=0.9, dash_array="8 8", tooltip=f"{nombre}: plan actual").add_to(mapa)
        # Plan nuevo o que se mantiene (sólido)
        if resultado and nueva:
            folium.PolyLine([punto(n) for n in [estado.inicio] + nueva + [0]], color=color, weight=4,
                            opacity=0.85,
                            tooltip=f"{nombre}: {'plan nuevo' if cambia else 'sigue igual'}").add_to(mapa)
        # Tramos afectados por el tráfico, en rojo
        pares = set(tramos(estado.inicio, actual)) | set(tramos(estado.inicio, nueva)) if actual or nueva else set()
        for a, b in pares:
            if trafico.afectados[a][b]:
                folium.PolyLine([punto(a), punto(b)], color=ROJO, weight=7, opacity=0.45,
                                tooltip=f"Tramo afectado: {trafico.minutos_base[a][b]:.0f} → "
                                        f"{trafico.minutos[a][b]:.0f} min").add_to(mapa)
        # Paradas
        for n in estado.visitadas:
            p = paradas.iloc[n - 1]
            folium.CircleMarker(punto(n), radius=6, color=GRIS, fill=True, fill_opacity=0.9,
                                tooltip=f"{nombre} · ya entregada (simulado): {p.client_id}").add_to(mapa)
        for orden, n in enumerate(nueva, start=1):
            p = paradas.iloc[n - 1]
            folium.CircleMarker(punto(n), radius=7, color=color, fill=True, fill_opacity=0.9,
                                tooltip=f"{nombre} · pendiente {orden}: {p.client_id} ({p.boxes} cajas)").add_to(mapa)
        # Dónde va la camioneta (simulado)
        if estado.en_ruta and actual:
            folium.Marker(punto(estado.inicio), tooltip=f"{nombre}: posición simulada",
                          icon=folium.DivIcon(html='<div style="font-size:22px">🚚</div>',
                                              icon_size=(24, 24), icon_anchor=(12, 12))).add_to(mapa)

    for i in (resultado or {}).get("no_atendidas", []):
        p = paradas.iloc[i]
        folium.CircleMarker((p.lat, p.lon), radius=9, color="black", weight=3, fill=True, fill_color=ROJO,
                            fill_opacity=1, tooltip=f"REQUIERE URGENTE: {p.client_id} ({p.boxes} cajas)").add_to(mapa)

    for inc in trafico.incidentes:
        if inc.puntos and len(inc.puntos) > 1:  # TomTom: el tramo de calle afectado
            folium.PolyLine(inc.puntos, color=ROJO, weight=5, opacity=0.8,
                            tooltip=folium.Tooltip(_tooltip_incidente(inc))).add_to(mapa)
        if inc.fuente == "simulado":
            folium.Circle((inc.lat, inc.lon), radius=inc.radio_m, color=ROJO, weight=1, fill=True,
                          fill_opacity=0.08, dash_array="4 4").add_to(mapa)
        tamano = 16 + 5 * (inc.magnitud or 0)
        folium.Marker((inc.lat, inc.lon), tooltip=folium.Tooltip(_tooltip_incidente(inc)),
                      icon=folium.DivIcon(html=f'<div style="font-size:{tamano}px">{ICONOS.get(inc.tipo, "⚠️")}</div>',
                                          icon_size=(tamano, tamano),
                                          icon_anchor=(tamano // 2, tamano // 2))).add_to(mapa)

    mapa.fit_bounds([punto(0)] + list(zip(paradas.lat, paradas.lon)))
    return mapa


# --- Tabla y resumen ---

def tabla_decision(resultado):
    """La tabla de replanear() con nombres legibles. Sin pendientes no se muestran tiempos."""
    filas = []
    for _, f in resultado["tabla"].iterrows():
        hay = f["pendientes"] > 0
        filas.append({
            "Camioneta": f["vehiculo"],
            "Paradas pendientes": int(f["pendientes"]),
            "Sin tráfico (min)": f["min_sin_trafico"] if hay else None,
            "Con tráfico (min)": f["min_con_trafico"] if hay else None,
            "Retraso (min)": f["retraso_min"] if hay else None,
            "Retraso (% del manejo)": f["retraso_pct"] / 100 if hay else None,
            "¿Afectada?": "sí" if f["afectada"] else "no",
            "Criterio": f["criterios"],
            "Plan nuevo (min)": f["min_plan_nuevo"],
            "Ahorro (min)": f["ahorro_min"],
            "Decisión": DECISIONES[f["decision"]],
            "Razón": f["nota"],
        })
    return pd.DataFrame(filas)


FORMATO_TABLA = {
    "Sin tráfico (min)": st.column_config.NumberColumn(format="%.0f"),
    "Con tráfico (min)": st.column_config.NumberColumn(format="%.0f"),
    "Retraso (min)": st.column_config.NumberColumn(format="%.0f"),
    "Retraso (% del manejo)": st.column_config.NumberColumn(format="percent"),
    "Plan nuevo (min)": st.column_config.NumberColumn(format="%.0f"),
    "Ahorro (min)": st.column_config.NumberColumn(format="%.0f"),
    "Razón": st.column_config.TextColumn(width="large"),
    "Criterio": st.column_config.TextColumn(width="medium"),
}


def explicacion_criterios(config):
    c = config["trafico"]
    return f"""
1. **Detección.** Una camioneta está afectada si cumple cualquiera de estas:
   - el tráfico le agrega al menos **{c['umbral_retraso_pct']} %** de su manejo pendiente **y** al menos
     **{c['umbral_retraso_min']} min** (las dos a la vez);
   - hay un incidente de magnitud **{c['magnitud_minima']}** o más a menos de **{c['radio_incidente_m']} m**
     de un tramo que le falta;
   - con tráfico ya no termina dentro de la jornada (**{config['tiempos']['jornada_h']} h**).
2. **Reoptimización.** Solo las afectadas, con los tiempos con tráfico. Antes de salir pueden pasarse
   paradas entre ellas; ya en ruta cada una solo reordena las suyas (lleva sus cajas cargadas).
3. **Aceptación.** Se cambia la ruta solo si ahorra al menos **{c['mejora_minima_min']} min** frente a
   seguir igual con el mismo tráfico (o si con el plan actual se pasaría de la jornada).
4. **Restricciones.** Nunca se pasa de la capacidad ni de la jornada. Lo que no quepa se marca
   **requiere urgente**.

Los umbrales están en `config.yaml` (sección `trafico`): son valores iniciales razonables,
no calibrados con datos de la empresa.
"""


# --- Pestaña completa ---

def panel_tomtom(cliente, config):
    """Qué se va a consultar y cuánto se lleva gastado de la cuota gratuita."""
    tt = config["tomtom"]
    st.markdown("**TomTom en vivo**")
    st.caption("Al presionar el botón se consulta: los incidentes del recuadro que cubre las rutas (1 consulta), "
               "una ruta con tráfico por camioneta con paradas pendientes, y Matrix solo para las afectadas "
               f"(tope {tt['max_transacciones_matrix']} transacciones por cálculo). Las consultas repetidas en "
               f"menos de {tt['minutos_cache']} min salen de la caché y no gastan cuota.")
    uso = cliente.uso()
    st.dataframe(pd.DataFrame([{
        "API": nombre, "Hoy": uso[api]["hoy"], "Este mes": uso[api]["mes"],
        "Gratis al mes": tt["limite_mensual"][api],
        "Usado": uso[api]["mes"] / tt["limite_mensual"][api],
    } for api, nombre in APIS.items()]), hide_index=True,
        column_config={"Usado": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    st.caption("TomTom cuenta la cuota gratuita por mes y por API (revisado el 12/11/2026). "
               "Matrix cobra por celdas: una matriz de 10 × 10 cuenta como 50 transacciones.")


def mostrar(paradas, escenario, nombre_escenario, config, clave, colores, ruta_ejemplo,
            modo="simulado", cliente=None):
    """Dibuja la pestaña. escenario: resultado del escenario cuyas rutas se replanean.

    modo: "simulado" (incidentes puestos a mano) o "tomtom" (necesita `cliente`).
    """
    en_vivo = modo == "tomtom"
    plan = {"rutas": escenario["rutas"]}
    lats, lons, _, minutos = puntos_del_dia(paradas, config)

    izquierda, derecha = st.columns([3, 2])
    with derecha:
        salida, fin = horario(config)
        hora = st.slider("Hora actual (simulada)", min_value=time(salida // 60, salida % 60),
                         max_value=time(fin // 60 % 24, fin % 60), value=time(salida // 60, salida % 60),
                         step=timedelta(minutes=15), format="HH:mm",
                         help="Qué paradas ya entregó cada camioneta se calcula con los tiempos sin tráfico "
                              "del plan. A la hora de salida se replanea el día completo.")
        transcurrido = hora.hour * 60 + hora.minute - salida
        if en_vivo:
            lugar_contador = st.container()  # se llena después de consultar, para mostrar el uso al día
            st.caption("El tráfico es el de **este momento**, aunque la posición de las camionetas se "
                       "simule a la hora elegida.")
        else:
            nuevo = panel_incidentes(config, ruta_ejemplo)
        st.divider()
        pulsado = st.button("🚦 Recalcular con tráfico", type="primary", use_container_width=True)

    estados = estados_a_la_hora(plan, minutos, config["tiempos"]["servicio_min"], transcurrido)
    if en_vivo:
        clave_calculo = (clave, transcurrido, "tomtom")
        trafico = trafico_simulado(lats, lons, minutos, [])  # sin datos hasta consultar
    else:
        clave_calculo = (clave, transcurrido, repr(_incidentes()))
        trafico = trafico_simulado(lats, lons, minutos, _objetos(_incidentes()),
                                   config["trafico"]["simulado"]["minutos_cerrado"])

    if pulsado:
        try:
            with st.spinner("Consultando TomTom y reoptimizando las afectadas..." if en_vivo
                            else "Revisando el tráfico y reoptimizando las camionetas afectadas..."):
                if en_vivo:
                    resultado, trafico = replanear_con_tomtom(plan, paradas, config, estados, cliente)
                else:
                    resultado = replanear(plan, paradas, trafico, config, estados=estados)
                costos = resumen_costos(escenario, resultado, paradas, config)
            st.session_state[CLAVE_RESULTADO] = (clave_calculo, resultado, costos, trafico)
        except ErrorTomTom as error:
            st.error(f"**No se pudo usar TomTom:** {error}")
    if en_vivo:
        with lugar_contador:
            panel_tomtom(cliente, config)

    guardado = st.session_state.get(CLAVE_RESULTADO)
    vigente = guardado if guardado and guardado[0] == clave_calculo else None
    resultado = vigente[1] if vigente else None
    if vigente:
        trafico = vigente[3]

    # Aviso fijo (TRAFICO.md)
    if en_vivo:
        consulta = f" (consultado a las {trafico.consultado:%H:%M})" if vigente and trafico.consultado else ""
        st.warning(f"**Prueba de concepto:** posición de camionetas simulada; tráfico de **TomTom en vivo**"
                   f"{consulta}. La cobertura de tráfico de TomTom en Pachuca puede ser limitada.")
    else:
        st.warning("**Prueba de concepto:** posición de camionetas simulada; tráfico de **incidente simulado**.")
    st.caption(f"Se replanean las rutas de **{nombre_escenario}**. No es un sistema en tiempo real: "
               "no hay GPS de las camionetas ni app para el chofer.")

    with izquierda:
        st.caption(f"🕗 {texto_hora(salida + transcurrido)} · línea punteada = plan actual · sólida = plan "
                   "nuevo · rojo = tramo afectado · gris = ya entregado · 🚚 = posición simulada")
        clic = st_folium(dibujar_mapa(paradas, config, plan, estados, trafico, resultado, colores),
                         height=560, use_container_width=True,
                         returned_objects=[] if en_vivo else ["last_clicked"], key=f"mapa-trafico-{clave}-{modo}")
    # Modo simulado: un clic nuevo en el mapa agrega un incidente con los valores del panel
    punto = None if en_vivo else (clic or {}).get("last_clicked")
    if punto and punto != st.session_state.get(CLAVE_CLIC):
        st.session_state[CLAVE_CLIC] = punto
        _incidentes().append(_a_diccionario(Incidente(punto["lat"], punto["lng"], **nuevo,
                                                      descripcion=f"{nuevo['tipo']} (clic en el mapa)")))
        st.rerun()

    if guardado and not vigente:
        st.info("Cambiaste la hora, la fuente o los incidentes: presiona **Recalcular con tráfico** para actualizar.")
    if not vigente:
        if not en_vivo and not _incidentes():
            st.caption("Agrega uno o más incidentes y presiona **Recalcular con tráfico**.")
        with st.expander("¿Cómo decide qué camionetas cambiar?"):
            st.markdown(explicacion_criterios(config))
        return

    _, resultado, costos, trafico = vigente
    for aviso in trafico.avisos:
        st.caption(f"ℹ️ {aviso}")
    antes, despues = costos["antes"], costos["despues"]
    decisiones = resultado["tabla"]["decision"]
    c = st.columns(5)
    c[0].metric("Afectadas", int(resultado["tabla"]["afectada"].sum()))
    c[1].metric("Cambian de ruta", int((decisiones == CAMBIAR).sum()))
    c[2].metric("Requieren urgente", int((decisiones == URGENTE).sum()),
                f"+{costos['urgentes_extra']} unidad(es)" if costos["urgentes_extra"] else None,
                delta_color="inverse")
    c[3].metric("Km del día", f"{despues['km']:.1f}", f"{despues['km'] - antes['km']:+.1f} km",
                delta_color="inverse", help=f"Antes: {antes['km']:.1f} km")
    c[4].metric("Costo del día", f"${despues['costo']:,.0f}", f"{despues['costo'] - antes['costo']:+,.0f} MXN",
                delta_color="inverse", help=f"Antes: ${antes['costo']:,.0f}")
    st.dataframe(tabla_decision(resultado), hide_index=True, column_config=FORMATO_TABLA)
    st.caption("Costo = costo por km × km + costo propia × propias + tarifa reservada × reservadas + "
               "tarifa urgente × urgentes. Las paradas que requieren urgente se rutean aparte con tiempos "
               "sin tráfico. El ahorro de una camioneta puede salir negativo si recibió paradas de otra: "
               "la decisión se toma con el total del grupo.")
    if en_vivo:
        with st.expander(f"Lo que respondió TomTom ({len(trafico.incidentes)} incidentes, "
                         f"{len(trafico.rutas_tomtom)} rutas)"):
            st.caption("Tiempos de TomTom por calles reales. Para comparar con el plan solo se usa cuánto más "
                       "lento va cada tramo (con tráfico ÷ sin tráfico), aplicado a los tiempos del modelo.")
            if trafico.rutas_tomtom:
                st.dataframe(pd.DataFrame(trafico.rutas_tomtom).rename(columns={
                    "vehiculo": "Camioneta", "min_con_trafico": "Con tráfico (min)",
                    "min_sin_trafico": "Sin tráfico (min)", "retraso_min": "Retraso (min)", "km": "Km por calle",
                }).round(1), hide_index=True)
            if trafico.incidentes:
                st.dataframe(pd.DataFrame([{
                    "": ICONOS.get(i.tipo, "⚠️"), "Tipo": i.tipo, "Magnitud": MAGNITUDES.get(i.magnitud, "?"),
                    "Retraso (min)": None if i.retraso_seg is None else round(i.retraso_seg / 60, 1),
                    "Descripción": i.descripcion,
                } for i in trafico.incidentes]), hide_index=True)
            else:
                st.caption("TomTom no reportó incidentes en la zona a esta hora.")
            medidos = int(trafico.medidos.sum() - len(trafico.medidos)) if trafico.medidos is not None else 0
            st.caption(f"Tramos medidos por TomTom: {medidos} de {len(trafico.minutos) ** 2 - len(trafico.minutos)}; "
                       "el resto se estimó con los incidentes.")
    with st.expander("¿Cómo decide qué camionetas cambiar?"):
        st.markdown(explicacion_criterios(config))
