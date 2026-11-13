"""App de ruteo de última milla (Streamlit).

Arrancar con:  .venv/bin/streamlit run app.py
"""

import hashlib
from io import BytesIO
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from src.load import ErrorDatos, cargar_config, cargar_fuentes, paradas_del_dia, valores_supuestos
from src.scenarios import (
    ESCENARIOS, brecha_cerrada, escenarios_dia, escenarios_rango, kpis_pronostico, reserva_del_dia,
    resumir, tabla_kpis,
)
import vista_trafico
from src.tomtom import ClienteTomTom, leer_clave

RAIZ = Path(__file__).parent
RUTA_CONFIG = RAIZ / "config.yaml"
CARPETA_REAL = RAIZ / "data"
CARPETA_EJEMPLO = RAIZ / "data" / "ejemplo"
ARCHIVOS = ["pedidos.csv", "clientes.csv", "operacion_real.csv", "reserva.csv"]

NOMBRES = {
    "E1": "E1 · Práctica actual",
    "E2": "E2 · VRP reactivo",
    "E3": "E3 · Pronóstico + VRP",
    "E4": "E4 · Información perfecta",
}
DIAS_SEMANA = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
COLORES = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


# --- Carga (se recuerda mientras los archivos no cambien) ---

@st.cache_data(show_spinner="Leyendo y validando los datos...")
def cargar_datos(contenidos, config):
    """contenidos: {nombre de archivo: bytes o None}."""
    fuente = {n: BytesIO(b) if b is not None else None for n, b in contenidos.items()}
    return cargar_fuentes(fuente["pedidos.csv"], fuente["clientes.csv"], config,
                          fuente["operacion_real.csv"], fuente["reserva.csv"])


@st.cache_data(show_spinner=False)
def calcular_dia(_datos, clave_datos, fecha, config, segundos):
    """escenarios_dia, pero recordando el resultado (clave_datos identifica los archivos)."""
    return escenarios_dia(_datos, fecha, config, segundos)


def leer_carpeta(carpeta):
    return {n: (carpeta / n).read_bytes() if (carpeta / n).exists() else None for n in ARCHIVOS}


def elegir_fuente():
    """Barra lateral: de dónde vienen los datos. Devuelve ({archivo: bytes}, es_ejemplo) o (None, _)."""
    opciones = ["Datos de ejemplo (inventados)", "Subir archivos"]
    if (CARPETA_REAL / "pedidos.csv").exists():
        opciones.insert(0, "Datos de la empresa (carpeta data/)")
    origen = st.sidebar.radio("Datos", opciones)

    if origen.startswith("Datos de ejemplo"):
        return leer_carpeta(CARPETA_EJEMPLO), True
    if origen.startswith("Datos de la empresa"):
        return leer_carpeta(CARPETA_REAL), False

    contenidos = {}
    for nombre in ARCHIVOS:
        opcional = nombre in ("operacion_real.csv", "reserva.csv")
        archivo = st.sidebar.file_uploader(nombre + (" (opcional)" if opcional else ""), type="csv", key=nombre)
        contenidos[nombre] = archivo.getvalue() if archivo else None
    if contenidos["pedidos.csv"] is None or contenidos["clientes.csv"] is None:
        return None, False
    return contenidos, False


def etiqueta_fecha(fecha, paradas):
    dia = paradas[paradas["date"] == fecha]
    return f"{DIAS_SEMANA[fecha.weekday()]} {fecha:%d/%m/%Y} · {len(dia)} paradas · {dia['boxes'].sum()} cajas"


def dinero(x):
    return f"${x:,.0f}"


# --- Mapa ---

def dibujar_mapa(paradas, resultado, config):
    almacen = (config["almacen"]["lat"], config["almacen"]["lon"])
    mapa = folium.Map(location=almacen, zoom_start=13, tiles="OpenStreetMap")
    folium.Marker(almacen, tooltip="Almacén", icon=folium.Icon(color="black", icon="home")).add_to(mapa)

    for k, ruta in enumerate(resultado["rutas"] or []):
        color = COLORES[k % len(COLORES)]
        puntos = [almacen] + [(paradas.lat[i], paradas.lon[i]) for i in ruta["paradas"]] + [almacen]
        folium.PolyLine(puntos, color=color, weight=4, opacity=0.8,
                        tooltip=f"{ruta['vehiculo']}: {ruta['cajas']} cajas, {ruta['km']:.1f} km").add_to(mapa)
        for orden, i in enumerate(ruta["paradas"], start=1):
            p = paradas.iloc[i]
            folium.CircleMarker(
                (p.lat, p.lon), radius=7, color=color, fill=True, fill_opacity=0.9,
                tooltip=f"{ruta['vehiculo']} · parada {orden}: {p.client_id} ({p.boxes} cajas)",
            ).add_to(mapa)

    visitadas = {i for r in (resultado["rutas"] or []) for i in r["paradas"]}
    for i, p in paradas.iterrows():
        if i not in visitadas:
            folium.CircleMarker((p.lat, p.lon), radius=7, color="red", fill=True,
                                tooltip=f"SIN ATENDER: {p.client_id} ({p.boxes} cajas)").add_to(mapa)
    mapa.fit_bounds([almacen] + list(zip(paradas.lat, paradas.lon)))
    return mapa


def tabla_camionetas(paradas, rutas, capacidad):
    filas = []
    for ruta in rutas:
        filas.append({
            "Camioneta": ruta["vehiculo"],
            "Paradas": len(ruta["paradas"]),
            "Cajas": ruta["cajas"],
            "Llenado": ruta["cajas"] / capacidad,
            "Km": ruta["km"],
            "Horas": ruta["horas"],
            "Orden de visita": " → ".join(paradas.client_id[i] for i in ruta["paradas"]),
        })
    return pd.DataFrame(filas)


def tabla_escenarios(resultados):
    filas = []
    base = resultados["E1"]["costo"] if resultados.get("E1") else None
    for nombre in ESCENARIOS:
        r = resultados.get(nombre)
        if r is None:
            filas.append({"Escenario": NOMBRES[nombre], "Fuente": "no disponible"})
            continue
        filas.append({
            "Escenario": NOMBRES[nombre], "Fuente": r["fuente"], "Km": r["km"], "Propias": r["propias"],
            "Reservadas": r["reservadas"], "Urgentes": r["urgentes"], "Utilización": r["utilizacion"],
            "Costo": r["costo"], "vs E1": (r["costo"] / base - 1) if base else None,
        })
    return pd.DataFrame(filas)


def mostrar_kpis(resumen):
    """Tabla de KPI × escenario, con cada fila en su formato (dinero, %, km...)."""
    tabla = tabla_kpis(resumen)
    formatos = {
        "Costo logístico (MXN)": dinero,
        "Costo por caja entregada (MXN)": lambda x: f"${x:,.2f}",
        "Km recorridos": lambda x: f"{x:,.1f}",
        "Utilización de la flota": lambda x: f"{x:.1%}",
        "Costo vs E1": lambda x: f"{x:+.1%}",
        "Costo vs E2": lambda x: f"{x:+.1%}",
        "Km vs E1": lambda x: f"{x:+.1%}",
    }
    texto = pd.DataFrame(index=tabla.index, columns=[NOMBRES[e] for e in tabla.columns])
    for kpi, fila in tabla.iterrows():
        formato = formatos.get(kpi, lambda x: f"{x:,.0f}")
        texto.loc[kpi] = ["—" if pd.isna(v) else formato(v) for v in fila]
    st.dataframe(texto, column_config={"_index": st.column_config.Column("KPI", width="medium")})


GLOSARIO_KPI = """
| KPI | Qué mide |
|---|---|
| **Costo logístico** | km × costo por km + propias × costo propia + reservados × tarifa reservada + urgentes × tarifa urgente |
| **Costo por caja entregada** | Costo logístico ÷ cajas entregadas |
| **Km recorridos** | Distancia total de todas las rutas |
| **Utilización de la flota** | Cajas entregadas ÷ capacidad de las camionetas que salieron |
| **Camionetas propias** | Cuántas veces salió una camioneta propia (suma de todos los días) |
| **Fletes reservados** | Unidades pagadas el día anterior, se usen o no |
| **Fletes urgentes** | Unidades contratadas el mismo día (la tarifa más cara) |
| **Paradas no atendidas** | Paradas imposibles de cubrir (por ejemplo, fuera de la jornada) |
| **Costo / Km vs E1, E2** | Diferencia porcentual frente a la práctica actual o al VRP sin pronóstico |
| **Brecha cerrada** | (costo E2 − costo E3) ÷ (costo E2 − costo E4): qué parte del ahorro posible logra el pronóstico |
| **WAPE** | Error del pronóstico: suma de \\|previsto − real\\| ÷ suma de lo real (ec. 1 del artículo) |
| **Sesgo** | Si el pronóstico tiende a pasarse (+) o a quedarse corto (−) |
| **Días con reserva exacta / de más / de menos** | Reserva de E3 frente a lo que hizo falta (urgentes de E2) |
"""


FORMATO = {
    "Km": st.column_config.NumberColumn(format="%.1f"),
    "Horas": st.column_config.NumberColumn(format="%.2f"),
    "Llenado": st.column_config.NumberColumn(format="percent"),
    "Utilización": st.column_config.NumberColumn(format="percent"),
    "Costo": st.column_config.NumberColumn(format="dollar"),
    "vs E1": st.column_config.NumberColumn(format="percent"),
}


# --- Página ---

st.set_page_config(page_title="Ruteo última milla", page_icon="🚚", layout="wide")
st.title("🚚 Ruteo de última milla para PYMES")

config = cargar_config(RUTA_CONFIG)

contenidos, es_ejemplo = elegir_fuente()
if contenidos is None:
    st.info("Sube al menos **pedidos.csv** y **clientes.csv** en la barra lateral.")
    st.stop()

try:
    datos = cargar_datos(contenidos, config)
except ErrorDatos as e:
    st.error(f"**Hay problemas en {e.archivo}:**\n\n" + "\n".join(f"- {p}" for p in e.problemas))
    st.stop()
clave_datos = hashlib.md5(b"".join(b or b"-" for b in contenidos.values())).hexdigest()

st.sidebar.divider()
segundos = st.sidebar.slider("Segundos de búsqueda por día (OR-Tools)", 1, 10,
                             int(config["solver"]["segundos_por_dia"]),
                             help="Más tiempo = rutas un poco mejores, pero más espera.")

fechas = sorted(datos["paradas"]["date"].unique())
fecha = st.sidebar.selectbox("Fecha", fechas, index=len(fechas) - 1,
                             format_func=lambda f: etiqueta_fecha(f, datos["paradas"]))
escenario = st.sidebar.radio("Escenario del mapa", ESCENARIOS, index=2, format_func=lambda e: NOMBRES[e])

# Fuente de tráfico para la pestaña 🚦 (TomTom solo si hay clave en .env)
st.sidebar.divider()
clave_tomtom = leer_clave(RAIZ / ".env")
fuentes = {"simulado": "Incidente simulado (sin internet)", "tomtom": "TomTom en vivo"}
modo_trafico = st.sidebar.radio("Fuente de tráfico", list(fuentes) if clave_tomtom else ["simulado"],
                                format_func=fuentes.get,
                                help="Solo afecta la pestaña 🚦 Tráfico. El modo simulado no necesita internet.")
if not clave_tomtom:
    st.sidebar.caption("⚠️ TomTom en vivo está deshabilitado: falta `TOMTOM_API_KEY` en `.env` "
                       "(copia `.env.example`).")
cliente_tomtom = (ClienteTomTom(clave_tomtom, RAIZ / ".cache_tomtom", config["tomtom"]["minutos_cache"],
                                config["tomtom"]["modo_viaje"]) if clave_tomtom else None)

# Avisos visibles
supuestos = valores_supuestos(RUTA_CONFIG)
with st.sidebar.expander(f"⚠️ {len(supuestos)} valores de config.yaml son supuestos"):
    st.dataframe(pd.DataFrame(supuestos, columns=["Parámetro", "Valor", "Nota"]), hide_index=True)
if es_ejemplo:
    st.warning("Estás viendo **datos de ejemplo inventados**, no datos de una empresa.")
if supuestos:
    st.caption(f"⚠️ {len(supuestos)} parámetros de costo y operación son supuestos del artículo "
               "(ver barra lateral). Los resultados cambian cuando se reemplacen por datos reales.")
for aviso in datos["avisos"]:
    st.caption(f"ℹ️ {aviso}")

paradas = paradas_del_dia(datos["paradas"], fecha)
with st.spinner(f"Calculando rutas del {fecha:%d/%m/%Y} ({segundos} s por cada búsqueda)..."):
    resultados, avisos_dia = calcular_dia(datos, clave_datos, fecha, config, segundos)
for aviso in avisos_dia:
    st.warning(aviso)

pestana_dia, pestana_trafico, pestana_comparacion, pestana_kpi = st.tabs(
    ["🗺️ Rutas del día", "🚦 Tráfico", "📊 Comparación de escenarios", "📈 KPI"])

with pestana_dia:
    r = resultados[escenario]
    st.subheader(f"{NOMBRES[escenario]} — {etiqueta_fecha(fecha, datos['paradas'])}")
    if r is None:
        st.warning(f"{NOMBRES[escenario]} no está disponible este día (ver avisos arriba).")
    else:
        st.caption(f"Fuente: {r['fuente']}")
        c = st.columns(6)
        c[0].metric("Costo", dinero(r["costo"]))
        c[1].metric("Km", f"{r['km']:.1f}")
        c[2].metric("Propias", r["propias"])
        c[3].metric("Reservadas (pagadas)", r["reservadas"])
        c[4].metric("Urgentes", r["urgentes"])
        c[5].metric("Utilización", f"{r['utilizacion']:.0%}")
        if escenario == "E4":
            st.caption("E4 usa las mismas rutas que E2; solo cambia cómo se pagó la unidad extra.")

        if not r["rutas"]:
            st.info("La operación real solo trae totales del día; no hay rutas para dibujar.")
        else:
            st_folium(dibujar_mapa(paradas, r, config), height=520, use_container_width=True,
                      returned_objects=[], key=f"mapa-{fecha}-{escenario}")
            st.dataframe(tabla_camionetas(paradas, r["rutas"], config["flota"]["capacidad_cajas"]),
                         hide_index=True, column_config=FORMATO)

with pestana_trafico:
    # Se replanean las rutas del escenario del mapa; si no tiene rutas (E1 real), las de E2
    base = escenario if resultados.get(escenario) and resultados[escenario]["rutas"] else "E2"
    st.subheader(f"Recalcular con tráfico — {etiqueta_fecha(fecha, datos['paradas'])}")
    vista_trafico.mostrar(paradas, resultados[base], NOMBRES[base], config, f"{clave_datos}-{fecha}-{base}",
                          COLORES, RAIZ / "data" / "incidentes_ejemplo.csv",
                          modo=modo_trafico, cliente=cliente_tomtom)

with pestana_comparacion:
    st.subheader(f"Día {fecha:%d/%m/%Y}")
    dia = tabla_escenarios(resultados)
    st.dataframe(dia, hide_index=True, column_config=FORMATO)
    disponibles = dia.dropna(subset=["Costo"])
    st.bar_chart(disponibles, x="Escenario", y="Costo", y_label="Costo del día (MXN)", x_label="", height=300)

    st.divider()
    st.subheader("Rango de fechas")
    # Por defecto, desde el primer día en que E3 tiene con qué calcular la reserva
    con_reserva = [f for f in fechas if reserva_del_dia(datos, f, config)[0] is not None]
    desde_def = con_reserva[0] if con_reserva else fechas[0]
    rango = st.date_input("Desde / hasta", (desde_def, fechas[-1]), min_value=fechas[0], max_value=fechas[-1])
    if len(rango) == 2:
        dias_rango = [f for f in fechas if rango[0] <= f <= rango[1]]
        st.caption(f"{len(dias_rango)} días con pedidos. Cada día nuevo tarda hasta {2 * segundos} s.")
        clave_rango = (clave_datos, tuple(dias_rango), segundos)
        if st.button("Calcular rango", type="primary"):
            barra = st.progress(0.0, text="Calculando...")
            st.session_state["rango"] = (clave_rango, escenarios_rango(
                datos, dias_rango, config, segundos,
                al_avanzar=lambda i, n: barra.progress(i / n, text=f"Día {i} de {n}"),
                calcular_dia=lambda d, f, c, s: calcular_dia(d, clave_datos, f, c, s),
            ))
            barra.empty()

        guardado = st.session_state.get("rango")
        if guardado and guardado[0] == clave_rango:
            detalle, resumen, avisos_rango = guardado[1]
            for aviso in avisos_rango:
                if "no disponible" not in aviso:  # los de cada día ya se ven en el detalle
                    st.caption(f"ℹ️ {aviso}")
            if resumen.empty:
                st.warning("No hay días con los cuatro escenarios disponibles en este rango.")
            else:
                tabla = pd.DataFrame({
                    "Escenario": [NOMBRES[e] for e in resumen.index],
                    "Días": resumen["dias"], "Km": resumen["km"], "Propias": resumen["propias"],
                    "Reservadas": resumen["reservadas"], "Urgentes": resumen["urgentes"],
                    "Utilización": resumen["utilizacion"], "Costo": resumen["costo"],
                    "vs E1": resumen["costo_vs_E1_%"] / 100, "vs E2": resumen["costo_vs_E2_%"] / 100,
                })
                st.dataframe(tabla, hide_index=True,
                             column_config={**FORMATO, "vs E2": FORMATO["vs E1"]})
                brecha = brecha_cerrada(resumen)
                if brecha is not None:
                    st.metric("Ahorro posible que logra el pronóstico (brecha cerrada E2 → E4)", f"{brecha:.1f}%",
                              help="(costo E2 − costo E3) / (costo E2 − costo E4). 100% = tan bueno como "
                                   "saber el futuro; 0% = igual que no pronosticar.")
                st.bar_chart(tabla, x="Escenario", y="Costo", y_label="Costo total del rango (MXN)",
                             x_label="", height=300)

                st.markdown("**Costo por día**")
                por_dia = detalle.pivot(index="date", columns="escenario", values="costo")
                # Formato año-mes-día para que la gráfica los ordene bien
                por_dia.index = [f"{f:%Y-%m-%d} {DIAS_SEMANA[f.weekday()]}" for f in por_dia.index]
                st.bar_chart(por_dia, stack=False, y_label="MXN", height=320)
                st.download_button("Descargar detalle por día (CSV)",
                                   detalle.to_csv(index=False).encode("utf-8"), "escenarios_por_dia.csv")

with pestana_kpi:
    st.subheader(f"KPI del día {fecha:%d/%m/%Y}")
    filas_dia = pd.DataFrame([{"date": fecha, **r} for r in resultados.values() if r is not None])
    mostrar_kpis(resumir(filas_dia, config))

    st.divider()
    guardado = st.session_state.get("rango")
    if not guardado or guardado[0][0] != clave_datos:
        st.subheader("KPI del rango")
        st.info("Calcula un rango en la pestaña **📊 Comparación de escenarios** para ver aquí sus KPI.")
    else:
        detalle, resumen, _ = guardado[1]
        dias_calc = guardado[0][1]
        st.subheader(f"KPI del rango {dias_calc[0]:%d/%m/%Y} – {dias_calc[-1]:%d/%m/%Y}")
        if resumen.empty:
            st.warning("No hay días con los cuatro escenarios disponibles en este rango.")
        else:
            st.caption(f"{int(resumen['dias'].iloc[0])} días comparados con los mismos pedidos.")
            c = st.columns(4)
            c[0].metric("Ahorro E3 vs práctica actual", dinero(resumen.loc["E1", "costo"] - resumen.loc["E3", "costo"]),
                        f"{resumen.loc['E3', 'costo_vs_E1_%']:+.1f}%", delta_color="inverse")
            c[1].metric("Km E3 vs práctica actual", f"{resumen.loc['E3', 'km']:,.0f}",
                        f"{(resumen.loc['E3', 'km'] / resumen.loc['E1', 'km'] - 1):+.1%}", delta_color="inverse")
            c[2].metric("Utilización E3", f"{resumen.loc['E3', 'utilizacion']:.1%}",
                        f"{(resumen.loc['E3', 'utilizacion'] - resumen.loc['E1', 'utilizacion']) * 100:+.1f} pts vs E1")
            brecha = brecha_cerrada(resumen)
            c[3].metric("Brecha cerrada", "—" if brecha is None else f"{brecha:.1f}%")
            mostrar_kpis(resumen)

            st.markdown("**Pronóstico y reserva (E3)**")
            kp = kpis_pronostico(datos, detalle, config)
            if kp is None:
                st.info("E3 no está disponible en este rango.")
            else:
                c = st.columns(4)
                c[0].metric("WAPE del pronóstico (cajas/día)", "—" if kp["wape"] is None else f"{kp['wape']:.1%}")
                c[1].metric("Sesgo", "—" if kp["sesgo"] is None else f"{kp['sesgo']:+.1%}")
                c[2].metric("Urgentes evitados", kp["urgentes_evitados"])
                c[3].metric("Reservas sin usar", kp["reservadas_sin_usar"])
                st.write(f"Reserva **exacta** {kp['dias_exactos']} días, **de más** {kp['dias_de_mas']}, "
                         f"**de menos** {kp['dias_de_menos']} (de {kp['dias']}). Fuente: {kp['fuente']}.")
                if kp["wape"] is None:
                    st.caption("El WAPE no se puede medir con reserva.csv porque trae unidades, no cajas previstas.")
                else:
                    st.caption("WAPE y sesgo son del pronóstico **sustituto** (promedio de semanas previas), "
                               "no del modelo del equipo.")

    with st.expander("¿Qué significa cada KPI?"):
        st.markdown(GLOSARIO_KPI)
