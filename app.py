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

from src.load import ErrorDatos, cargar_config, cargar_fuentes, paradas_del_dia
from src.scenarios import ESCENARIOS, escenarios_dia, resumir
import interfaz
import vista_ahorro
import vista_config
import vista_datos
import vista_trafico
from src.tomtom import ClienteTomTom, leer_clave

RAIZ = Path(__file__).parent
RUTA_CONFIG = RAIZ / "config.yaml"
CARPETA_REAL = RAIZ / "data"
CARPETA_CASO = RAIZ / "data" / "caso_estudio"
ARCHIVOS = ["pedidos.csv", "clientes.csv", "operacion_real.csv", "reserva.csv"]

NOMBRES = {e: interfaz.etiqueta(e) for e in ESCENARIOS}  # "Como se hace hoy (E1)"...
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


OPCION_CASO = "Caso de estudio (demostración)"
OPCION_MIOS = "Mis datos"


def elegir_origen():
    """Barra lateral: de dónde vienen los datos ("Caso de estudio (demostración)" o "Mis datos").

    Los datos de ejemplo (data/ejemplo) ya no se ofrecen aquí: solo los usan las pruebas, subiéndolos como "Mis datos".
    """
    opciones = ([OPCION_CASO] if (CARPETA_CASO / "pedidos.csv").exists() else []) + [OPCION_MIOS]
    return st.sidebar.radio("Datos", opciones, key="origen_datos",
                            help="El caso de estudio es una demostración simulada; en «Mis datos» usas tus archivos.")


def etiqueta_fecha(fecha, paradas):
    dia = paradas[paradas["date"] == fecha]
    return f"{DIAS_SEMANA[fecha.weekday()]} {fecha:%d/%m/%Y} · {len(dia)} paradas · {dia['boxes'].sum()} {UN}"


dinero = interfaz.dinero


# --- Mapa ---

def dibujar_mapa(paradas, resultado, config, colores=None):
    """Mapa de un escenario. colores: uno por camioneta (por omisión, la paleta general)."""
    colores = colores or COLORES
    almacen = (config["almacen"]["lat"], config["almacen"]["lon"])
    mapa = folium.Map(location=almacen, zoom_start=13, tiles="OpenStreetMap")
    folium.Marker(almacen, tooltip="Almacén", icon=folium.Icon(color="black", icon="home")).add_to(mapa)

    for k, ruta in enumerate(resultado["rutas"] or []):
        color = colores[k % len(colores)]
        puntos = [almacen] + [(paradas.lat[i], paradas.lon[i]) for i in ruta["paradas"]] + [almacen]
        folium.PolyLine(puntos, color=color, weight=4, opacity=0.8,
                        tooltip=f"{ruta['vehiculo']}: {ruta['cajas']} {UN}, {ruta['km']:.1f} km").add_to(mapa)
        for orden, i in enumerate(ruta["paradas"], start=1):
            p = paradas.iloc[i]
            folium.CircleMarker(
                (p.lat, p.lon), radius=7, color=color, fill=True, fill_opacity=0.9,
                tooltip=f"{ruta['vehiculo']} · parada {orden}: {p.client_id} ({p.boxes} {UN})",
            ).add_to(mapa)

    visitadas = {i for r in (resultado["rutas"] or []) for i in r["paradas"]}
    for i, p in paradas.iterrows():
        if i not in visitadas:
            folium.CircleMarker((p.lat, p.lon), radius=7, color="red", fill=True,
                                tooltip=f"SIN ATENDER: {p.client_id} ({p.boxes} {UN})").add_to(mapa)
    mapa.fit_bounds([almacen] + list(zip(paradas.lat, paradas.lon)))
    return mapa


def tabla_camionetas(paradas, rutas, capacidad):
    filas = []
    for ruta in rutas:
        filas.append({
            "Camioneta": ruta["vehiculo"],
            "Paradas": len(ruta["paradas"]),
            UN.capitalize(): ruta["cajas"],
            "Llenado": ruta["cajas"] / capacidad,
            "Km": ruta["km"],
            "Horas": ruta["horas"],
            "Orden de visita": " → ".join(paradas.client_id[i] for i in ruta["paradas"]),
        })
    return pd.DataFrame(filas)


FORMATO = {
    "Km": st.column_config.NumberColumn(format="%.1f"),
    "Horas": st.column_config.NumberColumn(format="%.2f"),
    "Llenado": st.column_config.NumberColumn(format="percent"),
    "Utilización": st.column_config.NumberColumn(format="percent"),
    "Costo": st.column_config.NumberColumn(format="dollar"),
    "vs E1": st.column_config.NumberColumn(format="percent"),
}


# --- Pestañas ---

def elegir_dia():
    """Selector del día, en la pestaña de rutas. El día elegido se recuerda aparte ("dia") para que la
    pestaña de tráfico use el mismo y para no perderlo al cambiar de pestaña."""
    recordado = st.session_state.get("dia")
    indice = fechas.index(recordado) if recordado in fechas else len(fechas) - 1  # por omisión, el último día
    fecha = st.selectbox("Día", fechas, index=indice, format_func=lambda f: etiqueta_fecha(f, datos["paradas"]))
    st.session_state["dia"] = fecha
    return fecha


def dia_elegido():
    recordado = st.session_state.get("dia")
    return recordado if recordado in fechas else fechas[-1]


def rutas_del_dia(fecha):
    """(paradas, resultados de los 4 escenarios, avisos) de ese día; lo ya calculado se recuerda."""
    with st.spinner(f"Calculando rutas del {fecha:%d/%m/%Y} ({segundos} s por cada búsqueda)..."):
        resultados, avisos = calcular_dia(datos, clave_datos, fecha, config, segundos)
    return paradas_del_dia(datos["paradas"], fecha), resultados, avisos


def mapa_y_tabla(paradas, r, escenario, fecha, alto=460):
    """Mapa de un escenario con sus rutas en tonos de su color (o una nota si no trae rutas)."""
    if r is None:
        st.caption("No disponible este día.")
        return
    if not r["rutas"]:
        st.caption("La operación real solo trae totales del día; no hay rutas para dibujar.")
        return
    st_folium(dibujar_mapa(paradas, r, config, interfaz.tonos(escenario, len(r["rutas"]))), height=alto,
              use_container_width=True, returned_objects=[], key=f"mapa-{fecha}-{escenario}")


def unidades_que_salen(r):
    return r["propias"] + r["reservadas"] + r["urgentes"]


def cifra_antes_despues(columna, titulo, hoy, prop, campo, formato, con_porcentaje=True, ayuda=None):
    """Una métrica 'hoy → propuesta' (con el % de cambio). campo: clave del resultado o función."""
    leer = lambda r: None if r is None else (campo(r) if callable(campo) else r[campo])  # noqa: E731
    antes, despues = leer(hoy), leer(prop)
    if antes is None and despues is None:
        texto = "—"
    elif antes is None or despues is None:
        texto = formato(despues if antes is None else antes)
    else:
        texto = f"{formato(antes)} → {formato(despues)}"
    porcentaje = (interfaz.pct((despues / antes - 1) * 100)
                  if con_porcentaje and antes and despues is not None else None)
    columna.metric(titulo, texto, porcentaje, delta_color="inverse", help=ayuda)


def pagina_rutas():
    st.caption("Compara cómo se reparte hoy contra la propuesta en un día concreto.")
    fecha = elegir_dia()
    paradas, resultados, avisos_dia = rutas_del_dia(fecha)
    if avisos_dia:
        st.warning(" · ".join(avisos_dia))

    # Hoy contra la propuesta; si la propuesta no está disponible ese día, contra solo rutas optimizadas
    hoy = resultados.get("E1")
    otro = "E3" if resultados.get("E3") else "E2"
    prop = resultados.get(otro)
    c = st.columns(4)
    cifra_antes_despues(c[0], "Costo del día", hoy, prop, "costo", dinero)
    cifra_antes_despues(c[1], "Kilómetros", hoy, prop, "km", lambda x: f"{x:,.1f}")
    cifra_antes_despues(c[2], "Fletes urgentes", hoy, prop, "urgentes", str, con_porcentaje=False)
    cifra_antes_despues(c[3], "Unidades que salen", hoy, prop, unidades_que_salen, str, con_porcentaje=False,
                        ayuda="Camionetas propias + fletes reservados + fletes urgentes.")
    st.caption(f"Izquierda: «{interfaz.NOMBRES['E1']}» · derecha: «{interfaz.NOMBRES[otro]}».")

    for columna, e in zip(st.columns(2), ("E1", otro)):
        with columna:
            r = resultados.get(e)
            interfaz.titulo_escenario(e, interfaz.km(r["km"], 1) if r else "")
            mapa_y_tabla(paradas, r, e, fecha)
    st.caption("Cada tono es una camioneta; los puntos rojos son paradas sin atender.")
    if prop and prop["rutas"]:
        st.markdown(f"**Camionetas de «{interfaz.NOMBRES[otro]}»**")
        st.dataframe(tabla_camionetas(paradas, prop["rutas"], config["flota"]["capacidad_cajas"]),
                     hide_index=True, column_config=FORMATO)

    # Antes pestañas 📊 Comparación y 📈 KPI: ahora una sola tabla del día, plegada
    with st.expander("Ver detalle: los 4 escenarios este día"):
        disponibles = {e: r for e, r in resultados.items() if r is not None}
        interfaz.grafica_costos({e: r["costo"] for e, r in disponibles.items()}, "Costo del día (MXN)")
        filas = pd.DataFrame([{"date": fecha, **r} for r in disponibles.values()])
        interfaz.mostrar_kpis(resumir(filas, config), UN, UN1)
        st.caption("Cómo se calculó cada uno: " + " · ".join(
            f"{e}: {resultados[e]['fuente'] if resultados.get(e) else 'no disponible'}" for e in ESCENARIOS) +
            ". Qué significa cada número: ver «¿Qué significa cada número?» en 💰 Ahorro.")

    with st.expander("Ver detalle: ver otro escenario en el mapa"):
        e = st.radio("Escenario", ESCENARIOS, index=1, horizontal=True, key="otro_escenario",
                     format_func=interfaz.etiqueta)
        r = resultados.get(e)
        if r is None:
            st.caption(f"{interfaz.etiqueta(e)} no está disponible este día.")
            return
        c = st.columns(6)
        c[0].metric("Costo", dinero(r["costo"]))
        c[1].metric("Km", f"{r['km']:,.1f}")
        c[2].metric("Propias", r["propias"])
        c[3].metric("Reservadas (pagadas)", r["reservadas"])
        c[4].metric("Urgentes", r["urgentes"])
        c[5].metric("Utilización", f"{r['utilizacion']:.0%}")
        if e == "E4":
            st.caption(f"«{interfaz.NOMBRES['E4']}» usa las mismas rutas que «{interfaz.NOMBRES['E2']}»; solo cambia "
                       "cómo se pagó la unidad extra.")
        mapa_y_tabla(paradas, r, e, fecha, alto=420)
        if r["rutas"]:
            st.dataframe(tabla_camionetas(paradas, r["rutas"], config["flota"]["capacidad_cajas"]),
                         hide_index=True, column_config=FORMATO)


def pagina_trafico():
    fecha = dia_elegido()
    st.caption("¿Qué pasa si hay un accidente a media ruta? Pon un incidente en el mapa y la app decide qué "
               "camionetas conviene cambiar de ruta.")
    lugar_aviso = st.container()  # el aviso único de prueba de concepto (lo escribe vista_trafico)
    # La fuente de tráfico se elige aquí (TomTom solo si hay clave en .env)
    clave_tomtom = leer_clave(RAIZ / ".env")
    fuentes = {"simulado": "Incidente simulado (sin internet)", "tomtom": "TomTom en vivo"}
    modo_trafico = st.radio("Fuente de tráfico", list(fuentes) if clave_tomtom else ["simulado"],
                            format_func=fuentes.get, horizontal=True, key="fuente_trafico",
                            help="El modo simulado no necesita internet.")
    if not clave_tomtom:
        st.caption("TomTom en vivo está deshabilitado: falta `TOMTOM_API_KEY` en `.env` (copia `.env.example`).")
    cliente_tomtom = (ClienteTomTom(clave_tomtom, RAIZ / ".cache_tomtom", config["tomtom"]["minutos_cache"],
                                    config["tomtom"]["modo_viaje"]) if clave_tomtom else None)
    paradas, resultados, _ = rutas_del_dia(fecha)
    # Se replanean las rutas de la propuesta; si no está disponible ese día, las de solo rutas optimizadas
    base = "E3" if resultados.get("E3") and resultados["E3"]["rutas"] else "E2"
    st.subheader(f"Recalcular con tráfico — {etiqueta_fecha(fecha, datos['paradas'])}")
    st.caption(f"Se replanean las rutas de «{interfaz.NOMBRES[base]}» del día elegido en 🗺️ Rutas de un día.")
    vista_trafico.mostrar(paradas, resultados[base], NOMBRES[base], config, f"{clave_datos}-{fecha}-{base}",
                          interfaz.tonos(base, len(resultados[base]["rutas"])),
                          RAIZ / "data" / "incidentes_ejemplo.csv",
                          modo=modo_trafico, cliente=cliente_tomtom, lugar_aviso=lugar_aviso)


# --- Página ---

st.set_page_config(page_title="Ruteo última milla", page_icon="🚚", layout="wide")
st.title("🚚 Ruteo de última milla para PYMES")
st.markdown("**Pronostica los pedidos de mañana, reserva fletes a tiempo y arma rutas más cortas**, y compara cuánto "
            "cuesta frente a como se reparte hoy.")

config_base = cargar_config(RUTA_CONFIG)  # los valores de config.yaml (los del artículo)
origen = elegir_origen()
# Parámetros de la empresa: parten de config.yaml o de un perfil y se pueden cambiar desde la barra lateral
config, perfil, cambios, supuestos, ajustes = vista_config.panel(config_base, RAIZ / "perfiles", RUTA_CONFIG)
segundos = ajustes["segundos"]
UN, UN1 = config["negocio"]["unidad"], config["negocio"]["unidad_singular"]  # p. ej. "cajas" y "caja"
lugar_encabezado = st.container()  # el aviso único; se llena cuando ya se sabe qué datos se usan

# Pestañas "perezosas": solo se calcula la que está abierta (las rutas de un día tardan unos segundos)
PESTANAS = ["💰 Ahorro", "🗺️ Rutas de un día", "📂 Mis datos", "🚦 Tráfico (demo)"]
pestana_ahorro, pestana_dia, pestana_datos, pestana_trafico = st.tabs(PESTANAS, key="pestana", on_change="rerun")

# 📂 Mis datos se dibuja siempre, para que los archivos subidos no se pierdan al cambiar de pestaña
with pestana_datos:
    mios, lugar_revision, lugar_paso4 = vista_datos.pagina((CARPETA_REAL / "pedidos.csv").exists())
carpeta_mia = None
subio_algo = any(st.session_state.get(a) is not None for a in vista_datos.ARCHIVOS)
if mios is None and not subio_algo and (CARPETA_REAL / "pedidos.csv").exists():
    mios, carpeta_mia = leer_carpeta(CARPETA_REAL), CARPETA_REAL  # los de la empresa, en la carpeta data/
datos_mios = None
if mios is not None:
    try:
        datos_mios = cargar_datos(mios, config)
    except ErrorDatos as e:
        with lugar_revision:
            st.error(f"**Al juntar tus archivos encontré problemas en {e.archivo}.** Corrígelos en tu archivo y "
                     "vuelve a subirlo:\n\n" + "\n".join(f"- {p}" for p in e.problemas))
vista_datos.paso_calcular(lugar_paso4, datos_mios, lambda d: vista_ahorro.estimar(d, ajustes))

# Los datos que usan las demás pestañas
if origen == OPCION_CASO:
    contenidos, carpeta_datos, es_caso = leer_carpeta(CARPETA_CASO), CARPETA_CASO, True
    try:
        datos = cargar_datos(contenidos, config)
    except ErrorDatos as e:
        datos = None
        st.error(f"**Con estos parámetros hay problemas en {e.archivo}:**\n\n" + "\n".join(f"- {p}" for p in e.problemas))
else:
    contenidos, carpeta_datos, es_caso, datos = mios, carpeta_mia, False, datos_mios
if datos is None:
    for pestana in (pestana_ahorro, pestana_dia, pestana_trafico):
        if pestana.open:
            with pestana:
                st.info("Todavía no hay datos tuyos: ve a **📂 Mis datos** (4 pasos) o elige **Caso de estudio** en la "
                        "barra lateral.")
    st.stop()
clave_datos = hashlib.md5(b"".join(b or b"-" for b in contenidos.values())).hexdigest()
fechas = sorted(datos["paradas"]["date"].unique())

# Aviso único: una línea gris (demostración, parámetros propios, supuestos) y el detalle plegado
with lugar_encabezado:
    interfaz.encabezado(es_caso, [
        f"Usas **parámetros de tu empresa** (perfil: {perfil}; {len(cambios)} valores distintos a los del artículo)"
        if cambios else None,
        f"{len(supuestos)} costos y parámetros son **supuestos**" if supuestos else None,
    ], supuestos, datos["avisos"])

if pestana_ahorro.open:
    with pestana_ahorro:
        # Si se acaba de presionar "Calcular mi ahorro" en 📂 Mis datos, se calcula aquí, con su barra de avance
        calcular_ya = st.session_state.pop(vista_datos.CALCULAR_AHORA, False) and origen == OPCION_MIOS
        vista_ahorro.pestana(vista_ahorro.cargar_resultados(carpeta_datos), config_base, config, datos, clave_datos,
                             lambda f, cfg: calcular_dia(datos, clave_datos, f, cfg, segundos), dibujar_mapa,
                             ajustes, calcular_ya)
if pestana_dia.open:
    with pestana_dia:
        pagina_rutas()
if pestana_trafico.open:
    with pestana_trafico:
        pagina_trafico()
