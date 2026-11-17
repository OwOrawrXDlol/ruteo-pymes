"""Pestaña 🐄 Rancho Ramada: el modelo aplicado al acopio de ganado (programación de jaulas + ruteo).

Se apoya en src/ganadero.py. No usa los datos de las demás pestañas: tiene sus propias órdenes de
compra (data/ejemplo_ganadero/), sus distancias por carretera (OSRM) y sus parámetros.
"""

import json
from datetime import timedelta
from pathlib import Path

import altair as alt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import interfaz
from src import ganadero as g

RAIZ = Path(__file__).parent
LOGO = RAIZ / "img" / "logo.jpeg"
FOTO = RAIZ / "img" / "ganado_rancho.jpg"
CREDITO = "Logo y fotografía: cortesía de Rancho Ramada."
FUENTE_REEMO = "https://reemo.senasica.gob.mx/ProtocoloemergenciaREEMO.pdf"
COLOR = {"A": "#E07B39", "B": "#4C78A8", "C": "#2E9E5B", "I": "#9E9E9E"}   # mismos tonos que E1-E4
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
dinero = interfaz.dinero


# --- Cálculo (se recuerda mientras no cambien los supuestos) ---

@st.cache_data(show_spinner=False)
def _ordenes():
    return g.leer_ordenes(g.CARPETA_DATOS / "ordenes.csv")


@st.cache_data(show_spinner=False)
def _pronostico():
    return g.pronostico(_ordenes())


@st.cache_data(show_spinner="Programando las jaulas del año...")
def calcular(supuestos):
    """supuestos: tupla ordenada de (nombre, valor) para que se pueda recordar."""
    s = dict(supuestos)
    ordenes, pron = _ordenes(), _pronostico()
    if s["pct_aviso_largo"] != g.BASE["pct_aviso_largo"] or s["pct_parcial"] != g.BASE["pct_parcial"]:
        # Mismas órdenes, solo cambian los días de aviso o las cabezas: el conteo semanal (y el pronóstico)
        # es el mismo; la fracción de órdenes avisadas tarde sí cambia
        ordenes = g.generar_ordenes(pct_aviso_largo=s["pct_aviso_largo"], pct_parcial=s["pct_parcial"])
        historial = ordenes[ordenes["fecha_embarque"] < g.INICIO_EVALUACION]
        pron = {**pron, "frac_corto": float((historial["dias_aviso"] < g.ANTICIPACION).mean())}
    resultado, _ = g.comparar(ordenes, s, pron=pron)
    return resultado


def ahorro(res, e="C"):
    a, x = g.costo_total(res["A"]["indicadores"]), g.costo_total(res[e]["indicadores"])
    return a - x, (a - x) / a


# --- Piezas de la pantalla ---

def encabezado():
    c1, c2 = st.columns([1, 6], vertical_alignment="center")
    c1.image(str(LOGO), width=110)
    c2.markdown("### Rancho Ramada · acopio de ganado")
    c2.caption("Aplicación del modelo (pronóstico + programación de jaulas + ruteo) con los parámetros de "
               "operación de Rancho Ramada: 3 jaulas propias, viajes de 3 días desde la engorda en Cardel, Ver., "
               "permisos de 5 días y fletera apartada con una semana de anticipación.")


# Barras de parámetros: (clave, etiqueta, mínimo, máximo, paso, formato, es porcentaje). Se dibujan al final
# de la página, pero se leen al principio (st.session_state) para calcular con ellas.
GRUPOS = {
    "Espera y merma": [
        ("merma_dia", "Merma de peso por día de espera", 0.5, 2.0, 0.1, "%.1f %%", True),
        ("peso_kg", "Peso por cabeza", 200, 350, 10, "%d kg", False),
        ("precio_kg", "Precio por kg en pie", 45, 70, 1, "$%d", False),
        ("costo_permiso", "Re-tramitar un permiso vencido", 500, 3_000, 100, "$%d", False),
    ],
    "Apartar fleteras": [
        ("sobrecosto", "Sobrecosto de una fletera de último momento", 0, 30_000, 1_000, "$%d", False),
        ("p_fletera", "Fletera disponible de un día para otro", 20, 90, 5, "%d %%", True),
        ("costo_sin_usar", "Pago de una fletera apartada que no se usa", 0, 100, 10, "%d %% del flete", True),
        ("pct_aviso_largo", "Órdenes avisadas con 7 días o más", 30, 90, 5, "%d %%", True),
    ],
    "Ruteo (juntar lotes)": [
        ("pct_parcial", "Órdenes que no llenan la jaula", 5, 40, 5, "%d %%", True),
        ("max_desvio_km", "Km máximos entre los dos puntos de carga", 50, 300, 10, "%d km", False),
        ("max_dias_espera", "Días que un lote puede esperar al otro", 0, 4, 1, "%d días", False),
        ("costo_km", "Costo por km del desvío", 15, 45, 1, "$%d/km", False),
    ],
}
AYUDA = {"sobrecosto": "En Rancho Ramada es $0: el flete cuesta igual apartado o no. Otra empresa con tarifa urgente "
                       "más cara pondría aquí la diferencia."}
CONTROLES = [c for grupo in GRUPOS.values() for c in grupo]


def _a_pantalla(clave, porcentaje, valor):
    """Valor del modelo → valor de la barra (0.01 → 1.0 %; 0.5 → 50 %). La merma lleva decimales; lo demás, enteros."""
    v = valor * 100 if porcentaje else valor
    return round(float(v), 1) if clave == "merma_dia" else int(round(v))


def parametros():
    """Los supuestos con los valores de las barras (o los de la empresa si nadie las ha movido)."""
    s = dict(g.BASE)
    for clave, _, _, _, _, _, porcentaje in CONTROLES:
        valor = st.session_state.get(f"g_{clave}", _a_pantalla(clave, porcentaje, g.BASE[clave]))
        s[clave] = valor / 100 if porcentaje else valor
    return s


def ajustados(s):
    """Línea gris con los parámetros que se movieron (vacía con los valores de la empresa)."""
    cambios = [f"{etiqueta} {formato % _a_pantalla(clave, porcentaje, s[clave])}"
               for clave, etiqueta, _, _, _, formato, porcentaje in CONTROLES
               if abs(s[clave] - g.BASE[clave]) > 1e-9]
    if cambios:
        st.caption("Parámetros ajustados (al final de la página): " + " · ".join(cambios).replace("%%", "%"))


def _restablecer():
    for clave, *_ in CONTROLES:
        st.session_state.pop(f"g_{clave}", None)


def controles():
    """Las barras, al final de la página y plegadas."""
    with st.expander("Ajustar parámetros"):
        for columna, (grupo, lista) in zip(st.columns(3), GRUPOS.items()):
            columna.markdown(f"**{grupo}**")
            for clave, etiqueta, minimo, maximo, paso, formato, porcentaje in lista:
                columna.slider(etiqueta, minimo, maximo, _a_pantalla(clave, porcentaje, g.BASE[clave]), paso,
                               format=formato, key=f"g_{clave}", help=AYUDA.get(clave))
        st.button("Volver a los valores base", on_click=_restablecer, key="g_restablecer")


def cifras(res):
    ia, ic = res["A"]["indicadores"], res["C"]["indicadores"]
    monto, pct = ahorro(res)
    c = st.columns(5)
    c[0].metric("Costo del año", f"{dinero(g.costo_total(ia))} → {dinero(g.costo_total(ic))}",
                interfaz.pct(-pct * 100), delta_color="inverse",
                help="Fletes + merma + permisos re-tramitados + fleteras apartadas sin usar.")
    c[1].metric("Viajes", f"{ia['Viajes']} → {ic['Viajes']}", f"{ic['Viajes'] - ia['Viajes']:+d}", delta_color="inverse",
                help="Jaulas que salen en el año. Bajan cuando el ruteo junta dos lotes chicos en una jaula.")
    c[2].metric("Días de retraso", f"{ia['Días de retraso']} → {ic['Días de retraso']}",
                f"{ic['Días de retraso'] - ia['Días de retraso']:+d}", delta_color="inverse")
    c[3].metric("Merma", f"{dinero(ia['Merma ($)'])} → {dinero(ic['Merma ($)'])}",
                help="Cabezas × días de espera × peso × merma por día × precio por kg.")
    c[4].metric("Fletes contratados", f"{ia['Fletes contratados']} → {ic['Fletes contratados']}",
                f"{ic['Fletes contratados'] - ia['Fletes contratados']:+d}", delta_color="inverse")
    st.caption(f"Izquierda: «{g.ESCENARIOS_G['A']}» · derecha: «{g.ESCENARIOS_G['C']}». "
               f"Ahorro: **{dinero(monto)} ({interfaz.pct(pct * 100, signo=False)})** en 52 semanas.")


def rango_merma(s):
    """El ahorro con la merma en su mínimo y su máximo (lo demás igual)."""
    bajo, alto = (calcular(tuple(sorted({**s, "merma_dia": m}.items()))) for m in (0.005, 0.02))
    (mb, pb), (ma, pa) = ahorro(bajo), ahorro(alto)
    st.caption(f"Con la merma entre 0.5 % y 2 % por día, el ahorro va de **{dinero(mb)}** "
               f"({interfaz.pct(pb * 100, signo=False)}) a **{dinero(ma)}** ({interfaz.pct(pa * 100, signo=False)}).")


FILAS = ["Viajes", "Viajes con 2 puntos de carga", "OCs con retraso", "Días de retraso", "Retraso máximo (días)",
         "Permisos vencidos", "Cabezas con retraso", "Merma (kg)", "Merma ($)", "Fletes contratados", "  apartados",
         "  de último momento", "Fleteras apartadas sin usar", "Costo de fletes ($)", "Sobrecosto de último momento ($)",
         "Pago por fleteras sin usar ($)", "Re-trámite de permisos ($)", "Utilización de jaulas propias"]


def tabla_escenarios(res):
    tabla = {}
    for e, nombre in g.ESCENARIOS_G.items():
        i = res[e]["indicadores"]
        col = {}
        for k in FILAS:
            v = i[k]
            col[k.strip()] = (f"{v:.0%}" if k.startswith("Utilización") else
                              dinero(v) if "$" in k else f"{v:,.0f}")
        col["Costo total"] = dinero(g.costo_total(i))
        tabla[f"{nombre} ({e})"] = col
    st.dataframe(pd.DataFrame(tabla), column_config={"_index": st.column_config.Column("Indicador", width="medium")})
    st.caption("El costo de fletes casi no cambia entre A y B: cada viaje cuesta lo mismo con jaula propia o con "
               "fletera, apartada o no. Lo que cambia es la espera (merma) y, con el ruteo, los viajes que se juntan. "
               "El ideal supone saber desde antes todas las órdenes: es el tope de la programación, no es alcanzable.")


def mapa(res):
    lugares = pd.read_csv(g.CARPETA_DATOS / "lugares.csv")
    trazos = json.loads((g.CARPETA_DATOS / "rutas.json").read_text())
    pos = {f.nombre: (f.lat, f.lon) for f in lugares.itertuples()}

    def trazo(a, b):
        return trazos.get(f"{a}|{b}") or list(reversed(trazos.get(f"{b}|{a}", [pos[a], pos[b]])))

    ordenes = res["A"]["resultado"]
    cuantas = ordenes["origen"].value_counts()
    m = folium.Map(location=(18.4, -93.8), zoom_start=6, tiles="OpenStreetMap")
    for origen, n in cuantas.items():   # viaje sencillo: engorda → origen → engorda
        folium.PolyLine(trazo(g.ENGORDA, origen), color="#888", weight=2 + n / 30, opacity=0.6,
                        tooltip=f"Engorda ↔ {origen}: {n} viajes en el año").add_to(m)
    for f in res["C"]["resultado"][res["C"]["resultado"]["lotes"] > 1].itertuples():
        partes = f.origen.split(" + ")
        if len(partes) == 2:            # dos puntos de carga: engorda → lejano → cercano → engorda
            puntos = trazo(g.ENGORDA, partes[0]) + trazo(partes[0], partes[1]) + trazo(partes[1], g.ENGORDA)
            folium.PolyLine(puntos, color=COLOR["C"], weight=4, opacity=0.9,
                            tooltip=f"{f.oc_id}: {f.origen}, {f.cabezas} cabezas").add_to(m)
    folium.Marker(pos[g.ENGORDA], tooltip=g.ENGORDA, icon=folium.Icon(color="black", icon="home")).add_to(m)
    for origen, n in cuantas.items():
        folium.CircleMarker(pos[origen], radius=5 + n / 10, color="#E07B39", fill=True, fill_opacity=0.85,
                            tooltip=f"{origen} ({g.ESTADO[origen]}): {n} órdenes, flete ${g.FLETE[origen]:,}").add_to(m)
    st_folium(m, height=430, use_container_width=True, returned_objects=[], key="mapa_ganadero")
    st.caption("Gris: viajes de un solo punto de carga (el grosor es cuántos hubo en el año). Verde: viajes del ruteo "
               "que juntan dos lotes chicos. Rutas y km por carretera (OpenStreetMap / OSRM).")


def calendario(res):
    mes = st.select_slider("Mes", options=list(range(12)), value=0, key="g_mes",
                           format_func=lambda k: f"{MESES[(g.INICIO_EVALUACION.month - 1 + k) % 12]} "
                                                 f"{g.INICIO_EVALUACION.year + (g.INICIO_EVALUACION.month - 1 + k) // 12}")
    inicio = (pd.Timestamp(g.INICIO_EVALUACION) + pd.DateOffset(months=mes)).replace(day=1).date()
    fin = (pd.Timestamp(inicio) + pd.DateOffset(months=1)).date()
    for e in ("A", "C"):
        v = res[e]["viajes"].merge(res[e]["resultado"][["oc_id", "origen", "cabezas"]], on="oc_id")
        v = v[(v["fin"] >= inicio) & (v["inicio"] < fin)].copy()
        v["desde"] = pd.to_datetime(v["inicio"])
        v["hasta"] = pd.to_datetime(v["fin"]) + pd.Timedelta(days=1)
        v["estado"] = v["retraso"].map(lambda r: f"{r} día(s) de retraso" if r else "a tiempo")
        st.markdown(f"<span style='color:{COLOR[e]}'>■</span> **{g.ESCENARIOS_G[e]}**", unsafe_allow_html=True)
        if v.empty:
            st.caption("Sin viajes este mes.")
            continue
        orden = sorted(v["recurso"].unique(), key=lambda r: (not r.startswith("Propia"), r))
        barras = alt.Chart(v).mark_bar(cornerRadius=3, height=16).encode(
            x=alt.X("desde:T", title=None, scale=alt.Scale(domain=[str(inicio), str(fin)])),
            x2="hasta:T",
            y=alt.Y("recurso:N", sort=orden, title=None),
            color=alt.Color("origen:N", title="Origen"),
            stroke=alt.condition("datum.retraso > 0", alt.value("#C62828"), alt.value(None)),
            strokeWidth=alt.condition("datum.retraso > 0", alt.value(2.5), alt.value(0)),
            tooltip=["oc_id", "origen", "cabezas", alt.Tooltip("inicio:T", title="Carga"), "estado"])
        st.altair_chart(barras.properties(height=34 * len(orden) + 30), width="stretch")
    st.caption("Cada barra es un viaje de 3 días; borde rojo = salió con retraso.")


def tabla_ordenes(res):
    a = res["A"]["resultado"].set_index("oc_id")
    c = res["C"]["resultado"]
    # En C una fila puede traer dos OCs ("OC-0101 + OC-0102"): se reparte a cada una
    viaje_c = {oc: f for f in c.itertuples() for oc in f.oc_id.split(" + ")}
    filas = []
    for oc, f in a.iterrows():
        x = viaje_c[oc]
        filas.append({"OC": oc, "Origen": f["origen"], "Cabezas": f["cabezas"], "Aviso (días)": f["dias_aviso"],
                      "Embarque": f["fecha_embarque"], "Carga hoy": f["fecha_carga"], "Medio hoy": f["medio"],
                      "Retraso hoy": f["retraso"], "Carga con el modelo": x.fecha_carga, "Medio con el modelo": x.medio,
                      "Retraso con el modelo": (x.fecha_carga - f["fecha_embarque"]).days,
                      "Junto con": " + ".join(o for o in x.oc_id.split(" + ") if o != oc)})
    tabla = pd.DataFrame(filas)
    solo = st.toggle("Solo las que cambian", value=True, key="g_solo")
    if solo:
        tabla = tabla[(tabla["Retraso hoy"] != tabla["Retraso con el modelo"]) | (tabla["Junto con"] != "")]
    st.dataframe(tabla, hide_index=True, height=320)


def reglas_vrp(res, s):
    pares = res["C"]["pares"]
    st.markdown(
        f"El ruteo junta **dos lotes chicos en una sola jaula** ({len(pares)} veces en el año) solo si se cumplen "
        "todas estas reglas:\n"
        f"- caben juntos: hasta **{g.CAPACIDAD_CABEZAS} cabezas** por jaula;\n"
        f"- a lo más **{g.PUNTOS_MAX} puntos de carga** y a no más de **{s['max_desvio_km']:.0f} km** por carretera entre ellos;\n"
        f"- el primer lote espera al segundo a lo más **{s['max_dias_espera']:.0f} días**, dentro de la vigencia de 5 días "
        "del permiso (esa espera se cobra como merma);\n"
        "- los dos se conocen antes de que el primero esté listo para cargar;\n"
        "- **cada lote viaja con su propia guía REEMO** (una UPP de origen y una de destino) **y su certificado "
        "zoosanitario**, con el mismo destino (la engorda).")
    st.caption(f"⚠️ Juntar lotes requiere confirmar con el administrador REEMO del estado que ambos orígenes tienen el "
               f"mismo estatus zoosanitario y que la jaula puede llevar dos guías. Hoy Rancho Ramada carga en un solo "
               f"punto por permiso. Fuente de la guía REEMO: [protocolo REEMO, SENASICA]({FUENTE_REEMO}).")


def pronostico_y_reserva(s):
    p = _pronostico()
    t = p["tabla"].copy()
    nombres = {"ingenuo": "Ingenuo (hace 2 semanas)", "prom4": "Promedio 4 semanas", "ridge": "Regresión Ridge",
               "bosque": "Bosque aleatorio", "gboost": "Gradient boosting"}
    t["Modelo"] = t["modelo"].map(nombres) + t["modelo"].map(lambda m: " ✔" if m == p["elegido"] else "")
    st.dataframe(t[["Modelo", "wape_semana", "wape_validacion", "sesgo_semana", "wape_semana_origen"]].rename(columns={
        "wape_semana": "WAPE semanal (prueba)", "wape_validacion": "WAPE semanal (validación)",
        "sesgo_semana": "Sesgo", "wape_semana_origen": "WAPE por origen"}), hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="percent") for c in
                       ["WAPE semanal (prueba)", "WAPE semanal (validación)", "Sesgo", "WAPE por origen"]})
    alfa = g.alfa_apartar(s, _ordenes())
    st.markdown(
        f"Se pronostican las jaulas de cada semana con los mismos modelos de la pestaña 💰 Ahorro; el elegido (✔) es el "
        f"de menor error en validación. Con él se apartan fleteras para las órdenes que todavía no llegan "
        f"({p['frac_corto']:.0%} se avisan con menos de 7 días).\n\n"
        f"**α = {alfa:.2f}** (ec. 2 del artículo): costo de que falte una jaula ÷ (que falte + que sobre). En una PYME "
        "de reparto, que falte cuesta la tarifa urgente; **en Rancho Ramada el flete cuesta igual apartado o no**, así "
        "que lo que cuesta es la espera: merma de peso por cada día sin jaula. Que sobre cuesta lo que se paga por "
        "una fletera apartada que no se usa.")


def datos_y_supuestos():
    filas = [
        ("Jaulas propias", "3", "Dato de la empresa"),
        ("Viaje redondo", "3 días", "Dato de la empresa"),
        ("Vigencia del permiso", "5 días; sin dictámenes en fin de semana", "Dato de la empresa"),
        ("Fletera", "Se aparta con una semana; ~5 al mes; mismo precio apartada o no, propia o contratada",
         "Dato de la empresa"),
        ("Flete por viaje", "Balancán $45,000; Pichucalco $40,000; otras zonas $16,000-18,000", "Dato de la empresa"),
        ("Cabezas por jaula", "70-87 (a veces menos)", "Dato de la empresa"),
        ("Engorda", "Carretera Cardel-Veracruz km 226, José Cardel, Ver.", "Dato de la empresa"),
        ("Órdenes de compra", f"{g.RITMO:.0f} jaulas por semana en promedio, 3-5 proveedores",
         "Supuesto (calibrado con ~5 fleteras al mes)"),
        ("Km y horas por carretera", "OSRM (OpenStreetMap)", "Calculado"),
    ] + [(g.NOMBRE_SUPUESTO[k], f"{v[0]:g} (de {v[1]:g} a {v[2]:g})", "Supuesto") for k, v in g.SUPUESTOS.items()] + [
        ("Costo por km del desvío", f"${g.COSTO_KM}/km", "Supuesto (flete de Balancán ÷ km de ida y vuelta)"),
    ]
    st.dataframe(pd.DataFrame(filas, columns=["Parámetro", "Valor", "Origen"]), hide_index=True)


# --- La pestaña ---

def pagina():
    encabezado()
    st.image(str(FOTO), width="stretch")
    st.caption(CREDITO)
    s = parametros()
    res = calcular(tuple(sorted(s.items())))
    cifras(res)
    ajustados(s)
    rango_merma(s)

    st.markdown("#### Escenarios")
    tabla_escenarios(res)

    st.markdown("#### Rutas desde la engorda")
    mapa(res)
    with st.expander("Reglas para juntar lotes en una jaula (ruteo)"):
        reglas_vrp(res, s)

    st.markdown("#### Calendario de jaulas")
    calendario(res)

    st.markdown("#### Órdenes de compra")
    tabla_ordenes(res)

    with st.expander("Pronóstico y reserva de fleteras"):
        pronostico_y_reserva(s)
    with st.expander("Datos de la empresa y supuestos"):
        datos_y_supuestos()
    controles()
