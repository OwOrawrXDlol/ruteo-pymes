"""Pestaña 🐄 Rancho Ramada: el modelo aplicado al acopio de ganado (programación de jaulas + ruteo).

Se apoya en src/ganadero.py. No usa los datos de las demás pestañas: tiene sus propias órdenes de
compra (data/ejemplo_ganadero/ o un archivo de la empresa), sus distancias por carretera (OSRM) y sus
parámetros (barras al final de la página).
"""

import json
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import altair as alt
import folium
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_folium import st_folium

import interfaz
from src import ganadero as g

RAIZ = Path(__file__).parent
LOGO = RAIZ / "img" / "logo.jpeg"
FOTO = RAIZ / "img" / "ganado_rancho.jpg"
CREDITO = "Logo y fotografía: cortesía de Rancho Ramada."
FUENTE_REEMO = "https://reemo.senasica.gob.mx/ProtocoloemergenciaREEMO.pdf"
COLOR = {"A": "#E07B39", "B": "#4C78A8", "C": "#2E9E5B", "D": "#9E9E9E"}   # mismos tonos que E1-E4
NOMBRE_CORTO = {"A": "A · Actual", "B": "B · Pronóstico", "C": "C · Pronóstico + ruteo",
                "D": "D · Información perfecta"}
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
MERMAS = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]   # puntos de la gráfica de sensibilidad
CLAVE_ARCHIVO, CLAVE_MERMA_MEDIDA = "g_archivo", "g_merma_medida"


def dinero(x):
    """$12,345 para escribir en st.markdown / st.caption: el $ va escapado para que Streamlit no lo lea
    como fórmula (dos $ en un mismo texto se vuelven LaTeX)."""
    return interfaz.dinero(x).replace("$", r"\$")


def dinero_con_signo(x):
    return ("+" if x >= 0 else "-") + interfaz.dinero(abs(x))


def fecha_corta(d):
    return f"{DIAS[d.weekday()]} {d:%d}/{MESES[d.month - 1]}/{d:%Y}"


# --- Datos: los de ejemplo o los de la empresa ---

@st.cache_data(show_spinner=False)
def leer_fuente(contenido):
    """contenido: bytes del CSV de la empresa, o None para las órdenes de ejemplo.
    Devuelve (órdenes, problemas, es_de_la_empresa)."""
    if contenido is None:
        return g.leer_ordenes(g.CARPETA_DATOS / "ordenes.csv"), [], False
    try:
        tabla = pd.read_csv(BytesIO(contenido), sep=None, engine="python")
    except Exception as e:  # noqa: BLE001
        return None, [f"No se pudo leer el archivo como CSV ({e})."], True
    ordenes, problemas = g.validar_ordenes(tabla)
    return ordenes, problemas, True


def contenido_subido():
    archivo = st.session_state.get(CLAVE_ARCHIVO)
    return archivo.getvalue() if archivo is not None else None


@st.cache_data(show_spinner=False)
def _pronostico(contenido):
    ordenes, problemas, _ = leer_fuente(contenido)
    return g.pronostico(ordenes)


@st.cache_data(show_spinner="Programando las jaulas...")
def calcular(supuestos, contenido):
    """supuestos: tupla ordenada de (nombre, valor) para que se pueda recordar."""
    s = dict(supuestos)
    ordenes, _, de_empresa = leer_fuente(contenido)
    pron = _pronostico(contenido)
    if not de_empresa and (s["pct_aviso_largo"] != g.BASE["pct_aviso_largo"]
                           or s["pct_parcial"] != g.BASE["pct_parcial"]):
        # Mismas órdenes de ejemplo, solo cambian los días de aviso o las cabezas: el conteo semanal (y el
        # pronóstico) es el mismo; la fracción de órdenes avisadas tarde sí cambia
        ordenes = g.generar_ordenes(pct_aviso_largo=s["pct_aviso_largo"], pct_parcial=s["pct_parcial"])
        historial = ordenes[ordenes["fecha_embarque"] < pron["inicio"]]
        pron = {**pron, "frac_corto": float((historial["dias_aviso"] < g.ANTICIPACION).mean())}
    resultado, _ = g.comparar(ordenes, s, pron=pron)
    return resultado


def ahorro(res, e="C"):
    a, x = g.costo_total(res["A"]["indicadores"]), g.costo_total(res[e]["indicadores"])
    return a - x, (a - x) / a if a else 0.0


# --- Barras de parámetros ---
# (clave, etiqueta, mínimo, máximo, paso, formato, es porcentaje). Se dibujan al final de la página, pero se
# leen al principio (st.session_state) para calcular con ellas.
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
DE_LOS_DATOS = {"pct_aviso_largo", "pct_parcial"}   # con órdenes de la empresa salen del archivo, no de la barra
CONTROLES = [c for grupo in GRUPOS.values() for c in grupo]


def _a_pantalla(clave, porcentaje, valor):
    """Valor del modelo → valor de la barra (0.01 → 1.0 %; 0.5 → 50 %). La merma lleva decimales; lo demás, enteros."""
    v = valor * 100 if porcentaje else valor
    return round(float(v), 1) if clave == "merma_dia" else int(round(v))


def parametros():
    """Los supuestos con los valores de las barras (o los base si nadie las ha movido). Si se capturó la
    merma medida por la empresa, esa reemplaza a la barra."""
    s = dict(g.BASE)
    for clave, _, _, _, _, _, porcentaje in CONTROLES:
        valor = st.session_state.get(f"g_{clave}", _a_pantalla(clave, porcentaje, g.BASE[clave]))
        s[clave] = valor / 100 if porcentaje else valor
    medida = st.session_state.get(CLAVE_MERMA_MEDIDA) or 0
    if medida > 0:
        s["merma_dia"] = medida / 100
    return s


def ajustados(s, de_empresa):
    """Línea gris con los parámetros que se movieron (no aparece con los valores base)."""
    medida = st.session_state.get(CLAVE_MERMA_MEDIDA) or 0
    cambios = [f"{etiqueta} {formato % _a_pantalla(clave, porcentaje, s[clave])}"
               for clave, etiqueta, _, _, _, formato, porcentaje in CONTROLES
               if abs(s[clave] - g.BASE[clave]) > 1e-9 and not (clave == "merma_dia" and medida > 0)
               and not (de_empresa and clave in DE_LOS_DATOS)]
    if medida > 0:
        cambios.insert(0, f"Merma medida por la empresa {medida:.2f} % por día")
    if cambios:
        st.caption(("Parámetros ajustados (al final de la página): " + " · ".join(cambios))
                   .replace("%%", "%").replace("$", r"\$"))


def _restablecer():
    for clave, *_ in CONTROLES:
        st.session_state.pop(f"g_{clave}", None)


def controles(de_empresa):
    """Las barras, al final de la página y plegadas."""
    with st.expander("Ajustar parámetros"):
        for columna, (grupo, lista) in zip(st.columns(3), GRUPOS.items()):
            columna.markdown(f"**{grupo}**")
            for clave, etiqueta, minimo, maximo, paso, formato, porcentaje in lista:
                bloqueada = de_empresa and clave in DE_LOS_DATOS
                columna.slider(etiqueta, minimo, maximo, _a_pantalla(clave, porcentaje, g.BASE[clave]), paso,
                               format=formato, key=f"g_{clave}", disabled=bloqueada,
                               help="Con las órdenes de la empresa, este valor sale del archivo." if bloqueada
                               else AYUDA.get(clave))
        if st.session_state.get(CLAVE_MERMA_MEDIDA):
            st.caption("La merma por día viene de la merma medida (en «Usar órdenes de la empresa»), no de la barra.")
        st.button("Volver a los valores base", on_click=_restablecer, key="g_restablecer")


# --- Piezas de la pantalla ---

def encabezado():
    c1, c2 = st.columns([1, 6], vertical_alignment="center")
    c1.image(str(LOGO), width=110)
    c2.markdown("### Rancho Ramada · acopio de ganado")
    c2.caption("Aplicación del modelo (pronóstico + programación de jaulas + ruteo) con los parámetros de "
               "operación de Rancho Ramada: 3 jaulas propias, viajes de 3 días desde la engorda en Cardel, Ver., "
               "permisos de 5 días y fletera apartada con una semana de anticipación.")


def texto_periodo(inicio, semanas):
    fin = inicio + timedelta(weeks=semanas) - timedelta(days=1)
    return f"{semanas} semanas, del {inicio:%d}/{MESES[inicio.month - 1]}/{inicio:%Y} al {fin:%d}/{MESES[fin.month - 1]}/{fin:%Y}"


def cifras(res, periodo):
    ia, ic = res["A"]["indicadores"], res["C"]["indicadores"]
    monto, pct = ahorro(res)
    c = st.columns(5)
    c[0].metric("Costo del año" if periodo[1] == 52 else "Costo del periodo",
                f"{interfaz.dinero(g.costo_total(ia))} → {interfaz.dinero(g.costo_total(ic))}",
                interfaz.pct(-pct * 100), delta_color="inverse",
                help="Fletes + merma + permisos re-tramitados + fleteras apartadas sin usar.")
    c[1].metric("Viajes", f"{ia['Viajes']} → {ic['Viajes']}", f"{ic['Viajes'] - ia['Viajes']:+d}", delta_color="inverse",
                help="Jaulas que salen. Bajan cuando el ruteo junta dos lotes chicos en una jaula.")
    c[2].metric("Días de retraso", f"{ia['Días de retraso']} → {ic['Días de retraso']}",
                f"{ic['Días de retraso'] - ia['Días de retraso']:+d}", delta_color="inverse")
    c[3].metric("Merma", f"{interfaz.dinero(ia['Merma ($)'])} → {interfaz.dinero(ic['Merma ($)'])}",
                dinero_con_signo(ic["Merma ($)"] - ia["Merma ($)"]), delta_color="inverse",
                help="Cabezas × días de espera × peso × merma por día × precio por kg.")
    c[4].metric("Fletes contratados", f"{ia['Fletes contratados']} → {ic['Fletes contratados']}",
                f"{ic['Fletes contratados'] - ia['Fletes contratados']:+d}", delta_color="inverse")
    st.caption(f"Izquierda: «{g.ESCENARIOS_G['A']}» · derecha: «{g.ESCENARIOS_G['C']}». "
               f"Ahorro: **{dinero(monto)} ({interfaz.pct(pct * 100, signo=False)})** en {texto_periodo(*periodo)}.")


def b_igual_a_a(res):
    """True si el pronóstico (B) no cambió nada frente a hoy (A)."""
    claves = ["Viajes", "  apartados", "  de último momento", "Días de retraso", "Fleteras apartadas sin usar"]
    return all(res["A"]["indicadores"][k] == res["B"]["indicadores"][k] for k in claves)


def curva(s, contenido):
    """Ahorro de B y de C, y α, para cada merma de MERMAS (lo demás igual)."""
    ordenes, _, _ = leer_fuente(contenido)
    filas = []
    for m in MERMAS:
        r = calcular(tuple(sorted({**s, "merma_dia": m}.items())), contenido)
        alfa = g.alfa_apartar({**s, "merma_dia": m}, ordenes)
        for e in ("B", "C"):
            monto, pct = ahorro(r, e)
            filas.append({"merma": m * 100, "escenario": NOMBRE_CORTO[e], "ahorro": monto, "pct": pct * 100,
                          "alfa": alfa, "fletes": r[e]["indicadores"]["Fletes contratados"],
                          "b_igual": b_igual_a_a(r)})
    return pd.DataFrame(filas)


def aviso_b_igual(res, s, contenido, tabla_curva):
    """Cambio 4: si B = A, decir por qué y qué haría que la reserva actúe."""
    if not b_igual_a_a(res):
        return
    ordenes, _, _ = leer_fuente(contenido)
    pron = _pronostico(contenido)
    falta, sobra, _ = g.costos_apartar(s, ordenes)
    sorpresa = pron["semanal"][pron["elegido"]].mean() * pron["frac_corto"] / 5
    actua = tabla_curva[~tabla_curva["b_igual"]]["merma"]
    desde = (f"con merma de **{actua.min():.2f} % por día** o más" if len(actua)
             else "con más merma de la que permite la barra (2 %)")
    st.info(
        f"**El pronóstico (B) sale igual que hoy (A) con estos valores.** Llegan en promedio "
        f"{sorpresa:.1f} órdenes sorpresa por día y α = {g.alfa_apartar(s, ordenes):.2f}: apartar una fletera que "
        f"probablemente sobre cuesta {dinero(sobra)}, más que la espera esperada de una jaula ({dinero(falta)}), así "
        f"que el modelo decide no apartar de más. La reserva sí actúa {desde}, con un sobrecosto de último momento "
        "(en Rancho Ramada es \\$0), si conseguir fletera de un día para otro es más difícil, o si una fletera "
        "apartada sin usar cuesta menos. Las barras están al final de la página.")


def grafica_merma(tabla, s):
    """Cambio 5: ahorro contra merma por día, con α en cada punto."""
    base = alt.Chart(tabla).encode(
        x=alt.X("merma:Q", title="Merma de peso por día de espera (%)", scale=alt.Scale(domain=[0.5, 2.0])),
        y=alt.Y("ahorro:Q", title="Ahorro contra A (MXN)", axis=alt.Axis(format="$,.0f")),
        color=alt.Color("escenario:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=[NOMBRE_CORTO["B"], NOMBRE_CORTO["C"]], range=[COLOR["B"], COLOR["C"]])),
        tooltip=[alt.Tooltip("merma:Q", title="Merma (%)", format=".2f"), "escenario",
                 alt.Tooltip("ahorro:Q", title="Ahorro", format="$,.0f"), alt.Tooltip("pct:Q", title="%", format=".1f"),
                 alt.Tooltip("alfa:Q", title="α", format=".2f"), alt.Tooltip("fletes:Q", title="Fletes contratados")])
    lineas = base.mark_line(point=True)
    alfas = alt.Chart(tabla[tabla["escenario"] == NOMBRE_CORTO["C"]]).mark_text(dy=-12, fontSize=11, color="#555").encode(
        x="merma:Q", y="ahorro:Q", text=alt.Text("alfa:Q", format="α=.2f"))
    actual = alt.Chart(pd.DataFrame({"merma": [s["merma_dia"] * 100]})).mark_rule(strokeDash=[4, 4], color="#999").encode(
        x="merma:Q")
    st.altair_chart((lineas + alfas + actual).properties(height=260), width="stretch")
    st.caption("Con más merma, cada día de espera cuesta más: sube α y el modelo aparta más fleteras (más fletes, "
               "menos merma). El ruteo junta menos lotes, porque esperar al segundo lote también cuesta más. Por eso el "
               "ahorro no sube parejo. Línea punteada: la merma en uso.")


FILAS_COSTO = [("Costo total", None), ("Ahorro contra A", None), ("Costo de fletes", "Costo de fletes ($)"),
               ("Costo de merma", "Merma ($)"), ("Re-trámite de permisos", "Re-trámite de permisos ($)"),
               ("Fleteras apartadas sin usar ($)", "Pago por fleteras sin usar ($)"),
               ("Sobrecosto de último momento", "Sobrecosto de último momento ($)")]
FILAS = ["Viajes", "Viajes con 2 puntos de carga", "OCs con retraso", "Días de retraso", "Retraso máximo (días)",
         "Permisos vencidos", "Cabezas con retraso", "Merma (kg)", "Fletes contratados", "  apartados",
         "  de último momento", "Fleteras apartadas sin usar", "Utilización de jaulas propias"]


def tabla_escenarios(res):
    """Indicador × escenario: primero los costos (cambio 6), luego la operación. Devuelve la tabla."""
    a = g.costo_total(res["A"]["indicadores"])
    tabla = {}
    for e in g.ESCENARIOS_G:
        i = res[e]["indicadores"]
        total = g.costo_total(i)
        col = {"Costo total": interfaz.dinero(total),
               "Ahorro contra A": "—" if e == "A" else
               f"{interfaz.dinero(a - total)} ({interfaz.pct((a - total) / a * 100, signo=False)})"}
        for nombre, clave in FILAS_COSTO[2:]:
            col[nombre] = interfaz.dinero(i[clave])
        for k in FILAS:
            col[k.strip()] = f"{i[k]:.0%}" if k.startswith("Utilización") else f"{i[k]:,.0f}"
        tabla[NOMBRE_CORTO[e]] = col
    tabla = pd.DataFrame(tabla)
    st.dataframe(tabla, width="stretch", height=35 * (len(tabla) + 1) + 3,
                 column_config={"_index": st.column_config.Column("Indicador", width="medium")})
    st.caption("El costo de fletes casi no cambia entre A y B: cada viaje cuesta lo mismo con jaula propia o con "
               "fletera, apartada o no. Lo que cambia es la espera (merma) y, con el ruteo, los viajes que se juntan. "
               "D supone saber desde antes todas las órdenes: es el tope de la programación, no es alcanzable.")
    return tabla


# --- Mapa ---

def _lugares():
    lugares = pd.read_csv(g.CARPETA_DATOS / "lugares.csv")
    trazos = json.loads((g.CARPETA_DATOS / "rutas.json").read_text())
    pos = {f.nombre: (f.lat, f.lon) for f in lugares.itertuples()}

    def trazo(a, b):
        return trazos.get(f"{a}|{b}") or list(reversed(trazos.get(f"{b}|{a}", [pos[a], pos[b]])))
    return pos, trazo


def combinaciones(res):
    """Viajes con 2 puntos de carga, agrupados por combinación de orígenes: {"Balancán + Tenosique": 3}."""
    c = res["C"]["resultado"]
    return c[c["lotes"] > 1]["origen"].value_counts().to_dict() if "lotes" in c else {}


def mapa_folium(res):
    pos, trazo = _lugares()
    cuantas = res["A"]["resultado"]["origen"].value_counts()
    juntas = combinaciones(res)
    m = folium.Map(location=(18.4, -93.8), zoom_start=6, tiles="OpenStreetMap")
    for origen, n in cuantas.items():   # viaje sencillo: engorda → origen → engorda
        folium.PolyLine(trazo(g.ENGORDA, origen), color="#888", weight=2 + n / 30, opacity=0.6,
                        tooltip=f"Engorda ↔ {origen}: {n} órdenes").add_to(m)
    for combinacion, n in juntas.items():
        partes = combinacion.split(" + ")
        if len(partes) == 2:            # dos puntos de carga: engorda → lejano → cercano → engorda
            puntos = trazo(g.ENGORDA, partes[0]) + trazo(partes[0], partes[1]) + trazo(partes[1], g.ENGORDA)
            folium.PolyLine(puntos, color=COLOR["C"], weight=3 + 1.5 * n, opacity=0.9,
                            tooltip=f"{combinacion} × {n}").add_to(m)
    folium.Marker(pos[g.ENGORDA], tooltip=g.ENGORDA, icon=folium.Icon(color="black", icon="home")).add_to(m)
    for origen, n in cuantas.items():
        mismo = juntas.get(origen, 0)   # dos lotes del mismo origen en una jaula
        folium.CircleMarker(pos[origen], radius=5 + n / 10, color="#E07B39", fill=True, fill_opacity=0.85,
                            tooltip=f"{origen} ({g.ESTADO[origen]}): {n} órdenes, flete ${g.FLETE[origen]:,}"
                                    + (f" · dos lotes juntos × {mismo}" if mismo else "")).add_to(m)
    return m


def mapa(res):
    st_folium(mapa_folium(res), height=430, use_container_width=True, returned_objects=[], key="mapa_ganadero")
    juntas = combinaciones(res)
    dos = {k: n for k, n in juntas.items() if " + " in k}
    mismo = sum(n for k, n in juntas.items() if " + " not in k)
    detalle = " · ".join(f"{k} × {n}" for k, n in sorted(dos.items(), key=lambda x: -x[1]))
    st.caption("Gris: viajes de un solo punto de carga (el grosor es cuántos hubo). Verde: viajes del ruteo con dos "
               f"puntos de carga; se juntaron lotes **{sum(juntas.values())} veces**"
               + (f": {detalle}" if detalle else "")
               + (f", y {mismo} con dos lotes del mismo origen (sin desvío, no se ven aparte en el mapa)" if mismo else "")
               + ". Rutas y km por carretera (OpenStreetMap / OSRM).")


def _fuente(tamano):
    """Source Sans (la letra de Streamlit, viene con el paquete) para que salgan los acentos; si no, DejaVu."""
    import glob
    import os
    carpeta = os.path.join(os.path.dirname(st.__file__), "static", "static", "media")
    for ruta in glob.glob(os.path.join(carpeta, "SourceSansVF-Upright*.woff2")) + \
            ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(ruta, tamano)
        except OSError:
            continue
    return ImageFont.load_default(size=tamano)


# Dónde va el nombre de cada lugar en el PNG (dx, dy, ancla de Pillow): en el grupo de Tabasco y Chiapas
# los puntos están muy juntos
ETIQUETA_PNG = {"Balancán": (24, -6, "lm"), "Tenosique": (24, 22, "lm"), "Emiliano Zapata": (-10, -34, "rm"),
                "Palenque/Ocosingo": (-24, 26, "rm"), "Villahermosa": (0, -34, "mm"), "Pichucalco": (0, 40, "mm"),
                "Medellín": (24, 0, "lm"), g.ENGORDA: (28, 0, "lm")}


def mapa_png(res):
    """Cambio 12: el mapa de rutas como imagen PNG (sin fondo de calles), para carteles.

    Gris: la ruta de cada origen (grosor = órdenes). Verde: el tramo entre los dos puntos de carga de los
    viajes que juntan lotes (grosor = veces); el resto de esos viajes va por la misma carretera gris.
    """
    pos, trazo = _lugares()
    ancho, alto = 2200, 1300
    izq, der, arriba, abajo = 120, 330, 170, 150
    lats = [p[0] for p in pos.values()]
    lons = [p[1] for p in pos.values()]
    escala = min((ancho - izq - der) / (max(lons) - min(lons)), (alto - arriba - abajo) / (max(lats) - min(lats)))

    def xy(lat, lon):
        return izq + (lon - min(lons)) * escala, alto - abajo - (lat - min(lats)) * escala

    imagen = Image.new("RGB", (ancho, alto), "white")
    dibujo = ImageDraw.Draw(imagen)
    letra, chica, titulo = _fuente(34), _fuente(28), _fuente(52)
    cuantas = res["A"]["resultado"]["origen"].value_counts()
    for origen, n in cuantas.items():
        dibujo.line([xy(*p) for p in trazo(g.ENGORDA, origen)], fill="#a8a8a8", width=int(4 + n / 12), joint="curve")
    for combinacion, n in combinaciones(res).items():
        partes = combinacion.split(" + ")
        if len(partes) == 2:
            dibujo.line([xy(*p) for p in trazo(partes[0], partes[1])], fill=COLOR["C"], width=int(8 + 4 * n),
                        joint="curve")
    for origen, n in cuantas.items():
        x, y = xy(*pos[origen])
        r = 12 + n / 7
        dibujo.ellipse([x - r, y - r, x + r, y + r], fill=COLOR["A"], outline="white", width=3)
        dx, dy, ancla = ETIQUETA_PNG.get(origen, (24, 0, "lm"))
        dibujo.text((x + dx, y + dy), f"{origen} ({n})", fill="#222", font=letra, anchor=ancla)
    x, y = xy(*pos[g.ENGORDA])
    dibujo.rectangle([x - 18, y - 18, x + 18, y + 18], fill="#111")
    dibujo.text((x + 28, y), "Engorda (Cardel, Ver.)", fill="#111", font=letra, anchor="lm")
    dibujo.text((izq, 60), "Rancho Ramada · rutas desde la engorda", fill="#111", font=titulo, anchor="lm")
    dibujo.text((izq, 115), "Entre paréntesis: órdenes de compra del periodo", fill="#666", font=chica, anchor="lm")
    dibujo.line([(izq, alto - 80), (izq + 60, alto - 80)], fill="#a8a8a8", width=10)
    dibujo.text((izq + 75, alto - 80), "Ruta por carretera de cada origen (un punto de carga)", fill="#444",
                font=chica, anchor="lm")
    dibujo.line([(izq, alto - 40), (izq + 60, alto - 40)], fill=COLOR["C"], width=14)
    dibujo.text((izq + 75, alto - 40), "Ruteo: tramo entre dos puntos de carga cuando se juntan lotes · "
                                       "Rutas: OpenStreetMap / OSRM", fill="#444", font=chica, anchor="lm")
    salida = BytesIO()
    imagen.save(salida, format="PNG")
    return salida.getvalue()


# --- Calendario ---

def meses_del_periodo(inicio, semanas):
    fin = inicio + timedelta(weeks=semanas) - timedelta(days=1)
    meses, d = [], date(inicio.year, inicio.month, 1)
    while d <= fin:
        meses.append(d)
        d = date(d.year + d.month // 12, d.month % 12 + 1, 1)
    return meses


def calendario(res, periodo):
    meses = meses_del_periodo(*periodo)
    inicio = st.select_slider("Mes", options=meses, value=meses[0], key="g_mes",
                              format_func=lambda d: f"{MESES[d.month - 1]} {d.year}")
    fin = date(inicio.year + inicio.month // 12, inicio.month % 12 + 1, 1)
    # Eje en español (cambio 2): Vega pondría "Mon 03" y "November"
    eje = alt.Axis(labelExpr="['dom','lun','mar','mié','jue','vie','sáb'][day(datum.value)] + ' ' + date(datum.value)",
                   tickCount={"interval": "day", "step": 2}, labelAngle=0, grid=True)
    for e in ("A", "C"):
        v = res[e]["viajes"].merge(res[e]["resultado"][["oc_id", "origen", "cabezas"]], on="oc_id")
        v = v[(v["fin"] >= inicio) & (v["inicio"] < fin)].copy()
        st.markdown(f"<span style='color:{COLOR[e]}'>■</span> **{g.ESCENARIOS_G[e]}**", unsafe_allow_html=True)
        if v.empty:
            st.caption("Sin viajes este mes.")
            continue
        v["desde"] = pd.to_datetime(v["inicio"])
        v["hasta"] = pd.to_datetime(v["fin"]) + pd.Timedelta(days=1)
        v["carga"] = v["inicio"].map(fecha_corta)
        v["estado"] = v["retraso"].map(lambda r: f"{r} día(s) de retraso" if r else "a tiempo")
        orden = sorted(v["recurso"].unique(), key=lambda r: (not r.startswith("Propia"), r))
        barras = alt.Chart(v).mark_bar(cornerRadius=3, height=16).encode(
            x=alt.X("desde:T", title=None, axis=eje, scale=alt.Scale(domain=[str(inicio), str(fin)])),
            x2="hasta:T",
            y=alt.Y("recurso:N", sort=orden, title=None),
            color=alt.Color("origen:N", title="Origen"),
            stroke=alt.condition("datum.retraso > 0", alt.value("#C62828"), alt.value(None)),
            strokeWidth=alt.condition("datum.retraso > 0", alt.value(2.5), alt.value(0)),
            tooltip=[alt.Tooltip("oc_id", title="OC"), alt.Tooltip("origen", title="Origen"),
                     alt.Tooltip("cabezas", title="Cabezas"), alt.Tooltip("carga", title="Carga"),
                     alt.Tooltip("estado", title="Estado")])
        st.altair_chart(barras.properties(height=34 * len(orden) + 30), width="stretch")
    st.caption("Cada barra es un viaje de 3 días; borde rojo = salió con retraso.")


# --- Tablas y textos ---

def tabla_ordenes(res):
    a = res["A"]["resultado"].set_index("oc_id")
    c = res["C"]["resultado"]
    # En C una fila puede traer dos OCs ("OC-0101 + OC-0102"): se reparte a cada una
    viaje_c = {oc: f for f in c.itertuples() for oc in str(f.oc_id).split(" + ")}
    filas = []
    for oc, f in a.iterrows():
        x = viaje_c[str(oc)]
        filas.append({"OC": oc, "Origen": f["origen"], "Cabezas": f["cabezas"], "Aviso (días)": f["dias_aviso"],
                      "Embarque": f["fecha_embarque"], "Carga hoy": f["fecha_carga"], "Medio hoy": f["medio"],
                      "Retraso hoy": f["retraso"], "Carga con el modelo": x.fecha_carga, "Medio con el modelo": x.medio,
                      "Retraso con el modelo": (x.fecha_carga - f["fecha_embarque"]).days,
                      "Junto con": " + ".join(o for o in str(x.oc_id).split(" + ") if o != str(oc))})
    tabla = pd.DataFrame(filas)
    if st.toggle("Solo las que cambian", value=True, key="g_solo"):
        tabla = tabla[(tabla["Retraso hoy"] != tabla["Retraso con el modelo"]) | (tabla["Junto con"] != "")]
    st.dataframe(tabla, hide_index=True, height=320)


def reglas_vrp(res, s):
    st.markdown(
        f"El ruteo junta **dos lotes chicos en una sola jaula** ({len(res['C']['pares'])} veces) solo si se cumplen "
        "todas estas reglas:\n"
        f"- caben juntos: hasta **{g.CAPACIDAD_CABEZAS} cabezas** por jaula;\n"
        f"- a lo más **{g.PUNTOS_MAX} puntos de carga** y a no más de **{s['max_desvio_km']:.0f} km** por carretera "
        "entre ellos;\n"
        f"- el primer lote espera al segundo a lo más **{s['max_dias_espera']:.0f} días**, dentro de la vigencia de "
        "5 días del permiso (esa espera se cobra como merma);\n"
        "- los dos se conocen antes de que el primero esté listo para cargar;\n"
        "- **cada lote viaja con su propia guía REEMO** (una UPP de origen y una de destino) **y su certificado "
        "zoosanitario**, con el mismo destino (la engorda).")
    st.caption("⚠️ Juntar lotes requiere confirmar con el administrador REEMO del estado que ambos orígenes tienen el "
               "mismo estatus zoosanitario y que la jaula puede llevar dos guías. Hoy Rancho Ramada carga en un solo "
               f"punto por permiso. Fuente de la guía REEMO: [protocolo REEMO, SENASICA]({FUENTE_REEMO}).")


NOMBRES_MODELO = {"ingenuo": "Ingenuo (hace 2 semanas)", "prom4": "Promedio 4 semanas", "ridge": "Regresión Ridge",
                  "bosque": "Bosque aleatorio", "gboost": "Gradient boosting",
                  "prom_hist": "Promedio del historial (sustituto)"}


def pronostico_y_reserva(s, contenido):
    p = _pronostico(contenido)
    ordenes, _, _ = leer_fuente(contenido)
    t = p["tabla"].copy()
    t["Modelo"] = t["modelo"].map(NOMBRES_MODELO) + t["modelo"].map(lambda m: " ✔" if m == p["elegido"] else "")
    columnas = {"wape_semana": "WAPE semanal (prueba)", "wape_validacion": "WAPE semanal (validación)",
                "sesgo_semana": "Sesgo"}
    for c in columnas:      # cambio 8: en porcentaje con un decimal
        t[c] = t[c] * 100
    st.dataframe(t[["Modelo"] + list(columnas)].rename(columns=columnas), hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.1f %%") for c in columnas.values()})
    if p.get("sustituto"):
        st.caption(f"Hay menos de {g.SEMANAS_MINIMAS} semanas de historial para entrenar los modelos: se usa el "
                   "promedio de jaulas por semana de lo ya conocido.")
    else:
        por_origen = p["tabla"]["wape_semana_origen"]
        st.caption(f"Se pronostica el total de jaulas de cada semana. Por origen el error es de {por_origen.min() * 100:.0f} % "
                   f"a {por_origen.max() * 100:.0f} %: cada origen recibe menos de una jaula por semana y es casi azar, así "
                   "que las fleteras se apartan con el total semanal.")
    alfa = g.alfa_apartar(s, ordenes)
    st.markdown(
        "Se pronostican las jaulas de cada semana con los mismos modelos de la pestaña 💰 Ahorro; el elegido (✔) es el "
        "de menor error en validación. Con él se apartan fleteras para las órdenes que todavía no llegan "
        f"({p['frac_corto']:.0%} se avisan con menos de 7 días).\n\n"
        f"**α = {alfa:.2f}** (ec. 2 del artículo): costo de que falte una jaula ÷ (que falte + que sobre). En una PYME "
        "de reparto, que falte cuesta la tarifa urgente; **en Rancho Ramada el flete cuesta igual apartado o no**, así "
        "que lo que cuesta es la espera: merma de peso por cada día sin jaula. Que sobre cuesta lo que se paga por "
        "una fletera apartada que no se usa.")


def datos_y_supuestos(ordenes, de_empresa, periodo):
    if de_empresa:
        fila_ordenes = ("Órdenes de compra", f"Archivo de la empresa: {len(ordenes)} órdenes, del "
                        f"{ordenes['fecha_embarque'].min():%d/%m/%Y} al {ordenes['fecha_embarque'].max():%d/%m/%Y}",
                        "Dato de la empresa")
    else:
        fila_ordenes = ("Órdenes de compra", f"{g.RITMO:.0f} jaulas por semana en promedio, 3-5 proveedores",
                        "Supuesto (calibrado con ~5 fleteras al mes)")
    filas = [
        ("Jaulas propias", "3", "Dato de la empresa"),
        ("Viaje redondo", "3 días", "Dato de la empresa"),
        ("Vigencia del permiso", "5 días; sin dictámenes en fin de semana", "Dato de la empresa"),
        ("Fletera", "Se aparta con una semana; ~5 al mes; mismo precio apartada o no, propia o contratada",
         "Dato de la empresa"),
        ("Flete por viaje", "Balancán $45,000; Pichucalco $40,000; otras zonas $16,000-18,000", "Dato de la empresa"),
        ("Cabezas por jaula", "70-87 (a veces menos)", "Dato de la empresa"),
        ("Engorda", "Carretera Cardel-Veracruz km 226, José Cardel, Ver.", "Dato de la empresa"),
        fila_ordenes,
        ("Periodo comparado", texto_periodo(*periodo), "Calculado"),
        ("Km y horas por carretera", "OSRM (OpenStreetMap)", "Calculado"),
    ] + [(g.NOMBRE_SUPUESTO[k], f"{v[0]:g} (de {v[1]:g} a {v[2]:g})", "Supuesto")
         for k, v in g.SUPUESTOS.items() if not (de_empresa and k in DE_LOS_DATOS)]
    st.dataframe(pd.DataFrame(filas, columns=["Parámetro", "Valor", "Origen"]), hide_index=True)


def cargar_ordenes(problemas):
    """Cambio 11: subir las órdenes de la empresa y la merma medida."""
    with st.expander("Usar órdenes de la empresa"):
        st.caption("CSV con una fila por orden de compra. Columnas: **oc_id, origen, cabezas, fecha_embarque** y "
                   "**fecha_aviso** (o **dias_aviso**). Opcionales: proveedor y flete_mxn. Orígenes: "
                   + ", ".join(g.FLETE) + ". Todo se recalcula con este archivo.")
        st.file_uploader("Órdenes de compra (CSV)", type=["csv"], key=CLAVE_ARCHIVO)
        st.number_input("Merma medida por día (%; 0 = usar la barra)", 0.0, 10.0, 0.0, 0.05, format="%.2f",
                        key=CLAVE_MERMA_MEDIDA,
                        help="Por ejemplo: (kg facturados en origen − kg al llegar) ÷ kg facturados ÷ días de espera.")
        for p in problemas:
            st.error(p)


def descargas(tabla, res):
    """Cambio 12: tabla, mapa y las imágenes, para carteles y presentaciones."""
    with st.expander("⬇️ Descargas para carteles y presentaciones"):
        c = st.columns(5)
        c[0].download_button("Escenarios (CSV)", tabla.to_csv().encode("utf-8-sig"), "escenarios_rancho_ramada.csv",
                             "text/csv", key="g_desc_csv")
        c[1].download_button("Mapa de rutas (PNG)", mapa_png(res), "rutas_rancho_ramada.png", "image/png",
                             key="g_desc_png")
        c[2].download_button("Mapa interactivo (HTML)", mapa_folium(res).get_root().render().encode("utf-8"),
                             "rutas_rancho_ramada.html", "text/html", key="g_desc_html")
        c[3].download_button("Logo", LOGO.read_bytes(), "logo_rancho_ramada.jpeg", "image/jpeg", key="g_desc_logo")
        c[4].download_button("Foto del ganado", FOTO.read_bytes(), "ganado_rancho_ramada.jpg", "image/jpeg",
                             key="g_desc_foto")
        st.caption("El logo (447 × 447 px) y la foto (1000 × 667 px) se descargan en la resolución original que se "
                   f"tiene. {CREDITO}")


def barra_lateral():
    """Cambio 10: en esta pestaña la barra lateral muestra los datos del rancho, no los de la demo de reparto."""
    st.sidebar.markdown("**🐄 Rancho Ramada**")
    st.sidebar.caption("3 jaulas propias · viaje redondo de 3 días · permisos de 5 días · fletera apartada con una "
                       "semana · engorda en Cardel, Ver.")
    st.sidebar.caption("Los parámetros de esta pestaña están al final de la página, en «Ajustar parámetros». Los "
                       "controles de las demás pestañas reaparecen al salir de aquí, con sus valores.")


# --- La pestaña ---

def pagina():
    encabezado()
    st.image(str(FOTO), width="stretch")
    st.caption(CREDITO)

    contenido = contenido_subido()
    ordenes, problemas, de_empresa = leer_fuente(contenido)
    if problemas:          # el archivo de la empresa tiene errores: se sigue con las órdenes de ejemplo
        contenido, (ordenes, _, de_empresa) = None, leer_fuente(None)
    elif de_empresa:
        st.caption(f"Con las órdenes de la empresa ({len(ordenes)} órdenes).")
    periodo = (_pronostico(contenido)["inicio"], _pronostico(contenido)["semanas"])

    s = parametros()
    res = calcular(tuple(sorted(s.items())), contenido)
    cifras(res, periodo)
    ajustados(s, de_empresa)
    tabla_curva = curva(s, contenido)
    aviso_b_igual(res, s, contenido, tabla_curva)

    st.markdown("#### Escenarios")
    tabla = tabla_escenarios(res)
    st.markdown("#### Ahorro según la merma por día")
    grafica_merma(tabla_curva, s)

    st.markdown("#### Rutas desde la engorda")
    mapa(res)
    with st.expander("Reglas para juntar lotes en una jaula (ruteo)"):
        reglas_vrp(res, s)

    st.markdown("#### Calendario de jaulas")
    calendario(res, periodo)

    st.markdown("#### Órdenes de compra")
    tabla_ordenes(res)

    with st.expander("Pronóstico y reserva de fleteras"):
        pronostico_y_reserva(s, contenido)
    with st.expander("Datos de la empresa y supuestos"):
        datos_y_supuestos(ordenes, de_empresa, periodo)
    cargar_ordenes(problemas)
    controles(de_empresa)
    descargas(tabla, res)
