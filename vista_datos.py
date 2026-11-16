"""Pestaña "📂 Mis datos": cualquier PYME usa sus archivos en 4 pasos numerados.

    ① Descargar la plantilla   ② Subir los archivos (CSV o Excel, con sus nombres de columna)
    ③ Revisar (qué columna es cada dato; solo se abre si hay algo que corregir)   ④ Calcular el ahorro

Esta pestaña se dibuja siempre, aunque no esté abierta, para que los archivos subidos no se pierdan al
cambiar de pestaña. Lo convertido se entrega a la app en el formato de siempre (pedidos.csv, clientes.csv...).
"""

import streamlit as st

from src.convert import CAMPOS, SIN_COLUMNA, convertir, faltantes, leer_tabla, plantillas_zip, sugerir_mapeo

ARCHIVOS = ["pedidos.csv", "clientes.csv", "operacion_real.csv", "reserva.csv"]
OPCIONALES = ("operacion_real.csv", "reserva.csv")
NOMBRE = {
    "pedidos.csv": "Pedidos",
    "clientes.csv": "Clientes",
    "operacion_real.csv": "Operación real",
    "reserva.csv": "Reserva",
}
DESCRIPCION = {
    "pedidos.csv": "Un renglón por pedido: fecha de entrega, cliente y cantidad.",
    "clientes.csv": "Un renglón por cliente, con su latitud y longitud.",
    "operacion_real.csv": "Lo que hizo la empresa cada día: km, camionetas y fletes.",
    "reserva.csv": "Unidades extra reservadas por día.",
}
PESTANA_AHORRO = "💰 Ahorro"
PESTANA_RUTAS = "🗺️ Rutas de un día"
CALCULAR_AHORA = "calcular_ahora"  # aviso a la pestaña 💰 Ahorro de que debe calcular en cuanto se abra


@st.cache_data(show_spinner=False)
def _leer(contenido, nombre):
    return leer_tabla(contenido, nombre)


@st.cache_data(show_spinner=False)
def _convertir(contenido, nombre, archivo, mapeo, redondear, incluye_fletes):
    """convertir(), pero recordando el resultado (mapeo va como tupla de pares para poder recordarlo)."""
    return convertir(_leer(contenido, nombre), archivo, dict(mapeo), redondear=redondear,
                     vehiculos_incluyen_fletes=incluye_fletes)


def _emparejar(archivo, subido, tabla, sugerido):
    """Selectores 'dato ← columna' para un archivo. Devuelve (mapeo, opciones)."""
    columnas = [SIN_COLUMNA] + list(tabla.columns)
    mapeo = {}
    izquierda, derecha = st.columns(2)
    for k, (campo, etiqueta, obligatorio, _) in enumerate(CAMPOS[archivo]):
        lado = izquierda if k % 2 == 0 else derecha
        elegida = lado.selectbox(etiqueta + (" *" if obligatorio else " (opcional)"), columnas,
                                 index=columnas.index(sugerido[campo]) if sugerido[campo] else 0,
                                 key=f"mapa|{archivo}|{subido.name}|{campo}")
        mapeo[campo] = None if elegida == SIN_COLUMNA else elegida
    opciones = {"redondear": False, "incluye_fletes": False}
    if archivo == "pedidos.csv":
        opciones["redondear"] = st.checkbox("Si una cantidad trae decimales (por ejemplo kilos), redondearla hacia arriba",
                                            key=f"redondear|{subido.name}")
    if archivo == "operacion_real.csv":
        opciones["incluye_fletes"] = st.checkbox(
            "La columna de vehículos cuenta también los fletes contratados (restárselos)",
            key=f"incluye|{subido.name}", help="La app necesita solo las camionetas propias que salieron.")
    return mapeo, opciones


def _revisar(archivo, subido):
    """Paso ③ de un archivo: columnas, vista previa y problemas. Devuelve los bytes convertidos o None."""
    nombre = NOMBRE[archivo]
    try:
        tabla = _leer(subido.getvalue(), subido.name)
    except ValueError as error:
        st.error(f"**{nombre}:** {error} Revisa que sea un CSV o un Excel con los encabezados en el primer renglón.")
        return None

    sugerido = sugerir_mapeo(tabla.columns, archivo)
    exacto = all(sugerido[c] == c for c, *_ in CAMPOS[archivo] if sugerido[c])
    _, problemas_sugeridos, _ = _convertir(subido.getvalue(), subido.name, archivo,
                                           tuple(sorted(sugerido.items())), False, False)
    # Se abre solo si hay algo que corregir o si la app tuvo que adivinar nombres de columna
    with st.expander(f"{nombre} · «{subido.name}»: qué columna es cada dato",
                     expanded=bool(problemas_sugeridos) or not exacto):
        st.caption("La app adivinó qué columna de tu archivo es cada dato; corrígelo si hace falta (* = obligatorio).")
        mapeo, opciones = _emparejar(archivo, subido, tabla, sugerido)
        convertido, problemas, avisos = _convertir(subido.getvalue(), subido.name, archivo,
                                                   tuple(sorted(mapeo.items())), opciones["redondear"],
                                                   opciones["incluye_fletes"])
        for aviso in avisos:
            st.caption(f"ℹ️ {aviso}")
        if convertido is not None:
            st.caption(f"Vista previa ({len(convertido):,} renglones, ya en el formato de la app):")
            st.dataframe(convertido.head(5), hide_index=True)
    if problemas:
        st.error(f"**{nombre} («{subido.name}»): corrige esto en tu archivo o elige otra columna arriba.**\n\n" +
                 "\n".join(f"- {p}" for p in problemas))
        return None
    st.caption(f"✅ {nombre}: {len(convertido):,} renglones listos.")
    return convertido.to_csv(index=False).encode("utf-8")


def pagina(hay_carpeta_data):
    """Pasos ①, ② y ③. Devuelve (contenidos, lugar_revision, lugar_paso4).

    contenidos: {archivo: bytes en el formato de la app o None}, o None si faltan pedidos o clientes, o si
    tienen problemas. lugar_revision y lugar_paso4 son espacios que llena app.py cuando ya cargó los datos
    (los errores al juntar pedidos con clientes y el botón de calcular, que necesita saber cuántos días hay).
    """
    st.caption("Usa tus propios archivos en 4 pasos. Los archivos solo viven en tu sesión: no se guardan en el proyecto.")

    with st.container(border=True):
        st.markdown("#### ① Descarga la plantilla")
        c = st.columns([3, 1], vertical_alignment="center")
        c[0].markdown("Necesitas dos archivos: **pedidos** (fecha de entrega, cliente y cantidad) y **clientes** "
                      "(latitud y longitud de cada uno). Sirven en CSV o Excel y con tus propios nombres de columna.")
        c[1].download_button("📄 Plantillas (.zip)", plantillas_zip(), "plantillas_ruteo.zip",
                             help="Un ejemplo de cada archivo con sus columnas.")

    with st.container(border=True):
        st.markdown("#### ② Sube tus archivos")
        subidos = {}
        for columna, archivo in zip(st.columns(2), ("pedidos.csv", "clientes.csv")):
            subidos[archivo] = columna.file_uploader(NOMBRE[archivo], type=["csv", "xlsx", "xls"], key=archivo,
                                                     help=DESCRIPCION[archivo])
        with st.expander("Opcionales: lo que hiciste cada día y tus reservas"):
            st.caption("Con **operación real**, «Como se hace hoy» usa tus km y fletes reales en vez de estimarlos. "
                       "Con **reserva**, la propuesta usa las unidades que tú reservaste.")
            for archivo in OPCIONALES:
                subidos[archivo] = st.file_uploader(f"{NOMBRE[archivo]} (opcional)", type=["csv", "xlsx", "xls"],
                                                    key=archivo, help=DESCRIPCION[archivo])
        subio_algo = any(subidos.values())
        if hay_carpeta_data and not subio_algo:
            st.caption("Se encontraron archivos en la carpeta data/ del proyecto; se usan esos salvo que subas otros.")

    with st.container(border=True):
        st.markdown("#### ③ Revisa")
        if not subio_algo:
            st.caption("Aquí verás si se reconocieron las columnas de tus archivos.")
        contenidos, listo = {}, True
        for archivo in ARCHIVOS:
            contenidos[archivo] = _revisar(archivo, subidos[archivo]) if subidos[archivo] else None
            if contenidos[archivo] is None and archivo not in OPCIONALES:
                listo = False
        if subio_algo and not (subidos["pedidos.csv"] and subidos["clientes.csv"]):
            st.caption("Falta subir " + " y ".join(NOMBRE[a].lower() for a in ("pedidos.csv", "clientes.csv")
                                                   if not subidos[a]) + ".")
        lugar_revision = st.container()

    lugar_paso4 = st.container(border=True)
    return (contenidos if listo else None), lugar_revision, lugar_paso4


def _ir(pestana, calcular):
    """Al presionar un botón del paso ④: usar "Mis datos" y abrir la pestaña (y calcular, si se pidió)."""
    st.session_state["origen_datos"] = "Mis datos"
    st.session_state["pestana"] = pestana
    if calcular:
        st.session_state[CALCULAR_AHORA] = True


def paso_calcular(lugar, datos, estimar):
    """Paso ④: el botón "Calcular mi ahorro" con su tiempo estimado.

    datos: los datos propios ya cargados (o None si aún no hay). estimar(datos) devuelve (días a evaluar,
    minutos) o lanza ValueError con lo que falta (por ejemplo, historial muy corto).
    """
    with lugar:
        st.markdown("#### ④ Calcula tu ahorro")
        if datos is None:
            st.caption("Cuando tus pedidos y clientes estén listos aparecerá aquí el botón.")
            return
        try:
            dias, minutos = estimar(datos)
            texto = (f"Evalúa {len(dias)} días ({dias[0]:%d/%m/%Y} a {dias[-1]:%d/%m/%Y}); tarda unos {minutos} min. "
                     "Las opciones del cálculo están en ⚙️ Opciones avanzadas.")
        except ValueError as error:
            dias, texto = None, None
            st.warning(str(error))
        c = st.columns([1, 1, 2], vertical_alignment="center")
        c[0].button("💰 Calcular mi ahorro", type="primary", disabled=dias is None, on_click=_ir,
                    args=(PESTANA_AHORRO, True))
        c[1].button("🗺️ Solo ver mis rutas", on_click=_ir, args=(PESTANA_RUTAS, False),
                    help="Ver las rutas de un día con tus datos, sin calcular el ahorro.")
        if texto:
            c[2].caption(texto)
