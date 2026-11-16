"""Barra lateral: "🏢 Mi empresa" (lo esencial) y "⚙️ Opciones avanzadas" (todo lo demás).

Permite que cualquier PYME ponga su almacén, su unidad (cajas, cabezas, kilos...), su flota y sus costos
sin editar config.yaml. Parte de un perfil (el del artículo o un ejemplo de otro negocio) y cada valor se
puede cambiar. Los cambios solo viven en la sesión de esa persona: no modifican el proyecto.
"""

from pathlib import Path

import streamlit as st
import yaml

from interfaz import dinero
from src.load import valores_supuestos
from src.perfiles import (CAMPOS, CAMPOS_TEXTO, POR_OMISION, aplicar_valores, cargar_perfiles, config_del_perfil,
                          diferencias, leer, problemas)

# Lo que se ve al abrir "Mi empresa", en este orden; el resto va en "Opciones avanzadas"
ESENCIALES = ["negocio.unidad", "flota.camionetas_propias", "flota.capacidad_cajas", "costos.tarifa_reservada",
              "costos.tarifa_urgente", "almacen.lat", "almacen.lon"]
NUMERICOS = {c[0]: c for c in CAMPOS}
TEXTOS = {c[0]: c for c in CAMPOS_TEXTO}
AVANZADOS = [r for r in TEXTOS if r not in ESENCIALES] + [r for r in NUMERICOS if r not in ESENCIALES]


def _campo(ruta, inicial, perfil):
    """Dibuja el control de un parámetro y devuelve su valor."""
    clave = f"{perfil}|{ruta}"  # cambia con el perfil, para que al elegir otro se carguen sus valores
    if ruta in TEXTOS:
        _, etiqueta, por_omision = TEXTOS[ruta]
        return st.text_input(etiqueta, value=str(leer(inicial, ruta)), key=clave).strip() or por_omision
    _, etiqueta, tipo, minimo, paso, ayuda = NUMERICOS[ruta]
    actual = leer(inicial, ruta)
    if tipo == "int":
        return int(st.number_input(etiqueta, min_value=int(minimo), value=int(actual), step=int(paso),
                                   help=ayuda, key=clave))
    return float(st.number_input(etiqueta, min_value=float(minimo), value=float(actual), step=float(paso),
                                 help=ayuda, key=clave, format="%g"))


def _rutas(config, prefijo=""):
    """Todas las rutas de un diccionario anidado: {"flota": {"capacidad_cajas": 80}} → ["flota.capacidad_cajas"]."""
    rutas = []
    for clave, valor in config.items():
        ruta = f"{prefijo}{clave}"
        rutas += _rutas(valor, ruta + ".") if isinstance(valor, dict) else [ruta]
    return rutas


def supuestos_vigentes(ruta_config, carpeta_perfiles, perfil, a_mano):
    """Los valores que siguen siendo supuestos: (parámetro, valor, nota).

    Parte de los marcados en config.yaml; si se eligió un perfil, sus valores reemplazan a los de config.yaml
    con su propia marca (real o supuesto); lo que la persona escribió a mano ya no cuenta como supuesto.
    """
    vigentes = {p: (p, v, n) for p, v, n in valores_supuestos(ruta_config)}
    for archivo in sorted(Path(carpeta_perfiles).glob("*.yaml")):
        contenido = yaml.safe_load(archivo.read_text(encoding="utf-8")) or {}
        if contenido.get("nombre") != perfil:
            continue
        for ruta in _rutas(contenido.get("config") or {}):
            vigentes.pop(ruta, None)
        for p, v, n in valores_supuestos(archivo):
            if p.startswith("config."):
                vigentes[p.removeprefix("config.")] = (p.removeprefix("config."), v, n)
    return [fila for ruta, fila in vigentes.items() if ruta not in a_mano]


def panel(config_base, carpeta_perfiles, ruta_config):
    """Dibuja los dos plegables. Devuelve (config, nombre del perfil, rutas distintas a config.yaml,
    supuestos vigentes, ajustes).

    ajustes: {"segundos": búsqueda de rutas por día, y las opciones del cálculo del ahorro:
              "semanas_prueba", "semanas_validacion", "segundos_ahorro"}.
    """
    perfiles = cargar_perfiles(carpeta_perfiles)
    # Se crean en el orden en que se ven, pero el perfil (en avanzadas) se elige antes de llenar "Mi empresa"
    caja_empresa = st.sidebar.expander("🏢 Mi empresa")
    lugar_resumen = st.sidebar.empty()
    caja_avanzadas = st.sidebar.expander("⚙️ Opciones avanzadas")

    with caja_avanzadas:
        nombre = st.selectbox("Empezar desde el perfil", [POR_OMISION] + list(perfiles), key="perfil_elegido",
                              help="Carga los valores de un negocio parecido al tuyo; luego ajusta cada uno.")
        if nombre in perfiles and perfiles[nombre]["descripcion"]:
            st.caption(perfiles[nombre]["descripcion"].strip())
    inicial = config_del_perfil(config_base, perfiles, nombre)

    valores = {}
    with caja_empresa:
        st.caption("Lo esencial de tu negocio. Los cambios solo valen en tu sesión.")
        for ruta in ESENCIALES:
            valores[ruta] = _campo(ruta, inicial, nombre)
    with caja_avanzadas:
        for ruta in AVANZADOS:
            valores[ruta] = _campo(ruta, inicial, nombre)
        segundos = st.slider("Segundos de búsqueda de rutas por día", 1, 10,
                             int(config_base["solver"]["segundos_por_dia"]), key="segundos",
                             help="Más tiempo = rutas un poco mejores, pero más espera.")
        st.markdown("**Cálculo del ahorro con tus datos o parámetros**")
        st.caption("Como en el artículo: las últimas semanas de tu historial son la prueba; con las anteriores se "
                   "entrena y se elige el pronóstico.")
        ajustes = {
            "segundos": segundos,
            "semanas_prueba": int(st.number_input("Semanas a evaluar", 1, 12, 4, key="semanas_prueba",
                                                  help="Las últimas semanas de tu historial.")),
            "semanas_validacion": int(st.number_input(
                "Semanas para elegir el modelo", 2, 12, 4, key="semanas_validacion",
                help="Las semanas anteriores a la prueba; con ellas se elige el modelo y el margen.")),
            "segundos_ahorro": int(st.number_input("Segundos de búsqueda por ruteo", 1, 4, 2, key="segundos_ahorro",
                                                   help="Más segundos = rutas un poco mejores, pero más espera.")),
        }

    config = aplicar_valores(inicial, valores)
    cambios = diferencias(config_base, config)
    f, c, n = config["flota"], config["costos"], config["negocio"]
    lugar_resumen.caption(f"{f['camionetas_propias']} camionetas de {f['capacidad_cajas']} {n['unidad']} · "
                          f"reservada {dinero(c['tarifa_reservada'])} · urgente {dinero(c['tarifa_urgente'])}"
                          + ("" if cambios else " (supuestos del artículo)"))
    for aviso in problemas(config):
        st.sidebar.error(aviso)
    supuestos = supuestos_vigentes(ruta_config, carpeta_perfiles, nombre, diferencias(inicial, config))
    return config, nombre, cambios, supuestos, ajustes
