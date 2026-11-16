"""Piezas comunes de la interfaz: nombres y colores de cada escenario, formatos de números y el aviso único.

Todas las pantallas usan estas mismas funciones para que el dinero, los porcentajes y los km se vean
igual en todas partes, y para que cada escenario tenga siempre el mismo nombre y el mismo color.
"""

import altair as alt
import pandas as pd
import streamlit as st

from src.scenarios import ESCENARIOS, tabla_kpis

# Nombre que entiende cualquiera; el código E1-E4 queda solo como etiqueta pequeña
NOMBRES = {
    "E1": "Como se hace hoy",
    "E2": "Solo rutas optimizadas",
    "E3": "Pronóstico + rutas (propuesta)",
    "E4": "Ideal: saber el futuro",
}
# Un color por escenario, el mismo en gráficas y mapas
COLORES = {
    "E1": "#E07B39",  # naranja: la práctica actual
    "E2": "#4C78A8",  # azul: solo el modelo de rutas
    "E3": "#2E9E5B",  # verde: la propuesta
    "E4": "#9E9E9E",  # gris: el ideal, que no es alcanzable
}


def etiqueta(escenario):
    """'Como se hace hoy (E1)': para tablas, selectores y gráficas."""
    return f"{NOMBRES[escenario]} ({escenario})"


# --- Formatos: los mismos en toda la app ---

def dinero(x):
    """$12,345 (sin centavos)."""
    return f"${x:,.0f}"


def dinero_centavos(x):
    """$12.34 (para costos por unidad)."""
    return f"${x:,.2f}"


def pct(x, signo=True):
    """x en porcentaje (15.7 → '+15.7 %'). Con signo=False: '15.7 %'.

    Usa el guion normal como signo menos: st.metric lo necesita para pintar la flecha del color correcto.
    """
    return f"{x:+.1f} %" if signo else f"{x:.1f} %"


def km(x, decimales=0):
    """1,234 km (totales) o 12.3 km (por ruta, con decimales=1)."""
    return f"{x:,.{decimales}f} km"


def tonos(escenario, n):
    """n tonos del color del escenario, de oscuro a claro: así cada camioneta se distingue en el mapa
    y el mapa entero sigue diciendo de qué escenario es."""
    base = COLORES[escenario]
    rgb = [int(base[k:k + 2], 16) for k in (1, 3, 5)]
    salida = []
    for k in range(n):
        t = 0 if n == 1 else -0.45 + 0.9 * k / (n - 1)  # −0.45 = más oscuro, +0.45 = más claro
        mezcla = [c * (1 + t) if t < 0 else c + (255 - c) * t for c in rgb]
        salida.append("#" + "".join(f"{round(c):02x}" for c in mezcla))
    return salida


def titulo_escenario(escenario, extra=""):
    """■ Nombre del escenario (E1) · extra, con el cuadrito del color del escenario (encima de cada mapa)."""
    st.markdown(f"<span style='color:{COLORES[escenario]}'>■</span> **{NOMBRES[escenario]}** "
                f"<small>({escenario})</small>" + (f" · {extra}" if extra else ""), unsafe_allow_html=True)


def grafica_costos(costos, titulo):
    """Barras horizontales de costo por escenario, cada una con su color y su cifra al final.

    costos: {"E1": 7642.3, "E2": ...} (los escenarios que falten no se dibujan).
    """
    tabla = pd.DataFrame([{"Escenario": etiqueta(e), "Costo": costos[e], "Cifra": dinero(costos[e])}
                          for e in ESCENARIOS if e in costos and pd.notna(costos[e])])
    orden = list(tabla["Escenario"])
    color = alt.Color("Escenario:N", legend=None,
                      scale=alt.Scale(domain=orden, range=[COLORES[e] for e in ESCENARIOS if etiqueta(e) in orden]))
    y = alt.Y("Escenario:N", sort=orden, title=None, axis=alt.Axis(labelLimit=260))
    barras = alt.Chart(tabla).mark_bar().encode(x=alt.X("Costo:Q", title=titulo), y=y, color=color,
                                                tooltip=["Escenario", "Cifra"])
    cifras = barras.mark_text(align="left", dx=4).encode(text="Cifra:N", color=alt.value("#444"))
    st.altair_chart((barras + cifras).properties(height=200), width="stretch")


# --- Tabla de indicadores (KPI) y su glosario ---

NOMBRE_KPI = {"Costo vs E1": "Costo frente a hoy (E1)", "Costo vs E2": "Costo frente a solo rutas (E2)",
              "Km vs E1": "Km frente a hoy (E1)"}
FORMATO_KPI = {
    "Costo logístico (MXN)": dinero,
    "Costo por caja entregada (MXN)": dinero_centavos,
    "Km recorridos": lambda x: km(x, 1),
    "Utilización de la flota": lambda x: f"{x:.1%}",
    "Costo vs E1": lambda x: pct(x * 100),
    "Costo vs E2": lambda x: pct(x * 100),
    "Km vs E1": lambda x: pct(x * 100),
}

GLOSARIO_KPI = """
| Indicador | Qué mide |
|---|---|
| **Costo logístico** | km × costo por km + propias × costo propia + reservados × tarifa reservada + urgentes × tarifa urgente |
| **Costo por caja entregada** | Costo logístico ÷ cajas entregadas |
| **Km recorridos** | Distancia total de todas las rutas |
| **Utilización de la flota** | Cajas entregadas ÷ capacidad de las camionetas que salieron |
| **Camionetas propias** | Cuántas veces salió una camioneta propia (suma de todos los días) |
| **Fletes reservados** | Unidades pagadas el día anterior, se usen o no |
| **Fletes urgentes** | Unidades contratadas el mismo día (la tarifa más cara) |
| **Paradas no atendidas** | Paradas imposibles de cubrir (por ejemplo, fuera de la jornada) |
| **Costo / km frente a E1, E2** | Diferencia porcentual frente a como se hace hoy o frente a solo rutas optimizadas |
| **Del ahorro máximo posible** (brecha cerrada) | (costo E2 − costo E3) ÷ (costo E2 − costo E4): qué parte del ahorro que daría saber el futuro logra el pronóstico |
| **WAPE** | Qué tanto se equivoca el pronóstico: suma de \\|previsto − real\\| ÷ suma de lo real (ec. 1 del artículo) |
| **Sesgo** | Si el pronóstico tiende a pasarse (+) o a quedarse corto (−) |
| **Días con reserva exacta / de más / de menos** | Reserva de E3 frente a lo que hizo falta (urgentes de E2) |
"""


def con_unidad(texto, unidad, singular):
    """Cambia "cajas"/"caja" por la unidad de la empresa en un texto fijo de la app."""
    return (texto.replace("Cajas", unidad.capitalize()).replace("cajas", unidad)
            .replace("Caja", singular.capitalize()).replace("caja", singular))


def mostrar_kpis(resumen, unidad, singular):
    """Tabla de indicador × escenario, cada fila en su formato (dinero, %, km...)."""
    tabla = tabla_kpis(resumen)
    texto = pd.DataFrame(index=tabla.index, columns=[etiqueta(e) for e in tabla.columns])
    for kpi, fila in tabla.iterrows():
        formato = FORMATO_KPI.get(kpi, lambda x: f"{x:,.0f}")
        texto.loc[kpi] = ["—" if pd.isna(v) else formato(v) for v in fila]
    texto.index = [con_unidad(NOMBRE_KPI.get(k, k), unidad, singular) for k in texto.index]
    st.dataframe(texto, column_config={"_index": st.column_config.Column("Indicador", width="medium")})


def glosario(unidad, singular):
    with st.expander("¿Qué significa cada número?"):
        st.markdown(con_unidad(GLOSARIO_KPI, unidad, singular))


# --- El aviso único de cada página ---

DEMOSTRACION = ("🧪 Demostración con el caso de estudio del artículo (simulación de una distribuidora urbana: "
                "560 clientes, 8 zonas)")


def encabezado(demostracion, frases=(), supuestos=(), avisos_datos=()):
    """El aviso único: una línea gris con lo que hay que saber y el detalle plegado debajo.

    demostracion: True con el caso de estudio (agrega la línea 🧪).
    frases: textos cortos (parámetros propios, valores supuestos...).
    supuestos: renglones (parámetro, valor, nota) de los valores que siguen siendo supuestos.
    avisos_datos: avisos de la carga de los archivos (columnas, pedidos partidos...).
    """
    partes = ([DEMOSTRACION] if demostracion else []) + [f for f in frases if f]
    if partes:
        st.caption(" · ".join(partes))
    if not supuestos and not avisos_datos:
        return
    cuenta = ([f"{len(supuestos)} valores supuestos"] if supuestos else []) + \
             ([f"{len(avisos_datos)} avisos de los datos"] if avisos_datos else [])
    with st.expander(f"¿Qué es real y qué es supuesto? ({', '.join(cuenta)})"):
        if supuestos:
            st.markdown("Estos parámetros son **supuestos** (la mayoría de la Tabla 5 del artículo). Los resultados "
                        "cambian cuando se reemplacen por los de tu empresa en la barra lateral.")
            st.dataframe(pd.DataFrame(list(supuestos), columns=["Parámetro", "Valor", "Nota"]), hide_index=True)
        for texto in avisos_datos:
            st.caption(f"ℹ️ {texto}")
