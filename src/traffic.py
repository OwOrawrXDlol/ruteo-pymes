"""Tráfico: convierte los tiempos "sin tráfico" en tiempos "con tráfico".

Hay dos fuentes posibles, y las dos entregan lo mismo (un objeto `Trafico`):
  - modo B, simulado: el usuario coloca incidentes en el mapa (o los toma de
    data/incidentes_ejemplo.csv). No necesita internet. Está en este archivo.
  - modo A, TomTom en vivo: rutas, incidentes y matriz con tráfico (abajo, y
    src/tomtom.py para las consultas). Necesita la clave en .env.

El resto de la app solo usa `Trafico.minutos` (la matriz con tráfico) y
`Trafico.incidentes` (para dibujarlos), sin saber de qué modo vinieron.

Cómo afecta un incidente simulado a un tramo i→j:
  - se traza el segmento recto entre el punto i y el punto j;
  - si el incidente queda a menos de su radio de ese segmento, el tramo está afectado;
  - el tiempo del tramo se multiplica por el factor de lentitud del incidente
    (2.5 = tarda 2.5 veces más). Si varios incidentes tocan el mismo tramo, se usa
    el factor más alto (no se multiplican entre sí, para no exagerar);
  - si el incidente es "cerrado", el tramo cuesta `minutos_cerrado` (más que la
    jornada completa), así que en la práctica ninguna ruta lo usa.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from src.distances import RADIO_TIERRA_KM
from src.tomtom import transacciones_matrix

# Tiempo de un tramo cerrado si config.yaml no dice otra cosa (más que una jornada de 9 h)
MINUTOS_CERRADO = 1000.0

TIPOS_INCIDENTE = ["accidente", "obra", "cierre", "embotellamiento", "inundación", "otro"]


@dataclass
class Incidente:
    """Un incidente de tráfico en un punto. Sirve igual para los simulados y los de TomTom."""
    lat: float
    lon: float
    tipo: str = "accidente"
    radio_m: float = 300.0       # simulado: distancia a la que afecta a un tramo
    factor: float = 2.5          # simulado: cuántas veces más tarda un tramo afectado
    cerrado: bool = False        # simulado: el tramo no se puede usar
    magnitud: int | None = None  # 0 a 4 como en TomTom (3 = mayor, 4 = cierre); None = calcularla
    retraso_seg: float | None = None  # retraso que reporta TomTom (None en los simulados)
    fuente: str = "simulado"     # "simulado" o "tomtom"
    descripcion: str = ""
    puntos: list | None = None   # TomTom: [(lat, lon), ...] del tramo de calle afectado; None = solo el punto

    def todos_los_puntos(self):
        """Puntos del incidente: el tramo completo si se conoce, o solo (lat, lon)."""
        return self.puntos or [(self.lat, self.lon)]

    def __post_init__(self):
        if self.radio_m <= 0:
            raise ValueError(f"El radio del incidente debe ser mayor que 0 m (vino {self.radio_m}).")
        if self.factor < 1:
            raise ValueError(f"El factor de lentitud debe ser 1 o más (vino {self.factor}); "
                             "menos de 1 querría decir que el tráfico acelera.")
        if self.magnitud is None:
            self.magnitud = magnitud_simulada(self.factor, self.cerrado)


def magnitud_simulada(factor, cerrado):
    """Traduce un incidente simulado a la escala 0-4 de TomTom, para tratar a ambos igual.

    1 = menor, 2 = moderado, 3 = mayor, 4 = cierre.
    """
    if cerrado:
        return 4
    if factor >= 2.5:
        return 3
    if factor >= 1.5:
        return 2
    return 1


@dataclass
class Trafico:
    """Resultado común de cualquier fuente de tráfico."""
    minutos: np.ndarray          # [i][j] = minutos de manejo con tráfico
    minutos_base: np.ndarray     # [i][j] = minutos de manejo sin tráfico (los del plan)
    incidentes: list = field(default_factory=list)
    fuente: str = "simulado"     # "simulado" o "tomtom"
    consultado: datetime | None = None  # hora de la consulta (solo TomTom)
    medidos: np.ndarray | None = None   # TomTom: tramos medidos (los demás se estiman con los incidentes)
    rutas_tomtom: list = field(default_factory=list)  # TomTom: resumen de la revisión de cada camioneta
    avisos: list = field(default_factory=list)

    @property
    def afectados(self):
        """Matriz de verdadero/falso: qué tramos tardan más que sin tráfico."""
        return self.minutos > self.minutos_base + 1e-9


# --- Geometría ---

def _a_metros(lats, lons, lat0, lon0):
    """Pasa coordenadas a metros (x hacia el este, y hacia el norte) alrededor del punto (lat0, lon0).

    Es una aproximación plana; para distancias de una ciudad el error es menor al 1 %.
    """
    r = RADIO_TIERRA_KM * 1000
    x = np.radians(np.asarray(lons, dtype=float) - lon0) * np.cos(np.radians(lat0)) * r
    y = np.radians(np.asarray(lats, dtype=float) - lat0) * r
    return x, y


def distancia_punto_segmento_m(lat, lon, lat_a, lon_a, lat_b, lon_b):
    """Distancia en metros del punto (lat, lon) al segmento recto A-B.

    Acepta arreglos para A y B (se calcula para todos los segmentos a la vez).
    Idea: se busca el punto del segmento más cercano al punto dado; si la
    perpendicular cae fuera del segmento, el más cercano es uno de sus extremos.
    """
    # Con el punto como origen, el problema queda en "distancia del origen al segmento"
    ax, ay = _a_metros(lat_a, lon_a, lat, lon)
    bx, by = _a_metros(lat_b, lon_b, lat, lon)
    dx, dy = bx - ax, by - ay
    largo2 = dx * dx + dy * dy
    # t = en qué parte del segmento cae la perpendicular (0 = en A, 1 = en B)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(largo2 > 0, -(ax * dx + ay * dy) / largo2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(cx, cy)


def distancias_a_tramos_m(incidente, lats, lons):
    """Matriz [i][j] = metros del incidente al segmento recto entre el punto i y el j.

    Si el incidente es un tramo de calle (varios puntos), se toma el punto más cercano.
    """
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    return np.min([distancia_punto_segmento_m(lat, lon, lats[:, None], lons[:, None], lats[None, :], lons[None, :])
                   for lat, lon in incidente.todos_los_puntos()], axis=0)


# --- Modo B: incidentes simulados ---

def trafico_simulado(lats, lons, minutos_base, incidentes, minutos_cerrado=MINUTOS_CERRADO):
    """Aplica los incidentes simulados a la matriz de tiempos sin tráfico.

    lats, lons: coordenadas de los puntos, en el mismo orden que la matriz.
    minutos_base: matriz de minutos sin tráfico (no se modifica).
    """
    base = np.asarray(minutos_base, dtype=float)
    n = len(lats)
    if base.shape != (n, n):
        raise ValueError(f"La matriz de tiempos es de {base.shape} pero hay {n} puntos.")

    factor = np.ones((n, n))
    cerrado = np.zeros((n, n), dtype=bool)
    for inc in incidentes:
        toca = distancias_a_tramos_m(inc, lats, lons) <= inc.radio_m
        factor = np.where(toca, np.maximum(factor, inc.factor), factor)
        if inc.cerrado:
            cerrado |= toca

    minutos = base * factor
    minutos = np.where(cerrado, np.maximum(minutos, minutos_cerrado), minutos)
    np.fill_diagonal(minutos, np.diag(base))  # quedarse en el mismo punto no cambia
    return Trafico(minutos=minutos, minutos_base=base, incidentes=list(incidentes), fuente="simulado")


def cargar_incidentes(ruta, radio_m=300.0, factor=2.5):
    """Lee incidentes simulados de un CSV (por ejemplo data/incidentes_ejemplo.csv).

    Columnas obligatorias: lat, lon. Opcionales: nombre, tipo, radio_m, factor, cerrado.
    Si falta radio_m o factor se usan los valores dados (los de config.yaml).
    """
    tabla = pd.read_csv(ruta)
    faltan = {"lat", "lon"} - set(tabla.columns)
    if faltan:
        raise ValueError(f"{ruta}: faltan las columnas {sorted(faltan)}.")
    incidentes = []
    for n, fila in enumerate(tabla.to_dict("records"), start=2):  # renglón 1 = encabezados
        try:
            incidentes.append(Incidente(
                lat=float(fila["lat"]),
                lon=float(fila["lon"]),
                tipo=str(fila.get("tipo") or "accidente"),
                radio_m=float(fila.get("radio_m") if pd.notna(fila.get("radio_m")) else radio_m),
                factor=float(fila.get("factor") if pd.notna(fila.get("factor")) else factor),
                cerrado=_a_booleano(fila.get("cerrado", False)),
                descripcion=str(fila.get("nombre") or ""),
            ))
        except ValueError as error:
            raise ValueError(f"{ruta}, renglón {n}: {error}") from error
    return incidentes


def _a_booleano(valor):
    """Acepta sí/si/true/1/x como verdadero; vacío como falso."""
    if pd.isna(valor):
        return False
    return str(valor).strip().lower() in {"sí", "si", "true", "1", "x", "verdadero"}


# --- Modo A: TomTom en vivo ---
#
# TomTom mide tiempos por calles reales; el plan usa línea recta × circuidad a velocidad fija.
# Para que se puedan comparar, de TomTom se toma solo CUÁNTO MÁS LENTO va cada tramo:
#     razón = tiempo con tráfico / tiempo sin tráfico   (sin tráfico = con tráfico − retraso)
# y esa razón se aplica al tiempo del modelo: minutos[i][j] = minutos_base[i][j] × razón.
# Los tramos que TomTom no midió se estiman con sus incidentes, igual que en el modo simulado.

TIPOS_TOMTOM = {0: "otro", 1: "accidente", 2: "niebla", 3: "condiciones peligrosas", 4: "lluvia",
                5: "hielo", 6: "embotellamiento", 7: "carril cerrado", 8: "cierre", 9: "obra",
                10: "viento", 11: "inundación", 14: "vehículo descompuesto"}


def _razon(segundos_con_trafico, retraso_seg):
    sin_trafico = max(segundos_con_trafico - retraso_seg, 1.0)
    return max(segundos_con_trafico / sin_trafico, 1.0)


def incidentes_de_tomtom(respuesta, config):
    """Convierte la respuesta de Incident Details v5 en objetos Incidente."""
    tt = config["tomtom"]
    factores = {int(k): float(v) for k, v in tt["factor_por_magnitud"].items()}
    incidentes = []
    for item in respuesta.get("incidents") or []:
        geometria = item.get("geometry") or {}
        props = item.get("properties") or {}
        coords = geometria.get("coordinates") or []
        if geometria.get("type") == "Point":
            coords = [coords]
        puntos = [(float(c[1]), float(c[0])) for c in coords if len(c) >= 2]  # TomTom da [lon, lat]
        if not puntos:
            continue
        if len(puntos) > 20:  # basta con unos cuantos puntos del tramo
            paso = len(puntos) / 20
            puntos = [puntos[int(k * paso)] for k in range(20)] + [puntos[-1]]
        categoria = int(props.get("iconCategory") or 0)
        magnitud = int(props.get("magnitudeOfDelay") or 0)
        eventos = "; ".join(e.get("description", "") for e in props.get("events") or [] if e.get("description"))
        lugar = " → ".join(x for x in (props.get("from"), props.get("to")) if x)
        centro = puntos[len(puntos) // 2]
        incidentes.append(Incidente(
            lat=centro[0], lon=centro[1],
            tipo=TIPOS_TOMTOM.get(categoria, "otro"),
            radio_m=float(tt["radio_estimado_m"]),
            factor=factores.get(magnitud, 1.0),
            cerrado=categoria == 8,
            magnitud=magnitud,
            retraso_seg=float(props["delay"]) if props.get("delay") is not None else None,
            fuente="tomtom",
            descripcion=" · ".join(x for x in (eventos, lugar) if x),
            puntos=puntos,
        ))
    return incidentes


def recuadro(lats, lons, margen_km):
    """(min_lon, min_lat, max_lon, max_lat) que cubre todos los puntos, con un margen."""
    lat0 = float(np.mean(lats))
    dlat = margen_km / 111.195
    dlon = margen_km / (111.195 * np.cos(np.radians(lat0)))
    return (min(lons) - dlon, min(lats) - dlat, max(lons) + dlon, max(lats) + dlat)


def trafico_tomtom(lats, lons, minutos_base, estados, cliente, config):
    """Revisión barata: 1 consulta de incidentes + 1 ruta por camioneta con paradas pendientes.

    estados: lista de replan.EstadoCamioneta (dónde va cada una y qué le falta).
    """
    base = np.asarray(minutos_base, dtype=float)
    tt = config["tomtom"]
    incidentes = incidentes_de_tomtom(cliente.incidentes(*recuadro(lats, lons, tt["margen_bbox_km"])), config)
    # Estimación para todos los tramos con los incidentes; luego se sobrescribe lo que TomTom mida
    estimado = trafico_simulado(lats, lons, base, incidentes, config["trafico"]["simulado"]["minutos_cerrado"])
    minutos = estimado.minutos.copy()
    medidos = np.zeros(base.shape, dtype=bool)
    np.fill_diagonal(medidos, True)

    resumen = []
    for e in estados:
        if not e.pendientes:
            continue
        nodos = [e.inicio] + list(e.pendientes) + [0]
        respuesta = cliente.ruta([(lats[n], lons[n]) for n in nodos])
        ruta = (respuesta.get("routes") or [None])[0]
        if ruta is None:
            continue
        for (a, b), tramo in zip(zip(nodos, nodos[1:]), ruta.get("legs") or []):
            s = tramo["summary"]
            minutos[a][b] = base[a][b] * _razon(s["travelTimeInSeconds"], s.get("trafficDelayInSeconds", 0))
            medidos[a][b] = True
        s = ruta["summary"]
        resumen.append({
            "vehiculo": e.vehiculo,
            "min_con_trafico": s["travelTimeInSeconds"] / 60,
            "min_sin_trafico": s.get("noTrafficTravelTimeInSeconds",
                                     s["travelTimeInSeconds"] - s.get("trafficDelayInSeconds", 0)) / 60,
            "retraso_min": s.get("trafficDelayInSeconds", 0) / 60,
            "km": s.get("lengthInMeters", 0) / 1000,
        })

    return Trafico(minutos=minutos, minutos_base=base, incidentes=incidentes, fuente="tomtom",
                   consultado=datetime.now(), medidos=medidos, rutas_tomtom=resumen)


def _bloques(nodos, tamano):
    return [nodos[k:k + tamano] for k in range(0, len(nodos), tamano)]


def costo_matriz(n_nodos):
    """(llamadas, transacciones) para medir todos los tramos entre n nodos en bloques de 10 × 10."""
    bloques = _bloques(list(range(n_nodos)), 10)
    return (len(bloques) ** 2,
            sum(transacciones_matrix(len(o), len(d)) for o in bloques for d in bloques))


def medir_con_matrix(trafico, lats, lons, nodos, cliente, config):
    """Mide con Matrix Routing v2 todos los tramos entre los nodos dados y actualiza `trafico`.

    Se parte en bloques de 10 × 10 (100 celdas, el máximo con tráfico en vivo).
    Si costaría más de `max_transacciones_matrix` o de lo que queda de la cuota del mes, no
    se consulta y se deja un aviso: esos tramos se quedan con la estimación por incidentes.
    Devuelve las transacciones usadas.
    """
    tt = config["tomtom"]
    nodos = list(nodos)
    llamadas, costo = costo_matriz(len(nodos))
    queda = tt["limite_mensual"]["matrix"] - cliente.uso()["matrix"]["mes"]
    if costo > tt["max_transacciones_matrix"] or costo > queda:
        motivo = (f"el tope por cálculo es {tt['max_transacciones_matrix']}" if costo > tt["max_transacciones_matrix"]
                  else f"solo quedan {queda} este mes")
        trafico.avisos.append(f"No se consultó Matrix para {len(nodos)} puntos: costaría {costo} transacciones "
                              f"y {motivo}. Esos tramos usan la estimación por incidentes de TomTom.")
        return 0
    usadas = 0
    for origenes in _bloques(nodos, 10):
        for destinos in _bloques(nodos, 10):
            respuesta = cliente.matriz([(lats[n], lons[n]) for n in origenes],
                                       [(lats[n], lons[n]) for n in destinos])
            usadas += transacciones_matrix(len(origenes), len(destinos))
            for celda in respuesta.get("data") or []:
                a, b = origenes[celda["originIndex"]], destinos[celda["destinationIndex"]]
                resumen = celda.get("routeSummary")
                if a == b or not resumen:
                    continue  # celda fallida: se queda la estimación
                trafico.minutos[a][b] = trafico.minutos_base[a][b] * _razon(
                    resumen["travelTimeInSeconds"], resumen.get("trafficDelayInSeconds", 0))
                trafico.medidos[a][b] = True
    return usadas
