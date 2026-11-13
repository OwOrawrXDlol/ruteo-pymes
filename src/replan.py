"""Decide qué camionetas conviene reoptimizar por tráfico y, si conviene, les da una ruta nueva.

Para cada camioneta, en este orden (los umbrales están en config.yaml, sección "trafico"):

1. Detección: se marca como afectada si cumple cualquiera de estas
   - retraso: el tráfico le agrega al menos umbral_retraso_pct % de su manejo pendiente
     Y al menos umbral_retraso_min minutos (las dos a la vez, para no reaccionar a
     retrasos chicos en rutas cortas ni a porcentajes chicos en rutas largas);
   - incidente: hay un incidente de magnitud >= magnitud_minima a menos de
     radio_incidente_m de alguno de sus tramos pendientes;
   - jornada: con tráfico ya no termina dentro de la jornada.
2. Reoptimización: solo las afectadas, con los tiempos con tráfico. Las no afectadas no se
   tocan. Las afectadas que todavía no salen del almacén se reoptimizan juntas y pueden
   pasarse paradas, sin pasar de la capacidad (aún se decide qué cajas sube cada una).
   Las que ya van en ruta traen cargadas sus cajas: cada una solo reordena las suyas.
3. Aceptación: el plan nuevo se adopta solo si ahorra al menos mejora_minima_min minutos
   frente a seguir con el plan actual bajo el mismo tráfico. Excepción: si con el plan
   actual se pasa de la jornada y con el nuevo no, se adopta aunque ahorre poco
   (la jornada es una regla, no una preferencia).
4. Restricciones: nunca se pasa de la capacidad ni de la jornada. Si aun reoptimizando
   quedan paradas que no caben, se reporta "requiere urgente" con esas paradas.

Numeración de puntos (igual que en vrp.py): nodo 0 = almacén, nodo i+1 = parada i.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.distances import matriz_distancias, matriz_tiempos_min
from src.scenarios import costo_dia
from src.traffic import distancia_punto_segmento_m, medir_con_matrix, trafico_tomtom
from src.vrp import resolver_dia

CASTIGO_NO_ATENDIDA = 10**9

MANTENER = "mantener"
CAMBIAR = "cambiar"
URGENTE = "requiere urgente"


@dataclass
class EstadoCamioneta:
    """Dónde va una camioneta y qué le falta. Los nodos usan la numeración de arriba."""
    vehiculo: str
    tipo: str
    inicio: int                     # nodo donde está ahora: 0 = almacén, o su última parada visitada
    pendientes: list                # nodos que le faltan, en el orden planeado
    visitadas: list = field(default_factory=list)  # nodos ya entregados (no se tocan)
    minutos_transcurridos: float = 0.0             # tiempo de jornada ya usado
    en_ruta: bool = False           # ya salió del almacén con sus cajas cargadas


def puntos_del_dia(paradas, config):
    """Coordenadas, km y minutos sin tráfico de almacén + paradas, en la numeración de nodos."""
    lats = [config["almacen"]["lat"]] + list(paradas["lat"])
    lons = [config["almacen"]["lon"]] + list(paradas["lon"])
    km = matriz_distancias(lats, lons, config["distancias"]["factor_circuidad"])
    minutos = matriz_tiempos_min(km, config["tiempos"]["velocidad_kmh"])
    return lats, lons, km, minutos


def estados_al_salir(plan):
    """Estado de cada camioneta del plan antes de salir del almacén (nada entregado aún)."""
    return [EstadoCamioneta(r["vehiculo"], r["tipo"], 0, [p + 1 for p in r["paradas"]])
            for r in plan["rutas"]]


# --- Hora actual (posición simulada: no hay GPS de las camionetas) ---

def _a_minutos(hora):
    """'08:30' -> 510 (minutos desde la medianoche)."""
    horas, minutos = str(hora).split(":")
    return int(horas) * 60 + int(minutos)


def horario(config):
    """(salida, fin de jornada) en minutos desde la medianoche, p. ej. (480, 1020) = 8:00 a 17:00."""
    salida = _a_minutos(config["tiempos"]["hora_salida"])
    return salida, salida + int(round(config["tiempos"]["jornada_h"] * 60))


def texto_hora(minutos_del_dia):
    """510 -> '08:30'."""
    minutos_del_dia = int(round(minutos_del_dia))
    return f"{minutos_del_dia // 60:02d}:{minutos_del_dia % 60:02d}"


def estados_a_la_hora(plan, minutos_base, servicio_min, minutos_desde_salida):
    """Dónde va cada camioneta a cierta hora, suponiendo que todo salió según el plan sin tráfico.

    Todas salen del almacén al mismo tiempo (minuto 0). Una parada cuenta como visitada
    cuando la camioneta ya llegó y terminó de descargar. La camioneta "reinicia" desde su
    última parada visitada, con el reloj en el momento en que terminó ahí. Si a esa hora
    iba manejando hacia la siguiente, se toma como si siguiera en la última: es una
    simplificación, porque no hay GPS para saber dónde va de verdad.
    minutos_desde_salida = 0 equivale a replanear antes de salir.
    """
    if minutos_desde_salida <= 0:
        return estados_al_salir(plan)
    estados = []
    for r in plan["rutas"]:
        nodos = [p + 1 for p in r["paradas"]]
        reloj, actual, visitadas = 0.0, 0, []
        for n in nodos:
            termina_aqui = reloj + minutos_base[actual][n] + servicio_min
            if termina_aqui > minutos_desde_salida + 1e-9:
                break
            reloj, actual = termina_aqui, n
            visitadas.append(n)
        estados.append(EstadoCamioneta(r["vehiculo"], r["tipo"], actual, nodos[len(visitadas):],
                                       visitadas, reloj, en_ruta=True))
    return estados


# --- Medir una ruta ---

def tramos(inicio, nodos):
    """Tramos de una ruta pendiente: desde donde está, por cada parada, y de regreso al almacén."""
    recorrido = [inicio] + list(nodos) + [0]
    return list(zip(recorrido, recorrido[1:]))


def minutos_manejo(inicio, nodos, minutos):
    return float(sum(minutos[a][b] for a, b in tramos(inicio, nodos)))


def km_ruta(inicio, nodos, km):
    return float(sum(km[a][b] for a, b in tramos(inicio, nodos)))


# --- 1. Detección ---

def cumple_retraso(retraso_min, manejo_sin_trafico_min, criterios):
    """True si el retraso pasa los DOS umbrales: porcentaje y minutos."""
    if manejo_sin_trafico_min <= 0:
        return False
    pct = 100 * retraso_min / manejo_sin_trafico_min
    return pct >= criterios["umbral_retraso_pct"] and retraso_min >= criterios["umbral_retraso_min"]


def incidentes_cercanos(estado, incidentes, lats, lons, criterios):
    """Incidentes de magnitud suficiente a menos de radio_incidente_m de un tramo pendiente."""
    pares = tramos(estado.inicio, estado.pendientes)
    a = np.array([p[0] for p in pares])
    b = np.array([p[1] for p in pares])
    lats, lons = np.asarray(lats), np.asarray(lons)
    cercanos = []
    for inc in incidentes:
        if inc.magnitud < criterios["magnitud_minima"]:
            continue
        distancia = min(distancia_punto_segmento_m(lat, lon, lats[a], lons[a], lats[b], lons[b]).min()
                        for lat, lon in inc.todos_los_puntos())
        if distancia <= criterios["radio_incidente_m"]:
            cercanos.append((inc, float(distancia)))
    return cercanos


def detectar(estado, trafico, lats, lons, config):
    """Mide la ruta pendiente sin y con tráfico y dice si la camioneta está afectada y por qué."""
    criterios = config["trafico"]
    servicio = config["tiempos"]["servicio_min"] * len(estado.pendientes)
    jornada = config["tiempos"]["jornada_h"] * 60

    manejo_base = minutos_manejo(estado.inicio, estado.pendientes, trafico.minutos_base)
    manejo_trafico = minutos_manejo(estado.inicio, estado.pendientes, trafico.minutos)
    retraso = manejo_trafico - manejo_base
    termina = estado.minutos_transcurridos + manejo_trafico + servicio

    razones = []
    if cumple_retraso(retraso, manejo_base, criterios):
        razones.append(f"retraso {retraso:.0f} min ({100 * retraso / manejo_base:.0f} %)")
    for inc, distancia in incidentes_cercanos(estado, trafico.incidentes, lats, lons, criterios):
        razones.append(f"incidente magnitud {inc.magnitud} ({inc.tipo}) a {distancia:.0f} m")
    if estado.pendientes and termina > jornada + 1e-6:
        razones.append(f"excede jornada ({termina / 60:.1f} h de {jornada / 60:.0f} h)")

    return {
        "min_sin_trafico": manejo_base + servicio,
        "min_con_trafico": manejo_trafico + servicio,
        "retraso_min": retraso,
        "retraso_pct": 100 * retraso / manejo_base if manejo_base > 0 else 0.0,
        "termina_min": termina,
        "afectada": bool(estado.pendientes) and bool(razones),
        "razones": razones,
    }


# --- 2. Reoptimización ---

def sin_salir(estado):
    """True si la camioneta sigue en el almacén y todavía se puede cambiar qué cajas sube."""
    return not estado.en_ruta


def agrupar_para_intercambio(estados):
    """Grupos de camionetas que se reoptimizan juntas (y por lo tanto pueden pasarse paradas).

    Las que no han salido van todas en un grupo: todavía se puede cambiar qué cajas sube cada una.
    Las que ya van en ruta van solas, cada una con lo que ya trae cargado.
    """
    en_almacen = [e for e in estados if sin_salir(e)]
    en_ruta = [[e] for e in estados if not sin_salir(e)]
    return ([en_almacen] if en_almacen else []) + en_ruta


def reoptimizar_grupo(grupo, minutos, cajas, config, segundos):
    """Reparte y ordena las paradas pendientes del grupo minimizando el tiempo con tráfico.

    minutos: matriz con tráfico (todos los nodos). cajas: cajas por nodo.
    Devuelve (lista de nodos pendientes por camioneta, nodos que no cupieron).
    """
    servicio_min = config["tiempos"]["servicio_min"]
    jornada_min = config["tiempos"]["jornada_h"] * 60
    capacidad = config["flota"]["capacidad_cajas"]

    # Modelo pequeño solo con los nodos que importan: almacén, posiciones actuales y pendientes
    pendientes = [n for e in grupo for n in e.pendientes]
    locales = [0] + sorted({e.inicio for e in grupo if e.inicio != 0}) + pendientes
    local = {nodo: i for i, nodo in enumerate(locales)}
    por_visitar = set(pendientes)
    inicios = [local[e.inicio] for e in grupo]

    gestor = pywrapcp.RoutingIndexManager(len(locales), len(grupo), inicios, [0] * len(grupo))
    modelo = pywrapcp.RoutingModel(gestor)

    def segundos_tramo(desde, hasta):
        i, j = locales[gestor.IndexToNode(desde)], locales[gestor.IndexToNode(hasta)]
        descarga = servicio_min if i in por_visitar else 0  # la descarga se cuenta al salir de la parada
        return int(round((minutos[i][j] + descarga) * 60))

    tiempo = modelo.RegisterTransitCallback(segundos_tramo)
    modelo.SetArcCostEvaluatorOfAllVehicles(tiempo)  # objetivo: menos tiempo total con tráfico
    restante = [max(0, int((jornada_min - e.minutos_transcurridos) * 60)) for e in grupo]
    modelo.AddDimensionWithVehicleCapacity(tiempo, 0, restante, True, "Tiempo")

    def cajas_nodo(desde):
        return cajas[locales[gestor.IndexToNode(desde)]] if locales[gestor.IndexToNode(desde)] in por_visitar else 0

    modelo.AddDimensionWithVehicleCapacity(
        modelo.RegisterUnaryTransitCallback(cajas_nodo), 0, [capacidad] * len(grupo), True, "Cajas")

    for nodo in pendientes:
        modelo.AddDisjunction([gestor.NodeToIndex(local[nodo])], CASTIGO_NO_ATENDIDA)

    parametros = pywrapcp.DefaultRoutingSearchParameters()
    parametros.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parametros.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parametros.time_limit.FromMilliseconds(int(segundos * 1000))
    solucion = modelo.SolveWithParameters(parametros)
    if solucion is None:
        return [[] for _ in grupo], list(pendientes)

    rutas = []
    for v in range(len(grupo)):
        nodos = []
        indice = solucion.Value(modelo.NextVar(modelo.Start(v)))
        while not modelo.IsEnd(indice):
            nodos.append(locales[gestor.IndexToNode(indice)])
            indice = solucion.Value(modelo.NextVar(indice))
        rutas.append(nodos)
    atendidas = {n for r in rutas for n in r}
    return rutas, [n for n in pendientes if n not in atendidas]


# --- 3 y 4. Aceptación y restricciones ---

def replanear(plan, paradas, trafico, config, estados=None, segundos=None):
    """Aplica los cuatro criterios a todas las camionetas del plan.

    plan: resultado de vrp.resolver_dia. paradas: la tabla de paradas del día.
    trafico: objeto Trafico calculado sobre puntos_del_dia(paradas, config).
    estados: dónde va cada camioneta; si no se da, se replanea antes de salir.
    Devuelve un diccionario con:
      tabla: una fila por camioneta con números, criterios y decisión;
      rutas: por camioneta, paradas visitadas, pendientes actuales y las que quedan adoptadas;
      no_atendidas: paradas que ni reoptimizando caben (necesitan unidad urgente).
    Las paradas se devuelven como índices de la tabla de paradas (nodo - 1).
    """
    lats, lons, km, _ = puntos_del_dia(paradas, config)
    if trafico.minutos.shape != (len(lats), len(lats)):
        raise ValueError("La matriz de tráfico no corresponde a las paradas de este día.")
    if estados is None:
        estados = estados_al_salir(plan)
    if segundos is None:
        segundos = config["trafico"]["segundos_reoptimizar"]
    criterios = config["trafico"]
    jornada = config["tiempos"]["jornada_h"] * 60
    servicio_min = config["tiempos"]["servicio_min"]
    cajas = [0] + [int(b) for b in paradas["boxes"]]

    deteccion = [detectar(e, trafico, lats, lons, config) for e in estados]
    filas = []
    adoptadas = [list(e.pendientes) for e in estados]
    nuevas = [list(e.pendientes) for e in estados]  # lo que propuso el solver (aunque no se adopte)
    decision = [MANTENER] * len(estados)
    nota = ["no afectada" if not d["razones"] else "" for d in deteccion]
    for i, e in enumerate(estados):
        if not e.pendientes:
            nota[i] = "ya entregó todo; solo le falta regresar al almacén" if e.en_ruta else "sin paradas"
    no_atendidas = []

    afectadas = [e for e, d in zip(estados, deteccion) if d["afectada"]]
    indice = {id(e): i for i, e in enumerate(estados)}
    for grupo in agrupar_para_intercambio(afectadas):
        ids = [indice[id(e)] for e in grupo]
        rutas, sobran = reoptimizar_grupo(grupo, trafico.minutos, cajas, config, segundos)

        def total(i, nodos):
            return minutos_manejo(estados[i].inicio, nodos, trafico.minutos) + servicio_min * len(nodos)

        actual = sum(deteccion[i]["min_con_trafico"] for i in ids)
        actual_cabe = all(deteccion[i]["termina_min"] <= jornada + 1e-6 for i in ids)
        propuesto = sum(total(i, r) for i, r in zip(ids, rutas))
        ahorro = actual - propuesto

        if sobran and actual_cabe:
            # El solver no encontró algo mejor que lo que ya cabía: seguir con el plan actual
            rutas, sobran, propuesto, ahorro = [estados[i].pendientes for i in ids], [], actual, 0.0
        for i, r in zip(ids, rutas):
            nuevas[i] = list(r)

        if sobran:
            veredicto = URGENTE
            texto = (f"aun reoptimizando, {len(sobran)} parada(s) no caben en la jornada o capacidad: "
                     "requieren unidad urgente")
            no_atendidas.extend(sobran)
        elif not actual_cabe:
            veredicto = CAMBIAR
            texto = f"con el plan actual excede la jornada; el nuevo sí cabe (ahorra {ahorro:.0f} min)"
        elif ahorro >= criterios["mejora_minima_min"]:
            veredicto = CAMBIAR
            texto = f"ahorra {ahorro:.0f} min (mínimo {criterios['mejora_minima_min']} min)"
        else:
            veredicto = MANTENER
            texto = (f"ahorraría solo {max(ahorro, 0):.1f} min, menos que el mínimo de "
                     f"{criterios['mejora_minima_min']} min")
        if len(grupo) > 1:
            texto += f" · reoptimizadas juntas: {', '.join(estados[j].vehiculo for j in ids)}"
        for i in ids:
            decision[i] = veredicto
            nota[i] = texto
            if veredicto != MANTENER:
                adoptadas[i] = list(nuevas[i])

    for i, (e, d) in enumerate(zip(estados, deteccion)):
        min_nuevo = minutos_manejo(e.inicio, nuevas[i], trafico.minutos) + servicio_min * len(nuevas[i])
        filas.append({
            "vehiculo": e.vehiculo,
            "pendientes": len(e.pendientes),
            "min_sin_trafico": d["min_sin_trafico"],
            "min_con_trafico": d["min_con_trafico"],
            "retraso_min": d["retraso_min"],
            "retraso_pct": d["retraso_pct"],
            "afectada": d["afectada"],
            "criterios": "; ".join(d["razones"]) if d["afectada"] else "",
            "min_plan_nuevo": min_nuevo if d["afectada"] else None,
            "ahorro_min": d["min_con_trafico"] - min_nuevo if d["afectada"] else None,
            "decision": decision[i],
            "nota": nota[i],
            "km_actual": km_ruta(e.inicio, e.pendientes, km),
            "km_adoptado": km_ruta(e.inicio, adoptadas[i], km),
        })

    rutas = [{
        "vehiculo": e.vehiculo,
        "tipo": e.tipo,
        "visitadas": [n - 1 for n in e.visitadas],
        "pendientes_actuales": [n - 1 for n in e.pendientes],
        "pendientes_adoptadas": [n - 1 for n in adoptadas[i]],
        "cambia": decision[i] != MANTENER,
    } for i, e in enumerate(estados)]
    return {"tabla": pd.DataFrame(filas), "rutas": rutas, "no_atendidas": [n - 1 for n in no_atendidas]}


# --- Resumen de km y costo antes y después ---

def resumen_costos(escenario, replan, paradas, config, segundos=1):
    """Km y costo del día antes y después de aplicar las decisiones, con la ec. de costo del proyecto.

    escenario: resultado del escenario cuyas rutas se replanearon (km, propias, reservadas,
    urgentes, costo). replan: resultado de replanear().
    - Km después = km del plan − km pendientes que se dejaron + km pendientes adoptados.
    - Una camioneta que antes de salir se queda sin paradas ya no sale (no se paga).
    - Las reservadas ya se pagaron: su tarifa no cambia.
    - Las paradas que "requieren urgente" se rutean con unidades urgentes (con vrp.resolver_dia y
      tiempos sin tráfico, porque saldrían después, cuando ya no se sabe cómo estará el tráfico).
    """
    tabla = replan["tabla"]
    km = escenario["km"] - tabla["km_actual"].sum() + tabla["km_adoptado"].sum()
    propias, reservadas, urgentes = escenario["propias"], escenario["reservadas"], escenario["urgentes"]
    for r in replan["rutas"]:
        if r["pendientes_actuales"] and not r["visitadas"] and not r["pendientes_adoptadas"]:
            if r["tipo"] == "propia":
                propias -= 1
            elif r["tipo"] == "urgente" and urgentes > 0:
                urgentes -= 1

    urgentes_extra, km_extra = 0, 0.0
    if replan["no_atendidas"]:
        solo_urgentes = {**config, "flota": {**config["flota"], "camionetas_propias": 0}}
        pendientes = paradas.iloc[replan["no_atendidas"]].reset_index(drop=True)
        extra = resolver_dia(pendientes, solo_urgentes, segundos=segundos)
        urgentes_extra, km_extra = extra["urgentes"], extra["km"]

    despues = {
        "km": float(km + km_extra),
        "propias": propias,
        "reservadas": reservadas,
        "urgentes": urgentes + urgentes_extra,
    }
    despues["costo"] = costo_dia(despues["km"], despues["propias"], despues["reservadas"],
                                 despues["urgentes"], config)
    antes = {k: escenario[k] for k in ("km", "propias", "reservadas", "urgentes", "costo")}
    return {"antes": antes, "despues": despues, "urgentes_extra": urgentes_extra}


# --- Modo A: TomTom en vivo ---

def replanear_con_tomtom(plan, paradas, config, estados, cliente, segundos=None):
    """Revisión con TomTom y, solo para las afectadas, matriz con tráfico de sus paradas pendientes.

    1. Revisión barata: incidentes (1 consulta) + 1 ruta por camioneta (src.traffic.trafico_tomtom).
    2. Detección con esos tiempos.
    3. Matrix Routing v2 solo para los puntos de las afectadas (de cada grupo que se reoptimiza junto).
    4. replanear() con la matriz completa.
    Devuelve (resultado de replanear, objeto Trafico usado).
    """
    lats, lons, _, minutos = puntos_del_dia(paradas, config)
    trafico = trafico_tomtom(lats, lons, minutos, estados, cliente, config)
    afectadas = [e for e in estados if detectar(e, trafico, lats, lons, config)["afectada"]]
    for grupo in agrupar_para_intercambio(afectadas):
        nodos = sorted({0} | {e.inicio for e in grupo} | {n for e in grupo for n in e.pendientes})
        medir_con_matrix(trafico, lats, lons, nodos, cliente, config)
    return replanear(plan, paradas, trafico, config, estados=estados, segundos=segundos), trafico
