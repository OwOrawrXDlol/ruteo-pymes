"""Ruteo de un día con Google OR-Tools (CVRP).

CVRP = problema de ruteo de vehículos con capacidad: dado un almacén, un grupo
de clientes con cajas por entregar y varias camionetas con capacidad limitada,
decidir qué camioneta visita a qué clientes y en qué orden, para que el costo
total sea el menor posible.

Aquí cada camioneta además tiene una jornada máxima (manejo + tiempo de
descarga en cada parada) y un costo fijo si sale:
  - propia:    costo_propia (operador, seguro, desgaste)
  - reservada: 0, porque ya se pagó el día anterior
  - urgente:   tarifa_urgente
Por eso el solver solo usa urgentes cuando las demás no alcanzan.
"""

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.distances import matriz_distancias, matriz_tiempos_min

# OR-Tools solo trabaja con enteros: los costos se pasan en centavos y los tiempos en segundos
CENTAVOS = 100
# Castigo por dejar una parada sin atender (enorme, para que solo pase si es imposible atenderla)
CASTIGO_NO_ATENDIDA = 10**9


def _lista_vehiculos(config, reservadas, max_urgentes):
    """Una entrada por camioneta disponible: (tipo, nombre, costo fijo en MXN)."""
    costos = config["costos"]
    vehiculos = []
    for i in range(config["flota"]["camionetas_propias"]):
        vehiculos.append(("propia", f"Propia {i + 1}", costos["costo_propia"]))
    for i in range(reservadas):
        vehiculos.append(("reservada", f"Reservada {i + 1}", 0))
    for i in range(max_urgentes):
        vehiculos.append(("urgente", f"Urgente {i + 1}", costos["tarifa_urgente"]))
    return vehiculos


def resolver_dia(paradas, config, reservadas=0, segundos=None, max_urgentes=None):
    """Arma las rutas de un día.

    paradas: tabla con lat, lon y boxes (una fila por parada, índice 0..n-1),
             como la que devuelve load.paradas_del_dia.
    reservadas: camionetas de terceros reservadas el día anterior.
    Devuelve un diccionario con las rutas, las paradas que no se pudieron atender y los totales.
    """
    capacidad = config["flota"]["capacidad_cajas"]
    if max_urgentes is None:
        max_urgentes = config["flota"]["max_urgentes"]
    if segundos is None:
        segundos = config["solver"]["segundos_por_dia"]
    vehiculos = _lista_vehiculos(config, reservadas, max_urgentes)

    resultado = {"rutas": [], "no_atendidas": [], "km": 0.0, "cajas": 0,
                 "propias": 0, "reservadas_usadas": 0, "urgentes": 0}
    if len(paradas) == 0:
        return resultado

    # Nodo 0 = almacén; nodo i+1 = parada i
    lats = [config["almacen"]["lat"]] + list(paradas["lat"])
    lons = [config["almacen"]["lon"]] + list(paradas["lon"])
    km = matriz_distancias(lats, lons, config["distancias"]["factor_circuidad"])
    minutos = matriz_tiempos_min(km, config["tiempos"]["velocidad_kmh"])
    servicio_min = config["tiempos"]["servicio_min"]
    cajas = [0] + [int(b) for b in paradas["boxes"]]

    gestor = pywrapcp.RoutingIndexManager(len(lats), len(vehiculos), 0)
    modelo = pywrapcp.RoutingModel(gestor)

    # Costo de recorrer un tramo = km × costo por km (en centavos)
    costo_km = config["costos"]["costo_km"]

    def costo_tramo(desde, hasta):
        i, j = gestor.IndexToNode(desde), gestor.IndexToNode(hasta)
        return int(round(km[i][j] * costo_km * CENTAVOS))

    modelo.SetArcCostEvaluatorOfAllVehicles(modelo.RegisterTransitCallback(costo_tramo))

    # Costo fijo de cada camioneta, solo si sale del almacén
    for v, (_tipo, _nombre, costo_fijo) in enumerate(vehiculos):
        modelo.SetFixedCostOfVehicle(int(costo_fijo * CENTAVOS), v)

    # Capacidad: las cajas cargadas no pueden pasar de la capacidad de la camioneta
    def cajas_parada(desde):
        return cajas[gestor.IndexToNode(desde)]

    modelo.AddDimensionWithVehicleCapacity(
        modelo.RegisterUnaryTransitCallback(cajas_parada), 0, [capacidad] * len(vehiculos), True, "Cajas")

    # Jornada: manejo + descarga en cada parada no puede pasar de la jornada máxima
    def segundos_tramo(desde, hasta):
        i, j = gestor.IndexToNode(desde), gestor.IndexToNode(hasta)
        descarga = servicio_min if i != 0 else 0
        return int(round((minutos[i][j] + descarga) * 60))

    jornada_seg = int(config["tiempos"]["jornada_h"] * 3600)
    modelo.AddDimension(modelo.RegisterTransitCallback(segundos_tramo), 0, jornada_seg, True, "Tiempo")

    # Permitir dejar una parada sin atender, con un castigo enorme, para que el solver siempre
    # devuelva algo y podamos decir cuál es imposible (por ejemplo, demasiado lejos para la jornada)
    for nodo in range(1, len(lats)):
        modelo.AddDisjunction([gestor.NodeToIndex(nodo)], CASTIGO_NO_ATENDIDA)

    # Cómo buscar: primero una solución rápida (ir siempre al tramo más barato) y luego
    # mejorarla con búsqueda local guiada durante el tiempo indicado
    parametros = pywrapcp.DefaultRoutingSearchParameters()
    parametros.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parametros.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parametros.time_limit.FromMilliseconds(int(segundos * 1000))

    solucion = modelo.SolveWithParameters(parametros)
    if solucion is None:
        raise RuntimeError("OR-Tools no encontró ninguna solución para este día.")

    # Leer las rutas de la solución
    visitadas = set()
    for v, (tipo, nombre, _costo) in enumerate(vehiculos):
        if not modelo.IsVehicleUsed(solucion, v):
            continue
        nodos = []
        indice = solucion.Value(modelo.NextVar(modelo.Start(v)))
        while not modelo.IsEnd(indice):
            nodos.append(gestor.IndexToNode(indice))
            indice = solucion.Value(modelo.NextVar(indice))
        recorrido = [0] + nodos + [0]
        km_ruta = sum(km[a][b] for a, b in zip(recorrido, recorrido[1:]))
        min_ruta = sum(minutos[a][b] for a, b in zip(recorrido, recorrido[1:])) + servicio_min * len(nodos)
        resultado["rutas"].append({
            "vehiculo": nombre,
            "tipo": tipo,
            "paradas": [n - 1 for n in nodos],  # índices de la tabla de paradas, en orden de visita
            "cajas": sum(cajas[n] for n in nodos),
            "km": float(km_ruta),
            "horas": float(min_ruta / 60),
        })
        visitadas.update(nodos)

    # Renumerar por tipo (Urgente 1, Urgente 2...) sin importar cuál de las disponibles eligió el solver
    contador = {}
    for ruta in resultado["rutas"]:
        contador[ruta["tipo"]] = contador.get(ruta["tipo"], 0) + 1
        ruta["vehiculo"] = f"{ruta['tipo'].capitalize()} {contador[ruta['tipo']]}"

    resultado["no_atendidas"] = [n - 1 for n in range(1, len(lats)) if n not in visitadas]
    resultado["km"] = sum(r["km"] for r in resultado["rutas"])
    resultado["cajas"] = sum(r["cajas"] for r in resultado["rutas"])  # solo las entregadas
    resultado["propias"] = sum(r["tipo"] == "propia" for r in resultado["rutas"])
    resultado["reservadas_usadas"] = sum(r["tipo"] == "reservada" for r in resultado["rutas"])
    resultado["urgentes"] = sum(r["tipo"] == "urgente" for r in resultado["rutas"])
    return resultado
