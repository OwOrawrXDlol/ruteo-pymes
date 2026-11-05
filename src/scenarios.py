"""Escenarios E1 a E4 para un día y para un rango de fechas.

  E1  Práctica actual. "Real" si hay operacion_real.csv (lo que hizo la empresa);
      si no, "simulado": un despachador que recorre las zonas en orden alrededor
      del almacén, llena cada camioneta en ese orden y contrata urgentes para lo
      que no quepa en la flota propia (como en el artículo).
  E2  VRP reactivo: rutas con OR-Tools, solo flota propia + urgentes.
  E3  Pronóstico + VRP: se reservan unidades el día anterior y luego se rutea.
  E4  Información perfecta: rutas de E2, pero lo que hizo falta se reservó a
      tarifa reservada en vez de urgente. No es alcanzable; es el piso de costo.

Costo del día = costo_km·km + costo_propia·propias + tarifa_reservada·reservadas + tarifa_urgente·urgentes
"""

import math
from datetime import timedelta

import numpy as np
import pandas as pd

from src.distances import matriz_distancias, matriz_tiempos_min
from src.load import paradas_del_dia
from src.vrp import resolver_dia

ESCENARIOS = ["E1", "E2", "E3", "E4"]


def costo_dia(km, propias, reservadas, urgentes, config):
    c = config["costos"]
    return (c["costo_km"] * km + c["costo_propia"] * propias
            + c["tarifa_reservada"] * reservadas + c["tarifa_urgente"] * urgentes)


def _fila(nombre, fuente, km, propias, reservadas, urgentes, cajas, config,
          rutas=None, no_atendidas=0, unidades_despachadas=None):
    """Resultado de un escenario en un día, con costo y utilización ya calculados.

    utilización = cajas entregadas / capacidad de las unidades que salieron.
    """
    if unidades_despachadas is None:
        unidades_despachadas = propias + reservadas + urgentes
    capacidad = config["flota"]["capacidad_cajas"] * unidades_despachadas
    return {
        "escenario": nombre,
        "fuente": fuente,
        "km": float(km),
        "propias": int(propias),
        "reservadas": int(reservadas),  # pagadas, se usen o no
        "urgentes": int(urgentes),
        "cajas": int(cajas),
        "capacidad": int(capacidad),
        "utilizacion": cajas / capacidad if capacidad else 0.0,
        "costo": costo_dia(km, propias, reservadas, urgentes, config),
        "no_atendidas": int(no_atendidas),
        "rutas": rutas,
    }


# --- Reserva para E3 ---

def pronostico_sustituto(pedidos, fecha, config):
    """Cajas previstas = promedio de cajas del mismo día de la semana en las semanas previas.

    Devuelve None si los datos no alcanzan a cubrir todas esas semanas.
    Un día de la historia sin pedidos cuenta como 0 cajas.
    """
    semanas = config["reserva"]["semanas_historia"]
    fechas = [fecha - timedelta(weeks=k) for k in range(1, semanas + 1)]
    if min(fechas) < pedidos["date"].min():
        return None
    cajas_por_dia = pedidos.groupby("date")["boxes"].sum()
    return float(np.mean([cajas_por_dia.get(f, 0) for f in fechas]))


def unidades_a_reservar(cajas_previstas, config):
    """Ec. (3) del artículo: K = ceil(cajas · q_alfa / (capacidad · u_max)); reservar = max(0, K − propias)."""
    r = config["reserva"]
    k = math.ceil(cajas_previstas * r["q_alfa"] / (config["flota"]["capacidad_cajas"] * r["u_max"]))
    return max(0, k - config["flota"]["camionetas_propias"])


def reserva_del_dia(datos, fecha, config):
    """(unidades reservadas, de dónde salió el dato), o (None, motivo) si no hay forma de saberlo."""
    reserva = datos.get("reserva")
    if reserva is not None:
        fila = reserva[reserva["date"] == fecha]
        if len(fila):
            return int(fila["reserved_extra"].iloc[0]), "reserva.csv"
        return None, "la fecha no está en reserva.csv"
    previstas = pronostico_sustituto(datos["pedidos"], fecha, config)
    if previstas is None:
        semanas = config["reserva"]["semanas_historia"]
        return None, f"no hay {semanas} semanas previas de pedidos para el pronóstico sustituto"
    return unidades_a_reservar(previstas, config), f"sustituto ({previstas:.0f} cajas previstas)"


# --- E1 simulado: práctica empírica ---

def _angulos(lats, lons, almacen):
    """Ángulo (grados, 0 a 360) de cada punto visto desde el almacén."""
    return np.degrees(np.arctan2(np.asarray(lats) - almacen["lat"], np.asarray(lons) - almacen["lon"])) % 360


def orden_empirico(paradas, config):
    """Orden en que el despachador toma las paradas: zona por zona, girando alrededor del almacén.

    Las zonas se ordenan por el ángulo de su centro; dentro de cada zona, por el ángulo de la parada.
    Sin zona, cada parada cuenta por su propio ángulo.
    """
    tabla = paradas.copy()
    tabla["angulo"] = _angulos(tabla["lat"], tabla["lon"], config["almacen"])
    centros = tabla.groupby("zone")[["lat", "lon"]].mean()
    angulo_zona = dict(zip(centros.index, _angulos(centros["lat"], centros["lon"], config["almacen"])))
    tabla["angulo_zona"] = [angulo_zona[z] if z else a for z, a in zip(tabla["zone"], tabla["angulo"])]
    return list(tabla.sort_values(["angulo_zona", "angulo"]).index)


def _vecino_mas_cercano(nodos, km):
    """Ordena las paradas yendo siempre a la más cercana que falte (nodo 0 = almacén)."""
    pendientes, actual, orden = list(nodos), 0, []
    while pendientes:
        actual = min(pendientes, key=lambda n: km[actual][n])
        pendientes.remove(actual)
        orden.append(actual)
    return orden


def simular_e1(paradas, config):
    """Rutas como las haría un despachador: llena camionetas en orden angular, sin optimizar."""
    capacidad = config["flota"]["capacidad_cajas"]
    servicio = config["tiempos"]["servicio_min"]
    jornada_min = config["tiempos"]["jornada_h"] * 60

    lats = [config["almacen"]["lat"]] + list(paradas["lat"])
    lons = [config["almacen"]["lon"]] + list(paradas["lon"])
    km = matriz_distancias(lats, lons, config["distancias"]["factor_circuidad"])
    minutos = matriz_tiempos_min(km, config["tiempos"]["velocidad_kmh"])

    def medir(nodos):
        recorrido = [0] + nodos + [0]
        tramos = list(zip(recorrido, recorrido[1:]))
        return sum(km[a][b] for a, b in tramos), sum(minutos[a][b] for a, b in tramos) + servicio * len(nodos)

    camionetas, actual, no_atendidas = [], [], []
    for i in orden_empirico(paradas, config):
        nodo = i + 1
        if medir([nodo])[1] > jornada_min:  # ni sola cabe en la jornada
            no_atendidas.append(i)
            continue
        candidata = _vecino_mas_cercano(actual + [nodo], km)
        cabe = sum(paradas["boxes"].iloc[n - 1] for n in candidata) <= capacidad
        if cabe and medir(candidata)[1] <= jornada_min:
            actual = candidata
        else:
            camionetas.append(actual)
            actual = [nodo]
    if actual:
        camionetas.append(actual)

    propias = config["flota"]["camionetas_propias"]
    rutas = []
    for k, nodos in enumerate(camionetas):
        tipo, numero = ("propia", k + 1) if k < propias else ("urgente", k - propias + 1)
        km_ruta, min_ruta = medir(nodos)
        rutas.append({"vehiculo": f"{tipo.capitalize()} {numero}", "tipo": tipo,
                      "paradas": [n - 1 for n in nodos],
                      "cajas": int(sum(paradas["boxes"].iloc[n - 1] for n in nodos)),
                      "km": float(km_ruta), "horas": float(min_ruta / 60)})
    return rutas, no_atendidas


# --- Escenarios de un día ---

def escenarios_dia(datos, fecha, config, segundos=None):
    """Calcula E1 a E4 para una fecha.

    Devuelve (resultados, avisos): resultados es {"E1": fila o None, ...}; None = no disponible.
    """
    paradas = paradas_del_dia(datos["paradas"], fecha)
    cajas_dia = int(paradas["boxes"].sum())
    resultados, avisos = {}, []

    # E1
    operacion = datos.get("operacion_real")
    if operacion is not None:
        fila = operacion[operacion["date"] == fecha]
        if len(fila):
            o = fila.iloc[0]
            resultados["E1"] = _fila("E1", "real (operacion_real.csv)", o.km, o.vehicles, o.reserved_hires,
                                     o.urgent_hires, cajas_dia, config)
        else:
            resultados["E1"] = None
            avisos.append(f"E1 no disponible el {fecha}: la fecha no está en operacion_real.csv.")
    else:
        rutas, fuera = simular_e1(paradas, config)
        resultados["E1"] = _fila(
            "E1", "simulado (despachador por zonas)", sum(r["km"] for r in rutas),
            sum(r["tipo"] == "propia" for r in rutas), 0, sum(r["tipo"] == "urgente" for r in rutas),
            sum(r["cajas"] for r in rutas), config, rutas=rutas, no_atendidas=len(fuera))

    # E2: solo propias + urgentes
    e2 = resolver_dia(paradas, config, reservadas=0, segundos=segundos)
    resultados["E2"] = _fila("E2", "OR-Tools", e2["km"], e2["propias"], 0, e2["urgentes"], e2["cajas"],
                             config, rutas=e2["rutas"], no_atendidas=len(e2["no_atendidas"]))

    # E3: propias + reservadas (pagadas aunque no salgan) + urgentes si aún no alcanza
    reservar, fuente = reserva_del_dia(datos, fecha, config)
    if reservar is None:
        resultados["E3"] = None
        avisos.append(f"E3 no disponible el {fecha}: {fuente}.")
    else:
        e3 = e2 if reservar == 0 else resolver_dia(paradas, config, reservadas=reservar, segundos=segundos)
        resultados["E3"] = _fila(
            "E3", f"OR-Tools + reserva de {reservar} ({fuente})", e3["km"], e3["propias"], reservar,
            e3["urgentes"], e3["cajas"], config, rutas=e3["rutas"], no_atendidas=len(e3["no_atendidas"]),
            unidades_despachadas=e3["propias"] + e3["reservadas_usadas"] + e3["urgentes"])

    # E4: mismas rutas que E2, pero las urgentes se habrían reservado a tiempo
    resultados["E4"] = _fila("E4", "rutas de E2 con reserva exacta", e2["km"], e2["propias"], e2["urgentes"], 0,
                             e2["cajas"], config, rutas=e2["rutas"], no_atendidas=len(e2["no_atendidas"]))

    for nombre, r in resultados.items():
        if r and r["no_atendidas"]:
            avisos.append(f"{nombre} el {fecha}: {r['no_atendidas']} paradas no se pudieron atender.")
    return resultados, avisos


# --- Rango de fechas ---

def escenarios_rango(datos, fechas, config, segundos=None, al_avanzar=None, calcular_dia=escenarios_dia):
    """Calcula E1 a E4 en varias fechas.

    Devuelve (detalle, resumen, avisos):
      detalle: una fila por fecha y escenario.
      resumen: totales por escenario, solo en los días donde todos los escenarios están disponibles,
               para comparar con los mismos pedidos, más diferencias porcentuales.
    al_avanzar(i, total): función opcional para mostrar el avance (la usa la app).
    calcular_dia: por defecto escenarios_dia; la app pasa una versión que recuerda días ya calculados.
    """
    filas, avisos = [], []
    for i, fecha in enumerate(fechas):
        resultados, avisos_dia = calcular_dia(datos, fecha, config, segundos)
        avisos += avisos_dia
        for nombre in ESCENARIOS:
            r = resultados[nombre]
            if r is not None:
                filas.append({"date": fecha, **{k: v for k, v in r.items() if k != "rutas"}})
        if al_avanzar:
            al_avanzar(i + 1, len(fechas))

    detalle = pd.DataFrame(filas)
    if detalle.empty:
        return detalle, pd.DataFrame(), avisos

    # Comparar solo en días con los cuatro escenarios
    completos = detalle.groupby("date")["escenario"].nunique()
    dias = completos[completos == len(ESCENARIOS)].index
    if len(dias) < len(fechas):
        avisos.append(f"El resumen usa {len(dias)} de {len(fechas)} días: en los demás falta algún escenario.")
    return detalle, resumir(detalle[detalle["date"].isin(dias)], config), avisos


def resumir(detalle, config):
    """Totales por escenario y diferencias porcentuales frente a E1 y E2."""
    if detalle.empty:
        return pd.DataFrame()
    resumen = detalle.groupby("escenario").agg(
        dias=("date", "nunique"), km=("km", "sum"), propias=("propias", "sum"),
        reservadas=("reservadas", "sum"), urgentes=("urgentes", "sum"), cajas=("cajas", "sum"),
        capacidad=("capacidad", "sum"), costo=("costo", "sum"), no_atendidas=("no_atendidas", "sum"),
    ).reindex(ESCENARIOS)
    resumen["utilizacion"] = resumen["cajas"] / resumen["capacidad"]
    resumen["costo_vs_E1_%"] = (resumen["costo"] / resumen.loc["E1", "costo"] - 1) * 100
    resumen["costo_vs_E2_%"] = (resumen["costo"] / resumen.loc["E2", "costo"] - 1) * 100
    resumen["km_vs_E1_%"] = (resumen["km"] / resumen.loc["E1", "km"] - 1) * 100
    return resumen


def brecha_cerrada(resumen):
    """Qué parte del ahorro posible (de E2 a E4) logra el pronóstico (E3), en %. Como en el artículo."""
    e2, e3, e4 = (resumen.loc[e, "costo"] for e in ("E2", "E3", "E4"))
    if e2 == e4:
        return None
    return (e2 - e3) / (e2 - e4) * 100


# --- KPI ---

def tabla_kpis(resumen):
    """Todos los indicadores por escenario: una fila por KPI y una columna por escenario."""
    kpis = {
        "Costo logístico (MXN)": resumen["costo"],
        "Costo por caja entregada (MXN)": resumen["costo"] / resumen["cajas"],
        "Km recorridos": resumen["km"],
        "Utilización de la flota": resumen["utilizacion"],
        "Camionetas propias (salidas)": resumen["propias"],
        "Fletes reservados": resumen["reservadas"],
        "Fletes urgentes": resumen["urgentes"],
        "Paradas no atendidas": resumen["no_atendidas"],
        "Costo vs E1": resumen["costo_vs_E1_%"] / 100,
        "Costo vs E2": resumen["costo_vs_E2_%"] / 100,
        "Km vs E1": resumen["km_vs_E1_%"] / 100,
    }
    return pd.DataFrame(kpis).T[ESCENARIOS]


def kpis_pronostico(datos, detalle, config):
    """Qué tan bien acertó la reserva de E3 frente a lo que de verdad hizo falta.

    Lo que hizo falta en cada día = fletes urgentes de E2 (lo que no cupo en la flota propia).
    El error del pronóstico (WAPE) solo se puede medir con el sustituto, porque reserva.csv
    trae unidades y no cajas previstas.
    """
    e3 = detalle[detalle["escenario"] == "E3"].set_index("date")
    if e3.empty:
        return None
    e2 = detalle[detalle["escenario"] == "E2"].set_index("date")
    reservadas = e3["reservadas"]
    necesarias = e2["urgentes"].reindex(e3.index)

    kpis = {
        "dias": len(e3),
        "fuente": "reserva.csv" if datos.get("reserva") is not None else "sustituto",
        "dias_exactos": int((reservadas == necesarias).sum()),
        "dias_de_mas": int((reservadas > necesarias).sum()),
        "dias_de_menos": int((reservadas < necesarias).sum()),
        "reservadas_sin_usar": int((reservadas - necesarias).clip(lower=0).sum()),
        "urgentes_evitados": int(necesarias.sum() - e3["urgentes"].sum()),
        "wape": None,
        "sesgo": None,
    }
    if kpis["fuente"] == "sustituto":
        reales = datos["pedidos"].groupby("date")["boxes"].sum().reindex(e3.index, fill_value=0)
        previstas = pd.Series([pronostico_sustituto(datos["pedidos"], f, config) for f in e3.index], index=e3.index)
        # WAPE = suma de errores absolutos / suma de lo real (ec. 1 del artículo)
        kpis["wape"] = float((previstas - reales).abs().sum() / reales.sum())
        # Sesgo: positivo = el pronóstico tiende a pasarse; negativo = se queda corto
        kpis["sesgo"] = float((previstas.sum() - reales.sum()) / reales.sum())
    return kpis
