"""Ejemplo ganadero: programación de jaulas para acopio de ganado, antes y después del pronóstico.

No es un problema de rutas de reparto: cada jaula carga en UN solo punto (permisos de movilización),
va y regresa. Lo que se decide es QUÉ DÍA sale cada jaula y CUÁNDO apartar fletera, para recoger
el ganado a tiempo (antes de que venza el permiso y sin merma de peso).

Partes:
  1. generar_ordenes(): órdenes de compra (OC) con semilla fija.
  2. simular(): despacho día por día con una regla para apartar fleteras (escenario A o B).
  3. indicadores(): retraso, permisos vencidos, merma, fletes y utilización de jaulas.
  4. pronostico() y apartar_con_pronostico(): el escenario B.

Por qué apartar puede ahorrar (en general y en este ejemplo):
  En una PYME de reparto, una unidad de último momento cuesta más que una apartada (en el artículo,
  $3,200 contra $1,500): ese "sobrecosto" es lo que el pronóstico ahorra. Este ejemplo NO tiene
  sobrecosto (dato: el flete cuesta igual apartado o no), así que el costo de que falte una jaula es
  la ESPERA: merma de peso por día y permisos vencidos. El modelo usa los dos: costo de que falte =
  sobrecosto + costo de la espera. Con sobrecosto = 0 solo cuenta la espera.

Cada valor dice si es DATO de la empresa o SUPUESTO.
"""

import math
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from src.forecast import metricas, nuevos_modelos

CARPETA_DATOS = Path(__file__).resolve().parent.parent / "data" / "ejemplo_ganadero"

# --- Datos de la empresa ---
JAULAS_PROPIAS = 3          # dato: 3 jaulas propias
DIAS_VIAJE = 3              # dato: un viaje redondo dura 3 días (la jaula queda ocupada 3 días)
VIGENCIA_PERMISO = 5        # dato: los permisos valen 5 días (el de emisión y los 4 siguientes)
ANTICIPACION = 7            # dato: la fletera se aparta con una semana de anticipación

# Flete por viaje (dato: solo depende de la distancia y cuesta igual propio que contratado).
# Balancán y Pichucalco son datos exactos; "otras zonas" es un rango de $16,000-18,000 y el valor
# de cada origen dentro del rango es supuesto.
FLETE = {
    "Balancán": 45_000,
    "Pichucalco": 40_000,
    "Tenosique": 18_000,
    "Palenque/Ocosingo": 18_000,
    "Emiliano Zapata": 17_000,
    "Medellín": 17_000,
    "Villahermosa": 16_000,
}
ESTADO = {"Balancán": "Tab.", "Tenosique": "Tab.", "Villahermosa": "Tab.", "Emiliano Zapata": "Tab.",
          "Palenque/Ocosingo": "Chis.", "Pichucalco": "Chis.", "Medellín": "Ver."}
CLAVE = {"Balancán": "BAL", "Tenosique": "TEN", "Villahermosa": "VHS", "Emiliano Zapata": "EZP",
         "Palenque/Ocosingo": "PAL", "Pichucalco": "PIC", "Medellín": "MED"}
# Qué tan seguido compra en cada origen (supuesto: la empresa no dio proporciones)
PESO_ORIGEN = {"Balancán": 0.20, "Palenque/Ocosingo": 0.20, "Tenosique": 0.15, "Emiliano Zapata": 0.15,
               "Villahermosa": 0.10, "Pichucalco": 0.10, "Medellín": 0.10}

# --- Supuestos: (valor base, mínimo, máximo) para el análisis de sensibilidad ---
SUPUESTOS = {
    "pct_aviso_largo": (0.60, 0.30, 0.90),   # OCs avisadas con 7 días o más
    "p_fletera": (0.50, 0.20, 0.90),         # conseguir fletera de un día para otro
    "costo_sin_usar": (0.50, 0.00, 1.00),    # fracción del flete que se paga si se aparta y no se usa
    "merma_dia": (0.010, 0.005, 0.020),      # merma de peso por día de retraso
    "peso_kg": (250, 200, 350),              # peso por cabeza
    "precio_kg": (55, 45, 70),               # precio por kg en pie (MXN)
    "costo_permiso": (1_500, 500, 3_000),    # re-tramitar un permiso vencido (MXN)
    # Ruteo (juntar lotes chicos en una jaula)
    "pct_parcial": (0.15, 0.05, 0.40),       # OCs que no llenan la jaula (30-55 cabezas)
    "max_desvio_km": (150, 50, 300),         # km máximos por carretera entre los dos puntos de carga
    "max_dias_espera": (2, 0, 4),            # días que el primer lote puede esperar al segundo
    "costo_km": (31, 15, 45),                # costo por km del desvío (MXN)
}
NOMBRE_SUPUESTO = {
    "pct_aviso_largo": "OCs avisadas con ≥ 7 días",
    "p_fletera": "Fletera disponible de un día para otro",
    "costo_sin_usar": "Pago de una fletera apartada que no se usa",
    "merma_dia": "Merma por día de retraso",
    "peso_kg": "Peso por cabeza (kg)",
    "precio_kg": "Precio por kg en pie",
    "costo_permiso": "Re-tramitar un permiso vencido",
    "pct_parcial": "OCs que no llenan la jaula",
    "max_desvio_km": "Km máximos entre los dos puntos de carga",
    "max_dias_espera": "Días que un lote puede esperar al otro",
    "costo_km": "Costo por km del desvío",
}
BASE = {k: v[0] for k, v in SUPUESTOS.items()}
# Sobrecosto de una fletera de último momento frente a una apartada (MXN por viaje).
# Dato de este ejemplo: 0 ("es el mismo costo"). Otra PYME pondría aquí su diferencia.
BASE["sobrecosto"] = 0

# Periodos: un año de historial (para entrenar el pronóstico) y un año que se compara
INICIO_HISTORIAL = date(2024, 11, 4)    # lunes
INICIO_EVALUACION = date(2025, 11, 3)   # lunes
SEMANAS = 52
RITMO = 6.0      # jaulas por semana en promedio (supuesto, calibrado: con A salen ≈ 5-6 fleteras al mes)
SEMILLA = 7


def habil(d):
    """El primer día hábil (lunes a viernes) desde d: los fines de semana no se expiden dictámenes."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


# --- 1. Órdenes de compra ---

def generar_ordenes(semilla=SEMILLA, pct_aviso_largo=BASE["pct_aviso_largo"], ritmo=RITMO,
                    inicio=INICIO_HISTORIAL, semanas=2 * SEMANAS, pct_parcial=BASE["pct_parcial"]):
    """Una OC por jaula, para `semanas` semanas desde `inicio` (un lunes).

    Cada semana: unas `ritmo` jaulas (entre 4 y 10), de 3 a 5 proveedores, cada uno en un día distinto
    de lunes a viernes; las jaulas de un mismo proveedor salen del mismo origen el mismo día. Los números aleatorios se sacan siempre en el mismo
    orden, así que cambiar pct_aviso_largo (o pct_parcial) solo cambia los días de aviso (o las cabezas),
    no el resto de las órdenes.
    """
    rng = np.random.default_rng(semilla)
    origenes = list(PESO_ORIGEN)
    pesos = np.array(list(PESO_ORIGEN.values()))
    pesos = pesos / pesos.sum()
    filas = []
    for s in range(semanas):
        lunes = inicio + timedelta(weeks=s)
        # Temporada suave (supuesto): ±12 % a lo largo del año
        media = ritmo * (1 + 0.12 * math.sin(2 * math.pi * (lunes.isocalendar()[1] - 10) / 52))
        jaulas = int(np.clip(round(rng.normal(media, 1.0)), 4, 10))
        proveedores = min(int(rng.integers(3, 6)), jaulas)
        reparto = 1 + rng.multinomial(jaulas - proveedores, [1 / proveedores] * proveedores)
        dias = rng.permutation(5)          # cada proveedor embarca un día distinto de lunes a viernes
        for cuantas, dia in zip(reparto, dias):
            origen = origenes[rng.choice(len(origenes), p=pesos)]
            proveedor = f"{CLAVE[origen]}-{int(rng.integers(1, 9)):02d}"
            embarque = lunes + timedelta(days=int(dia))
            # Días de aviso: largo (7 a 14) o corto (1 a 6); se sortean los dos y se elige uno
            sorteo, largo, corto = rng.random(), int(rng.integers(7, 15)), int(rng.integers(1, 7))
            aviso = largo if sorteo < pct_aviso_largo else corto
            for _ in range(cuantas):
                completa, llena, parcial = rng.random(), int(rng.integers(70, 88)), int(rng.integers(30, 56))
                filas.append({
                    "proveedor": proveedor, "origen": origen,
                    "cabezas": llena if completa < 1 - pct_parcial else parcial,
                    "fecha_aviso": embarque - timedelta(days=aviso), "fecha_embarque": embarque,
                    "dias_aviso": aviso, "flete_mxn": FLETE[origen],
                })
    ordenes = pd.DataFrame(filas).sort_values(["fecha_embarque", "proveedor"], kind="stable")
    ordenes.insert(0, "oc_id", [f"OC-{k:04d}" for k in range(1, len(ordenes) + 1)])
    return ordenes.reset_index(drop=True)


def leer_ordenes(ruta):
    """Lee ordenes.csv con las fechas como fechas (no como texto)."""
    ordenes = pd.read_csv(ruta)
    for columna in ("fecha_aviso", "fecha_embarque"):
        ordenes[columna] = pd.to_datetime(ordenes[columna]).dt.date
    return ordenes


def _sin_acentos(texto):
    texto = unicodedata.normalize("NFKD", str(texto).strip().lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


# Cómo puede venir escrito cada origen en el archivo de la empresa
ALIAS_ORIGEN = {_sin_acentos(n): n for n in FLETE} | {
    "palenque": "Palenque/Ocosingo", "ocosingo": "Palenque/Ocosingo", "palenque / ocosingo": "Palenque/Ocosingo",
    "medellin de bravo": "Medellín", "tenosique de pino suarez": "Tenosique", "centro": "Villahermosa",
}


def validar_ordenes(tabla):
    """Revisa un archivo de órdenes de la empresa y lo deja como el de ejemplo.

    Columnas obligatorias: oc_id, origen, cabezas, fecha_embarque, y fecha_aviso o dias_aviso.
    Opcionales: proveedor (si falta, cada OC cuenta como proveedor distinto) y flete_mxn (si falta, el de
    su origen). Devuelve (órdenes, lista de problemas). Si hay problemas, las órdenes son None.
    """
    t = tabla.rename(columns=lambda c: _sin_acentos(c).replace(" ", "_"))
    faltan = [c for c in ("oc_id", "origen", "cabezas", "fecha_embarque") if c not in t]
    if "fecha_aviso" not in t and "dias_aviso" not in t:
        faltan.append("fecha_aviso o dias_aviso")
    if faltan:
        return None, [f"Faltan columnas: {', '.join(faltan)}."]
    problemas = []
    t = t.copy()
    t["fecha_embarque"] = pd.to_datetime(t["fecha_embarque"], dayfirst=True, errors="coerce", format="mixed")
    if t["fecha_embarque"].isna().any():
        problemas.append(f"{int(t['fecha_embarque'].isna().sum())} fechas de embarque no se entienden "
                         "(usa AAAA-MM-DD o DD/MM/AAAA).")
    t["fecha_embarque"] = t["fecha_embarque"].dt.date
    if "fecha_aviso" in t:
        t["fecha_aviso"] = pd.to_datetime(t["fecha_aviso"], dayfirst=True, errors="coerce", format="mixed").dt.date
        if t["fecha_aviso"].isna().any():
            problemas.append(f"{int(t['fecha_aviso'].isna().sum())} fechas de aviso no se entienden.")
        else:
            t["dias_aviso"] = [(e - a).days if e and a else 0 for e, a in zip(t["fecha_embarque"], t["fecha_aviso"])]
    else:
        t["dias_aviso"] = pd.to_numeric(t["dias_aviso"], errors="coerce")
        if t["dias_aviso"].isna().any():
            problemas.append("Hay días de aviso que no son números.")
        elif not problemas:
            t["fecha_aviso"] = [e - timedelta(days=int(a)) for e, a in zip(t["fecha_embarque"], t["dias_aviso"])]
    if "dias_aviso" in t and (pd.to_numeric(t["dias_aviso"], errors="coerce") < 0).any():
        problemas.append("Hay órdenes avisadas después de su fecha de embarque.")
    t["cabezas"] = pd.to_numeric(t["cabezas"], errors="coerce")
    malas = t[t["cabezas"].isna() | (t["cabezas"] <= 0) | (t["cabezas"] > CAPACIDAD_CABEZAS)]
    if len(malas):
        problemas.append(f"{len(malas)} órdenes con cabezas fuera de 1 a {CAPACIDAD_CABEZAS} "
                         f"(por ejemplo {', '.join(map(str, malas['oc_id'].head(3)))}).")
    t["origen"] = t["origen"].map(lambda o: ALIAS_ORIGEN.get(_sin_acentos(o), o))
    desconocidos = sorted(set(t["origen"]) - set(FLETE))
    if desconocidos:
        problemas.append(f"Orígenes sin distancia por carretera: {', '.join(map(str, desconocidos))}. "
                         f"Los conocidos son: {', '.join(FLETE)}.")
    if t["oc_id"].duplicated().any():
        problemas.append("Hay oc_id repetidos.")
    if problemas:
        return None, problemas
    if "proveedor" not in t:
        t["proveedor"] = t["oc_id"].astype(str)
    if "flete_mxn" not in t:
        t["flete_mxn"] = t["origen"].map(FLETE)
    t["cabezas"] = t["cabezas"].astype(int)
    t["dias_aviso"] = t["dias_aviso"].astype(int)
    columnas = ["oc_id", "proveedor", "origen", "cabezas", "fecha_aviso", "fecha_embarque", "dias_aviso", "flete_mxn"]
    return t[columnas].sort_values(["fecha_embarque", "oc_id"], kind="stable").reset_index(drop=True), []


def del_periodo(ordenes, inicio=INICIO_EVALUACION, semanas=SEMANAS):
    """Las OCs con embarque dentro del periodo que se compara."""
    fin = inicio + timedelta(weeks=semanas)
    return ordenes[(ordenes["fecha_embarque"] >= inicio) & (ordenes["fecha_embarque"] < fin)].reset_index(drop=True)


# --- 2. Despacho día por día ---

def _suerte(semilla, i, d):
    """Número al azar fijo para (OC, día): A y B tienen la misma suerte con la fletera."""
    return np.random.default_rng([semilla, i, d.toordinal()]).random()


def _orden_fila(oc):
    """Orden de atención: primero la que debía salir antes (y, si empatan, el permiso que vence antes)."""
    return (oc["embarque"], oc["vence"], oc["i"])


def plan_tentativo(d, objetivo, ocs, libre, apartadas, sabe_todo=False):
    """Simula de d+1 a `objetivo` con las OCs que ya se conocen el día d, sin fleteras de último momento.

    Devuelve (OCs del día objetivo que se quedarían sin jaula, jaulas propias que sobrarían ese día).
    Sirve para decidir cuántas fleteras apartar para el día objetivo.
    sabe_todo=True: usa también las OCs que todavía no se avisan (solo para el escenario ideal).
    """
    libre = list(libre)
    conocidas = sorted((oc for oc in ocs if oc["carga"] is None and (sabe_todo or oc["aviso"] <= d)
                        and oc["embarque"] <= objetivo), key=_orden_fila)
    asignada = set()
    x = d + timedelta(days=1)
    while x <= objetivo:
        fleteras = apartadas.get(x, 0)
        for oc in conocidas:
            if oc["i"] in asignada or oc["embarque"] > x:
                continue
            if fleteras:
                fleteras -= 1
            else:
                k = next((k for k in range(len(libre)) if libre[k] <= x), None)
                if k is None:
                    continue
                libre[k] = x + timedelta(days=DIAS_VIAJE)
            asignada.add(oc["i"])
        x += timedelta(days=1)
    faltan = sum(1 for oc in conocidas if oc["i"] not in asignada and oc["embarque"] == objetivo)
    sobran = 0 if faltan else sum(1 for v in libre if v <= objetivo)
    return faltan, sobran


def apartar_reactivo(d, objetivo, ocs, libre, apartadas):
    """Escenario A: se aparta fletera solo para las OCs ya confirmadas que no caben en las jaulas propias."""
    faltan, _ = plan_tentativo(d, objetivo, ocs, libre, apartadas)
    return faltan


def simular(ordenes, apartar, p_fletera=BASE["p_fletera"], semilla=SEMILLA):
    """Despacho día por día.

    apartar(d, objetivo, ocs, libre, apartadas) -> cuántas fleteras apartar HOY (día d) para el día
    objetivo = d + 7 (dato: se apartan con una semana de anticipación).

    Cada día, las OCs pendientes (ya llegó su fecha de embarque y su permiso está vigente) se atienden
    en orden con: 1) las fleteras apartadas para hoy, 2) las jaulas propias libres y 3) una fletera
    de último momento, que se consigue con probabilidad p_fletera. Si nada de eso hay, la OC espera.
    Si se pasa la vigencia del permiso, se re-tramita: un día hábil de trámite y otros 5 días de vigencia.

    Devuelve (ordenes con día de carga y medio, viajes para el calendario, fleteras apartadas sin usar).
    """
    ocs = [{"i": i, "oc_id": f.oc_id, "aviso": f.fecha_aviso, "embarque": f.fecha_embarque,
            "desde": f.fecha_embarque, "vence": f.fecha_embarque + timedelta(days=VIGENCIA_PERMISO - 1),
            "carga": None, "medio": None, "vencidos": 0}
           for i, f in enumerate(ordenes.itertuples())]
    if not ocs:
        return ordenes.assign(fecha_carga=None, medio=None, retraso=0, permisos_vencidos=0), pd.DataFrame(), []
    libre = [min(oc["embarque"] for oc in ocs)] * JAULAS_PROPIAS   # día en que cada jaula propia queda libre
    carriles = []                       # para el calendario: día en que cada fletera queda libre
    apartadas, sin_usar, viajes = {}, [], []
    d = min(oc["embarque"] for oc in ocs) - timedelta(days=ANTICIPACION)
    ultimo = max(oc["embarque"] for oc in ocs)

    while any(oc["carga"] is None for oc in ocs):
        # 1) Apartar fleteras para dentro de una semana (sin pasar de la última fecha de embarque)
        objetivo = d + timedelta(days=ANTICIPACION)
        if objetivo <= ultimo:
            n = apartar(d, objetivo, ocs, libre, apartadas)
            if n:
                apartadas[objetivo] = apartadas.get(objetivo, 0) + n

        # 2) Permisos vencidos: se re-tramitan (un día hábil de trámite; fines de semana no hay dictámenes)
        for oc in ocs:
            if oc["carga"] is None and d > oc["vence"]:
                emision = habil(habil(d) + timedelta(days=1))
                oc["desde"], oc["vence"] = emision, emision + timedelta(days=VIGENCIA_PERMISO - 1)
                oc["vencidos"] += 1

        # 3) Cargar lo pendiente de hoy
        pendientes = sorted((oc for oc in ocs if oc["carga"] is None and oc["embarque"] <= d
                             and oc["desde"] <= d), key=_orden_fila)
        fleteras = apartadas.get(d, 0)
        for oc in pendientes:
            if fleteras:
                fleteras -= 1
                oc["medio"] = "Fletera apartada"
            else:
                k = next((k for k in range(JAULAS_PROPIAS) if libre[k] <= d), None)
                if k is not None:
                    libre[k] = d + timedelta(days=DIAS_VIAJE)
                    oc["medio"] = f"Propia {k + 1}"
                elif _suerte(semilla, oc["i"], d) < p_fletera:
                    oc["medio"] = "Fletera de último momento"
                else:
                    continue
            oc["carga"] = d
            if oc["medio"].startswith("Propia"):
                recurso = oc["medio"]
            else:   # carril libre en el calendario de fleteras
                c = next((c for c in range(len(carriles)) if carriles[c] <= d), None)
                if c is None:
                    carriles.append(d)
                    c = len(carriles) - 1
                carriles[c] = d + timedelta(days=DIAS_VIAJE)
                recurso = f"Fletera {c + 1}"
            viajes.append({"oc_id": oc["oc_id"], "recurso": recurso, "inicio": d,
                           "fin": d + timedelta(days=DIAS_VIAJE - 1), "retraso": (d - oc["embarque"]).days})
        sin_usar += [d] * fleteras
        d += timedelta(days=1)
    # Fleteras apartadas para días posteriores a la última carga: tampoco se usaron
    sin_usar += [x for x, n in sorted(apartadas.items()) if x >= d for _ in range(n)]

    salida = ordenes.copy()
    salida["fecha_carga"] = [oc["carga"] for oc in ocs]
    salida["medio"] = [oc["medio"] for oc in ocs]
    salida["retraso"] = [(oc["carga"] - oc["embarque"]).days for oc in ocs]
    salida["permisos_vencidos"] = [oc["vencidos"] for oc in ocs]
    return salida, pd.DataFrame(viajes), sin_usar


# --- 3. Indicadores ---

def indicadores(resultado, sin_usar, s=BASE, inicio=INICIO_EVALUACION, semanas=SEMANAS):
    """Los números para comparar escenarios. s: valores de los supuestos (merma, peso, precio...)."""
    r = resultado
    # Merma: cabezas × días de retraso, más la espera del primer lote cuando se juntan dos (escenario C)
    espera = r["cab_dias_espera"] if "cab_dias_espera" in r else 0
    kg = (r["cabezas"] * r["retraso"] + espera) * s["peso_kg"] * s["merma_dia"]
    contratada = ~r["medio"].str.startswith("Propia")
    flete_medio = r["flete_mxn"].mean()
    dias = semanas * 7
    fin = inicio + timedelta(days=dias)
    ocupado = 0     # días-jaula propia ocupados dentro del periodo
    for f in r[~contratada].itertuples():
        for k in range(DIAS_VIAJE):
            if inicio <= f.fecha_carga + timedelta(days=k) < fin:
                ocupado += 1
    lotes = r["lotes"] if "lotes" in r else pd.Series(1, index=r.index)
    return {
        "OCs": int(lotes.sum()),
        "Viajes": len(r),
        "Viajes con 2 puntos de carga": int((lotes > 1).sum()),
        "OCs con retraso": int((r["retraso"] > 0).sum()),
        "Días de retraso": int(r["retraso"].sum()),
        "Retraso máximo (días)": int(r["retraso"].max()) if len(r) else 0,
        "Permisos vencidos": int(r["permisos_vencidos"].sum()),
        "Cabezas con retraso": int(r.loc[r["retraso"] > 0, "cabezas"].sum()),
        "Merma (kg)": float(kg.sum()),
        "Merma ($)": float(kg.sum() * s["precio_kg"]),
        "Re-trámite de permisos ($)": float(r["permisos_vencidos"].sum() * s["costo_permiso"]),
        "Fletes contratados": int(contratada.sum()),
        "  apartados": int((r["medio"] == "Fletera apartada").sum()),
        "  de último momento": int((r["medio"] == "Fletera de último momento").sum()),
        "Fleteras apartadas sin usar": len(sin_usar),
        "Costo de fletes ($)": float(r["flete_mxn"].sum()),
        "Sobrecosto de último momento ($)": float((r["medio"] == "Fletera de último momento").sum()
                                                  * s.get("sobrecosto", 0)),
        "Pago por fleteras sin usar ($)": float(len(sin_usar) * s["costo_sin_usar"] * flete_medio),
        "Utilización de jaulas propias": ocupado / (JAULAS_PROPIAS * dias),
    }


def costo_total(ind):
    """Fletes + sobrecosto + fleteras sin usar + merma + re-trámites (lo que cambia entre escenarios)."""
    return (ind["Costo de fletes ($)"] + ind["Sobrecosto de último momento ($)"]
            + ind["Pago por fleteras sin usar ($)"] + ind["Merma ($)"] + ind["Re-trámite de permisos ($)"])


# --- 4. Pronóstico y escenario B ---

def lunes_de(d):
    return d - timedelta(days=d.weekday())


def tabla_semanal(ordenes):
    """Jaulas por semana (la fecha es el lunes) y origen, con 0 en las semanas sin compras."""
    conteo = ordenes.assign(semana=ordenes["fecha_embarque"].map(lunes_de)).groupby(["semana", "origen"]).size()
    primera, ultima = lunes_de(ordenes["fecha_embarque"].min()), lunes_de(ordenes["fecha_embarque"].max())
    semanas = [primera + timedelta(weeks=k) for k in range((ultima - primera).days // 7 + 1)]
    indice = pd.MultiIndex.from_product([semanas, list(FLETE)], names=["semana", "origen"])
    return conteo.reindex(indice, fill_value=0).rename("y").reset_index()


def variables_semanales(t):
    """Variables de cada (semana, origen), con lo que se sabe UNA semana antes de que empiece.

    Las fleteras se apartan con 7 días de anticipación, así que al pronosticar la semana W todavía
    corre la semana W-1: el último dato completo es W-2. Por eso los rezagos empiezan en 2.
    """
    t = t.sort_values(["origen", "semana"]).copy()
    g = t.groupby("origen")["y"]
    for k in (2, 3, 4):
        t[f"rezago_{k}"] = g.shift(k)
    t["prom_4sem"] = g.transform(lambda y: y.shift(2).rolling(4).mean())
    semana_anio = t["semana"].map(lambda d: d.isocalendar()[1])
    t["sen"], t["cos"] = np.sin(2 * np.pi * semana_anio / 52), np.cos(2 * np.pi * semana_anio / 52)
    for o in FLETE:
        t[f"es_{CLAVE[o]}"] = (t["origen"] == o).astype(int)
    return t


COLUMNAS_SEMANA = (["rezago_2", "rezago_3", "rezago_4", "prom_4sem", "sen", "cos"]
                   + [f"es_{CLAVE[o]}" for o in FLETE])
MODELOS_SEMANA = ["ingenuo", "prom4", "ridge", "bosque", "gboost"]


def _pronosticar_semanas(t, entrena_hasta, predice_desde, predice_hasta):
    """Entrena con las semanas < entrena_hasta y pronostica [predice_desde, predice_hasta)."""
    base = t[t["semana"] < entrena_hasta].dropna(subset=COLUMNAS_SEMANA)
    objetivo = t[(t["semana"] >= predice_desde) & (t["semana"] < predice_hasta)].copy()
    objetivo["ingenuo"] = objetivo["rezago_2"]
    objetivo["prom4"] = objetivo["prom_4sem"]
    for nombre, modelo in nuevos_modelos().items():   # los mismos modelos de src/forecast.py
        modelo.fit(base[COLUMNAS_SEMANA], base["y"])
        objetivo[nombre] = np.clip(modelo.predict(objetivo[COLUMNAS_SEMANA]), 0, None)
    return objetivo


SEMANAS_MINIMAS = 20   # historial mínimo para entrenar los modelos (rezagos + 8 de validación)


def periodo_de(ordenes):
    """(lunes de inicio, semanas) del periodo que se compara.

    Con 60 semanas o más de órdenes: las últimas 52 (en el ejemplo, del 3-nov-2025 en adelante).
    Con menos: la segunda mitad de las semanas; la primera mitad queda como historial.
    """
    primera = lunes_de(ordenes["fecha_embarque"].min())
    ultima = lunes_de(ordenes["fecha_embarque"].max())
    n = (ultima - primera).days // 7 + 1
    evalua = SEMANAS if n >= 60 else max(1, math.ceil(n / 2))
    return ultima - timedelta(weeks=evalua - 1), evalua


def _frac_corto(ordenes, inicio):
    historial = ordenes[ordenes["fecha_embarque"] < inicio]
    base = historial if len(historial) else ordenes
    return float((base["dias_aviso"] < ANTICIPACION).mean())


def pronostico_sustituto(ordenes, inicio, semanas):
    """Cuando hay poco historial para entrenar: el promedio de jaulas por semana de todo lo que ya se
    conoce dos semanas antes (las semanas sin compras cuentan como 0)."""
    total = tabla_semanal(ordenes).groupby("semana")["y"].sum()
    previsto = total.shift(2).expanding().mean()
    fin = inicio + timedelta(weeks=semanas)
    semanal = pd.DataFrame({"y": total, "prom_hist": previsto.fillna(0)})
    semanal = semanal[(semanal.index >= inicio) & (semanal.index < fin)]
    m = metricas(semanal["y"], semanal["prom_hist"]) if semanal["y"].sum() else {"wape": np.nan, "sesgo": np.nan}
    tabla = pd.DataFrame([{"modelo": "prom_hist", "wape_semana_origen": np.nan, "wape_semana": m["wape"],
                           "sesgo_semana": m["sesgo"], "wape_validacion": np.nan}])
    return {"tabla": tabla, "elegido": "prom_hist", "semanal": semanal, "por_origen": None,
            "frac_corto": _frac_corto(ordenes, inicio), "inicio": inicio, "semanas": semanas, "sustituto": True}


def pronostico(ordenes, inicio=None, semanas=None, semanas_validacion=8):
    """Pronóstico de jaulas por semana y origen, igual que src/forecast.py pero por semana.

    1. Validación: entrena con el historial menos sus últimas 8 semanas y pronostica esas 8.
       El modelo con menor WAPE del total semanal (lo que sirve para apartar) es el elegido.
    2. Prueba: entrena con todo el historial y pronostica el periodo que se compara.
    Si hay menos de SEMANAS_MINIMAS semanas de historial, usa pronostico_sustituto().
    Devuelve {"tabla": métricas por modelo, "elegido", "semanal": real y previsto por semana,
              "por_origen": la tabla completa, "frac_corto": OCs avisadas con menos de 7 días,
              "inicio" y "semanas" del periodo, "sustituto": si se usó el promedio simple}.
    """
    if inicio is None:
        inicio, semanas = periodo_de(ordenes)
    if (inicio - lunes_de(ordenes["fecha_embarque"].min())).days // 7 < SEMANAS_MINIMAS:
        return pronostico_sustituto(ordenes, inicio, semanas)
    t = variables_semanales(tabla_semanal(ordenes))
    corte = inicio - timedelta(weeks=semanas_validacion)
    valida = _pronosticar_semanas(t, corte, corte, inicio)
    prueba = _pronosticar_semanas(t, inicio, inicio, inicio + timedelta(weeks=semanas))
    filas = []
    for m in MODELOS_SEMANA:
        total_v = valida.groupby("semana")[["y", m]].sum()
        total_p = prueba.groupby("semana")[["y", m]].sum()
        filas.append({"modelo": m,
                      "wape_semana_origen": metricas(prueba["y"], prueba[m])["wape"],
                      "wape_semana": metricas(total_p["y"], total_p[m])["wape"],
                      "sesgo_semana": metricas(total_p["y"], total_p[m])["sesgo"],
                      "wape_validacion": metricas(total_v["y"], total_v[m])["wape"]})
    tabla = pd.DataFrame(filas)
    elegido = tabla.loc[tabla["wape_validacion"].idxmin(), "modelo"]
    semanal = prueba.groupby("semana")[["y"] + MODELOS_SEMANA].sum()
    return {"tabla": tabla, "elegido": elegido, "semanal": semanal, "por_origen": prueba,
            "frac_corto": _frac_corto(ordenes, inicio), "inicio": inicio, "semanas": semanas, "sustituto": False}


def cuantil_poisson(alfa, media):
    """El menor k con P(X <= k) >= alfa, si X ~ Poisson(media): cuántas OCs sorpresa cubrir."""
    alfa = min(alfa, 0.999)      # con α = 1 habría que cubrir "todo": se corta en 99.9 %
    k, p = 0, math.exp(-media)
    acumulada = p
    while acumulada < alfa:
        k += 1
        p *= media / k
        acumulada += p
    return k


def costos_apartar(s, ordenes):
    """(costo de que falte una jaula, costo de que sobre una fletera apartada, costo de un día de espera).

    - Que falte: el sobrecosto de último momento (0 en este ejemplo) más la espera: con probabilidad p se
      consigue fletera cada día, así que se espera (1 − p) ÷ p días en promedio, y cada día cuesta la merma
      de una jaula (cabezas promedio × peso × merma × precio).
    - Que sobre: la fracción del flete que se paga por una fletera apartada que no se usa.
    """
    p = max(s["p_fletera"], 0.05)
    costo_dia = ordenes["cabezas"].mean() * s["peso_kg"] * s["merma_dia"] * s["precio_kg"]
    falta = s.get("sobrecosto", 0) + costo_dia * (1 - p) / p
    sobra = s["costo_sin_usar"] * ordenes["flete_mxn"].mean()
    return falta, sobra, costo_dia


def alfa_apartar(s, ordenes):
    """Ec. 2 del artículo (vendedor de periódicos): α = costo de que falte ÷ (falte + sobre)."""
    falta, sobra, _ = costos_apartar(s, ordenes)
    return falta / (falta + sobra) if falta + sobra > 0 else 0.5


def apartar_con_pronostico(prevision, frac_corto, alfa):
    """Escenario B: lo de A (OCs confirmadas que no caben) MÁS fleteras para las OCs que todavía no se
    avisan. prevision: {lunes: jaulas previstas esa semana}. De ellas, frac_corto se avisa con menos
    de 7 días: en promedio prevision × frac_corto ÷ 5 por día hábil. Se cubre el cuantil α de esas
    OCs sorpresa (Poisson), descontando las jaulas propias que sobran ese día."""
    def apartar(d, objetivo, ocs, libre, apartadas):
        faltan, sobran = plan_tentativo(d, objetivo, ocs, libre, apartadas)
        if objetivo.weekday() >= 5:
            return faltan
        sorpresa = prevision.get(lunes_de(objetivo), 0) * frac_corto / 5
        return faltan + max(0, cuantil_poisson(alfa, sorpresa) - sobran)
    return apartar


def apartar_ideal(d, objetivo, ocs, libre, apartadas):
    """Ideal (inalcanzable): aparta sabiendo desde hoy todas las OCs que se van a avisar."""
    faltan, _ = plan_tentativo(d, objetivo, ocs, libre, apartadas, sabe_todo=True)
    return faltan


# --- 5. Ruteo (VRP): juntar dos lotes chicos en una jaula ---
#
# Hoy cada jaula carga en un solo punto. Con las reglas de movilización, dos lotes podrían viajar
# juntos si CADA lote lleva su propia guía REEMO (una UPP de origen y una de destino, según el
# protocolo REEMO de SENASICA) y su certificado zoosanitario, y los dos van a la misma engorda.
# El modelo solo junta lotes cuando además se cumplen estas reglas (supuestos, ajustables):

CAPACIDAD_CABEZAS = 87      # dato: una jaula lleva hasta 87 cabezas
PUNTOS_MAX = 2              # supuesto: hasta 2 puntos de carga por jaula
MAX_DESVIO_KM = BASE["max_desvio_km"]       # supuesto: los dos orígenes a no más de 150 km por carretera
MAX_DIAS_ESPERA = BASE["max_dias_espera"]   # supuesto: el primer lote espera al segundo ≤ 2 días (permiso de 5)
KM_MISMO_ORIGEN = 20        # supuesto: dos proveedores del mismo municipio están a ~20 km
COSTO_KM = BASE["costo_km"]  # supuesto: $45,000 del flete a Balancán ÷ 1,448 km de ida y vuelta
ENGORDA = "Engorda (Cardel, Ver.)"   # dato: Carretera Cardel-Veracruz km 226, José Cardel, Ver.


def cargar_km(carpeta=CARPETA_DATOS):
    """{(desde, hasta): km por carretera} (de distancias.csv, obtenido con OSRM)."""
    d = pd.read_csv(carpeta / "distancias.csv")
    return {(f.desde, f.hasta): float(f.km) for f in d.itertuples()}


def pares_posibles(ordenes, km, s=BASE):
    """Pares de lotes que se pueden juntar y cuánto se ahorra con cada uno.

    Ahorro = flete del viaje que ya no se hace − km extra del desvío × costo por km − merma del lote que
    espera (cabezas × días de espera × peso × merma × precio). Solo se guardan los pares que ahorran.
    """
    o = ordenes.reset_index(drop=True)
    costo_cabeza_dia = s["peso_kg"] * s["merma_dia"] * s["precio_kg"]
    chicas = o.index[o["cabezas"] <= CAPACIDAD_CABEZAS - o["cabezas"].min()]
    pares = []
    for n, i in enumerate(chicas):
        a = o.loc[i]
        for j in chicas[n + 1:]:
            b = o.loc[j]
            if a["cabezas"] + b["cabezas"] > CAPACIDAD_CABEZAS:
                continue
            espera = abs((a["fecha_embarque"] - b["fecha_embarque"]).days)
            if espera > s.get("max_dias_espera", MAX_DIAS_ESPERA):
                continue
            # Los dos se tienen que conocer antes de que el primero esté listo para cargar
            if max(a["fecha_aviso"], b["fecha_aviso"]) >= min(a["fecha_embarque"], b["fecha_embarque"]):
                continue
            if a["origen"] == b["origen"]:
                entre = 0 if a["proveedor"] == b["proveedor"] else KM_MISMO_ORIGEN
            else:
                entre = km[(a["origen"], b["origen"])]
            if entre > s.get("max_desvio_km", MAX_DESVIO_KM):
                continue
            ida_a, ida_b = km[(ENGORDA, a["origen"])], km[(ENGORDA, b["origen"])]
            extra = ida_a + entre + ida_b - 2 * max(ida_a, ida_b)
            primero = a if a["fecha_embarque"] <= b["fecha_embarque"] else b
            ahorro = (min(a["flete_mxn"], b["flete_mxn"]) - extra * s.get("costo_km", COSTO_KM)
                      - primero["cabezas"] * espera * costo_cabeza_dia)
            if ahorro > 0:
                pares.append({"i": i, "j": j, "ahorro": ahorro, "km_extra": extra, "espera": espera,
                              "cab_dias_espera": int(primero["cabezas"]) * espera})
    return pd.DataFrame(pares, columns=["i", "j", "ahorro", "km_extra", "espera", "cab_dias_espera"])


def elegir_pares(pares):
    """Qué pares juntar para ahorrar lo más posible, sin usar un lote dos veces (OR-Tools CP-SAT)."""
    if pares.empty:
        return pares
    modelo = cp_model.CpModel()
    x = [modelo.NewBoolVar(f"par_{k}") for k in range(len(pares))]
    for lote in set(pares["i"]) | set(pares["j"]):
        modelo.Add(sum(x[k] for k, p in enumerate(pares.itertuples()) if lote in (p.i, p.j)) <= 1)
    modelo.Maximize(sum(int(round(p.ahorro)) * x[k] for k, p in enumerate(pares.itertuples())))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.random_seed = SEMILLA
    solver.Solve(modelo)
    return pares[[bool(solver.Value(v)) for v in x]].reset_index(drop=True)


def consolidar(ordenes, km=None, s=BASE):
    """Las órdenes como viajes: los pares elegidos se vuelven un solo viaje con 2 puntos de carga.

    El viaje junto sale el día del lote más tardío; su flete es el del origen más caro más el desvío.
    Devuelve (viajes, pares elegidos).
    """
    km = km or cargar_km()
    o = ordenes.reset_index(drop=True).assign(lotes=1, cab_dias_espera=0)
    elegidos = elegir_pares(pares_posibles(o, km, s))
    juntos, quitar = [], set()
    for p in elegidos.itertuples():
        a, b = o.loc[p.i], o.loc[p.j]
        lejos, cerca = (a, b) if km[(ENGORDA, a["origen"])] >= km[(ENGORDA, b["origen"])] else (b, a)
        juntos.append({
            "oc_id": f"{lejos['oc_id']} + {cerca['oc_id']}",
            "proveedor": f"{lejos['proveedor']} + {cerca['proveedor']}",
            "origen": lejos["origen"] if a["origen"] == b["origen"] else f"{lejos['origen']} + {cerca['origen']}",
            "cabezas": int(a["cabezas"] + b["cabezas"]),
            "fecha_aviso": max(a["fecha_aviso"], b["fecha_aviso"]),
            "fecha_embarque": max(a["fecha_embarque"], b["fecha_embarque"]),
            "flete_mxn": float(lejos["flete_mxn"] + p.km_extra * s.get("costo_km", COSTO_KM)),
            "lotes": 2, "cab_dias_espera": p.cab_dias_espera,
        })
        quitar |= {p.i, p.j}
    viajes = pd.concat([o.drop(index=list(quitar)), pd.DataFrame(juntos)], ignore_index=True)
    viajes["dias_aviso"] = [(e - a).days for e, a in zip(viajes["fecha_embarque"], viajes["fecha_aviso"])]
    viajes = viajes.sort_values(["fecha_embarque", "oc_id"], kind="stable").reset_index(drop=True)
    return viajes, elegidos


ESCENARIOS_G = {"A": "Despacho actual", "B": "Pronóstico + programación de jaulas",
                "C": "Pronóstico + programación + ruteo (VRP)", "D": "Información perfecta"}


def comparar(ordenes, s=BASE, pron=None, semilla=SEMILLA, km=None):
    """Corre A, B, C y D sobre el periodo que se compara. Devuelve {escenario: {resultado, viajes,
    sin_usar, indicadores}} y el pronóstico (se puede pasar ya calculado para no repetirlo; trae el periodo).

    C = B más juntar lotes chicos con el ruteo. D = información perfecta: saber desde antes todas las
    órdenes (el tope de la programación, sin juntar lotes; no es alcanzable).
    """
    pron = pron or pronostico(ordenes)
    inicio, semanas = pron["inicio"], pron["semanas"]
    periodo = del_periodo(ordenes, inicio, semanas)
    prevision = pron["semanal"][pron["elegido"]].to_dict()
    con_pronostico = apartar_con_pronostico(prevision, pron["frac_corto"], alfa_apartar(s, ordenes))
    juntas, pares = consolidar(periodo, km, s)
    corridas = {"A": (periodo, apartar_reactivo), "B": (periodo, con_pronostico),
                "C": (juntas, con_pronostico), "D": (periodo, apartar_ideal)}
    salida = {}
    for e, (tabla, regla) in corridas.items():
        resultado, viajes, sin_usar = simular(tabla, regla, s["p_fletera"], semilla)
        salida[e] = {"resultado": resultado, "viajes": viajes, "sin_usar": sin_usar,
                     "indicadores": indicadores(resultado, sin_usar, s, inicio, semanas)}
    salida["C"]["pares"] = pares
    return salida, pron
