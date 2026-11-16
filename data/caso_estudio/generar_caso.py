"""Genera el CASO DE ESTUDIO SIMULADO del artículo (distribuidora urbana en Pachuca).

NO son datos de ninguna empresa: es una simulación, como dice el artículo y el cartel.
Se reconstruyó a partir de lo que el artículo describe (Tabla 3 y la sección "Caso de estudio y datos"):
8 zonas, 560 clientes, del 2-ene-2024 al 31-dic-2025, 621 días operativos (lun a sáb, sin festivos)
y unos 48,948 pedidos. El código original del artículo no está disponible, así que lo que el artículo
NO especifica (ángulos de las zonas, clientes por zona, tamaño de cada efecto de temporada, grado de
sobredispersión) son SUPUESTOS de esta reconstrucción, marcados abajo.

Los pedidos de cada zona, día y franja salen de un conteo binomial negativo (Poisson con sobredispersión)
cuya tasa combina: nivel base de la zona × factor de día de la semana × factor mensual × eventos de
temporada × tendencia de 7 % anual × un choque diario común a todas las zonas.

Además de pedidos.csv y clientes.csv guarda tasa_real.csv: la tasa media verdadera con la que se
generó cada celda (zona, día, franja). En la vida real no existe; sirve solo para medir el "oráculo",
el error mínimo que cualquier modelo tendría por puro azar.

Uso:  .venv/bin/python data/caso_estudio/generar_caso.py
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path(__file__).parent
SEMILLA = 2025
ALMACEN = (20.1011, -98.7591)  # el mismo de config.yaml (supuesto: centro de Pachuca)

# Tabla 3 del artículo: (zona, tipo, km al almacén, pedidos/día, cajas/pedido, factor viernes, factor sábado, % pedidos)
ZONAS = [
    ("Z1", "comercial",   2.5, 14.8, 6.0, 1.67, 1.18, 19.3),
    ("Z2", "comercial",   4.3, 12.7, 5.5, 1.59, 1.16, 16.6),
    ("Z3", "residencial", 6.7, 10.2, 4.0, 1.33, 1.46, 13.3),
    ("Z4", "residencial", 5.1,  9.2, 4.0, 1.33, 1.48, 12.1),
    ("Z5", "residencial", 4.9,  9.2, 4.5, 1.27, 1.41, 12.1),
    ("Z6", "mixta",       5.7,  8.1, 5.1, 1.36, 1.13, 10.6),
    ("Z7", "industrial",  6.7,  5.5, 8.9, 1.00, 0.55,  7.2),
    ("Z8", "mixta",       8.1,  6.7, 5.0, 1.34, 1.26,  8.7),
]
# SUPUESTO: el artículo da la distancia pero no la dirección; se reparten alrededor del almacén
ANGULOS_GRADOS = [20, 95, 160, 215, 270, 320, 190, 55]
# El artículo: "dispersión de 0.9 a 1.5 km" alrededor del centroide de cada zona (la asignación es supuesta)
DISPERSION_KM = [0.9, 1.0, 1.4, 1.3, 1.2, 1.1, 1.5, 1.2]
N_CLIENTES = 560

INICIO, FIN = date(2024, 1, 2), date(2025, 12, 31)
# 5 festivos sin operación: con ellos salen los 621 días operativos del artículo (103 en la prueba)
FESTIVOS = {date(2024, 9, 16), date(2024, 12, 25), date(2025, 1, 1), date(2025, 9, 16), date(2025, 12, 25)}
ENTRENAMIENTO = (date(2024, 2, 5), date(2025, 8, 31))  # el artículo descarta las primeras semanas (rezagos)
PRUEBA = (date(2025, 9, 1), date(2025, 12, 31))
PEDIDOS_DIA_ENTRENAMIENTO = 76.4  # Tabla 3 y texto del artículo
CRECIMIENTO_ANUAL = 0.07

# SUPUESTOS: el artículo nombra los eventos pero no su tamaño
FACTOR_MES = {1: 0.95, 2: 0.97, 3: 1.00, 4: 1.00, 5: 1.03, 6: 0.98, 7: 0.97, 8: 1.00, 9: 1.00, 10: 1.02, 11: 1.05, 12: 1.05}
BUEN_FIN = [(date(2024, 11, 15), date(2024, 11, 18)), (date(2025, 11, 14), date(2025, 11, 17))]  # viernes a lunes
SEMANA_SANTA = [(date(2024, 3, 25), date(2024, 3, 30)), (date(2025, 4, 14), date(2025, 4, 19))]
REGRESO_CLASES = [(date(2024, 8, 26), date(2024, 9, 7)), (date(2025, 8, 25), date(2025, 9, 6))]
F_BUEN_FIN, F_DICIEMBRE, F_FIN_ANIO, F_SEMANA_SANTA, F_CLASES, F_QUINCENA = 1.40, 1.20, 0.80, 0.85, 1.10, 1.08

# Calibrados con calibrar() para que el oráculo dé ~40.6 % por zona/día/franja y ~12.5 % por día (artículo)
SIGMA_CHOQUE = 0.105  # desviación del choque diario común a todas las zonas
R_DISPERSION = 13.0    # binomial negativa: varianza = media + media² / R (R grande = casi Poisson)

# El artículo: 43.9 % de los pedidos llegan por la mañana (8 a 13 h) en un día normal y 33.1 % los viernes
SHARE_MANANA = {"normal": 0.439, "viernes": 0.331}
CAJAS_FORMA = 3.0  # SUPUESTO: forma de la distribución del tamaño del pedido


def dias_operativos():
    dias = pd.date_range(INICIO, FIN)
    return [d.date() for d in dias if d.day_of_week < 6 and d.date() not in FESTIVOS]


def _en(rangos, d):
    return any(a <= d <= b for a, b in rangos)


def factor_evento(d):
    """Efecto de temporada de un día: Buen Fin, diciembre, fin de año, Semana Santa, regreso a clases, quincena."""
    f = 1.0
    if _en(BUEN_FIN, d):
        f *= F_BUEN_FIN
    if d.month == 12:
        f *= F_DICIEMBRE if d.day <= 23 else F_FIN_ANIO
    if _en(SEMANA_SANTA, d):
        f *= F_SEMANA_SANTA
    if _en(REGRESO_CLASES, d):
        f *= F_CLASES
    if d.day in (14, 15, 16, 29, 30, 31, 1):  # quincena
        f *= F_QUINCENA
    return f


def tasas_por_zona_y_dia(dias):
    """DataFrame (día × zona) con el nº esperado de pedidos de cada zona cada día (sin el choque aleatorio)."""
    centro = pd.Timestamp(ENTRENAMIENTO[0]) + (pd.Timestamp(ENTRENAMIENTO[1]) - pd.Timestamp(ENTRENAMIENTO[0])) / 2
    filas = []
    for d in dias:
        tendencia = (1 + CRECIMIENTO_ANUAL) ** ((pd.Timestamp(d) - centro).days / 365.25)
        comun = FACTOR_MES[d.month] * factor_evento(d) * tendencia
        fila = {"date": d}
        for zona, _t, _km, media, _cajas, f_vie, f_sab, _pct in ZONAS:
            # nivel base de lun-jue tal que el promedio lun-sáb sea el de la Tabla 3
            base = media * 6 / (4 + f_vie + f_sab)
            dia_sem = {4: f_vie, 5: f_sab}.get(d.weekday(), 1.0)
            fila[zona] = base * dia_sem * comun
        filas.append(fila)
    tabla = pd.DataFrame(filas).set_index("date")
    # Escala global para que el promedio de entrenamiento sea de 76.4 pedidos por día
    entrena = [d for d in tabla.index if ENTRENAMIENTO[0] <= d <= ENTRENAMIENTO[1]]
    return tabla * (PEDIDOS_DIA_ENTRENAMIENTO / tabla.loc[entrena].sum(axis=1).mean())


def clientes_por_zona():
    """560 clientes repartidos en proporción al % de pedidos de cada zona (SUPUESTO)."""
    pct = np.array([z[7] for z in ZONAS])
    cuota = pct / pct.sum() * N_CLIENTES
    n = np.floor(cuota).astype(int)
    for k in np.argsort(-(cuota - n))[: N_CLIENTES - n.sum()]:
        n[k] += 1
    return n


def generar_clientes(rng):
    lat0, lon0 = ALMACEN
    filas = []
    for (zona, _t, km, *_), ang, disp, n in zip(ZONAS, ANGULOS_GRADOS, DISPERSION_KM, clientes_por_zona()):
        a = np.radians(ang)
        dlat = km * np.cos(a) / 111.195
        dlon = km * np.sin(a) / (111.195 * np.cos(np.radians(lat0)))
        for i in range(n):
            ex, ey = rng.normal(0, disp, 2)  # km al este y al norte del centro de la zona
            filas.append({
                "client_id": f"SIM-{zona}-{i + 1:03d}",
                "lat": round(lat0 + dlat + ey / 111.195, 6),
                "lon": round(lon0 + dlon + ex / (111.195 * np.cos(np.radians(lat0))), 6),
                "zone": zona,
            })
    return pd.DataFrame(filas)


def generar_conteos(rng, tasas, sigma=SIGMA_CHOQUE, r=R_DISPERSION):
    """Pedidos por zona, día y franja (binomial negativa) y la tasa verdadera de cada celda."""
    filas = []
    for d, fila in tasas.iterrows():
        choque = rng.lognormal(-sigma ** 2 / 2, sigma)  # el mismo para todas las zonas ese día
        manana = SHARE_MANANA["viernes" if d.weekday() == 4 else "normal"]
        for zona in tasas.columns:
            for franja, share in (("manana", manana), ("tarde", 1 - manana)):
                tasa = fila[zona] * share
                media = tasa * choque
                p = r / (r + media)
                filas.append({"date": d, "zone": zona, "franja": franja, "tasa": tasa,
                              "pedidos": rng.negative_binomial(r, p)})
    return pd.DataFrame(filas)


def _hora(rng, franja):
    """Hora de recepción: pico a las 11 h por la mañana (8-13) y a las 16 h por la tarde (13-19)."""
    inicio, fin, pico, desv = (8, 13, 11.0, 1.2) if franja == "manana" else (13, 19, 16.0, 1.4)
    while True:
        h = rng.normal(pico, desv)
        if inicio <= h < fin:
            return f"{int(h):02d}:{int((h % 1) * 60):02d}"


def generar_pedidos(rng, conteos, clientes):
    media_cajas = {z[0]: z[4] for z in ZONAS}
    por_zona = {z: g["client_id"].to_numpy() for z, g in clientes.groupby("zone")}
    # Algunos clientes piden más seguido que otros
    pesos = {z: rng.lognormal(0, 0.5, len(ids)) for z, ids in por_zona.items()}
    pesos = {z: p / p.sum() for z, p in pesos.items()}
    filas = []
    for c in conteos[conteos["pedidos"] > 0].itertuples(index=False):
        n = c.pedidos
        ids = rng.choice(por_zona[c.zone], size=n, p=pesos[c.zone])
        # Tamaño: 1 + Poisson con tasa gamma, de media = cajas por pedido de la zona (Tabla 3)
        gamma = rng.gamma(CAJAS_FORMA, (media_cajas[c.zone] - 1) / CAJAS_FORMA, size=n)
        cajas = np.minimum(1 + rng.poisson(gamma), 60)
        for cliente, caja in zip(ids, cajas):
            filas.append({"date": c.date, "time": _hora(rng, c.franja), "client_id": cliente, "boxes": int(caja)})
    pedidos = pd.DataFrame(filas).sort_values(["date", "time"], kind="stable").reset_index(drop=True)
    pedidos.insert(0, "order_id", [f"SIM-{i:06d}" for i in range(1, len(pedidos) + 1)])
    return pedidos


def generar(carpeta=CARPETA, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    dias = dias_operativos()
    clientes = generar_clientes(rng)
    conteos = generar_conteos(rng, tasas_por_zona_y_dia(dias))
    pedidos = generar_pedidos(rng, conteos, clientes)
    carpeta = Path(carpeta)
    clientes.to_csv(carpeta / "clientes.csv", index=False)
    pedidos.to_csv(carpeta / "pedidos.csv", index=False)
    conteos[["date", "zone", "franja", "tasa"]].round(4).to_csv(carpeta / "tasa_real.csv", index=False)
    return clientes, pedidos, conteos


def calibrar():
    """Error del oráculo (tasa real contra pedidos) por zona/día/franja y por día, en la prueba."""
    rng = np.random.default_rng(SEMILLA)
    c = generar_conteos(rng, tasas_por_zona_y_dia(dias_operativos()))
    c = c[(c["date"] >= PRUEBA[0]) & (c["date"] <= PRUEBA[1])]
    celda = (c["pedidos"] - c["tasa"]).abs().sum() / c["pedidos"].sum()
    dia = c.groupby("date")[["pedidos", "tasa"]].sum()
    return celda, (dia["pedidos"] - dia["tasa"]).abs().sum() / dia["pedidos"].sum()


if __name__ == "__main__":
    clientes, pedidos, _ = generar()
    print(f"{len(clientes)} clientes, {len(pedidos)} pedidos, {pedidos['date'].nunique()} días, "
          f"{pedidos['boxes'].sum()} cajas")
