"""Genera los datos INVENTADOS de ejemplo (clientes y pedidos en Pachuca).

No son datos de ninguna empresa: solo sirven para probar la app mientras
llegan los reales. Se usa una semilla fija para que siempre salga lo mismo.

Uso:  .venv/bin/python data/ejemplo/generar_ejemplo.py
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42
CARPETA = Path(__file__).parent

# Zonas inventadas alrededor del centro de Pachuca: (nombre, lat, lon, tipo)
ZONAS = [
    ("Z1", 20.1230, -98.7360, "comercial"),
    ("Z2", 20.0880, -98.7650, "comercial"),
    ("Z3", 20.0700, -98.7400, "residencial"),
    ("Z4", 20.1350, -98.7700, "residencial"),
    ("Z5", 20.0950, -98.7100, "mixta"),
    ("Z6", 20.0600, -98.7850, "industrial"),
]
CLIENTES_POR_ZONA = 10
DISPERSION_GRADOS = 0.006  # ~0.6 km alrededor del centro de la zona

# 6 semanas, lunes a sábado: 4 de historia (para el pronóstico sustituto) + 2 para evaluar
INICIO = date(2025, 9, 1)  # lunes
SEMANAS = 6

# Probabilidad base de que un cliente pida en un día y ajuste por día de la semana (0 = lunes)
PROB_BASE = 0.55
AJUSTE_DIA = {0: 0.95, 1: 0.9, 2: 1.0, 3: 1.0, 4: 1.35, 5: 0.8}
AJUSTE_SABADO_RESIDENCIAL = 1.4


def generar_clientes(rng):
    filas = []
    for zona, lat, lon, _tipo in ZONAS:
        for i in range(CLIENTES_POR_ZONA):
            filas.append({
                "client_id": f"EJ-{zona}-{i + 1:02d}",
                "lat": round(lat + rng.normal(0, DISPERSION_GRADOS), 6),
                "lon": round(lon + rng.normal(0, DISPERSION_GRADOS), 6),
                "zone": zona,
            })
    return pd.DataFrame(filas)


def generar_pedidos(rng, clientes):
    tipo_zona = {z: t for z, _, _, t in ZONAS}
    filas = []
    numero = 1
    for d in range(SEMANAS * 7):
        dia = INICIO + timedelta(days=d)
        if dia.weekday() == 6:  # domingo no hay reparto
            continue
        for _, c in clientes.iterrows():
            p = PROB_BASE * AJUSTE_DIA[dia.weekday()]
            if dia.weekday() == 5 and tipo_zona[c.zone] == "residencial":
                p *= AJUSTE_SABADO_RESIDENCIAL
            if rng.random() > min(p, 0.95):
                continue
            hora = int(np.clip(rng.normal(12, 3), 7, 19))
            filas.append({
                "order_id": f"EJ-P{numero:05d}",
                "date": dia.isoformat(),
                "time": f"{hora:02d}:{rng.integers(0, 60):02d}",
                "client_id": c.client_id,
                "boxes": int(rng.integers(3, 21)),
            })
            numero += 1

    pedidos = pd.DataFrame(filas)
    # Un pedido más grande que una camioneta (130 cajas) para probar que se parte
    pedidos.loc[pedidos.index[-5], "boxes"] = 130
    return pedidos


def main():
    rng = np.random.default_rng(SEMILLA)
    clientes = generar_clientes(rng)
    pedidos = generar_pedidos(rng, clientes)
    clientes.to_csv(CARPETA / "clientes.csv", index=False)
    pedidos.to_csv(CARPETA / "pedidos.csv", index=False)
    print(f"{len(clientes)} clientes y {len(pedidos)} pedidos guardados en {CARPETA}")


if __name__ == "__main__":
    main()
