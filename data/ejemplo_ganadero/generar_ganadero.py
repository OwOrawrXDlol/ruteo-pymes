"""Genera ordenes.csv: dos años de órdenes de compra (OC) de acopio de ganado, con semilla fija.

El primer año es historial (para entrenar el pronóstico); el segundo es el que se compara.
Los rangos vienen de lo que contó la empresa; lo demás son supuestos (ver src/ganadero.py y LEEME.md).

Uso:  .venv/bin/python data/ejemplo_ganadero/generar_ganadero.py
"""

import sys
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA.parent.parent))   # para poder importar src/

from src.ganadero import generar_ordenes  # noqa: E402

if __name__ == "__main__":
    ordenes = generar_ordenes()
    ordenes.to_csv(CARPETA / "ordenes.csv", index=False)
    semanas = ordenes["fecha_embarque"].map(lambda d: d.isocalendar()[:2]).nunique()
    print(f"{len(ordenes)} OCs en {semanas} semanas ({len(ordenes) / semanas:.1f} jaulas por semana), "
          f"{ordenes['cabezas'].sum():,} cabezas, del {ordenes['fecha_embarque'].min()} "
          f"al {ordenes['fecha_embarque'].max()}")
