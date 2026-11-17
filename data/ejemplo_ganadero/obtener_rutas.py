"""Ubica la engorda y los orígenes y guarda las distancias y rutas por carretera (se corre una vez).

- Coordenadas: Nominatim (OpenStreetMap), máximo 1 petición por segundo.
- Distancias, horas y trazo de cada ruta: OSRM (router.project-osrm.org), por carreteras reales.

Escribe en esta carpeta:
  lugares.csv     nombre, lat, lon (la engorda primero)
  distancias.csv  km y horas por carretera entre cada par de lugares
  rutas.json      trazo [lat, lon] de cada ruta, para dibujarla en el mapa

Uso:  .venv/bin/python data/ejemplo_ganadero/obtener_rutas.py
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

CARPETA = Path(__file__).resolve().parent
AGENTE = {"User-Agent": "ruteo-pymes-itp/1.0 (proyecto escolar)"}

# Dirección de la engorda (dato de la empresa): Carretera Cardel-Veracruz km 226, José Cardel, Ver.
# Nominatim no ubica el kilómetro exacto: se usa José Cardel y se marca como aproximado.
BUSQUEDAS = {
    "Engorda (Cardel, Ver.)": "José Cardel, Veracruz, México",
    "Balancán": "Balancán, Tabasco, México",
    "Tenosique": "Tenosique de Pino Suárez, Tabasco, México",
    "Villahermosa": "Villahermosa, Tabasco, México",
    "Emiliano Zapata": "Emiliano Zapata, Tabasco, México",
    "Palenque/Ocosingo": "Palenque, Chiapas, México",
    "Pichucalco": "Pichucalco, Chiapas, México",
    "Medellín": "Medellín de Bravo, Veracruz, México",
}


def pedir(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENTE), timeout=30) as r:
        return json.loads(r.read())


def ubicar():
    filas = []
    for nombre, texto in BUSQUEDAS.items():
        r = pedir("https://nominatim.openstreetmap.org/search?" +
                  urllib.parse.urlencode({"q": texto, "format": "json", "limit": 1}))
        if not r:
            raise SystemExit(f"Nominatim no encontró «{texto}»")
        filas.append({"nombre": nombre, "lat": float(r[0]["lat"]), "lon": float(r[0]["lon"]), "busqueda": texto})
        time.sleep(1.1)   # máximo 1 petición por segundo
    return pd.DataFrame(filas)


def rutas(lugares):
    coords = ";".join(f"{f.lon},{f.lat}" for f in lugares.itertuples())
    tabla = pedir(f"https://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance,duration")
    nombres = list(lugares["nombre"])
    filas = [{"desde": a, "hasta": b, "km": tabla["distances"][i][j] / 1000, "horas": tabla["durations"][i][j] / 3600}
             for i, a in enumerate(nombres) for j, b in enumerate(nombres) if i != j]
    trazos = {}
    for i, a in enumerate(nombres):
        for j, b in enumerate(nombres):
            if j <= i:
                continue
            p, q = lugares.iloc[i], lugares.iloc[j]
            r = pedir(f"https://router.project-osrm.org/route/v1/driving/{p.lon},{p.lat};{q.lon},{q.lat}"
                      "?overview=simplified&geometries=geojson")
            trazos[f"{a}|{b}"] = [[round(y, 5), round(x, 5)] for x, y in r["routes"][0]["geometry"]["coordinates"]]
            time.sleep(1.1)
    return pd.DataFrame(filas), trazos


if __name__ == "__main__":
    lugares = ubicar()
    lugares.to_csv(CARPETA / "lugares.csv", index=False)
    distancias, trazos = rutas(lugares)
    distancias.round(2).to_csv(CARPETA / "distancias.csv", index=False)
    (CARPETA / "rutas.json").write_text(json.dumps(trazos))
    print(lugares)
    print(distancias[distancias["desde"] == lugares["nombre"][0]].round(1))
