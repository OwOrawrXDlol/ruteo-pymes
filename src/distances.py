"""Distancias y tiempos de viaje entre puntos.

Por ahora se usa la distancia en línea recta sobre la Tierra (fórmula de
haversine) multiplicada por un factor de circuidad (las calles no van en línea
recta), igual que en el artículo. Esta es la única parte que habría que cambiar
para usar distancias por calle reales (por ejemplo, con OSRM).
"""

import numpy as np

RADIO_TIERRA_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia en línea recta (km) entre dos puntos dados en grados. Acepta arreglos."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(a))


def matriz_distancias(lats, lons, factor_circuidad):
    """Matriz de km por calle aproximados: entrada [i][j] = km de ir del punto i al j."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    linea_recta = haversine_km(lats[:, None], lons[:, None], lats[None, :], lons[None, :])
    return linea_recta * factor_circuidad


def matriz_tiempos_min(distancias_km, velocidad_kmh):
    """Minutos de manejo para cada par de puntos, a velocidad constante."""
    return distancias_km / velocidad_kmh * 60
