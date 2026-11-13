"""Cliente de TomTom para el modo A (tráfico en vivo).

Revisado contra la documentación de docs.tomtom.com el 12 de noviembre de 2026:
  - Routing API v1, calculateRoute: GET, una llamada por camioneta con sus paradas en orden.
    traffic=true, departAt=now, computeTravelTimeFor=all. Hasta 150 puntos por ruta.
  - Traffic Incident Details API v5: GET con bbox (minLon,minLat,maxLon,maxLat), máximo 10,000 km².
  - Matrix Routing v2 síncrona: POST. Con tráfico en vivo (traffic=live, departAt=now) el
    límite es de 100 celdas por llamada, así que las matrices grandes se parten en bloques.
    Cobro: si orígenes y destinos pasan de 5 se cobra (el mayor) × 5; si no, orígenes × destinos.
  - Plan gratuito: cuota MENSUAL por API (Routing 20,000; Incident Details 2,500; Matrix 2,500).
    TRAFICO.md hablaba de un límite diario: TomTom lo cambió. Por eso el contador lleva el día y el mes.

Ahorro de cuota: cada respuesta se guarda en disco y, si se repite la misma consulta antes de
`minutos_cache` minutos, se usa la guardada sin llamar a TomTom (y sin contarla).
La clave nunca se guarda en la caché ni aparece en los mensajes de error.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

BASE = "https://api.tomtom.com"
APIS = {"routing": "Routing", "incidentes": "Incident Details", "matrix": "Matrix Routing v2"}
CELDAS_MAX_SINCRONA = 100
CAMPOS_INCIDENTES = ("{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,"
                     "events{description,code},startTime,endTime,from,to,length,delay}}}")


class ErrorTomTom(Exception):
    """Algo falló al hablar con TomTom. El mensaje ya viene en español para mostrarlo."""


def leer_clave(ruta_env=None):
    """La clave TOMTOM_API_KEY de .env (o del entorno). None si no hay."""
    load_dotenv(ruta_env, override=False)
    clave = os.environ.get("TOMTOM_API_KEY", "").strip()
    return clave or None


def transacciones_matrix(origenes, destinos):
    """Cuántas transacciones cobra TomTom por una matriz (regla de 'discounted transaction billing')."""
    if origenes > 5 and destinos > 5:
        return max(origenes, destinos) * 5
    return origenes * destinos


def _coordenadas(puntos):
    return ":".join(f"{lat:.6f},{lon:.6f}" for lat, lon in puntos)


def _punto_json(lat, lon):
    return {"point": {"latitude": round(float(lat), 6), "longitude": round(float(lon), 6)}}


MENSAJES_HTTP = {
    400: "la consulta no es válida (revisa las coordenadas)",
    403: "la clave no es válida o ya se acabó la cuota gratuita",
    429: "demasiadas consultas seguidas; espera unos segundos",
}


class ClienteTomTom:
    def __init__(self, clave, carpeta_cache, minutos_cache=5, modo_viaje="car", abrir=None,
                 reloj=time.time, hoy=date.today):
        """abrir, reloj y hoy se pueden reemplazar en las pruebas para no usar internet."""
        if not clave:
            raise ErrorTomTom("Falta la clave TOMTOM_API_KEY en .env.")
        self.clave = clave
        self.carpeta = Path(carpeta_cache)
        self.minutos_cache = minutos_cache
        self.modo_viaje = modo_viaje
        self.abrir = abrir or self._abrir_http
        self.reloj = reloj
        self.hoy = hoy

    # --- HTTP, caché y contador ---

    def _abrir_http(self, url, cuerpo):
        datos = json.dumps(cuerpo).encode("utf-8") if cuerpo is not None else None
        pedido = urllib.request.Request(url, data=datos, method="POST" if datos else "GET",
                                        headers={"Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(pedido, timeout=20) as respuesta:
                return json.load(respuesta)
        except urllib.error.HTTPError as error:
            motivo = MENSAJES_HTTP.get(error.code, "error del servicio")
            raise ErrorTomTom(f"TomTom respondió {error.code}: {motivo}.") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ErrorTomTom("No se pudo conectar con TomTom (¿hay internet?). "
                              "Puedes usar el modo de incidente simulado.") from None

    def _consultar(self, api, ruta, parametros=None, cuerpo=None, transacciones=1):
        """Consulta con caché. ruta y parámetros NO llevan la clave (se agrega solo al llamar)."""
        parametros = dict(parametros or {})
        firma = json.dumps([ruta, sorted(parametros.items()), cuerpo], sort_keys=True)
        archivo = self.carpeta / f"{hashlib.sha256(firma.encode()).hexdigest()[:32]}.json"
        if archivo.exists():
            guardado = json.loads(archivo.read_text(encoding="utf-8"))
            if self.reloj() - guardado["guardado"] < self.minutos_cache * 60:
                return guardado["respuesta"]

        url = f"{BASE}{ruta}?{urllib.parse.urlencode({**parametros, 'key': self.clave})}"
        respuesta = self.abrir(url, cuerpo)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        archivo.write_text(json.dumps({"guardado": self.reloj(), "respuesta": respuesta}), encoding="utf-8")
        self._contar(api, transacciones)
        return respuesta

    @property
    def _archivo_contador(self):
        return self.carpeta / "contador.json"

    def _leer_contador(self):
        if self._archivo_contador.exists():
            return json.loads(self._archivo_contador.read_text(encoding="utf-8"))
        return {}

    def _contar(self, api, transacciones):
        contador = self._leer_contador()
        dia = self.hoy().isoformat()
        contador.setdefault(api, {})
        contador[api][dia] = contador[api].get(dia, 0) + transacciones
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self._archivo_contador.write_text(json.dumps(contador, indent=1), encoding="utf-8")

    def uso(self):
        """{api: {"hoy": n, "mes": n}} en transacciones (lo que cobra TomTom)."""
        contador = self._leer_contador()
        hoy = self.hoy()
        mes = hoy.isoformat()[:7]
        return {api: {"hoy": contador.get(api, {}).get(hoy.isoformat(), 0),
                      "mes": sum(n for d, n in contador.get(api, {}).items() if d.startswith(mes))}
                for api in APIS}

    # --- Las tres consultas ---

    def ruta(self, puntos):
        """Routing API con tráfico, saliendo ahora, por los puntos dados en ese orden."""
        if len(puntos) < 2:
            raise ValueError("Una ruta necesita al menos 2 puntos.")
        if len(puntos) > 150:
            raise ErrorTomTom("TomTom acepta hasta 150 puntos por ruta.")
        return self._consultar("routing", f"/routing/1/calculateRoute/{_coordenadas(puntos)}/json", {
            "traffic": "true", "departAt": "now", "computeTravelTimeFor": "all",
            "routeType": "fastest", "travelMode": self.modo_viaje,
        })

    def incidentes(self, min_lon, min_lat, max_lon, max_lat):
        """Traffic Incident Details v5 dentro del recuadro."""
        return self._consultar("incidentes", "/traffic/services/5/incidentDetails", {
            "bbox": f"{min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}",
            "fields": CAMPOS_INCIDENTES, "language": "es-ES", "timeValidityFilter": "present",
        })

    def matriz(self, origenes, destinos):
        """Matrix Routing v2 síncrona con tráfico en vivo. origenes/destinos: [(lat, lon), ...]."""
        if len(origenes) * len(destinos) > CELDAS_MAX_SINCRONA:
            raise ValueError(f"Con tráfico en vivo la matriz síncrona acepta hasta {CELDAS_MAX_SINCRONA} celdas.")
        modo = self.modo_viaje if self.modo_viaje in ("car", "truck") else "car"  # Matrix no acepta "van"
        cuerpo = {
            "origins": [_punto_json(*p) for p in origenes],
            "destinations": [_punto_json(*p) for p in destinos],
            "options": {"departAt": "now", "traffic": "live", "routeType": "fastest", "travelMode": modo},
        }
        return self._consultar("matrix", "/routing/matrix/2", cuerpo=cuerpo,
                               transacciones=transacciones_matrix(len(origenes), len(destinos)))
