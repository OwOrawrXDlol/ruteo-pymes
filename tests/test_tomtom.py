"""Pruebas del paso 5 de TRAFICO.md: modo TomTom, sin internet.

Las respuestas de TomTom se imitan con el formato de su documentación (docs.tomtom.com,
revisada el 12/11/2026). Así se prueba la caché, el contador y cómo se arma la matriz.
"""

import io
import json
import urllib.error
import urllib.parse
from datetime import date

import numpy as np
import pytest

from src.replan import CAMBIAR, estados_al_salir, puntos_del_dia, replanear_con_tomtom
from src.tomtom import ClienteTomTom, ErrorTomTom, leer_clave, transacciones_matrix
from src.traffic import costo_matriz, incidentes_de_tomtom, medir_con_matrix, trafico_tomtom
from tests.test_replan import CONFIG, CUADRO, lugar, paradas, plan

CLAVE = "CLAVE-SECRETA-DE-PRUEBA"
TOMTOM = {
    "minutos_cache": 5, "modo_viaje": "car", "max_transacciones_matrix": 500, "margen_bbox_km": 1,
    "radio_estimado_m": 150, "factor_por_magnitud": {0: 1.2, 1: 1.2, 2: 1.5, 3: 2.5, 4: 2.5},
    "limite_mensual": {"routing": 20000, "incidentes": 2500, "matrix": 2500},
}
CFG = {**CONFIG, "tomtom": TOMTOM,
       "trafico": {**CONFIG["trafico"], "simulado": {"radio_m": 300, "factor": 2.5, "minutos_cerrado": 1000}}}


class TomTomFalso:
    """Imita a TomTom. `lento` = pares de puntos (lat, lon) cuyo tramo va `razon` veces más lento."""

    def __init__(self, lento=(), razon=5.0, incidentes=None):
        self.lento = {tuple(sorted(p)) for p in lento}
        self.razon = razon
        self.respuesta_incidentes = incidentes or {"incidents": []}
        self.llamadas = []

    def _segundos(self, a, b):
        sin = 600.0
        con = sin * self.razon if tuple(sorted((a, b))) in self.lento else sin
        return con, con - sin

    def __call__(self, url, cuerpo):
        partes = urllib.parse.urlparse(url)
        self.llamadas.append((partes.path, cuerpo))
        if "incidentDetails" in partes.path:
            return self.respuesta_incidentes
        if "calculateRoute" in partes.path:
            texto = partes.path.split("/calculateRoute/")[1].rsplit("/", 1)[0]
            puntos = [tuple(round(float(x), 6) for x in p.split(",")) for p in texto.split(":")]
            tramos = [self._segundos(a, b) for a, b in zip(puntos, puntos[1:])]
            return {"routes": [{
                "summary": {"travelTimeInSeconds": sum(t for t, _ in tramos),
                            "trafficDelayInSeconds": sum(d for _, d in tramos),
                            "noTrafficTravelTimeInSeconds": 600 * len(tramos), "lengthInMeters": 1000},
                "legs": [{"summary": {"travelTimeInSeconds": t, "trafficDelayInSeconds": d}} for t, d in tramos],
            }]}
        if "matrix" in partes.path:
            o = [(p["point"]["latitude"], p["point"]["longitude"]) for p in cuerpo["origins"]]
            d = [(p["point"]["latitude"], p["point"]["longitude"]) for p in cuerpo["destinations"]]
            datos = []
            for i, a in enumerate(o):
                for j, b in enumerate(d):
                    t, r = self._segundos(a, b)
                    datos.append({"originIndex": i, "destinationIndex": j,
                                  "routeSummary": {"travelTimeInSeconds": t, "trafficDelayInSeconds": r}})
            return {"data": datos}
        raise AssertionError(f"consulta inesperada: {partes.path}")


def cliente(tmp_path, falso, reloj=lambda: 1000.0, hoy=lambda: date(2026, 11, 12)):
    return ClienteTomTom(CLAVE, tmp_path / "cache", minutos_cache=5, abrir=falso, reloj=reloj, hoy=hoy)


def redondo(lat, lon):
    return (round(lat, 6), round(lon, 6))


# --- Reglas de cobro y clave ---

def test_transacciones_de_matrix_como_en_la_documentacion():
    assert transacciones_matrix(2, 3) == 6
    assert transacciones_matrix(5, 5) == 25
    assert transacciones_matrix(6, 100) == 500
    assert transacciones_matrix(1000, 1000) == 5000


def test_costo_de_matriz_en_bloques_de_100_celdas():
    assert costo_matriz(4) == (1, 16)    # 4 × 4, un solo bloque
    # 12 puntos: bloques 10×10 (50), 10×2 (20), 2×10 (20) y 2×2 (4)
    assert costo_matriz(12) == (4, 94)
    # Antes de salir con 40 paradas + almacén: 881 transacciones, más que el tope de 500 por cálculo
    assert costo_matriz(41) == (25, 881)


def test_leer_clave(tmp_path, monkeypatch):
    monkeypatch.delenv("TOMTOM_API_KEY", raising=False)
    assert leer_clave(tmp_path / "no-existe.env") is None
    env = tmp_path / ".env"
    env.write_text("TOMTOM_API_KEY=abc123\n")
    assert leer_clave(env) == "abc123"


def test_sin_clave_no_se_crea_el_cliente(tmp_path):
    with pytest.raises(ErrorTomTom, match="TOMTOM_API_KEY"):
        ClienteTomTom("", tmp_path)


# --- Caché y contador ---

def test_cache_de_5_minutos_y_contador(tmp_path):
    falso = TomTomFalso()
    ahora = [1000.0]
    c = cliente(tmp_path, falso, reloj=lambda: ahora[0])
    puntos = [lugar(0, 0), lugar(1, 0)]
    c.ruta(puntos)
    c.ruta(puntos)                 # misma consulta al instante: sale de la caché
    assert len(falso.llamadas) == 1 and c.uso()["routing"] == {"hoy": 1, "mes": 1}
    ahora[0] += 6 * 60             # 6 minutos después ya no sirve
    c.ruta(puntos)
    assert len(falso.llamadas) == 2 and c.uso()["routing"]["hoy"] == 2


def test_contador_separa_dia_y_mes(tmp_path):
    falso = TomTomFalso()
    dia = [date(2026, 11, 11)]
    c = cliente(tmp_path, falso, hoy=lambda: dia[0])
    c.incidentes(-98.1, 19.9, -97.9, 20.1)
    dia[0] = date(2026, 11, 12)
    c.matriz([lugar(0, 0), lugar(1, 0)], [lugar(0, 0), lugar(1, 0)])
    uso = c.uso()
    assert uso["incidentes"] == {"hoy": 0, "mes": 1}
    assert uso["matrix"] == {"hoy": 4, "mes": 4}  # 2 × 2 = 4 transacciones


def test_la_clave_no_queda_en_la_cache(tmp_path):
    c = cliente(tmp_path, TomTomFalso())
    c.ruta([lugar(0, 0), lugar(1, 0)])
    for archivo in (tmp_path / "cache").iterdir():
        assert CLAVE not in archivo.read_text()


def test_errores_http_en_espanol_y_sin_la_clave(tmp_path, monkeypatch):
    def falla(pedido, timeout):
        raise urllib.error.HTTPError(pedido.full_url, 403, "Forbidden", {}, io.BytesIO(b""))
    monkeypatch.setattr("urllib.request.urlopen", falla)
    c = ClienteTomTom(CLAVE, tmp_path / "cache")
    with pytest.raises(ErrorTomTom) as error:
        c.ruta([lugar(0, 0), lugar(1, 0)])
    assert "403" in str(error.value) and "cuota" in str(error.value) and CLAVE not in str(error.value)


def test_matriz_rechaza_mas_de_100_celdas(tmp_path):
    c = cliente(tmp_path, TomTomFalso())
    with pytest.raises(ValueError, match="100 celdas"):
        c.matriz([lugar(k, 0) for k in range(11)], [lugar(k, 0) for k in range(10)])


# --- Incidentes ---

def test_incidentes_de_tomtom():
    respuesta = {"incidents": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-98.0, 20.0]},
         "properties": {"iconCategory": 1, "magnitudeOfDelay": 3, "delay": 420,
                        "events": [{"description": "Accidente", "code": 1}], "from": "Av. A", "to": "Calle B"}},
        {"type": "Feature", "geometry": {"type": "LineString",
                                         "coordinates": [[-98.0, 20.0], [-98.01, 20.01], [-98.02, 20.02]]},
         "properties": {"iconCategory": 8, "magnitudeOfDelay": 4}},
    ]}
    choque, cierre = incidentes_de_tomtom(respuesta, CFG)
    assert (choque.lat, choque.lon) == (20.0, -98.0)  # TomTom da [lon, lat]
    assert choque.tipo == "accidente" and choque.magnitud == 3 and choque.factor == 2.5
    assert choque.retraso_seg == 420 and choque.fuente == "tomtom"
    assert "Accidente" in choque.descripcion and "Av. A → Calle B" in choque.descripcion
    assert cierre.tipo == "cierre" and cierre.cerrado and len(cierre.puntos) == 3
    assert (cierre.lat, cierre.lon) == (20.01, -98.01)  # el punto de en medio del tramo


# --- Revisión y matriz ---

def puntos_cuadro():
    tabla = paradas(*CUADRO)
    lats, lons, _, minutos = puntos_del_dia(tabla, CFG)
    nodo = [redondo(la, lo) for la, lo in zip(lats, lons)]
    return tabla, lats, lons, minutos, nodo


def test_revision_aplica_la_razon_de_tomtom_a_los_tramos_de_la_ruta(tmp_path):
    tabla, lats, lons, minutos, nodo = puntos_cuadro()
    falso = TomTomFalso(lento=[(nodo[1], nodo[2])], razon=5.0)  # el lado A-B va 5 veces más lento
    c = cliente(tmp_path, falso)
    t = trafico_tomtom(lats, lons, minutos, estados_al_salir(plan([0, 1, 2])), c, CFG)
    assert t.fuente == "tomtom" and t.consultado is not None
    assert t.minutos[1][2] == pytest.approx(minutos[1][2] * 5)      # A→B medido: ×5
    assert t.minutos[0][1] == pytest.approx(minutos[0][1])          # almacén→A medido: sin cambio
    assert t.medidos[1][2] and not t.medidos[1][3]                  # A→C no está en la ruta
    assert c.uso()["routing"]["hoy"] == 1 and c.uso()["incidentes"]["hoy"] == 1
    assert t.rutas_tomtom[0]["retraso_min"] == pytest.approx(40)    # 2400 s de retraso


def test_matrix_mide_los_tramos_que_faltaban(tmp_path):
    tabla, lats, lons, minutos, nodo = puntos_cuadro()
    c = cliente(tmp_path, TomTomFalso(lento=[(nodo[1], nodo[2])]))
    t = trafico_tomtom(lats, lons, minutos, estados_al_salir(plan([0, 1, 2])), c, CFG)
    usadas = medir_con_matrix(t, lats, lons, [0, 1, 2, 3], c, CFG)
    assert usadas == 16 and t.medidos.all()
    assert t.minutos[2][1] == pytest.approx(minutos[2][1] * 5)      # B→A también va lento
    assert t.minutos[1][3] == pytest.approx(minutos[1][3])          # A→C, sin tráfico


def test_matrix_no_se_consulta_si_rebasa_el_tope(tmp_path):
    tabla, lats, lons, minutos, nodo = puntos_cuadro()
    falso = TomTomFalso()
    c = cliente(tmp_path, falso)
    t = trafico_tomtom(lats, lons, minutos, estados_al_salir(plan([0, 1, 2])), c, CFG)
    tope = {**CFG, "tomtom": {**TOMTOM, "max_transacciones_matrix": 10}}
    antes = len(falso.llamadas)
    assert medir_con_matrix(t, lats, lons, [0, 1, 2, 3], c, tope) == 0
    assert len(falso.llamadas) == antes
    assert "costaría 16 transacciones" in t.avisos[0]


def test_replanear_con_tomtom_cambia_la_ruta(tmp_path):
    tabla, lats, lons, minutos, nodo = puntos_cuadro()
    incidente = {"incidents": [{"type": "Feature",
                                "geometry": {"type": "Point", "coordinates": [lugar(4, 2)[1], lugar(4, 2)[0]]},
                                "properties": {"iconCategory": 1, "magnitudeOfDelay": 3, "delay": 2400}}]}
    falso = TomTomFalso(lento=[(nodo[1], nodo[2])], incidentes=incidente)
    c = cliente(tmp_path, falso)
    p = plan([0, 1, 2])
    r, t = replanear_con_tomtom(p, tabla, CFG, estados_al_salir(p), c)
    fila = r["tabla"].iloc[0]
    assert fila["afectada"] and "incidente" in fila["criterios"] and fila["decision"] == CAMBIAR
    assert r["rutas"][0]["pendientes_adoptadas"] in ([0, 2, 1], [1, 2, 0])
    # Matrix solo para los puntos de la camioneta afectada: almacén + 3 paradas = 4 × 4
    assert c.uso()["matrix"]["hoy"] == 16
    assert sum("matrix" in ruta for ruta, _ in falso.llamadas) == 1
