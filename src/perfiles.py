"""Perfiles de empresa: parámetros que una PYME puede cambiar desde la app, sin editar archivos.

Un PERFIL es un archivo de perfiles/*.yaml con solo los valores que cambian respecto de config.yaml
(el almacén, la unidad que se reparte, la capacidad, los costos...). Sirve de punto de partida: la
PYME elige el que más se parece a su negocio y ajusta cada valor a mano en la barra lateral.

Dentro del código la capacidad se llama `capacidad_cajas` y los pedidos `boxes` por historia del
proyecto, pero en pantalla se muestra la unidad de cada empresa (cajas, cabezas, kilos, paquetes...)
que viene de negocio.unidad.
"""

import copy
from pathlib import Path

import yaml

# (ruta en config, etiqueta, tipo, mínimo, paso, ayuda)
CAMPOS = [
    ("almacen.lat", "Latitud del almacén", "float", -90.0, 0.0001, "Punto de salida y regreso de las camionetas."),
    ("almacen.lon", "Longitud del almacén", "float", -180.0, 0.0001, "Con signo negativo en México (ej. -98.7591)."),
    ("flota.camionetas_propias", "Camionetas propias", "int", 1, 1, "Las que la empresa ya tiene."),
    ("flota.capacidad_cajas", "Capacidad por camioneta (en su unidad)", "int", 1, 1,
     "Cuánto cabe en una camioneta, en la unidad que reparte (cajas, cabezas, kilos...)."),
    ("flota.max_urgentes", "Máximo de fletes urgentes al día", "int", 0, 1, "Tope de unidades que se pueden contratar el mismo día."),
    ("tiempos.jornada_h", "Jornada máxima por ruta (horas)", "float", 1.0, 0.5, "Incluye manejo y descargas."),
    ("tiempos.servicio_min", "Minutos por parada", "float", 0.0, 1.0, "Descarga y firma en cada cliente."),
    ("tiempos.velocidad_kmh", "Velocidad promedio (km/h)", "float", 5.0, 1.0, "Urbana ~22; carretera ~80-100."),
    ("distancias.factor_circuidad", "Factor de circuidad", "float", 1.0, 0.01,
     "Km por calle ÷ km en línea recta. Típico: 1.3 en ciudad."),
    ("costos.costo_km", "Costo por km (MXN)", "float", 0.0, 0.1, "Combustible y mantenimiento."),
    ("costos.costo_propia", "Costo diario de una camioneta propia que sale (MXN)", "float", 0.0, 50.0,
     "Operador, seguro y desgaste."),
    ("costos.tarifa_reservada", "Tarifa de un flete reservado el día anterior (MXN)", "float", 0.0, 50.0,
     "Se paga aunque no se use."),
    ("costos.tarifa_urgente", "Tarifa de un flete urgente del mismo día (MXN)", "float", 0.0, 50.0,
     "Debe ser mayor que la reservada para que reservar convenga."),
    ("reserva.u_max", "Llenado máximo alcanzable (0 a 1)", "float", 0.1, 0.01,
     "Una camioneta casi nunca se llena al 100 %."),
]
CAMPOS_TEXTO = [
    ("negocio.unidad", "Unidad que reparte (plural)", "cajas"),
    ("negocio.unidad_singular", "Unidad (singular)", "caja"),
]
POR_OMISION = "Distribuidora urbana del artículo (por omisión)"


def leer(config, ruta):
    """config['costos.costo_km'] → config['costos']['costo_km']."""
    valor = config
    for parte in ruta.split("."):
        valor = valor[parte]
    return valor


def escribir(config, ruta, valor):
    """Cambia un valor (modifica `config`)."""
    *camino, hoja = ruta.split(".")
    destino = config
    for parte in camino:
        destino = destino.setdefault(parte, {})
    destino[hoja] = valor


def fusionar(base, cambios):
    """Copia de `base` con `cambios` encima, sección por sección."""
    resultado = copy.deepcopy(base)
    for clave, valor in (cambios or {}).items():
        if isinstance(valor, dict) and isinstance(resultado.get(clave), dict):
            resultado[clave] = fusionar(resultado[clave], valor)
        else:
            resultado[clave] = copy.deepcopy(valor)
    return resultado


def cargar_perfiles(carpeta):
    """{nombre: {"descripcion": ..., "config": {...}}} de cada perfiles/*.yaml, ordenados por nombre."""
    perfiles = {}
    for archivo in sorted(Path(carpeta).glob("*.yaml")):
        contenido = yaml.safe_load(archivo.read_text(encoding="utf-8")) or {}
        if "nombre" not in contenido or "config" not in contenido:
            raise ValueError(f"{archivo.name}: un perfil necesita 'nombre' y 'config'.")
        perfiles[contenido["nombre"]] = {"descripcion": contenido.get("descripcion", ""),
                                         "config": contenido["config"], "archivo": archivo.name}
    return perfiles


def config_del_perfil(base, perfiles, nombre):
    """La configuración resultante de partir del perfil elegido (o de config.yaml si es el de por omisión)."""
    if nombre == POR_OMISION or nombre not in perfiles:
        return copy.deepcopy(base)
    return fusionar(base, perfiles[nombre]["config"])


def aplicar_valores(config, valores):
    """Copia de `config` con los valores {ruta: valor} que escribió la persona en la barra lateral."""
    nueva = copy.deepcopy(config)
    for ruta, valor in valores.items():
        escribir(nueva, ruta, valor)
    return nueva


def diferencias(config_a, config_b):
    """Rutas de los parámetros editables que valen distinto en las dos configuraciones."""
    rutas = [c[0] for c in CAMPOS] + [c[0] for c in CAMPOS_TEXTO]
    return [r for r in rutas if leer(config_a, r) != leer(config_b, r)]


def problemas(config):
    """Combinaciones de valores que no tienen sentido, en español (lista vacía si todo está bien)."""
    avisos = []
    c = config["costos"]
    if c["tarifa_urgente"] <= c["tarifa_reservada"]:
        avisos.append("La tarifa urgente debe ser mayor que la reservada; si no, reservar nunca conviene "
                      "y el pronóstico no puede ahorrar.")
    if not -90 <= config["almacen"]["lat"] <= 90 or not -180 <= config["almacen"]["lon"] <= 180:
        avisos.append("Las coordenadas del almacén no son válidas.")
    if config["flota"]["camionetas_propias"] < 1:
        avisos.append("Debe haber al menos una camioneta propia.")
    return avisos
