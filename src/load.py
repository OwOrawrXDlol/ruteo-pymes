"""Carga y validación de los datos de entrada.

Lee los CSV del contrato (pedidos, clientes y los opcionales), revisa que estén
bien y los deja listos para el ruteo: una "parada" por cliente y día, con sus
cajas sumadas. Si un cliente pide más de lo que cabe en una camioneta, su
pedido se parte en varias paradas en el mismo lugar.

Cuando algo está mal se lanza ErrorDatos con TODOS los problemas encontrados,
explicados en español, para corregirlos de una sola vez.
"""

from pathlib import Path

import pandas as pd
import yaml

# Columnas obligatorias de cada archivo (las opcionales no se exigen)
COLUMNAS_PEDIDOS = ["order_id", "date", "client_id", "boxes"]
COLUMNAS_CLIENTES = ["client_id", "lat", "lon"]
COLUMNAS_OPERACION = ["date", "km", "vehicles", "urgent_hires", "reserved_hires"]
COLUMNAS_RESERVA = ["date", "reserved_extra"]

# Cuántos ejemplos mostrar en cada mensaje de error
MAX_EJEMPLOS = 10


class ErrorDatos(Exception):
    """Error en los datos de entrada. El mensaje lista todos los problemas."""

    def __init__(self, archivo, problemas):
        self.archivo = archivo
        self.problemas = problemas
        lista = "\n".join(f"  - {p}" for p in problemas)
        super().__init__(f"Problemas en {archivo}:\n{lista}")


def _ejemplos(valores):
    """Texto corto con algunos valores, para los mensajes de error."""
    valores = [str(v) for v in valores]
    texto = ", ".join(valores[:MAX_EJEMPLOS])
    if len(valores) > MAX_EJEMPLOS:
        texto += f" (y {len(valores) - MAX_EJEMPLOS} más)"
    return texto


def _leer_csv(fuente, nombre, columnas):
    """Lee un CSV (ruta o archivo subido) como texto y revisa que tenga las columnas."""
    try:
        tabla = pd.read_csv(fuente, dtype=str, keep_default_na=False)
    except Exception as e:
        raise ErrorDatos(nombre, [f"no se pudo leer el archivo: {e}"])
    tabla.columns = [c.strip() for c in tabla.columns]
    faltan = [c for c in columnas if c not in tabla.columns]
    if faltan:
        raise ErrorDatos(nombre, [
            f"faltan columnas: {', '.join(faltan)} "
            f"(columnas encontradas: {', '.join(tabla.columns)})"
        ])
    # Quitar espacios sobrantes en todas las celdas
    for c in tabla.columns:
        tabla[c] = tabla[c].str.strip()
    return tabla


def _revisar_fechas(tabla, problemas, id_col=None):
    """Convierte la columna date a fecha (YYYY-MM-DD) y anota las que no sirven."""
    fechas = pd.to_datetime(tabla["date"], format="%Y-%m-%d", errors="coerce")
    malas = tabla.loc[fechas.isna(), id_col or "date"]
    if len(malas):
        problemas.append(f"{len(malas)} fechas no tienen formato YYYY-MM-DD: {_ejemplos(malas)}")
    tabla["date"] = fechas.dt.date


def _revisar_enteros(tabla, columna, problemas, id_col, minimo):
    """Convierte una columna a entero y anota los valores no válidos (vacíos, texto, decimales, < minimo)."""
    numeros = pd.to_numeric(tabla[columna], errors="coerce")
    malos = numeros.isna() | (numeros % 1 != 0) | (numeros < minimo)
    if malos.any():
        ejemplos = [f"{i} ('{v}')" for i, v in zip(tabla.loc[malos, id_col], tabla.loc[malos, columna])]
        regla = "un entero mayor que 0" if minimo == 1 else f"un entero mayor o igual a {minimo}"
        problemas.append(f"{malos.sum()} valores de '{columna}' no son {regla}: {_ejemplos(ejemplos)}")
    tabla[columna] = numeros.where(~malos).astype("Int64")


def _revisar_repetidos(tabla, columna, problemas):
    repetidos = tabla.loc[tabla[columna].duplicated(), columna].unique()
    if len(repetidos):
        problemas.append(f"{len(repetidos)} valores de '{columna}' están repetidos: {_ejemplos(repetidos)}")


def cargar_config(ruta="config.yaml"):
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def valores_supuestos(ruta="config.yaml"):
    """Lista de (parámetro, valor, comentario) de config.yaml marcados como supuesto."""
    supuestos, seccion = [], ""
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        if not linea.strip() or linea.strip().startswith("#"):
            continue
        clave, _, resto = linea.partition(":")
        valor, _, comentario = resto.partition("#")
        if not valor.strip():  # encabezado de sección, p. ej. "flota:"
            seccion = clave.strip()
        elif "supuesto" in comentario:
            supuestos.append((f"{seccion}.{clave.strip()}", valor.strip(), comentario.strip()))
    return supuestos


def cargar_pedidos(fuente, nombre="pedidos.csv"):
    """Lee y valida pedidos.csv. Devuelve una tabla con date como fecha y boxes como entero."""
    pedidos = _leer_csv(fuente, nombre, COLUMNAS_PEDIDOS)
    problemas = []

    sin_cliente = pedidos.loc[pedidos["client_id"] == "", "order_id"]
    if len(sin_cliente):
        problemas.append(f"{len(sin_cliente)} pedidos no tienen client_id: {_ejemplos(sin_cliente)}")
    _revisar_repetidos(pedidos, "order_id", problemas)
    _revisar_fechas(pedidos, problemas, id_col="order_id")
    _revisar_enteros(pedidos, "boxes", problemas, "order_id", minimo=1)

    # La hora es opcional; si viene, debe ser HH:MM (se permiten vacías)
    if "time" in pedidos.columns:
        con_hora = pedidos["time"] != ""
        horas = pd.to_datetime(pedidos["time"], format="%H:%M", errors="coerce")
        malas = pedidos.loc[con_hora & horas.isna(), "order_id"]
        if len(malas):
            problemas.append(f"{len(malas)} horas no tienen formato HH:MM: {_ejemplos(malas)}")

    if problemas:
        raise ErrorDatos(nombre, problemas)
    return pedidos


def cargar_clientes(fuente, nombre="clientes.csv"):
    """Lee y valida clientes.csv. Todo cliente debe tener coordenadas válidas."""
    clientes = _leer_csv(fuente, nombre, COLUMNAS_CLIENTES)
    problemas = []

    _revisar_repetidos(clientes, "client_id", problemas)
    lat = pd.to_numeric(clientes["lat"], errors="coerce")
    lon = pd.to_numeric(clientes["lon"], errors="coerce")
    sin_coordenadas = clientes.loc[lat.isna() | lon.isna(), "client_id"]
    if len(sin_coordenadas):
        problemas.append(
            f"{len(sin_coordenadas)} clientes no tienen coordenadas (lat/lon vacías o no numéricas): "
            f"{_ejemplos(sin_coordenadas)}"
        )
    fuera = clientes.loc[(lat.abs() > 90) | (lon.abs() > 180), "client_id"]
    if len(fuera):
        problemas.append(f"{len(fuera)} clientes tienen coordenadas imposibles: {_ejemplos(fuera)}")

    if problemas:
        raise ErrorDatos(nombre, problemas)
    clientes["lat"] = lat
    clientes["lon"] = lon
    if "zone" not in clientes.columns:
        clientes["zone"] = ""
    return clientes


def unir_pedidos_clientes(pedidos, clientes):
    """Pega a cada pedido las coordenadas y zona de su cliente."""
    desconocidos = pedidos.loc[~pedidos["client_id"].isin(clientes["client_id"]), "client_id"].unique()
    if len(desconocidos):
        raise ErrorDatos("pedidos.csv", [
            f"{len(desconocidos)} clientes de los pedidos no están en clientes.csv: {_ejemplos(desconocidos)}"
        ])
    return pedidos.merge(clientes[["client_id", "lat", "lon", "zone"]], on="client_id", how="left")


def armar_paradas(datos, capacidad):
    """Agrupa los pedidos por día y cliente: una parada con la suma de cajas.

    Si un cliente pide más cajas que la capacidad de una camioneta, se parte en
    varias paradas en el mismo lugar (por ejemplo, 130 cajas con capacidad 110
    da una parada de 110 y otra de 20), porque una sola camioneta no puede llevarlo.
    """
    grupos = (
        datos.groupby(["date", "client_id"], as_index=False)
        .agg(lat=("lat", "first"), lon=("lon", "first"), zone=("zone", "first"),
             boxes=("boxes", "sum"), pedidos=("order_id", "count"))
    )
    filas = []
    for g in grupos.itertuples(index=False):
        restantes = int(g.boxes)
        partes = -(-restantes // capacidad)  # división hacia arriba
        for parte in range(1, partes + 1):
            cajas = min(restantes, capacidad)
            restantes -= cajas
            filas.append({**g._asdict(), "boxes": cajas, "parte": parte, "partes": partes})
    return pd.DataFrame(filas)


def paradas_del_dia(paradas, fecha):
    """Las paradas de una fecha, con un índice limpio (0, 1, 2...)."""
    return paradas[paradas["date"] == fecha].reset_index(drop=True)


def _cargar_opcional(fuente, nombre, columnas, enteras):
    """Lee un archivo opcional. Devuelve None si no se entregó."""
    if fuente is None or (isinstance(fuente, (str, Path)) and not Path(fuente).exists()):
        return None
    tabla = _leer_csv(fuente, nombre, columnas)
    problemas = []
    _revisar_repetidos(tabla, "date", problemas)
    _revisar_fechas(tabla, problemas)
    for columna in enteras:
        _revisar_enteros(tabla, columna, problemas, "date", minimo=0)
    if problemas:
        raise ErrorDatos(nombre, problemas)
    return tabla


def cargar_operacion_real(fuente, nombre="operacion_real.csv"):
    tabla = _cargar_opcional(fuente, nombre, COLUMNAS_OPERACION,
                             enteras=["vehicles", "urgent_hires", "reserved_hires"])
    if tabla is not None:
        km = pd.to_numeric(tabla["km"], errors="coerce")
        malos = tabla.loc[km.isna() | (km < 0), "date"]
        if len(malos):
            raise ErrorDatos(nombre, [f"{len(malos)} valores de 'km' no son números ≥ 0: {_ejemplos(malos)}"])
        tabla["km"] = km
    return tabla


def cargar_reserva(fuente, nombre="reserva.csv"):
    return _cargar_opcional(fuente, nombre, COLUMNAS_RESERVA, enteras=["reserved_extra"])


def cargar_carpeta(carpeta, config):
    """Carga todo lo que haya en una carpeta de datos (ver cargar_fuentes)."""
    carpeta = Path(carpeta)
    for obligatorio in ["pedidos.csv", "clientes.csv"]:
        if not (carpeta / obligatorio).exists():
            raise ErrorDatos(obligatorio, [f"no existe el archivo en {carpeta}"])
    return cargar_fuentes(carpeta / "pedidos.csv", carpeta / "clientes.csv", config,
                          carpeta / "operacion_real.csv", carpeta / "reserva.csv")


def cargar_fuentes(pedidos, clientes, config, operacion_real=None, reserva=None):
    """Carga los datos desde rutas o archivos subidos (los opcionales pueden ser None).

    Devuelve un diccionario con pedidos (ya unidos a clientes), clientes,
    paradas, operacion_real y reserva (None si no existen) y una lista de avisos.
    """
    clientes = cargar_clientes(clientes)
    pedidos = unir_pedidos_clientes(cargar_pedidos(pedidos), clientes)
    capacidad = config["flota"]["capacidad_cajas"]
    paradas = armar_paradas(pedidos, capacidad)

    avisos = []
    partidos = paradas.loc[paradas["partes"] > 1, ["date", "client_id", "partes"]].drop_duplicates()
    for p in partidos.itertuples(index=False):
        avisos.append(f"{p.date}: el cliente {p.client_id} pidió más de {capacidad} cajas; "
                      f"se reparte en {p.partes} viajes.")

    operacion_real = cargar_operacion_real(operacion_real)
    reserva = cargar_reserva(reserva)
    if operacion_real is None:
        avisos.append("No hay operacion_real.csv: E1 no se puede tomar de la operación real.")
    if reserva is None:
        avisos.append("No hay reserva.csv: E3 usará el pronóstico sustituto (promedio de semanas previas).")

    return {
        "pedidos": pedidos,
        "clientes": clientes,
        "paradas": paradas,
        "operacion_real": operacion_real,
        "reserva": reserva,
        "avisos": avisos,
    }
