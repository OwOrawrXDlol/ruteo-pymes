"""Adaptador de datos: convierte los archivos de cualquier PYME (CSV o Excel, con sus propios nombres de
columna) al formato que usa la app (pedidos.csv, clientes.csv, operacion_real.csv, reserva.csv).

Pasos:
1. leer_tabla(): abre CSV (separado por coma o punto y coma, UTF-8 o Latin-1) o Excel.
2. sugerir_mapeo(): adivina qué columna corresponde a cada campo, por su nombre ("fecha" → date,
   "cabezas" → boxes, "latitud" → lat...). La persona lo revisa y corrige en la app.
3. convertir(): renombra, convierte fechas (AAAA-MM-DD o DD/MM/AAAA), horas y números (acepta coma
   decimal) y devuelve la tabla en el formato del contrato, junto con la lista de problemas.

Nunca inventa datos: si una fecha, coordenada o cantidad no se entiende, se reporta como problema y la
fila no se usa hasta que se corrija. La única columna que se genera es el número de pedido (order_id)
cuando el archivo no lo trae, porque solo es un identificador.
"""

import io
import math
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd

# Campos de cada archivo: (campo, etiqueta, obligatorio, nombres con que suele venir)
CAMPOS = {
    "pedidos.csv": [
        ("order_id", "Número de pedido", False, ["order_id", "id_pedido", "pedido", "folio", "id_viaje", "viaje",
                                                 "no_pedido", "num_pedido", "id"]),
        ("date", "Fecha de entrega", True, ["date", "fecha", "fecha_entrega", "dia", "fecha_pedido"]),
        ("time", "Hora en que llegó el pedido", False, ["time", "hora", "hora_recepcion", "hora_pedido"]),
        ("client_id", "Cliente (o punto de entrega/carga)", True,
         ["client_id", "cliente", "id_cliente", "clave_cliente", "lugar_carga", "destino", "punto", "tienda",
          "sucursal"]),
        ("boxes", "Cantidad (cajas, cabezas, kilos...)", True,
         ["boxes", "cajas", "cantidad", "cabezas", "piezas", "kilos", "kg", "unidades", "bultos", "paquetes"]),
    ],
    "clientes.csv": [
        ("client_id", "Cliente (o punto de entrega/carga)", True,
         ["client_id", "cliente", "id_cliente", "clave_cliente", "lugar_carga", "destino", "punto", "tienda",
          "sucursal"]),
        ("lat", "Latitud", True, ["lat", "latitud", "latitude"]),
        ("lon", "Longitud", True, ["lon", "lng", "long", "longitud", "longitude"]),
        ("zone", "Zona", False, ["zone", "zona", "region", "sector", "ruta", "estado", "municipio", "colonia"]),
    ],
    "operacion_real.csv": [
        ("date", "Fecha", True, ["date", "fecha", "dia"]),
        ("km", "Km recorridos en el día", True, ["km", "kilometros", "distancia", "km_totales"]),
        ("vehicles", "Camionetas propias que salieron", True,
         ["vehicles", "camionetas", "vehiculos", "unidades", "camiones", "jaulas_usadas", "jaulas"]),
        ("urgent_hires", "Fletes contratados el mismo día", True, ["urgent_hires", "fletes_urgentes", "urgentes"]),
        ("reserved_hires", "Fletes contratados con anticipación", True,
         ["reserved_hires", "fletes_reservados", "reservados"]),
    ],
    "reserva.csv": [
        ("date", "Fecha", True, ["date", "fecha", "dia"]),
        ("reserved_extra", "Unidades extra reservadas", True,
         ["reserved_extra", "reservadas", "reserva", "unidades_extra", "extra"]),
    ],
}
SIN_COLUMNA = "(no la tengo)"
ENTEROS = {"boxes", "vehicles", "urgent_hires", "reserved_hires", "reserved_extra"}
MAX_EJEMPLOS = 8


def normalizar(nombre):
    """'Fecha de Entrega ' → 'fecha_de_entrega' (sin acentos, minúsculas, guiones bajos)."""
    texto = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", texto.lower()).strip("_")


def leer_tabla(contenido, nombre_archivo):
    """Abre un archivo subido (bytes) como tabla de texto. Acepta .csv, .xlsx y .xls."""
    nombre = nombre_archivo.lower()
    if nombre.endswith((".xlsx", ".xls")):
        try:
            tabla = pd.read_excel(io.BytesIO(contenido), dtype=object)
        except Exception as error:
            raise ValueError(f"No se pudo abrir el Excel {nombre_archivo}: {error}") from None
    else:
        tabla = None
        for codificacion in ("utf-8-sig", "latin-1"):
            try:
                tabla = pd.read_csv(io.BytesIO(contenido), sep=None, engine="python", dtype=str,
                                    keep_default_na=False, encoding=codificacion)
                break
            except UnicodeDecodeError:
                continue
            except Exception as error:
                raise ValueError(f"No se pudo leer {nombre_archivo}: {error}") from None
        if tabla is None:
            raise ValueError(f"No se pudo leer {nombre_archivo}: codificación desconocida.")
    tabla.columns = [str(c).strip() for c in tabla.columns]
    tabla = tabla.dropna(how="all")
    if tabla.empty:
        raise ValueError(f"{nombre_archivo} no tiene renglones.")
    return tabla


def sugerir_mapeo(columnas, archivo):
    """{campo: columna del archivo o None}. Primero el nombre exacto, luego los nombres comunes."""
    por_normal = {normalizar(c): c for c in columnas}
    mapeo, usadas = {}, set()
    for campo, _etiqueta, _obligatorio, nombres in CAMPOS[archivo]:
        elegida = None
        for candidato in nombres:
            columna = por_normal.get(normalizar(candidato))
            if columna is not None and columna not in usadas:
                elegida = columna
                break
        mapeo[campo] = elegida
        if elegida is not None:
            usadas.add(elegida)
    return mapeo


def faltantes(mapeo, archivo):
    """Campos obligatorios que no tienen columna."""
    return [etiqueta for campo, etiqueta, obligatorio, _ in CAMPOS[archivo] if obligatorio and not mapeo.get(campo)]


# --- Conversión de valores ---

def _ejemplos(valores):
    valores = list(valores)
    texto = ", ".join(str(v) for v in valores[:MAX_EJEMPLOS])
    return texto + (f" y {len(valores) - MAX_EJEMPLOS} más" if len(valores) > MAX_EJEMPLOS else "")


def a_fecha(serie):
    """Convierte a fecha. Acepta AAAA-MM-DD, DD/MM/AAAA (día primero, como en México) y fechas de Excel."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return pd.to_datetime(serie, errors="coerce")
    texto = serie.astype(str).str.strip()
    iso = pd.to_datetime(texto, format="%Y-%m-%d", errors="coerce")
    # Excel a veces trae la hora pegada ("2025-09-01 00:00:00")
    con_hora = pd.to_datetime(texto, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    dia_primero = pd.to_datetime(texto, dayfirst=True, errors="coerce", format="mixed")
    return iso.fillna(con_hora).fillna(dia_primero)


def a_hora(serie):
    """Convierte a HH:MM. Acepta '9:30', '09:30:15' y horas de Excel. Vacío se queda vacío."""
    def una(valor):
        if valor is None or (isinstance(valor, float) and math.isnan(valor)):
            return ""
        if hasattr(valor, "hour") and hasattr(valor, "minute"):
            return f"{valor.hour:02d}:{valor.minute:02d}"
        texto = str(valor).strip()
        if texto == "":
            return ""
        m = re.match(r"^(\d{1,2}):(\d{2})(:\d{2})?(\s*[ap]\.?\s*m\.?)?$", texto.lower())
        if not m:
            return None
        hora, minuto = int(m.group(1)), int(m.group(2))
        sufijo = (m.group(4) or "").replace(".", "").replace(" ", "")
        if sufijo == "pm" and hora < 12:
            hora += 12
        if sufijo == "am" and hora == 12:
            hora = 0
        return f"{hora:02d}:{minuto:02d}" if hora < 24 and minuto < 60 else None
    return serie.map(una)


def a_numero(serie):
    """Convierte a número. Acepta coma decimal ('19,37') y miles con coma ('1,200')."""
    def una(valor):
        if isinstance(valor, (int, float)) and not isinstance(valor, bool):
            return float(valor)
        texto = str(valor).strip().replace(" ", "").replace("$", "")
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", texto):
            texto = texto.replace(",", "")  # miles
        elif "," in texto and "." not in texto:
            texto = texto.replace(",", ".")  # coma decimal
        try:
            return float(texto)
        except ValueError:
            return math.nan
    return serie.map(una).astype(float)


def convertir(tabla, archivo, mapeo, redondear=False, vehiculos_incluyen_fletes=False):
    """Devuelve (tabla en el formato de la app, problemas, avisos).

    redondear: si una cantidad trae decimales (por ejemplo kilos), redondearla hacia arriba en vez de
               reportarla como problema.
    vehiculos_incluyen_fletes (operacion_real): la columna de vehículos cuenta también los fletes
               contratados; se les restan para quedarse con las camionetas propias.
    """
    problemas, avisos = [], []
    falta = faltantes(mapeo, archivo)
    if falta:
        return None, [f"Falta indicar qué columna es: {', '.join(falta)}."], avisos

    salida = pd.DataFrame(index=tabla.index)
    for campo, etiqueta, _obligatorio, _ in CAMPOS[archivo]:
        columna = mapeo.get(campo)
        if not columna:
            continue
        valores = tabla[columna]
        if campo == "date":
            fechas = a_fecha(valores)
            malas = valores[fechas.isna()]
            if len(malas):
                problemas.append(f"{len(malas)} fechas no se entienden (use AAAA-MM-DD o DD/MM/AAAA): {_ejemplos(malas)}")
            salida[campo] = fechas.dt.strftime("%Y-%m-%d")
        elif campo == "time":
            horas = a_hora(valores)
            malas = valores[horas.isna()]
            if len(malas):
                problemas.append(f"{len(malas)} horas no se entienden (use HH:MM): {_ejemplos(malas)}")
            salida[campo] = horas
        elif campo in ("lat", "lon", "km") or campo in ENTEROS:
            numeros = a_numero(valores)
            malos = valores[numeros.isna()]
            if len(malos):
                problemas.append(f"{len(malos)} valores de '{columna}' ({etiqueta}) no son números: {_ejemplos(malos)}")
            if campo in ENTEROS:
                decimales = numeros.notna() & (numeros % 1 != 0)
                if decimales.any():
                    if redondear:
                        avisos.append(f"{int(decimales.sum())} valores de '{columna}' traían decimales y se "
                                      "redondearon hacia arriba.")
                        numeros = numeros.where(~decimales, np.ceil(numeros))
                    else:
                        problemas.append(f"{int(decimales.sum())} valores de '{columna}' tienen decimales: "
                                         f"{_ejemplos(valores[decimales])}. La app usa cantidades enteras; marque "
                                         "\"redondear hacia arriba\" o use una unidad más chica.")
                enteros = numeros.notna() & (numeros % 1 == 0)
                salida[campo] = numeros.astype("Int64") if enteros.equals(numeros.notna()) else numeros
            else:
                salida[campo] = numeros
        else:
            salida[campo] = valores.astype(str).str.strip()

    if archivo == "pedidos.csv" and "order_id" not in salida:
        salida.insert(0, "order_id", [f"P{i:06d}" for i in range(1, len(salida) + 1)])
        avisos.append("El archivo no traía número de pedido; se numeraron P000001, P000002...")
    if archivo == "operacion_real.csv" and vehiculos_incluyen_fletes and not problemas:
        propias = salida["vehicles"] - salida["urgent_hires"] - salida["reserved_hires"]
        if (propias < 0).any():
            problemas.append("Al restar los fletes, algunos días quedan con camionetas propias negativas.")
        salida["vehicles"] = propias
        avisos.append("A la columna de vehículos se le restaron los fletes urgentes y reservados.")
    if archivo == "clientes.csv" and not problemas:
        avisos += revisar_coordenadas(salida)

    orden = [c for c, *_ in CAMPOS[archivo] if c in salida]
    return (None if problemas else salida[orden].reset_index(drop=True)), problemas, avisos


def revisar_coordenadas(clientes):
    """Avisos sobre coordenadas sospechosas (no las cambia)."""
    avisos = []
    lat, lon = clientes["lat"], clientes["lon"]
    if (lat.abs() > 90).any() or ((lat < 0).all() and (lon > 0).all() and lon.between(14, 33).all()):
        avisos.append("Parece que latitud y longitud están invertidas: en México la latitud va de 14 a 33 y la "
                      "longitud es negativa (−86 a −118). Revise el emparejamiento de columnas.")
    elif not (lat.between(14, 33).all() and lon.between(-118, -86).all()):
        fuera = clientes.loc[~(lat.between(14, 33) & lon.between(-118, -86)), "client_id"]
        avisos.append(f"{len(fuera)} clientes quedan fuera de México según sus coordenadas: {_ejemplos(fuera)}")
    return avisos


# --- Plantillas para descargar ---

EJEMPLOS = {
    "pedidos.csv": "order_id,date,time,client_id,boxes\nP0001,2025-10-06,09:15,C001,12\nP0002,2025-10-06,15:40,C002,5\n",
    "clientes.csv": "client_id,lat,lon,zone\nC001,20.1230,-98.7360,Centro\nC002,20.0880,-98.7650,Sur\n",
    "operacion_real.csv": "date,km,vehicles,urgent_hires,reserved_hires\n2025-10-06,184.5,4,1,0\n",
    "reserva.csv": "date,reserved_extra\n2025-10-07,1\n",
}
LEEME_PLANTILLAS = """PLANTILLAS DE DATOS PARA LA APP DE RUTEO

Obligatorios: pedidos.csv y clientes.csv. Opcionales: operacion_real.csv y reserva.csv.
Pueden usar sus propios nombres de columna o Excel: la app les pide indicar qué columna es cada cosa.

pedidos.csv   order_id (opcional), date (AAAA-MM-DD o DD/MM/AAAA), time (HH:MM, opcional; sirve para el
              pronóstico por franja), client_id, boxes (cantidad entera: cajas, cabezas, kilos...)
clientes.csv  client_id, lat, lon (en grados; en México la longitud es negativa), zone (opcional)
operacion_real.csv  date, km, vehicles (camionetas propias que salieron), urgent_hires, reserved_hires
reserva.csv   date, reserved_extra (unidades extra reservadas el día anterior)

Los renglones de ejemplo son inventados: bórrenlos y pongan los suyos.
"""


def plantillas_zip():
    """Un .zip con las 4 plantillas y un LEEME, para descargar desde la app."""
    memoria = io.BytesIO()
    with zipfile.ZipFile(memoria, "w", zipfile.ZIP_DEFLATED) as z:
        for nombre, texto in EJEMPLOS.items():
            z.writestr(nombre, texto)
        z.writestr("LEEME.txt", LEEME_PLANTILLAS)
    return memoria.getvalue()
