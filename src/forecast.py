"""Módulo 1 del artículo: modelo predictivo de demanda.

Qué se pronostica: cuántos pedidos llegarán a cada ZONA, cada DÍA y en cada FRANJA horaria
(mañana de 8 a 13 h y tarde de 13 a 19 h), con información disponible al cierre del día anterior.
Así el pronóstico se puede usar para decidir la flota con un día de anticipación.

Se comparan cinco modelos, como en el artículo:
  - ingenuo estacional: repite lo que pasó hace 7 días (mismo día de la semana pasada);
  - promedio 4 semanas: promedio del mismo día de la semana en las 4 semanas anteriores;
  - regresión Ridge, bosque aleatorio (400 árboles) y gradient boosting (Scikit-learn).
Y, si hay simulación con tasa verdadera (tasa_real.csv), el "oráculo": el error que tendría quien
supiera la tasa exacta. No es un modelo usable; mide el error que nadie puede evitar (azar).

Variables de entrada (Tabla 4 del artículo):
  calendario (día de la semana, mes, día del mes, semana del año), franja, zona, temporada
  (Buen Fin y semana posterior, diciembre 1-23, fin de año, Semana Santa, regreso a clases, quincena)
  e historial (rezagos de 1, 7, 14 y 364 días, media móvil de 6 días, media del mismo día de la
  semana en las 4 semanas previas). La zona ya identifica su tipo, así que no se agrega aparte.

Periodos (cronológicos, para no "ver el futuro"):
  entrenamiento 5-feb-2024 a 31-ago-2025; validación jul-ago 2025 con un modelo entrenado hasta
  junio (sirve para elegir el modelo); prueba 1-sep a 31-dic-2025.

Uso:  .venv/bin/python -m src.forecast data/caso_estudio
      (escribe los resultados en data/caso_estudio/resultados/)
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
FRANJAS = ("manana", "tarde")
HORA_CORTE = 13  # antes de las 13 h = mañana

ENTRENAMIENTO = (date(2024, 2, 5), date(2025, 8, 31))
VALIDACION = (date(2025, 7, 1), date(2025, 8, 31))
PRUEBA = (date(2025, 9, 1), date(2025, 12, 31))

MODELOS = ["ingenuo", "prom4", "ridge", "gboost", "bosque"]
NOMBRES = {
    "ingenuo": "Ingenuo estacional (t−7)",
    "prom4": "Promedio 4 semanas",
    "ridge": "Regresión Ridge",
    "gboost": "Gradient boosting",
    "bosque": "Bosque aleatorio",
    "oraculo": "Oráculo (tasa real del simulador)",
}

# Calendario de temporadas en México (fechas conocidas, no datos de la empresa)
BUEN_FIN = {2024: date(2024, 11, 15), 2025: date(2025, 11, 14), 2026: date(2026, 11, 13)}  # viernes de inicio
SEMANA_SANTA = {2024: date(2024, 3, 25), 2025: date(2025, 4, 14), 2026: date(2026, 3, 30)}  # lunes santo
REGRESO_CLASES = {2024: date(2024, 8, 26), 2025: date(2025, 8, 25), 2026: date(2026, 8, 24)}  # lunes de inicio


# --- Tabla zona × día × franja ---

def franja_de(horas):
    """'09:30' -> 'manana', '15:10' -> 'tarde'."""
    return np.where(horas.str[:2].astype(int) < HORA_CORTE, "manana", "tarde")


SIN_ZONA = "sin zona"
DIA_COMPLETO = "dia"


def zonas_de(clientes):
    """Zona de cada cliente (client_id → zona). Los clientes sin zona quedan en "sin zona"."""
    zona = clientes.get("zone", pd.Series("", index=clientes.index)).fillna("").astype(str).str.strip()
    return pd.Series(zona.where(zona != "", SIN_ZONA).to_numpy(), index=clientes["client_id"])


def tiene_horas(pedidos):
    """True si todos los pedidos traen hora (HH:MM): entonces se pronostica por franja."""
    return "time" in pedidos and pedidos["time"].fillna("").astype(str).str.match(r"^\d{1,2}:\d{2}").all()


def panel(pedidos, clientes):
    """Pedidos por zona, día y franja, con ceros donde no hubo pedidos.

    pedidos: date (fecha), time (HH:MM, opcional), client_id. clientes: client_id, zone (opcional).
    Los días son los que aparecen en los pedidos (días operativos).
    Si algún pedido no trae hora se pronostica el día completo (franja "dia"); si los clientes no
    traen zona, todos van en "sin zona" (se pronostica el total).
    """
    p = pedidos.drop(columns=["zone"], errors="ignore").copy()
    p["zone"] = p["client_id"].map(zonas_de(clientes))
    if p["zone"].isna().any():
        raise ValueError("Hay pedidos de clientes que no están en clientes.csv.")
    franjas = FRANJAS if tiene_horas(p) else (DIA_COMPLETO,)
    franja = franja_de(p["time"].astype(str)) if franjas == FRANJAS else DIA_COMPLETO
    p = p.assign(date=pd.to_datetime(p["date"]).dt.date, franja=franja)
    conteo = p.groupby(["date", "zone", "franja"]).size()
    indice = pd.MultiIndex.from_product([sorted(p["date"].unique()), sorted(p["zone"].unique()), franjas],
                                        names=["date", "zone", "franja"])
    return conteo.reindex(indice, fill_value=0).rename("y").reset_index()


def periodos_automaticos(fechas, semanas_prueba=4, semanas_validacion=4, semanas_minimas=8):
    """Periodos de entrenamiento, validación y prueba a partir de las fechas de una PYME.

    La prueba son las últimas `semanas_prueba` semanas; la validación, las `semanas_validacion` anteriores;
    el entrenamiento, todo lo anterior salvo las primeras 5 semanas (que solo sirven de historial).
    """
    fechas = sorted(pd.to_datetime(pd.Series(list(fechas))).dt.date.unique())
    inicio, fin = fechas[0], fechas[-1]
    prueba = (fin - timedelta(weeks=semanas_prueba) + timedelta(days=1), fin)
    validacion = (prueba[0] - timedelta(weeks=semanas_validacion), prueba[0] - timedelta(days=1))
    entrena = (inicio + timedelta(weeks=5), prueba[0] - timedelta(days=1))
    semanas_para_entrenar = (validacion[0] - entrena[0]).days / 7
    if semanas_para_entrenar < semanas_minimas:
        hace_falta = 5 + semanas_minimas + semanas_validacion + semanas_prueba
        tiene = ((fin - inicio).days + 1) / 7
        raise ValueError(f"Hace falta un historial de al menos {hace_falta} semanas para pronosticar y evaluar "
                         f"(se tienen {tiene:.0f}). Prueben con menos semanas de prueba o suban más historia.")
    return {"entrenamiento": entrena, "validacion": validacion, "prueba": prueba}


PERIODOS_ARTICULO = {"entrenamiento": ENTRENAMIENTO, "validacion": VALIDACION, "prueba": PRUEBA}


# --- Variables ---

def _rango(inicio, dias):
    return {inicio + timedelta(days=k) for k in range(dias)}


def banderas_temporada(d):
    """Variables binarias de temporada de un día."""
    buen_fin = BUEN_FIN.get(d.year)
    santa = SEMANA_SANTA.get(d.year)
    clases = REGRESO_CLASES.get(d.year)
    return {
        "buen_fin": int(buen_fin is not None and d in _rango(buen_fin, 4)),
        "post_buen_fin": int(buen_fin is not None and d in _rango(buen_fin + timedelta(days=4), 7)),
        "diciembre": int(d.month == 12 and d.day <= 23),
        "fin_anio": int(d.month == 12 and d.day >= 24),
        "semana_santa": int(santa is not None and d in _rango(santa, 6)),
        "regreso_clases": int(clases is not None and d in _rango(clases, 13)),
        "quincena": int(d.day in (14, 15, 16, 29, 30, 31, 1)),
    }


def variables(tabla):
    """Agrega las variables de la Tabla 4. Solo usa datos hasta el día anterior."""
    t = tabla.sort_values(["zone", "franja", "date"]).reset_index(drop=True)
    fechas = pd.to_datetime(t["date"])
    t["dia_semana"] = fechas.dt.day_of_week
    t["mes"] = fechas.dt.month
    t["dia_mes"] = fechas.dt.day
    t["semana_anio"] = fechas.dt.isocalendar().week.astype(int).to_numpy()
    temporada = pd.DataFrame([banderas_temporada(d) for d in t["date"]])
    t = pd.concat([t, temporada], axis=1)

    # Historial: buscar el valor de la misma zona y franja en otra fecha (NaN si ese día no se trabajó)
    valor = dict(zip(zip(t["date"], t["zone"], t["franja"]), t["y"]))

    def hace(dias):
        return [valor.get((d - timedelta(days=dias), z, f), np.nan) for d, z, f in zip(t["date"], t["zone"], t["franja"])]

    for dias in (7, 14, 21, 28, 364):
        t[f"_hace{dias}"] = hace(dias)
    grupo = t.groupby(["zone", "franja"])["y"]
    t["rezago_1"] = grupo.shift(1)  # el día operativo anterior
    t["media_6"] = grupo.transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    t["prom_4sem"] = t[["_hace7", "_hace14", "_hace21", "_hace28"]].mean(axis=1)
    # Si la empresa no trabaja todos los días, puede no haber "mismo día de la semana" en las 4 semanas
    # anteriores: entonces se usa el promedio de los últimos días con datos
    t["prom_4sem"] = t["prom_4sem"].fillna(t["media_6"])
    t["rezago_7"] = t["_hace7"].fillna(t["prom_4sem"])
    t["rezago_14"] = t["_hace14"].fillna(t["prom_4sem"])
    t["rezago_364"] = t["_hace364"].fillna(t["prom_4sem"])  # el primer año no tiene año anterior
    return t.drop(columns=[c for c in t.columns if c.startswith("_hace")])


NUMERICAS = ["dia_mes", "semana_anio", "buen_fin", "post_buen_fin", "diciembre", "fin_anio", "semana_santa",
             "regreso_clases", "quincena", "rezago_1", "rezago_7", "rezago_14", "rezago_364", "media_6", "prom_4sem"]
CATEGORICAS = ["dia_semana", "mes", "zone", "franja"]


def matriz(t, columnas=None):
    """Matriz numérica para Scikit-learn (las categorías se codifican en one-hot)."""
    x = pd.get_dummies(t[NUMERICAS + CATEGORICAS].astype({c: str for c in CATEGORICAS}),
                       columns=CATEGORICAS, dtype=float)
    if columnas is not None:
        x = x.reindex(columns=columnas, fill_value=0.0)
    return x


def _entre(t, rango):
    d = t["date"]
    return t[(d >= rango[0]) & (d <= rango[1])]


# --- Modelos ---

def nuevos_modelos():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "bosque": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=SEMILLA),
        "gboost": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=SEMILLA),
    }


def pronosticar(t, entrena, predice):
    """Entrena con el rango `entrena` y pronostica el rango `predice`. Devuelve una columna por modelo."""
    base = _entre(t, (min(t["date"]), entrena[1]))
    base = base[base["date"] >= entrena[0]].dropna(subset=["rezago_1", "prom_4sem"])
    if len(base) < 20:
        raise ValueError("No hay suficientes días con pedidos para entrenar el pronóstico "
                         f"({base['date'].nunique()} días antes de {entrena[1]:%d/%m/%Y}). Suban más historial.")
    objetivo = _entre(t, predice).copy()
    if objetivo.empty:
        raise ValueError(f"No hay días con pedidos entre el {predice[0]:%d/%m/%Y} y el {predice[1]:%d/%m/%Y}, "
                         "así que no se puede validar ni probar ahí. Cambien las semanas o suban más historial.")
    # Referencias sin aprendizaje
    objetivo["ingenuo"] = objetivo["rezago_7"]
    objetivo["prom4"] = objetivo["prom_4sem"]
    x = matriz(base)
    x_obj = matriz(objetivo.fillna({"rezago_1": objetivo["prom_4sem"], "media_6": objetivo["prom_4sem"]}), x.columns)
    for nombre, modelo in nuevos_modelos().items():
        modelo.fit(x, base["y"])
        objetivo[nombre] = np.clip(modelo.predict(x_obj), 0, None)
    return objetivo


# --- Métricas ---

def metricas(real, previsto):
    """MAE, RMSE, WAPE (ec. 1 del artículo) y sesgo, en las unidades de `real`."""
    real, previsto = np.asarray(real, float), np.asarray(previsto, float)
    error = previsto - real
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "wape": float(np.abs(error).sum() / real.sum()),
        "sesgo": float(error.sum() / real.sum()),
    }


def tabla_metricas(prueba, validacion, columnas):
    """Tabla 6 del artículo: métricas por modelo en la prueba, más el WAPE de validación."""
    filas = []
    viernes = pd.to_datetime(prueba["date"]).dt.day_of_week == 4
    diario = prueba.groupby("date")[["y"] + columnas].sum()
    for c in columnas:
        m = metricas(prueba["y"], prueba[c])
        filas.append({
            "modelo": c, "nombre": NOMBRES[c],
            "mae": m["mae"], "rmse": m["rmse"], "wape": m["wape"], "sesgo": m["sesgo"],
            "wape_diario": metricas(diario["y"], diario[c])["wape"],
            "wape_viernes": metricas(prueba.loc[viernes, "y"], prueba.loc[viernes, c])["wape"],
            "wape_validacion": metricas(validacion["y"], validacion[c])["wape"] if c in validacion else np.nan,
        })
    return pd.DataFrame(filas)


def sesgo_mensual(prueba, columna):
    """Sesgo del pronóstico por mes (negativo = se quedó corto). El artículo reporta −7.4 % en diciembre."""
    mes = pd.to_datetime(prueba["date"]).dt.to_period("M").astype(str)
    return prueba.groupby(mes).apply(lambda g: (g[columna].sum() - g["y"].sum()) / g["y"].sum(),
                                     include_groups=False)


# --- Todo junto ---

def evaluar(pedidos, clientes, tasa_real=None, periodos=None):
    """Entrena, valida y prueba los cinco modelos. Devuelve un diccionario con todo lo necesario.

    periodos: {"entrenamiento", "validacion", "prueba"} → (inicio, fin). Si no se dan, los del artículo.
    """
    periodos = periodos or PERIODOS_ARTICULO
    entrena, valida, prueba_rango = periodos["entrenamiento"], periodos["validacion"], periodos["prueba"]
    t = variables(panel(pedidos, clientes))

    # Validación: modelo entrenado hasta antes de la validación (en el artículo: hasta junio de 2025)
    validacion = pronosticar(t, (entrena[0], valida[0] - timedelta(days=1)), valida)
    wape_val = {c: metricas(validacion["y"], validacion[c])["wape"] for c in MODELOS}
    elegido = min(wape_val, key=wape_val.get)

    # Prueba: modelo entrenado con todo el periodo de entrenamiento
    prueba = pronosticar(t, entrena, prueba_rango)
    columnas = list(MODELOS)
    if tasa_real is not None:
        tasa = tasa_real.assign(date=pd.to_datetime(tasa_real["date"]).dt.date)
        prueba = prueba.merge(tasa.rename(columns={"tasa": "oraculo"}), on=["date", "zone", "franja"], how="left")
        columnas.append("oraculo")

    return {
        "elegido": elegido,
        "metricas": tabla_metricas(prueba, validacion, columnas),
        "prueba": prueba[["date", "zone", "franja", "y"] + columnas],
        "validacion": validacion[["date", "zone", "franja", "y"] + MODELOS],
        "sesgo_mensual": sesgo_mensual(prueba, elegido),
        "periodos": periodos,
    }


def guardar(resultado, carpeta):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    resultado["metricas"].round(4).to_csv(carpeta / "metricas_pronostico.csv", index=False)
    resultado["prueba"].round(4).to_csv(carpeta / "pronostico_prueba.csv", index=False)
    resultado["validacion"].round(4).to_csv(carpeta / "pronostico_validacion.csv", index=False)
    (pd.DataFrame({"mes": resultado["sesgo_mensual"].index, "sesgo": resultado["sesgo_mensual"].to_numpy()})
     .round(4).to_csv(carpeta / "sesgo_mensual.csv", index=False))
    (carpeta / "modelo_elegido.txt").write_text(resultado["elegido"] + "\n", encoding="utf-8")


if __name__ == "__main__":
    origen = Path(sys.argv[1] if len(sys.argv) > 1 else "data/caso_estudio")
    pedidos = pd.read_csv(origen / "pedidos.csv", dtype={"time": str})
    clientes = pd.read_csv(origen / "clientes.csv")
    tasa = pd.read_csv(origen / "tasa_real.csv") if (origen / "tasa_real.csv").exists() else None
    resultado = evaluar(pedidos, clientes, tasa)
    guardar(resultado, origen / "resultados")
    print(f"Modelo elegido en validación: {NOMBRES[resultado['elegido']]}")
    print(resultado["metricas"].assign(**{c: resultado["metricas"][c] * 100 for c in
                                          ("wape", "sesgo", "wape_diario", "wape_viernes", "wape_validacion")})
          .round(2).drop(columns="modelo").to_string(index=False))
    print("Sesgo mensual del modelo elegido (%):", (resultado["sesgo_mensual"] * 100).round(1).to_dict())
