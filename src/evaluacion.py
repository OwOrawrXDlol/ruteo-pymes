"""Evaluación del periodo de prueba (103 días): los indicadores del artículo y del cartel.

1. calcular(): resuelve E1 a E4 en cada día de la prueba con los MISMOS pedidos (unos 15-20 min con
   4 s de búsqueda por ruteo) y guarda una fila por día y escenario en resultados/escenarios_prueba.csv.
   Es lento, por eso se calcula una vez y la app solo lee el archivo.
2. analizar(): a partir de ese archivo saca la Tabla 7 del artículo y las cifras del cartel:
   km, unidades, utilización, urgentes y costo por escenario; ahorro del ruteo (E1 → E2) y del
   pronóstico (E2 → E3); qué parte del ahorro posible se logra (brecha cerrada); días en que E3 ganó o
   perdió frente a E2; viernes; septiembre-octubre frente a noviembre-diciembre; y la prueba de rangos
   con signo de Wilcoxon (¿la diferencia diaria es consistente o pudo ser suerte?).

Uso:  .venv/bin/python -m src.evaluacion data/caso_estudio
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from src.forecast import PRUEBA
from src.load import cargar_carpeta, cargar_config
from src.scenarios import ESCENARIOS, escenarios_rango

ARCHIVO = "escenarios_prueba.csv"
SUBPERIODOS = {"sep-oct": (9, 10), "nov-dic": (11, 12)}


def dias_de_prueba(datos):
    fechas = sorted(datos["paradas"]["date"].unique())
    return [f for f in fechas if PRUEBA[0] <= f <= PRUEBA[1]]


def calcular(carpeta, config, segundos=None, al_avanzar=None):
    """Resuelve los cuatro escenarios en cada día de prueba y guarda el detalle."""
    datos = cargar_carpeta(carpeta, config)
    detalle, _resumen, avisos = escenarios_rango(datos, dias_de_prueba(datos), config, segundos, al_avanzar)
    salida = Path(carpeta) / "resultados"
    salida.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(salida / ARCHIVO, index=False)
    return detalle, avisos


def leer(carpeta):
    """El detalle guardado, o None si todavía no se calculó."""
    ruta = Path(carpeta) / "resultados" / ARCHIVO
    if not ruta.exists():
        return None
    return pd.read_csv(ruta, parse_dates=["date"])


def _pct(nuevo, base):
    return (nuevo / base - 1) * 100


def tabla_escenarios(detalle, config):
    """Tabla 7 del artículo: una fila por escenario."""
    capacidad = config["flota"]["capacidad_cajas"]
    g = detalle.groupby("escenario")
    t = pd.DataFrame({
        "dias": g["date"].nunique(),
        "km_total": g["km"].sum(),
        "unidades": g["capacidad"].sum() / capacidad,
        "cajas": g["cajas"].sum(),
        "capacidad": g["capacidad"].sum(),
        "urgentes": g["urgentes"].sum(),
        "reservadas": g["reservadas"].sum(),
        "costo_total": g["costo"].sum(),
        "no_atendidas": g["no_atendidas"].sum(),
    }).reindex(ESCENARIOS)
    t["km_dia"] = t["km_total"] / t["dias"]
    t["unidades_dia"] = t["unidades"] / t["dias"]
    t["utilizacion"] = t["cajas"] / t["capacidad"]
    t["costo_dia"] = t["costo_total"] / t["dias"]
    t["costo_caja"] = t["costo_total"] / t["cajas"]
    return t


def _por_dia(detalle, columna):
    """Tabla día × escenario de una columna."""
    return detalle.pivot(index="date", columns="escenario", values=columna)


def analizar(detalle, config):
    """Todos los indicadores del cartel a partir del detalle por día y escenario."""
    completos = detalle.groupby("date")["escenario"].nunique()
    detalle = detalle[detalle["date"].isin(completos[completos == len(ESCENARIOS)].index)]
    t = tabla_escenarios(detalle, config)
    costo = _por_dia(detalle, "costo")
    km = _por_dia(detalle, "km")

    # E3 frente a E2, día por día (tolerancia de 1 peso)
    dif = costo["E3"] - costo["E2"]
    viernes = costo.index.day_of_week == 4
    tv = tabla_escenarios(detalle[detalle["date"].dt.day_of_week == 4], config)

    subperiodos = {}
    for nombre, meses in SUBPERIODOS.items():
        en = costo.index.month.isin(meses)
        if not en.any():
            continue  # el rango no incluye esos meses
        subperiodos[nombre] = {
            "dias": int(en.sum()),
            **{f"costo_dia_{e}": float(costo.loc[en, e].mean()) for e in ESCENARIOS},
            "ahorro_E3_vs_E1": float(-_pct(costo.loc[en, "E3"].sum(), costo.loc[en, "E1"].sum())),
        }

    ahorro_posible = t.loc["E2", "costo_total"] - t.loc["E4", "costo_total"]
    return {
        "tabla": t,
        "dias": int(len(costo)),
        "km_E2_vs_E1": float(_pct(t.loc["E2", "km_total"], t.loc["E1", "km_total"])),
        "km_E3_vs_E2": float(_pct(t.loc["E3", "km_total"], t.loc["E2", "km_total"])),
        "costo_E2_vs_E1": float(_pct(t.loc["E2", "costo_total"], t.loc["E1", "costo_total"])),
        "costo_E3_vs_E1": float(_pct(t.loc["E3", "costo_total"], t.loc["E1", "costo_total"])),
        "costo_E3_vs_E2": float(_pct(t.loc["E3", "costo_total"], t.loc["E2", "costo_total"])),
        "ahorro_E3_vs_E1_mxn": float(t.loc["E1", "costo_total"] - t.loc["E3", "costo_total"]),
        "brecha_cerrada": float((t.loc["E2", "costo_total"] - t.loc["E3", "costo_total"]) / ahorro_posible * 100)
        if ahorro_posible > 0 else None,
        "dias_E3_menor": int((dif < -1).sum()),
        "dias_E3_igual": int((dif.abs() <= 1).sum()),
        "dias_E3_mayor": int((dif > 1).sum()),
        "sobrecosto_E3_mayor": float(dif[dif > 1].sum()),
        "dias_km_E2_menor": int((km["E2"] < km["E1"] - 1e-6).sum()),
        "wilcoxon_km_E1_E2": float(wilcoxon(km["E1"], km["E2"]).pvalue),
        "wilcoxon_costo_E2_E3": float(wilcoxon(costo["E2"], costo["E3"]).pvalue),
        "viernes": {
            "dias": int(viernes.sum()),
            "costo_dia_E1": float(costo.loc[viernes, "E1"].mean()),
            "costo_dia_E3": float(costo.loc[viernes, "E3"].mean()),
            "ahorro_E3_vs_E1": float(-_pct(costo.loc[viernes, "E3"].sum(), costo.loc[viernes, "E1"].sum())),
            "urgentes_E1": int(tv.loc["E1", "urgentes"]),
            "urgentes_E3": int(tv.loc["E3", "urgentes"]),
            "utilizacion_E3": float(tv.loc["E3", "utilizacion"]),
        },
        "subperiodos": subperiodos,
    }


def resumen_texto(a):
    t = a["tabla"]
    lineas = [f"{a['dias']} días de prueba"]
    for e in ESCENARIOS:
        f = t.loc[e]
        lineas.append(f"  {e}: {f.km_dia:6.1f} km/día | {f.unidades_dia:.2f} unidades/día | utilización "
                      f"{f.utilizacion:.1%} | {int(f.urgentes):3d} urgentes | ${f.costo_dia:,.0f}/día | "
                      f"${f.costo_total:,.0f} total | ${f.costo_caja:.2f}/caja")
    v = a["viernes"]
    lineas += [
        f"km E2 vs E1: {a['km_E2_vs_E1']:+.1f} % (E2 menor en {a['dias_km_E2_menor']} días; Wilcoxon p = {a['wilcoxon_km_E1_E2']:.2g})",
        f"costo E2 vs E1: {a['costo_E2_vs_E1']:+.1f} % | E3 vs E1: {a['costo_E3_vs_E1']:+.1f} % | "
        f"E3 vs E2: {a['costo_E3_vs_E2']:+.1f} % (Wilcoxon p = {a['wilcoxon_costo_E2_E3']:.2g})",
        f"ahorro E3 vs E1: ${a['ahorro_E3_vs_E1_mxn']:,.0f} | brecha cerrada: {a['brecha_cerrada']:.0f} %",
        f"E3 frente a E2: menor {a['dias_E3_menor']} días, igual {a['dias_E3_igual']}, mayor {a['dias_E3_mayor']} "
        f"(sobrecosto ${a['sobrecosto_E3_mayor']:,.0f})",
        f"viernes ({v['dias']}): ${v['costo_dia_E1']:,.0f} → ${v['costo_dia_E3']:,.0f}/día ({-v['ahorro_E3_vs_E1']:+.1f} %), "
        f"urgentes {v['urgentes_E1']} → {v['urgentes_E3']}, utilización E3 {v['utilizacion_E3']:.1%}",
    ] + [f"{n}: ahorro E3 vs E1 {s['ahorro_E3_vs_E1']:.1f} %" for n, s in a["subperiodos"].items()]
    return "\n".join(lineas)


if __name__ == "__main__":
    carpeta = Path(sys.argv[1] if len(sys.argv) > 1 else "data/caso_estudio")
    config = cargar_config(Path(__file__).resolve().parent.parent / "config.yaml")
    detalle = leer(carpeta)
    if detalle is None or "--recalcular" in sys.argv:
        detalle, avisos = calcular(carpeta, config,
                                   al_avanzar=lambda i, n: print(f"día {i} de {n}", flush=True))
        for aviso in avisos:
            print("aviso:", aviso)
        detalle = leer(carpeta)
    print(resumen_texto(analizar(detalle, config)))
