# ⚠️ CASO DE ESTUDIO SIMULADO (NO SON DATOS DE UNA EMPRESA)

Estos archivos son una **simulación** del caso del artículo ("distribuidora urbana en Pachuca"), igual que dicen
el artículo y el cartel. Los genera `generar_caso.py` con una semilla fija; no provienen de ninguna empresa real.

- `clientes.csv`: 560 clientes en 8 zonas (ids `SIM-Z1-001`...).
- `pedidos.csv`: pedidos del 2-ene-2024 al 31-dic-2025, 621 días operativos (lun a sáb, sin 5 festivos).
- `tasa_real.csv`: la tasa media **verdadera** con la que se generó cada zona/día/franja. En la vida real no
  existe; sirve solo para calcular el "oráculo" (el error mínimo por puro azar).

- `reserva.csv`: unidades a reservar cada día de la prueba (1-sep a 31-dic-2025). **No es simulado a mano**: lo
  calcula el modelo de pronóstico (`src/forecast.py`) con la regla del artículo (`src/integracion.py`, ec. 2 y 3).
  La app lo usa para el escenario E3; fuera de esos 103 días, E3 aparece como no disponible.
- `resultados/`: error de cada modelo de pronóstico, pronósticos de validación y prueba, sesgo por mes y
  parámetros de la reserva (modelo usado, α y q_α).

Para recalcular el pronóstico y la reserva: `.venv/bin/python -m src.integracion data/caso_estudio`

## Qué tan parecido es al artículo

El código original del artículo no está disponible: esto es una **reconstrucción** a partir de su Tabla 3 y su
descripción, así que los números salen parecidos, no idénticos.

| | Artículo | Esta simulación |
|---|---|---|
| Clientes / zonas | 560 / 8 | 560 / 8 |
| Días operativos (total / prueba) | 621 / 103 | 621 / 103 |
| Pedidos en total | 48,948 | 47,580 |
| Pedidos por día (entrenamiento) | 76.4 | 75.5 |
| Cajas por día (entrenamiento) | 401 | 397 |
| Factor viernes Z1 / Z3 / Z7 | 1.67 / 1.33 / 1.00 | 1.69 / 1.34 / 0.97 |
| Pedidos por la mañana (día normal / viernes) | 43.9 % / 33.1 % | 44.0 % / 33.3 % |
| Error del oráculo por zona-día-franja | 40.6 % | 40.7 % |
| Error del oráculo por día | 12.5 % | 13.1 % |
| Viernes 14-nov-2025 (pedidos / paradas / cajas) | 112 / 101 / 592 | 123 / 109 / 580 |

## Supuestos de la reconstrucción (el artículo no los especifica)

Dirección de cada zona respecto del almacén, clientes por zona (proporcional a su % de pedidos), tamaño de los
efectos de temporada (Buen Fin ×1.40, diciembre ×1.20, fin de año ×0.80, Semana Santa ×0.85, regreso a clases
×1.10, quincena ×1.08), factor mensual, y la sobredispersión y el choque diario (calibrados para acercarse al
error del oráculo del artículo). Están marcados en `generar_caso.py`.

Para regenerar: `.venv/bin/python data/caso_estudio/generar_caso.py`
