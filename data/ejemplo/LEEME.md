# ⚠️ DATOS DE EJEMPLO (INVENTADOS)

Estos archivos **no son datos de ninguna empresa**. Los generó `generar_ejemplo.py` con números
aleatorios (semilla fija) solo para probar la app mientras llegan los datos reales.

- `clientes.csv`: 60 clientes inventados en 6 zonas alrededor de Pachuca (ids `EJ-...`).
- `pedidos.csv`: 6 semanas de pedidos, lunes a sábado (del 1 sep al 11 oct de 2025).
  Las 4 primeras semanas sirven de historia para el pronóstico sustituto y las 2 últimas para evaluar.
  Los viernes hay más demanda y algunos días no alcanzan las 4 camionetas propias.
  Incluye a propósito un pedido de 130 cajas (más de una camioneta) para probar que se parte.

No hay `operacion_real.csv` ni `reserva.csv` de ejemplo: la app debe avisar que faltan.

Para regenerarlos: `.venv/bin/python data/ejemplo/generar_ejemplo.py`
