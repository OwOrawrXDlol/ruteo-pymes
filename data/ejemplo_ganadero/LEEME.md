# Órdenes de acopio de ganado (escenario de demostración)

`ordenes.csv` lo generó `generar_ganadero.py` (semilla fija) con los rangos que contó una operación real de
acopio de ganado. No viene del sistema de la empresa: sirve para comparar dos formas de despachar jaulas.

| columna | descripción |
|---|---|
| oc_id | orden de compra (una jaula por OC) |
| proveedor | clave del origen + número |
| origen | Balancán, Tenosique, Villahermosa, Emiliano Zapata (Tab.), Palenque/Ocosingo, Pichucalco (Chis.), Medellín (Ver.) |
| cabezas | 70-87 la mayoría (85 %); 30-55 las jaulas a medio llenar |
| fecha_aviso | día en que se confirmó la OC |
| fecha_embarque | día en que el permiso queda listo para cargar (lunes a viernes) |
| dias_aviso | fecha_embarque − fecha_aviso |
| flete_mxn | flete del viaje (solo depende del origen) |

- Del 4-nov-2024 al 1-nov-2026: el primer año es historial para el pronóstico; el segundo (desde el
  3-nov-2025) es el que se compara.
- **Datos de la empresa:** 3 jaulas propias, viaje redondo de 3 días, un solo punto de carga, permisos de 5
  días, fletera apartada con una semana de anticipación, mismo precio de flete propio o contratado
  (Balancán $45,000; Pichucalco $40,000; otras zonas $16,000-18,000), 70-87 cabezas por jaula.
- **Supuestos:** el ritmo (6 jaulas por semana en promedio, con temporada de ±12 %, calibrado para que el
  despacho actual contrate unas 5-6 fleteras al mes), la proporción de cada origen, 3-5 proveedores por
  semana en días distintos, y que el 60 % de las OCs se avisa con 7 días o más.

Otros archivos (de `obtener_rutas.py`, con OpenStreetMap; se corre una vez y necesita internet):
`lugares.csv` (engorda y orígenes), `distancias.csv` (km y horas por carretera) y `rutas.json` (trazos para
el mapa). La engorda se ubicó en José Cardel (Nominatim no encuentra el km 226 exacto).

Para regenerarlo: `.venv/bin/python data/ejemplo_ganadero/generar_ganadero.py`
