# Ruteo de última milla para PYMES (demo)

Demo del proyecto "Modelo predictivo de demanda integrado con VRP para última milla en PYMES" (ITP).
Carga los pedidos de una PYME, arma rutas con Google OR-Tools, las muestra en un mapa y compara
km, costo y fletes urgentes en cuatro escenarios (E1 a E4).

## Cómo arrancar

Requisitos: Python 3.11 o más reciente y [uv](https://docs.astral.sh/uv/) (o `python -m venv`).

```bash
# 1. Crear el entorno e instalar dependencias (solo la primera vez)
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
#   sin uv:  python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Correr las pruebas
.venv/bin/python -m pytest

# 3. Abrir la app (se abre en el navegador en http://localhost:8501)
.venv/bin/streamlit run app.py
```

## Recalcular con tráfico (pestaña 🚦, prueba de concepto)

Funciona sin internet con **incidentes simulados** (clic en el mapa o lista de `data/incidentes_ejemplo.csv`).
Para usar **TomTom en vivo** (opcional):

1. Crea una cuenta gratuita en <https://developer.tomtom.com> y copia tu clave.
2. Copia `.env.example` como `.env` y pon la clave: `TOMTOM_API_KEY=...`. El archivo `.env` no se comparte.
3. Reinicia la app y elige "TomTom en vivo" en la barra lateral.

El plan gratuito de TomTom da una cuota **mensual** por API (Routing 20,000; incidentes 2,500;
Matrix 2,500). La app guarda cada consulta 5 minutos en `.cache_tomtom/` y muestra cuánto llevas usado.

## Carpetas

| carpeta         | contenido                                                        |
|-----------------|------------------------------------------------------------------|
| `src/`          | código: carga, distancias, ruteo, escenarios                     |
| `data/`         | datos reales de la empresa (cuando lleguen)                      |
| `data/ejemplo/` | datos **inventados** para probar (ver `data/ejemplo/LEEME.md`)   |
| `docs/`         | artículo de referencia, guía del proyecto y explicación          |
| `tests/`        | pruebas automáticas (pytest)                                     |
| `config.yaml`   | parámetros de costo y operación; cada uno dice si es real o supuesto |

## Compartir la app con otros (enlace temporal)

```bash
./compartir.sh
```
Arranca la app y crea un enlace público `https://....trycloudflare.com` con Cloudflare Tunnel
(requiere `cloudflared` en `~/.local/bin`). El enlace cambia cada vez y funciona mientras la
terminal siga abierta; Ctrl+C lo cierra. La app no tiene login: cualquiera con el enlace ve los
datos cargados, así que no lo compartas con datos de la empresa sin permiso.
