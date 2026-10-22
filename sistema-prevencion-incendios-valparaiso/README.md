# Sistema de apoyo a la prevención y clasificación de riesgo de incendios forestales en Valparaíso

Prototipo interactivo que evalúa condiciones atmosféricas y georreferencia un **nivel de riesgo de incendio forestal** (Bajo / Medio / Alto) sobre un mapa de la Región de Valparaíso, Chile. Expone una API REST, un frontend web con mapa, y una suite de monitoreo (Prometheus + Grafana).

> **Disclaimer — prototipo de investigación**
> El modelo de clasificación está **SIMULADO**: no existe inferencia de red neuronal en esta versión. La puntuación de riesgo la calcula una **heurística determinista** que emula la salida esperada de una red **CNN/LSTM** (imágenes satelitales + series meteorológicas). El campo `modelo` de la respuesta lo indica explícitamente: `heuristica-simulada-cnn-lstm`.
> Este sistema **no reemplaza ni sustituye** los servicios oficiales de **CONAF**, **SENAPRED** ni la Dirección Meteorológica de Chile. Los niveles de riesgo mostrados **no constituyen alertas oficiales**. En caso de emergencia llama al **132 (bomberos)** o **137 (SENAPRED)**.

---

## Índice

- [Arquitectura](#arquitectura)
- [Flujo de una petición](#flujo-entrada--procesamiento--resultado)
- [Capa de datos](#capa-de-datos-timeouts-threadpool-y-degradación)
- [Árbol de directorios](#árbol-de-directorios)
- [Prerrequisitos](#prerrequisitos)
- [Ejecución paso a paso](#ejecución-paso-a-paso)
- [Endpoints](#endpoints)
- [Métricas: el label `endpoint`](#métricas-el-label-endpoint-y-su-cardinalidad)
- [Ejemplo de llamada con curl](#ejemplo-de-llamada-con-curl)
- [Tests](#tests)
- [Guía de Grafana](#guía-de-grafana)
- [Variables de entorno](#variables-de-entorno)
- [Seguridad](#seguridad)
- [Documentación técnica](#documentación-técnica)

---

## Arquitectura

Cuatro servicios definidos en `docker-compose.yml` (red bridge `incendios-net`, proyecto `prevencion-incendios-valparaiso`):

| Servicio   | Contenedor             | Imagen / build            | Puerto expuesto        | Rol |
|------------|------------------------|---------------------------|------------------------|-----|
| `api`      | `incendios-api`        | `./backend` (Python 3.11, FastAPI) | **8000** por defecto (variable `API_HOST_PORT`; en esta máquina **18000**) | API REST + frontend estático + endpoint de métricas |
| `db`       | `incendios-db`         | `postgis/postgis:16-3.4`  | **5432 solo interno** (no está publicado al host) | PostgreSQL 16 + PostGIS: tabla `prediction` con geometría |
| `prometheus` | `incendios-prometheus` | `prom/prometheus:v2.54.1` | **9090**               | Scraping de `/metrics` cada 10 s |
| `grafana`  | `incendios-grafana`    | `grafana/grafana:11.4.0`  | **3000**               | Dashboard de métricas (auto-provisionado) |

```text
   Navegador (frontend Leaflet)             Cliente HTTP / curl / Swagger
          │                                          │
          │  GET /          (HTML+JS+CSS)            │  POST /predict  (header X-API-Key)
          │  GET /docs      (Swagger UI)             │  GET  /health   (sin auth)
          ▼                                          ▼  GET /metrics (sin auth)
  ┌──────────────────────────────  API FastAPI  :8000  ──────────────────────────────┐
  │  middleware de métricas → auth X-API-Key → validación Pydantic (rangos)          │
  │  → _score_riesgo() heurística simulada CNN/LSTM → clasifica Bajo/Medio/Alto      │
  │  → persistencia (si hay DB) → respuesta JSON explicada                           │
  └──────────────┬──────────────────────────────────────────┬────────────────────────┘
                 │ INSERT (punto PostGIS · SRID 4326)       │ expone /metrics
                 ▼                                          ▼
      PostgreSQL + PostGIS  :5432 (interno)                  texto Prometheus (generate_latest)
      tabla prediction (índice GIST + created_at)           │
                                                            ▼
                                                  Prometheus :9090 (scrape 10 s)
                                                            │
                                                            ▼
                                    Grafana :3000 → dashboard
                                    "Prevención Incendios Forestales - Valparaíso"
```

### Flujo entrada → procesamiento → resultado

1. **Entrada**: `POST /predict` con JSON (`temperatura_c`, `humedad_relativa`, `velocidad_viento_kmh`, `latitud`, `longitud`) y header `X-API-Key`. El frontend Leaflet rellena esos campos desde el formulario o desde un clic en el mapa.
2. **Procesamiento**:
   - Validación de rangos por Pydantic (422 si algo queda fuera).
   - Heurística determinista (`_score_riesgo`): cuatro bloques que suman máximo 100 puntos (temperatura 30, humedad 30, viento 25, proximidad/vegetación simulada 15).
   - Clasificación: `score >= 70 → Alto`, `>= 40 → Medio`, resto `Bajo`.
   - Persistencia del resultado en `prediction` con punto PostGIS (SRID 4326), ejecutada en un *threadpool* para no bloquear el event loop. Si PostgreSQL no responde, la predicción **se responde igualmente** con `id_prediccion: null` (degradación controlada).
3. **Resultado**: JSON con `nivel_riesgo`, `score`, lista `factores` explicativos, `timestamp`, coordenadas, `modelo` e `id_prediccion`. Además se incrementa el contador `predict_requests_total{riesgo="Bajo|Medio|Alto"}` y el frontend coloca un marcador en el mapa.

### Capa de datos: timeouts, threadpool y degradación

- **La URL de conexión lleva timeouts.** `_database_url()` normaliza la URL con `sqlalchemy.engine.make_url` y añade, si no vienen ya (lo que venga en `DATABASE_URL` se respeta), `connect_timeout=5` y `options=-c statement_timeout=10000`. URL efectiva verificada dentro del contenedor:

  ```text
  postgresql+psycopg2://incendios:incendios123@db:5432/prevencion_incendios?connect_timeout=5&options=-c+statement_timeout%3D10000
  ```

- **El arranque falla rápido.** La comprobación `SELECT 1` del `lifespan` está acotada por `connect_timeout`: contra un host que *filtra* el tráfico en lugar de rechazarlo (probado con `10.255.255.1`) falla en **≈5 s** en vez de colgarse indefinidamente. El servicio **no aborta**: registra un `WARNING` y arranca en **modo degradado** (sin persistencia), según el ADR-002 de `docs/NOTAS_TECNICAS.md`.
- **La persistencia no bloquea el event loop.** El `INSERT` usa `psycopg2` síncrono, así que se ejecuta con `starlette.concurrency.run_in_threadpool` (igual que la comprobación de BD del arranque). El motor se crea una sola vez, protegido por un lock para que varias peticiones concurrentes no creen motores duplicados.
- **Degradación controlada.** Si la BD no está disponible, `_persistir_prediccion()` captura la excepción, la registra como `WARNING` y devuelve `None`: la respuesta se sirve **igual** (HTTP 200) con `id_prediccion: null`. Los tests lo fijan como comportamiento esperado.

---

## Árbol de directorios

```text
sistema-prevencion-incendios-valparaiso/
├── README.md                      # este archivo
├── docker-compose.yml             # orquestación: api, db, prometheus, grafana
├── .env.example                   # plantilla de variables de entorno (copiar a .env)
├── .gitignore                     # ignora .env, __pycache__, .pytest_cache, .venv, venv
├── .dockerignore                  # excluye artefactos (cachés, .env, docs) de builds con contexto raíz
│
├── backend/
│   ├── Dockerfile                 # python:3.11-slim + uvicorn + healthcheck curl (WORKDIR /app)
│   ├── .dockerignore              # excluye artefactos del contexto de build del servicio api (contexto = backend/)
│   ├── main.py                    # aplicación FastAPI: /health, /predict, /metrics y frontend
│   ├── requirements.txt           # fastapi, uvicorn, prometheus-client, psycopg2, sqlalchemy, pydantic
│   ├── requirements-dev.txt       # requirements.txt + pytest + httpx (desarrollo/tests; no va a producción)
│   ├── pytest.ini                 # testpaths=tests, pythonpath=.
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py            # fixtures: cliente TestClient, API_KEY, DB caída deliberada
│       └── test_predict.py        # 12 tests: auth, esquema, clasificación, 422, métricas, key no-ASCII
│
├── database/
│   └── init/
│       └── init.sql               # extensión PostGIS + tabla prediction + índices
│
├── frontend/
│   └── index.html                 # SPA autocontenida (HTML/CSS/JS + Leaflet 1.9.4 vía CDN)
│
├── docs/
│   └── NOTAS_TECNICAS.md          # decisiones arquitectónicas y hoja de ruta a modelo real
│
└── monitoring/
    ├── prometheus/
    │   └── prometheus.yml         # scrape de api:8000/metrics cada 10 s
    └── grafana/
        ├── dashboards/
        │   └── incendios.json     # dashboard con 3 paneles (latencia, RPS, errores 5xx)
        └── provisioning/
            ├── datasources/
            │   └── prometheus.yml # datasource Prometheus → http://prometheus:9090
            └── dashboards/
                └── dashboards.yml # proveedor de archivos en /etc/grafana/dashboards
```

> Nota: también aparecerán artefactos locales ignorados por git (`.env`, `backend/__pycache__/`, `backend/.pytest_cache/`, `.ruff_cache/`, `.venv/`). No forman parte del repositorio.

> Nota sobre los `.dockerignore`: el servicio `api` se construye con `build: ./backend`, así que **el contexto de build es `backend/`** y Docker busca el `.dockerignore` **en la raíz de ese contexto**. Por tanto, el `.dockerignore` de la raíz del repo (el del bloque de arriba) **no se aplica a la imagen del API**: era inerte para ese servicio y sólo serviría para un build con contexto raíz (`docker build .`). Ahora existe **`backend/.dockerignore`**, que sí aplica a la imagen del API y excluye `__pycache__/`, `*.pyc`, `*.pyo`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.venv/`, `venv/`, `.env`, `.env.*` (salvo `.env.example`), `.git/`, `.gitignore` y carpetas de editor. `tests/` **no** se excluye a propósito: el README documenta ejecutar los tests dentro del contenedor.

---

## Prerrequisitos

- **Docker** 20.10 o superior.
- **Docker Compose v2** (plugin `docker compose`). Verifica con:

  ```bash
  docker compose version
  ```

  > En este equipo está verificado `docker compose` (plugin v2). Si tu sistema sólo tiene el binario antiguo con guion, sustituye `docker compose` por `docker-compose` en los comandos de este README.

- Puerto **8000** libre en el host (o define `API_HOST_PORT`, ver [más abajo](#puerto-alternativo-del-api-api_host_port)). En esta máquina el 8000 **está ocupado** por otro servicio y el `.env` local usa `API_HOST_PORT=18000`: el API se publica en `http://localhost:18000` (ver [la nota de URLs](#urls-de-servicio)).
- Puerto **3000** (Grafana) y **9090** (Prometheus) libres.

---

## Ejecución paso a paso

```bash
# 1. Clonar/entrar al proyecto
cd sistema-prevencion-incendios-valparaiso

# 2. Crear el archivo de entorno a partir de la plantilla
cp .env.example .env

# 3. Levantar todo (imágenes, API, DB, Prometheus y Grafana)
docker compose up --build
```

La primera ejecución descarga las imágenes, instala las dependencias de `backend/` y ejecuta `database/init/init.sql` en el arranque limpio de PostgreSQL. Espera a que `incendios-api` aparezca como `healthy` (healthcheck contra `GET /health`).

> En esta máquina hay que añadir `API_HOST_PORT=18000` al `.env` (el puerto 8000 del host lo ocupa otro servicio), con lo que el API queda publicado en `http://localhost:18000`. Ver [la nota de URLs](#urls-de-servicio).

### URLs de servicio

| Servicio | URL | Credenciales |
|----------|-----|--------------|
| Frontend (mapa + formulario) | http://localhost:8000/ | — |
| Swagger UI (OpenAPI) | http://localhost:8000/docs | — |
| ReDoc | http://localhost:8000/redoc | — |
| Métricas Prometheus (formato texto) | http://localhost:8000/metrics | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | `admin` / `admin` (por defecto) |
| PostgreSQL + PostGIS | sólo interno en la red `incendios-net` (puerto 5432 del contenedor `incendios-db`) | `incendios` / `incendios123` (por defecto) |

- La contraseña de Grafana viene de `GF_SECURITY_ADMIN_PASSWORD` (`.env.example` la deja en `admin`); el usuario es siempre `admin` (`GF_SECURITY_ADMIN_USER` no se sobreescribe).
- El registro de usuarios está deshabilitado en Grafana (`GF_USERS_ALLOW_SIGN_UP=false`).

> ### ⚠️ En esta máquina el API se publica en `http://localhost:18000`
>
> La tabla anterior (y todos los ejemplos de `curl` de este README) usan **`localhost:8000`**, que es el valor por defecto y sigue siendo válido en cualquier máquina donde el puerto 8000 esté libre.
>
> En **esta** máquina el puerto **8000 del host ya está ocupado por otro servicio**, así que el `.env` local define:
>
> ```bash
> API_HOST_PORT=18000
> ```
>
> Con esa variable el API queda en **`http://localhost:18000`** (el contenedor sigue escuchando en 8000 por dentro; sólo cambia el puerto publicado al host). Para ejecutar los ejemplos de este README aquí, **basta con sustituir el puerto**:
>
> ```bash
> # en vez de http://localhost:8000/predict  ->  http://localhost:18000/predict
> curl -s http://localhost:18000/health
> ```
>
> Prometheus (9090) y Grafana (3000) **no cambian**: hablan con `api:8000` por la red interna del compose. Detalle en [Puerto alternativo del API](#puerto-alternativo-del-api-api_host_port).

### Puerto alternativo del API (`API_HOST_PORT`)

En `docker-compose.yml` el servicio `api` publica:

```yaml
ports:
  # Por defecto 8000:8000; API_HOST_PORT permite evitar colisiones locales.
  - "${API_HOST_PORT:-8000}:8000"
```

Es decir:

- **Sin** `API_HOST_PORT` en `.env` → el API queda en `http://localhost:8000` (el contenedor siempre escucha en 8000 por dentro; sólo cambia el puerto del host).
- **Si 8000 ya está ocupado** por otro proceso (p. ej. otro contenedor), añade una línea a tu `.env`:

  ```bash
  API_HOST_PORT=18000
  ```

  y reinicia con `docker compose up --build`. La API y el frontend quedan entonces en `http://localhost:18000` (mismo cambio en `/docs`, `/metrics`, etc.).

> Detalle verificado: `.env.example` **no incluye** `API_HOST_PORT` (sólo `API_KEY`, `POSTGRES_*` y `GF_SECURITY_ADMIN_PASSWORD`), así que por defecto se usa 8000. La variable se añade a `.env` a mano únicamente cuando hace falta desplazar el puerto. Prometheus y Grafana **no** se ven afectados: hablan con `api:8000` por la red interna del compose.

> **Estado actual de esta máquina**: el `.env` local ya tiene `API_HOST_PORT=18000` porque el 8000 del host está ocupado por otro servicio. El stack levantado expone `0.0.0.0:18000->8000/tcp` (`docker compose ps`), de modo que aquí la API, el frontend, `/docs` y `/metrics` se sirven en `http://localhost:18000`. Todo lo que sigue en este README usa `localhost:8000` como **valor por defecto de referencia**: sustituye el puerto en los comandos si usas otro.

### Detener el stack

```bash
docker compose down        # detiene y borra contenedores (conserva volúmenes de datos)
docker compose down -v     # además borra los volúmenes pgdata/prometheus-data/grafana-data
```

---

## Endpoints

| Método | Ruta | Autenticación (`X-API-Key`) | Descripción |
|--------|------|------------------------------|-------------|
| `GET`  | `/health` | **No** | Estado del servicio. Respuesta: `{"status": "ok"}` |
| `POST` | `/predict` | **Sí** | Clasifica el riesgo. 401 sin header, 403 con key inválida, 422 con datos fuera de rango |
| `GET`  | `/metrics` | **No** | Métricas Prometheus (`http_requests_total`, `http_request_duration_seconds`, `predict_requests_total`) |
| `GET`  | `/` y ficheros estáticos | **No** | Frontend Leaflet (`frontend/index.html`) |
| `GET`  | `/docs`, `/redoc` | **No** | Documentación OpenAPI generada por FastAPI |

### Contrato de `POST /predict`

**Entrada** (rangos validados por Pydantic):

| Campo | Tipo | Rango |
|-------|------|-------|
| `temperatura_c` | number | -10 … 55 (°C) |
| `humedad_relativa` | number | 0 … 100 (%) |
| `velocidad_viento_kmh` | number | 0 … 200 (km/h) |
| `latitud` | number | -33.5 … -30 (zona de Valparaíso) |
| `longitud` | number | -72 … -70 |

**Salida** (`PrediccionResponse`): `nivel_riesgo` (`"Bajo" | "Medio" | "Alto"`), `score` (0–100), `factores[]` (`nombre`, `valor`, `aporte_puntos`, `descripcion`), `timestamp` (UTC ISO 8601), `latitud`, `longitud`, `modelo`, `id_prediccion` (entero o `null` si no hubo base de datos).

### Métricas: el label `endpoint` y su cardinalidad

`GET /metrics` expone tres familias de métricas:

| Métrica | Tipo | Labels |
|---------|------|--------|
| `http_requests_total` | Counter | `method`, `endpoint`, `status` |
| `http_request_duration_seconds` | Histogram (5 ms … 10 s) | `method`, `endpoint` |
| `predict_requests_total` | Counter | `riesgo` (`Bajo`/`Medio`/`Alto`) |

El label `endpoint` se **deriva de la ruta que atendió la petición**, nunca del path arbitrario que pidió el cliente. Los valores posibles son:

| Valor de `endpoint` | Cuándo se usa |
|---------------------|--------------|
| `/predict`, `/health`, `/metrics` | Path de una ruta de la API (FastAPI `APIRoute`) |
| `/docs`, `/redoc`, `/openapi.json` | Path de las rutas Starlette puras (documentación OpenAPI) |
| `static` | Peticiones servidas por el montaje de ficheros estáticos (el frontend Leaflet) |
| `unmatched` | 404/405 reales: la petición no encaja en ninguna ruta de la API |

> **La cardinalidad de labels está acotada por diseño**: el número de series posibles es finito y conocido (las rutas de la aplicación + `static` + `unmatched`). Nunca se usa `request.url.path` como valor de backup, porque el path lo elige el cliente y generaría un número de series ilimitado (ataque de cardinalidad contra Prometheus).

Salida real de `GET /metrics` (contadores acumulados del proceso en ejecución):

```text
http_requests_total{endpoint="/docs",method="GET",status="200"} 2.0
http_requests_total{endpoint="/health",method="GET",status="200"} 38.0
http_requests_total{endpoint="/metrics",method="GET",status="200"} 38.0
http_requests_total{endpoint="/openapi.json",method="GET",status="200"} 1.0
http_requests_total{endpoint="/predict",method="POST",status="200"} 9.0
http_requests_total{endpoint="/predict",method="POST",status="401"} 2.0
http_requests_total{endpoint="/predict",method="POST",status="403"} 5.0
http_requests_total{endpoint="/predict",method="POST",status="422"} 2.0
http_requests_total{endpoint="/redoc",method="GET",status="200"} 1.0
http_requests_total{endpoint="static",method="GET",status="200"} 11.0
http_requests_total{endpoint="unmatched",method="GET",status="404"} 4.0
http_requests_total{endpoint="unmatched",method="POST",status="405"} 1.0
```

---

## Ejemplo de llamada con curl

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-api-key-123" \
  -d '{
        "temperatura_c": 32.5,
        "humedad_relativa": 18.0,
        "velocidad_viento_kmh": 45.0,
        "latitud": -33.0472,
        "longitud": -71.6127
      }'
```

Respuesta real de ejemplo (HTTP 200):

```json
{
  "nivel_riesgo": "Alto",
  "score": 70.1,
  "factores": [
    {
      "nombre": "temperatura",
      "valor": 32.5,
      "aporte_puntos": 17.5,
      "descripcion": "Aporte térmico: escala creciente entre 15 °C y 45 °C."
    },
    {
      "nombre": "humedad_relativa",
      "valor": 18.0,
      "aporte_puntos": 24.6,
      "descripcion": "Aporte por sequedad: inverso de la humedad relativa (0-100%)."
    },
    {
      "nombre": "velocidad_viento",
      "valor": 45.0,
      "aporte_puntos": 14.06,
      "descripcion": "Aporte de dispersión: escala creciente hasta 80 km/h."
    },
    {
      "nombre": "proximidad_vegetacion",
      "valor": 0.9268,
      "aporte_puntos": 13.9,
      "descripcion": "Dentro del área de Valparaíso (0.0 km del punto de referencia más cercano); vegetación simulada = 0.84."
    }
  ],
  "timestamp": "2026-10-21T05:29:56.801770Z",
  "latitud": -33.0472,
  "longitud": -71.6127,
  "modelo": "heuristica-simulada-cnn-lstm",
  "id_prediccion": 2
}
```

Casos de error verificados (códigos de estado comprobados contra el stack en marcha):

```bash
# Sin header X-API-Key → 401 {"detail":"Falta el header X-API-Key."}
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"temperatura_c":32.5,"humedad_relativa":18,"velocidad_viento_kmh":45,"latitud":-33.0472,"longitud":-71.6127}'

# Key incorrecta → 403 {"detail":"API key inválida."}
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' -H 'X-API-Key: clave-mala' \
  -d '{"temperatura_c":32.5,"humedad_relativa":18,"velocidad_viento_kmh":45,"latitud":-33.0472,"longitud":-71.6127}'

# API Key con caracteres no-ASCII → 403 (comparación en bytes, nunca 500)
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' -H $'X-API-Key: cl\xe9-mala' \
  -d '{"temperatura_c":32.5,"humedad_relativa":18,"velocidad_viento_kmh":45,"latitud":-33.0472,"longitud":-71.6127}'

# Fuera de rango (temperatura 99 °C o latitud 0) → 422 (validación de Pydantic)
```

> Si usaste `API_HOST_PORT=18000`, sustituye `localhost:8000` por `localhost:18000` en todos los comandos (en esta máquina es el caso: ver [la nota de ports](#urls-de-servicio)).

---

## Tests

Los tests cubren 12 casos: `/health`, autenticación (401 sin header, 403 con key incorrecta, 403 con key no-ASCII), esquema de respuesta, clasificación Alto/Medio/Bajo, validaciones 422 y exposición de métricas. `backend/tests/conftest.py` apunta `DATABASE_URL` a un puerto inválido (`127.0.0.1:1`) para que sean **herméticas**: no tocan ninguna base de datos real y ejercitan la ruta degradada (`id_prediccion = null`).

**Método verificado** (venv local; ejecutado desde la raíz del repo, resultado `12 passed, 1 warning`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt

pytest backend/tests
```

Salida observada:

```text
12 passed, 1 warning in 0.17s
```

> El `pytest.ini` de `backend/` fija `testpaths = tests` y `pythonpath = .`, por eso el comando funciona tal cual desde la raíz (pytest localiza `backend/pytest.ini` y añade `backend/` al `sys.path`; además `conftest.py` lo inserta explícitamente).

> **Por qué `requirements-dev.txt` y no `requirements.txt`**: `backend/requirements-dev.txt` contiene `-r requirements.txt` + `pytest` + `httpx`. **`httpx` es obligatoria**: `fastapi.testclient` la importa en tiempo de importación, así que sin ella pytest aborta **antes de colectar ningún test** con `RuntimeError: The starlette.testclient module requires the httpx package to be installed.` Ese es exactamente el fallo que reproducía el comando que documentaba la versión anterior de este README (`pip install -r backend/requirements.txt pytest`): instalaba `pytest` pero **no** `httpx`, de modo que el método **no era reproducible** en un venv limpio.

**Alternativa en contenedor** (verificada): la imagen del API incluye `backend/requirements-dev.txt` (el `Dockerfile` hace `COPY . .` sobre el contexto `backend/`), pero **no** instala `pytest` ni `httpx` — la imagen de producción sólo instala `requirements.txt`. El `WORKDIR` de la imagen es `/app`, así que las rutas relativas son las del backend (`requirements-dev.txt`, `tests/`):

```bash
docker compose up --build                      # asegura que backend/tests/ esté en la imagen
docker compose exec api pip install -r requirements-dev.txt
docker compose exec api python -m pytest tests -q
```

Salida observada:

```text
12 passed, 1 warning in 0.18s
```

---

## Guía de Grafana

### Opción A — Auto-provisionada (la estándar)

Grafana se configura sola al arrancar mediante ficheros montados en `docker-compose.yml`:

- **Datasource**: `monitoring/grafana/provisioning/datasources/prometheus.yml` → crea el datasource `Prometheus` (uid `prometheus`) apuntando a `http://prometheus:9090`, marcado como por defecto y **no editable** desde la UI (`editable: false`), con `timeInterval: 10s`.
- **Proveedor de dashboards**: `monitoring/grafana/provisioning/dashboards/dashboards.yml` → carpeta **"Incendios Forestales"**, lee los JSON de `/etc/grafana/dashboards` cada 30 s y permite actualizaciones desde la UI (`allowUiUpdates: true`).
- **Dashboard**: `monitoring/grafana/dashboards/incendios.json` → **"Prevención Incendios Forestales - Valparaíso"** (uid `incendios-valparaiso`), con tres paneles de tipo *timeseries*.

Pasos:

1. Levanta el stack (`docker compose up --build`) y abre **http://localhost:3000**.
2. Inicia sesión con `admin` / `admin` (te pedirá cambiar la contraseña en el primer acceso; puedes omitirlo).
3. Menú lateral → **Dashboards** → carpeta **Incendios Forestales** → **Prevención Incendios Forestales - Valparaíso**.

El dashboard ya existe y apunta al datasource Prometheus correcto; no hay que crear nada. Genera tráfico (varias llamadas a `/predict`) y en unos segundos verás las tres gráficas con datos.

### Opción B — Montarlo a mano

Por si quieres replicarlo en otra instancia de Grafana (o entender qué hizo el provisioning):

**1. Data source**

1. Menú lateral → **Connections → Data sources → Add data source**.
2. Elige **Prometheus**.
3. URL: `http://prometheus:9090` (nombre de servicio de la red Docker, **no** `localhost`).
4. **Save & test** → debe responder *"Data source is working"*.

**2. Dashboard**

1. Menú lateral → **Dashboards → New dashboard → Add visualization** (o **New dashboard → Add panel**).
2. Selecciona el datasource **Prometheus** y añade tres paneles con estas queries (las mismas del dashboard auto-provisionado):

| Panel | Query PromQL |
|-------|--------------|
| Latencia promedio HTTP | `sum(rate(http_request_duration_seconds_sum[5m])) / sum(rate(http_request_duration_seconds_count[5m]))` |
| Solicitudes por segundo (RPS) | `sum(rate(http_requests_total[1m]))` |
| Errores HTTP (5xx) | `sum(rate(http_requests_total{status=~"5.."}[1m]))` |

Las tres queries **agregan sobre todos los labels** (`sum(...)` sin `by`), así que son indiferentes a cómo se etiquete `endpoint` y siguen funcionando con el esquema actual (rutas de la API + `static` + `unmatched`; ver [Métricas](#métricas-el-label-endpoint-y-su-cardinalidad)).

Para ver el desglose por endpoint (útil para distinguir el tráfico de la API del tráfico de assets y de los 404), añade un panel con `sum by (endpoint) (...)`:

| Panel | Query PromQL |
|-------|--------------|
| RPS por endpoint | `sum by (endpoint) (rate(http_requests_total[1m]))` |
| Latencia por endpoint | `sum by (endpoint) (rate(http_request_duration_seconds_sum[5m])) / sum by (endpoint) (rate(http_request_duration_seconds_count[5m]))` |

Salida real de la primera query contra el Prometheus del stack (verificada):

```text
{endpoint="/health"}       0.099998
{endpoint="/metrics"}      0.099998
{endpoint="/predict"}      0.019999
{endpoint="/docs"}         0
{endpoint="/openapi.json"} 0
{endpoint="/redoc"}        0
{endpoint="static"}        0
{endpoint="unmatched"}     0
```

3. Guarda con un título (p. ej. `Prevención Incendios Forestales - Valparaíso`).

**Comprobación rápida en Prometheus** (http://localhost:9090): la job `api` debe aparecer en **Status → Targets** con estado `up`, scrapeando `api:8000/metrics` cada 10 s. También puedes probar las queries directamente en el navegador de Prometheus.

---

## Variables de entorno

Plantilla oficial: **`.env.example`** (cópiala con `cp .env.example .env`; `.env` está ignorado por git).

Contenido de `.env.example`:

| Variable | Ejemplo / default | Descripción |
|----------|-------------------|-------------|
| `API_KEY` | `dev-api-key-123` | Clave que debe traer el header `X-API-Key` en `POST /predict`. En el compose: `${API_KEY:-dev-api-key-123}` |
| `POSTGRES_USER` | `incendios` | Usuario de PostgreSQL |
| `POSTGRES_PASSWORD` | `incendios123` | Contraseña de PostgreSQL |
| `POSTGRES_DB` | `prevencion_incendios` | Base de datos creada en el arranque |
| `GF_SECURITY_ADMIN_PASSWORD` | `admin` | Contraseña del usuario `admin` de Grafana |

Variables adicionales reconocidas (no están en la plantilla):

| Variable | Default | Descripción |
|----------|---------|-------------|
| `API_HOST_PORT` | `8000` | Puerto del host mapeado al 8000 interno del API. Añádela a `.env` si el 8000 está ocupado |
| `DATABASE_URL` | construida por compose | URL SQLAlchemy completa; si no existe se arma desde `POSTGRES_USER/PASSWORD/DB`, `DB_HOST` y `DB_PORT` |
| `DB_HOST` / `DB_PORT` | `localhost` / `5432` (fuera de compose) | Host/puerto de PostgreSQL. En el compose se fijan a `db` / `5432` |
| `STATIC_DIR` | resuelto automáticamente | Ruta del frontend dentro del contenedor; el compose la fija a `/app/frontend` |

---

## Seguridad

**Decisión: API Key en el header `X-API-Key`** (ver `docs/NOTAS_TECNICAS.md`, registros `219674af` y `8a940e7b`).

- Sólo `POST /predict` requiere autenticación. `/health`, `/metrics` y el frontend son públicos (son endpoints de sondeo/observabilidad sin datos sensibles).
- La clave se compara con `secrets.compare_digest` (comparación en tiempo constante) contra la variable de entorno `API_KEY` (default de desarrollo: `dev-api-key-123`).
- **La comparación se hace sobre los bytes UTF-8** de la clave recibida y la esperada (`_coincide_api_key()`): `secrets.compare_digest` sólo admite cadenas **ASCII**, por lo que una clave con caracteres no-ASCII lanzaba un `TypeError` y devolvía **HTTP 500** en lugar del 403 documentado. Comparando bytes se mantiene la comparación **en tiempo constante** y una clave no-ASCII devuelve **403** (nunca 5xx). Verificado:

  ```bash
  # clave con el byte 0xE9 (latin-1) -> 403 {"detail":"API key inválida."}
  curl -s -o /dev/null -w '%{http_code}\n' -X POST http://localhost:8000/predict \
    -H 'Content-Type: application/json' -H $'X-API-Key: cl\xe9-mala' \
    -d '{"temperatura_c":32.5,"humedad_relativa":18,"velocidad_viento_kmh":45,"latitud":-33.0472,"longitud":-71.6127}'
  # -> 403
  ```

- Códigos: **401** si falta el header (con `WWW-Authenticate: ApiKey`), **403** si la clave no coincide. Nunca un 5xx por contenido del header.
- **La clave se valida en el arranque**: la clave efectiva es `os.getenv("API_KEY") or DEFAULT_API_KEY` (una variable presente pero **vacía** ya no vale: `compare_digest("", "")` daba `True` y aceptaba un header vacío). Si la clave efectiva quedara en blanco, el arranque aborta con un `RuntimeError` explícito: es imposible levantar el servicio con `/predict` desprotegido.
- El frontend la expone en *Configuración avanzada: API Key* con `dev-api-key-123` precargada, y envía `X-API-Key` en cada `fetch` a `/predict`.
- En producción: cambia `API_KEY` en `.env`, no uses el default, termina el tráfico con TLS y valora migrar a JWT/OAuth2 cuando existan usuarios y roles (ese es el alcance de la decisión registrada).

---

## Documentación técnica

- [`docs/NOTAS_TECNICAS.md`](docs/NOTAS_TECNICAS.md) — decisiones arquitectónicas (heurística simulada, degradación sin DB, PostGIS SRID 4326, API Key vs JWT, métricas instrumentadas y derivación del label `endpoint`, comparación de la API key en bytes, timeouts de BD y persistencia en *threadpool*, separación de `requirements-dev.txt`, `.dockerignore` en la raíz del contexto de build), esquema de datos y hoja de ruta para sustituir la heurística por un modelo CNN/LSTM real entrenado con datos INIA/CONAF.

