# Notas técnicas — Prevención y clasificación de riesgo de incendios forestales (Valparaíso)

Documento de **explicación y registro de decisiones** del prototipo. Complementa al [`README.md`](../README.md) (uso y ejecución) con el *por qué* de cada elección arquitectónica y con la hoja de ruta para pasar de la heurística simulada a un modelo de aprendizaje automático real.

> **Disclaimer**: el clasificador actual es una **heurística determinista que simula** la salida de una red CNN/LSTM. No hay inferencia de red neuronal en esta versión. No reemplaza a CONAF, SENAPRED ni a la Dirección Meteorológica de Chile.

---

## 1. Contexto y alcance

- **Problema**: la Región de Valparaíso concentra un riesgo alto de incendios forestales (clima mediterráneo con estación seca estival, topografía de cerros, interfaz urbano-forestal). Se busca una herramienta de **apoyo** que, dadas condiciones atmosféricas y una ubicación, produzca una clasificación de riesgo *explicada* y georreferenciada.
- **Alcance del prototipo**: API REST de predicción, frontend web con mapa, persistencia geoespacial y observabilidad. Todo el stack es reproducible con un solo `docker compose up --build`.
- **No alcance**: alertamiento oficial, pronóstico operativo, sustitución de sistemas de CONAF/SENAPRED, modelo de ML entrenado.

## 2. Stack y decisiones globales (resumen)

| ADR | Decisión | Registro Lemoria |
|-----|----------|------------------|
| ADR-001 | Heurística determinista que simula el modelo CNN/LSTM | `4b2cc44b-071f-4b27-8240-80994e28732e` |
| ADR-002 | Degradación controlada cuando PostgreSQL no está disponible | `01fd0fb8-ac6f-43fd-ac53-2c1c1dcd3fa7` |
| ADR-003 | Persistencia geoespacial con PostGIS y SRID 4326 | `d426692e-1b55-42ff-8c0e-51e1ed45209c` |
| ADR-004 | API Key (`X-API-Key`) en lugar de JWT | `219674af-3bd2-429b-8cc6-61fe29c2f163` |
| ADR-005 | Métricas Prometheus instrumentadas en middleware (control de cardinalidad) | `05e1313a-d4b8-4775-902b-9f79a763ab2e` |
| ADR-006 | Frontend SPA autocontenido servido por FastAPI (sin build step) | `569889bb-859c-4e38-98af-ac6cfc41a140`, `78b91b73-ef5c-4afa-bfc3-98cd73431dfe` |
| ADR-007 | Puerto del API parametrizado con `API_HOST_PORT` (default 8000) | `8fbb3425-df59-4fd9-b740-8ecbffbbaee6` |
| ADR-008 | Stack tecnológico base: FastAPI + PostgreSQL/PostGIS + Prometheus/Grafana | `187faccd-e24b-403a-b5f1-aa3ece28d333` |
| ADR-009 | Comparación de la API Key en bytes UTF-8 y validación en el arranque | `8a940e7b-21c9-4212-b910-df90dab5325b`, `e536aa74-3c17-4807-b615-c8fd26891b48`, `49ed12d5-19b2-4b99-a653-5cbf49e426cb` |
| ADR-010 | Derivación explícita y acotada del label `endpoint` en las métricas | `e413a59f-0eeb-4235-abcc-ba04c1137785`, `aa33597b-62a1-43e0-9535-d68933bfa0fa` |
| ADR-011 | Dependencias de desarrollo separadas (`backend/requirements-dev.txt`) | `9e9e6da2-f415-4efe-a9ea-41e94c2925dc`, `666fc278-e956-4eb0-aa22-a43862e6cee4` |
| ADR-012 | Timeouts de la BD (`connect_timeout`/`statement_timeout`) y persistencia en threadpool | `78597414-25a3-46f0-87d8-2cb26decf51b`, `ebd9a789-c970-4687-9a7d-ab10e59e2958` |
| ADR-013 | `.dockerignore` en la raíz del contexto de build (`backend/.dockerignore`) | `ba79adb3-b619-4a79-be05-174cf94785bb` |

---

## 3. ADR-001 — Heurística determinista que simula el modelo CNN/LSTM

### Contexto

El objetivo final del sistema es clasificar el riesgo con una red **CNN** (imágenes satelitales / mapas de riesgo) + **LSTM** (series temporales meteorológicas). En el prototipo no existen datos etiquetados ni infraestructura de entrenamiento, pero sí se necesita demostrar de punta a punta el flujo: entrada → clasificación explicada → persistencia → observabilidad → frontend.

### Decisión

Sustituir la inferencia de la red por un módulo determinista `_score_riesgo()` en `backend/main.py` que produce un **score 0–100** a partir de cuatro bloques aditivos, y clasifica con umbrales fijos:

| Bloque | Fórmula (resumida) | Puntos máx. |
|--------|--------------------|-------------|
| Temperatura | `(T − 15) / 30`, acotado a [0, 1] | 30 |
| Humedad relativa | `(100 − HR) / 100`, acotado a [0, 1] | 30 |
| Velocidad del viento | `v / 80`, acotado a [0, 1] | 25 |
| Proximidad/vegetación simulada | mezcla 0.55 · proximidad + 0.45 · vegetación | 15 |

- **Clasificación**: `score >= 70 → Alto`, `score >= 40 → Medio`, resto `Bajo`.
- **Proximidad**: distancia (proyección equirrectangular) a tres puntos de referencia (`Valparaíso - cerros` peso 1.00, `Viña del Mar` 0.85, `Quilpué - zona agrícola` 0.70) con caída gaussiana (`exp(-(d/45)²)`). Si el punto más cercano está a más de 60 km, el factor se penaliza 60 % (se considera "fuera del área de Valparaíso").
- **Vegetación simulada**: índice determinista derivado de `SHA-256(lat,lon)` (mismo punto ⇒ siempre el mismo índice). Es un marcador de posición del índice NDVI real que aportará el modelo futuro.
- **Límites geográficos** (validados en Pydantic y en el frontend): latitud `-33.5 … -30` y longitud `-72 … -70`. El límite sur se amplió a `-33.5` para incluir el área urbana (Valparaíso, Viña del Mar, Quilpué, cuya latitud es ≈ −33.05) — registro `99267d88-1fdc-4899-b45a-906cc9f876b7`.
- **Explicabilidad**: la respuesta incluye `factores[]` con `nombre`, `valor`, `aporte_puntos` y `descripcion`, de modo que el usuario ve *por qué* el score es alto. El campo `modelo` identifica la versión: `heuristica-simulada-cnn-lstm`.

### Consecuencias

- (+) Sin datos ni GPU: el prototipo es ejecutable, testeable y demostrable.
- (+) Determinismo y explicabilidad totales: mismas entradas ⇒ misma salida; cada punto es rastreable a un factor.
- (+) El contrato de la API (`PrediccionResponse`) ya es el definitivo; cambiar la heurística por la red no rompe clientes.
- (−) No modela secuencias temporales, ni satélite, ni aprendizaje: es un **sustituto ilustrativo**.
- (−) La "vegetación" es ruido determinista, no información ecológica real; el factor de proximidad es una aproximación de la interfaz urbano-forestal.

---

## 4. ADR-002 — Degradación controlada cuando PostgreSQL no está disponible

### Contexto

Un prototipo de demostración no debe quedarse inoperativo porque la base de datos esté caída o tardía en arrancar (el arranque de PostGIS puede tardar; en CI no hay red). Tampoco debe mentir: si no se persistió, el cliente debe poder saberlo.

### Decisión

- El engine de SQLAlchemy se crea de forma **perezosa** y reutilizada (`_get_engine()`, `pool_pre_ping=True`, `pool_size=5`); un fallo al crearlo devuelve `None` sin lanzar.
- `_comprobar_base_de_datos()` sólo **registra** (log, nivel warning) el estado de la conexión en el `lifespan` de arranque; nunca aborta el proceso. La comprobación está acotada por `connect_timeout` y se ejecuta en un *threadpool* (ver **ADR-012**).
- `_persistir_prediccion()` captura `SQLAlchemyError` (y el `psycopg2.Error` subyacente) y cualquier excepción inesperada: log warning y retorno de `None`. La petición `/predict` **siempre responde 200** con el resultado calculado.
- **Contrato honesto**: si no hubo inserción, el campo `id_prediccion` de la respuesta es `null`; si hubo inserción, contiene el `id` devuelto por `RETURNING id`.
- **Tests herméticos**: `backend/tests/conftest.py` fija `DATABASE_URL=postgresql+psycopg2://test:test@127.0.0.1:1/testdb`, un puerto inválido que garantiza que los tests ejercitan la ruta degradada y no tocan ninguna base real.

### Consecuencias

- (+) La API es operable en demo, CI y desarrollo sin PostGIS.
- (+) Un test clave (`id_prediccion is None` con respuesta 200) fija la degradación como comportamiento, no como accidente.
- (−) Se pierde el histórico de predicciones cuando la DB está caída (no hay cola ni reintento): aceptable para un prototipo, debe resolverse antes de un entorno productivo (outbox/colas).
- (−) El log es la única señal del fallo: en producción habría que añadir un indicador de salud de DB y alertas.

---

## 5. ADR-003 — Persistencia geoespacial con PostGIS y SRID 4326

### Contexto

Las predicciones tienen una ubicación asociada y se espera consultarlas geográficamente (mapa, vecinos, zonas calientes). Una tabla con `lat`/`lon` sueltas no permite consultas espaciales eficientes.

### Decisión

- Imagen `postgis/postgis:16-3.4`; la extensión se habilita en `database/init/init.sql` (`CREATE EXTENSION IF NOT EXISTS postgis`), ejecutado automáticamente por el entrypoint de la imagen en el primer arranque (volumen `pgdata`).
- Tabla `prediction`:

  | Columna | Tipo | Notas |
  |---------|------|-------|
  | `id` | `SERIAL` PK | devuelto como `id_prediccion` |
  | `temperatura_c`, `humedad`, `viento` | `DOUBLE PRECISION` | condiciones de entrada |
  | `riesgo` | `TEXT` | `CHECK (riesgo IN ('Bajo','Medio','Alto'))` |
  | `score` | `DOUBLE PRECISION` | 0–100 |
  | `geom` | `geometry(Point, 4326)` | punto WGS84 |
  | `created_at` | `TIMESTAMPTZ` | default `now()` |

- La API inserta con `ST_SetSRID(ST_MakePoint(:longitud, :latitud), 4326)` — **orden (lon, lat)** propio de PostGIS, y SRID **4326 (WGS84)**, el mismo que usan el frontend (Leaflet), la API de entrada y los tiles de OpenStreetMap: cero transformaciones en el camino crítico.
- Índices: **GIST** sobre `geom` (`idx_prediction_geom`) para consultas espaciales y **B-tree** sobre `created_at DESC` (`idx_prediction_created_at`) para el histórico reciente.

### Consecuencias

- (+) Listo para consultas tipo `ST_DWithin`/`ST_Buffer`, mapas de calor y capas SIG sin migraciones.
- (+) Consistencia total de CRS entre frontend, API y base.
- (−) WGS84 (grados) no es apropiado para cálculos métricos finos: hoy la distancia se aproxima en el backend con proyección equirrectangular; para precisión regional usar el tipo `geography` de PostGIS o `ST_Distance`.
- (−) `SERIAL` y sin `updated_at`/borrado lógico: ampliar si el histórico se vuelve mutable.

---

## 6. ADR-004 — API Key (`X-API-Key`) en lugar de JWT

### Contexto

`POST /predict` es el único endpoint con valor protegible. Se valoraron: sin auth, API Key estática y JWT/OAuth2.

### Decisión

- API Key simple en el header **`X-API-Key`**, validada en `require_api_key()` con `secrets.compare_digest` (comparación en tiempo constante, mitiga *timing attacks*). La comparación se hace **sobre bytes UTF-8**: ver **ADR-009**.
- Valores: `401` si falta el header (con `WWW-Authenticate: ApiKey`) y `403` si no coincide. La clave viene de la variable de entorno `API_KEY` (default de desarrollo `dev-api-key-123`, propagada por el compose).
- Sólo se protege `/predict`. `/health`, `/metrics` y el frontend estático quedan abiertos: son sondeo/observabilidad sin datos sensibles (el `/metrics` expone agregados, no payloads).
- El frontend la maneja en un bloque "Configuración avanzada" precargada con el default de desarrollo.

### Por qué no JWT (todavía)

JWT/OAuth2 aportan con usuarios/roles, expiración, scopes y emisión propia. Con un único consumidor y un prototipo local, añadiríamos rotación de secretos, endpoints de token y complejidad de cliente sin beneficiario. La decisión queda **abierta a revisión** al aparecer múltiples consumidores, roles (consulta vs. administración) o multi-tenant.

### Consecuencias

- (+) Superficie de ataque e implementación mínimas; trivial de probar (los tests cubren 401/403/403 con key no-ASCII/200).
- (+) Fácil de rotar: cambiar `API_KEY` en `.env` y reiniciar.
- (−) Clave estática compartida, sin revocación por usuario ni auditoría individual: **no apta para producción** sin añadir TLS obligatorio, vault de secretos y, probablemente, JWT.

---

## 7. ADR-005 — Métricas Prometheus instrumentadas en middleware

### Contexto

No hay observabilidad sin métricas. Se quiere saber latencia, tráfico y errores de la API sin instalar un agente externo, y que el dashboard de Grafana muestre algo útil apenas se levanta el stack.

### Decisión

- `prometheus_client` dentro del proceso, expuesto en `GET /metrics` (sin auth).
- Middleware `@app.middleware("http")` que mide cada petición:
  - `http_request_duration_seconds` (**Histogram**, buckets 5 ms … 10 s) con labels `method`, `endpoint`.
  - `http_requests_total` (**Counter**) con labels `method`, `endpoint`, `status`.
- **Control de cardinalidad**: el label `endpoint` se deriva de la ruta emparejada, **nunca** del path arbitrario pedido por el cliente. La derivación explícita (rutas de API / rutas Starlette puras / `static` / `unmatched`) está documentada en el **ADR-010**. Sin esto, un atacante o un crawler con URLs aleatorias dispararía el número de series.
- `predict_requests_total{riesgo}` (**Counter**) añadido en `/predict`, con el nivel de riesgo como label: permite gráficos de distribución de riesgo (Bajo/Medio/Alto) por tiempo.
- Prometheus scrapea `api:8000/metrics` cada 10 s (`monitoring/prometheus/prometheus.yml`, job `api`, label `servicio: api-incendios`) y Grafana nace con un dashboard de tres paneles: **latencia promedio** (`sum(rate(http_request_duration_seconds_sum[5m])) / sum(rate(http_request_duration_seconds_count[5m]))`), **RPS** (`sum(rate(http_requests_total[1m]))`) y **errores 5xx** (`sum(rate(http_requests_total{status=~"5.."}[1m]))`).

### Consecuencias

- (+) Latencia, tráfico, errores y distribución de riesgo desde el primer arranque; los tests verifican la presencia de `http_requests_total`, `http_request_duration_seconds_bucket` y `predict_requests_total`.
- (+) Sin dependencias externas en el proceso (no se usa client push ni StatsD).
- (−) Métricas por proceso (no multiworker): si se pasa a varios workers o réplicas, hace falta un agregador o `multiprocess` mode.
- (−) No hay métricas de negocio de la base (duración del INSERT, fallos de conexión): candidato a añadir como contadores/histogramas propios.

---

## 8. ADR-006 — Frontend SPA autocontenido servido por FastAPI

### Contexto

Se quería una demo visual (mapa de Valparaíso + formulario) sin añadir un servidor estático ni una cadena de build (Node, bundlers) al stack.

### Decisión

- Un único `frontend/index.html` con HTML/CSS/JS embebidos y **Leaflet 1.9.4 por CDN** (con atributos `integrity`/`crossorigin`). Sin npm, sin build.
- FastAPI lo sirve mediante `StaticFiles` montado en `/` con `html=True`. La ruta del directorio se resuelve con `_resolver_directorio_frontend()`: `STATIC_DIR` (fijado por el compose a `/app/frontend`) → `../frontend` respecto de `backend/` → `frontend/` relativo al backend. Esto hace que funcione igual en local y en contenedor (el compose monta `./frontend:/app/frontend:ro`, en vivo, sin recompilar la imagen).
- El cliente tolera fallos: si Leaflet no carga (sin internet al CDN) muestra un mensaje y el formulario sigue operativo; normaliza los nombres de campos de respuesta (`nivel_riesgo|riesgo|risk_level`, `score|score_riesgo|riesgo_score|probabilidad`) y escapa el HTML antes de inyectarlo (mitiga XSS).

### Consecuencias

- (+) Cero toolchain; el frontend se edita con cualquier editor y se ve recargando la página.
- (+) Un solo contenedor sirve API + UI: menos piezas, menos puertos.
- (−) El CDN es un punto externo de fallo (mitigado parcialmente por el manejo de errores) y el HTML es una pieza única de ~800 líneas: convendría extraer componentes al crecer.

---

## 9. ADR-007 — Puerto del API parametrizado (`API_HOST_PORT`)

### Contexto

El 8000 es un puerto popular; en máquinas con otros servicios (p. ej. otro contenedor de IA) el `docker compose up` fallaba por colisión.

### Decisión

En `docker-compose.yml`: `- "${API_HOST_PORT:-8000}:8000"`. El contenedor siempre escucha en el 8000 interno (red `incendios-net`, de la que dependen Prometheus y Grafana); sólo cambia el puerto publicado al host. `.env.example` no incluye la variable (por defecto 8000); se agrega a `.env` a mano cuando hace falta (por ejemplo `API_HOST_PORT=18000`).

### Consecuencias

- (+) Sin colisiones y sin tocar la config interna; la documentación de URLs es idéntica salvo el puerto.
- (−) Es fácil olvidar cambiar la URL en el navegador/curl al fijar otro puerto; el README lo advierte explícitamente.

---

## 10. ADR-008 — Stack tecnológico base

| Capa | Elección | Motivo |
|------|----------|--------|
| API | **FastAPI 0.115 + Uvicorn 0.34** (Python 3.11-slim) | Async nativo, validación declarativa con Pydantic 2, OpenAPI automática (`/docs`) |
| Persistencia | **PostgreSQL 16 + PostGIS 3.4**, SQLAlchemy 2.0 + psycopg2 | Geodatos de primera clase; el ORM/SQL crítico es un `INSERT` parametrizado simple |
| Frontend | HTML/JS + Leaflet (CDN) | Ver ADR-006 |
| Métricas | **Prometheus 2.54** + **Grafana 11.4** | Estándar de facto, provisioning como código |
| Orquestación | **Docker Compose v2**, red bridge `incendios-net`, healthchecks y `depends_on` con `condition: service_healthy` | Arranque ordenado: db → api → prometheus → grafana |

---

## 11. ADR-009 — Comparación de la API Key en bytes UTF-8 y validación en el arranque

> Registro Lemoria: `8a940e7b-21c9-4212-b910-df90dab5325b` ("Comparación de API Key en bytes (fix no-ASCII)"), `e536aa74-3c17-4807-b615-c8fd26891b48` ("B1: comparación de API key en bytes UTF-8") y `49ed12d5-19b2-4b99-a653-5cbf49e426cb` ("M4: API_KEY vacía → valor por defecto y fallo claro en el arranque").

### Contexto

La validación de la API key del ADR-004 usaba `secrets.compare_digest(clave_recibida, clave_esperada)` sobre **cadenas**. La documentación de la API afirma que una clave incorrecta devuelve **403**; la realidad era que `secrets.compare_digest` **sólo admite cadenas ASCII** y lanza `TypeError: comparing strings with non-ASCII characters is not supported` ante una `X-API-Key` con caracteres no-ASCII. Como el `TypeError` se produce **dentro de la dependencia** (no en un manejador de errores del endpoint), FastAPI lo convertía en un **HTTP 500**: una entrada de cliente puede provocar un 5xx, y el contrato "401 sin header / 403 key inválida / nunca 5xx" quedaba roto.

En paralelo, la clave esperada se obtenía con `os.getenv("API_KEY", DEFAULT_API_KEY)`: una variable presente pero **vacía** (`API_KEY=`) devolvía `""`, y `compare_digest("", "")` es `True`, de modo que un header `X-API-Key` **vacío** pasaba la autenticación.

### Decisión

- La comparación se hace sobre los **bytes UTF-8** de la clave recibida y la esperada, en la función `_coincide_api_key()`:

  ```python
  recibido_bytes = recibida.encode("utf-8")
  esperado_bytes = esperada.encode("utf-8")
  return secrets.compare_digest(recibido_bytes, esperado_bytes)
  ```

  Se conserva la **comparación en tiempo constante** (misma propiedad anti-*timing*) y se acepta cualquier entrada válida sin excepción: una clave no-ASCII devuelve **403** y nunca 5xx.
- La clave efectiva es `os.getenv("API_KEY") or DEFAULT_API_KEY` (`_api_key_esperada()`): una variable presente pero vacía cae al valor por defecto en lugar de acceptarse como clave vacía.
- El `lifespan` llama a `_validar_api_key_de_arranque()`: si la clave efectiva quedara **en blanco**, el arranque aborta con un `RuntimeError` explícito ("API_KEY está vacía: define la variable de entorno API_KEY con un valor no vacío antes de arrancar el servicio"). Es imposible levantar el servicio con `/predict` desprotegido.
- Test de regresión `test_predict_with_non_ascii_api_key_returns_403_and_never_5xx` (`backend/tests/test_predict.py`): envía el byte latin-1 `0xE9` en crudo (`b"cl\xe9-mala"`) y comprueba `403` y `< 500`.

### Consecuencias

- (+) El contrato de la API es el documentado: **401** sin header, **403** con clave inválida (incluida la no-ASCII), **nunca 5xx** por contenido del header.
- (+) La comparación sigue siendo en tiempo constante: no se pierde la mitigación de *timing attacks*.
- (+) El error de configuración (clave vacía) se detecta **en el arranque**, no en la primera petición.
- (−) Comparar bytes UTF-8 es algo menos idiomático que comparar `str`: la seguridad depende de no volver a pasar cadenas a `compare_digest` en el futuro (el test de regresión lo fija).

---

## 12. ADR-010 — Derivación explícita y acotada del label `endpoint` en las métricas

> Registro Lemoria: `e413a59f-0eeb-4235-abcc-ba04c1137785` ("M1: label endpoint de las métricas derivado de forma explícita y acotada") y `aa33597b-62a1-43e0-9535-d68933bfa0fa` ("Derivación del label endpoint en métricas").

### Contexto

El ADR-005 define el label `endpoint` como control de cardinalidad: el path pedido por el cliente **nunca** puede ser el valor del label. La primera implementación usaba `request.scope["route"].path`, pero **sólo FastAPI** fija `scope["route"]` (en `fastapi.routing.APIRoute`): las rutas Starlette puras (`/docs`, `/redoc`, `/openapi.json`) y el montaje de `StaticFiles` (el frontend) **colapsaban todas en `unmatched`**, contaminando los paneles de RPS y latencia de Grafana con el tráfico de assets y de la documentación OpenAPI.

### Decisión

Se añade `_derive_endpoint(scope, status_code, app)`, que decide el valor del label **explícitamente**:

| Situación | Valor de `endpoint` |
|-----------|---------------------|
| Ruta de la API (`APIRoute`) | su path: `/predict`, `/health`, `/metrics` |
| Ruta Starlette pura (`/docs`, `/redoc`, `/openapi.json`) | su path, resuelto con un índice `endpoint → path` cacheado por aplicación (`_indice_rutas_starlette`) |
| Montaje de ficheros estáticos (`StaticFiles`) | `static` (y `unmatched` si la respuesta es ≥ 400, p. ej. un fichero inexistente) |
| Resto (404/405 reales) | `unmatched` |

- **Nunca** se usa `request.url.path` como valor de respaldo: el path lo elige el cliente y generaría un número de series ilimitado (ataque de cardinalidad). El número de valores posibles es finito y conocido.
- El *lifespan*/arranque **no** se instrumenta: el middleware sólo atiende peticiones `http`, de modo que no se crea ninguna serie al arrancar.
- Las queries del dashboard siguen agregando sobre todos los labels (`sum(...)` sin `by`), por lo que son indiferentes a esta decisión; el desglose por endpoint se obtiene con `sum by (endpoint) (...)`.

Salida real de `GET /metrics` (contadores acumulados del proceso):

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

### Consecuencias

- (+) La cardinalidad queda **acotada por diseño** y, a la vez, las series son interpretables: se distingue el tráfico de la API del de assets (`static`) y de los 404/405 (`unmatched`).
- (+) Los paneles de Grafana dejan de mezclar la latencia del frontend con la de la API.
- (−) `static` y `unmatched` son "sacos" que ocultan qué fichero o qué ruta falló: para ese nivel de detalle haría falta un log estructurado (no una métrica con cardinalidad variable).

---

## 13. ADR-011 — Dependencias de desarrollo separadas (`backend/requirements-dev.txt`)

> Registro Lemoria: `9e9e6da2-f415-4efe-a9ea-41e94c2925dc` ("Dependencias de desarrollo separadas (requirements-dev.txt)") y `666fc278-e956-4eb0-aa22-a43862e6cee4` ("backend/requirements-dev.txt separa dependencias de producción y de desarrollo").

### Contexto

El método de tests documentado en el README (`pip install -r backend/requirements.txt pytest`) **no era reproducible**: `requirements.txt` no incluye `httpx`, que `starlette.testclient` **exige en tiempo de importación** (`fastapi.testclient` la usa en `backend/tests/conftest.py`). En un venv limpio, pytest aborta **antes de colectar ningún test**:

```text
RuntimeError: The starlette.testclient module requires the httpx package to be installed.
```

Es un fallo silencioso desde el punto de vista de quien sigue el README: el comando "parece" el correcto y sólo falla al ejecutarlo.

### Decisión

- `backend/requirements.txt` queda **intacto**: es el fichero de producción y la imagen Docker sólo instala este (`RUN pip install --no-cache-dir -r requirements.txt`).
- Se crea `backend/requirements-dev.txt` con:

  ```text
  -r requirements.txt
  pytest
  httpx
  ```

  Es **declarativo** y **reproducible**: una sola línea instala el runtime, el runner y la dependencia implícita del `TestClient`.
- El README documenta el método nuevo (`pip install -r backend/requirements-dev.txt`) y la alternativa en contenedor (`docker compose exec api pip install -r requirements-dev.txt && docker compose exec api python -m pytest tests -q`), ambas verificadas (`12 passed, 1 warning`).
- `backend/.dockerignore` **no** excluye `tests/`: el README documenta correr los tests dentro del contenedor.

### Consecuencias

- (+) El método documentado funciona en un venv limpio y en el contenedor: la documentación y el código coinciden.
- (+) La imagen de producción no carga el stack de test (ni `pytest` ni `httpx`).
- (−) `requirements-dev.txt` no fija versiones (a diferencia de `requirements.txt`): es coherente con el carácter de "extra de desarrollo" del fichero, pero en un entorno con varias personas conviene fijar versiones para que el conjunto de tests sea reproducible en el tiempo.

---

## 14. ADR-012 — Timeouts de la base de datos y persistencia en threadpool

> Registro Lemoria: `78597414-25a3-46f0-87d8-2cb26decf51b` ("M5a: connect_timeout y statement_timeout en la URL de PostgreSQL") y `ebd9a789-c970-4687-9a7d-ab10e59e2958` ("M5b: persistencia de predicciones en run_in_threadpool").

### Contexto

Dos riesgos del acceso a PostgreSQL en un servicio async:

1. **Arranque colgado**: si el host de la base de datos *filtra* el tráfico en lugar de rechazarlo, el `SELECT 1` de comprobación del `lifespan` se queda bloqueado indefinidamente y el contenedor nunca llega a estar `healthy` (el healthcheck falla por *timeout*, no por error).
2. **Event loop bloqueado**: `psycopg2` es **síncrono**. Llamarlo directamente desde el handler `async` de `/predict` bloquea el event loop durante el `INSERT`, serializando todas las peticiones concurrentes del proceso.

### Decisión

- **Timeouts en la URL**: `_database_url()` normaliza la URL con `sqlalchemy.engine.make_url` y añade, con `setdefault` (lo que venga explícito en `DATABASE_URL` se respeta), `connect_timeout=5` y `options=-c statement_timeout=10000`. URL efectiva verificada en el contenedor:

  ```text
  postgresql+psycopg2://incendios:incendios123@db:5432/prevencion_incendios?connect_timeout=5&options=-c+statement_timeout%3D10000
  ```

  `connect_timeout` acota el **establecimiento** de la conexión; `statement_timeout` acota cualquier **consulta bloqueada en el servidor**.
- **El arranque falla rápido**: contra un host que filtra el tráfico (`10.255.255.1`) la comprobación de arranque falla en **≈5 s** (medido: 5,05 s) en lugar de colgarse. El servicio **no aborta**: registra un `WARNING` y continúa en modo degradado (ADR-002).
- **Persistencia en threadpool**: `predict()` llama a `_persistir_prediccion()` mediante `starlette.concurrency.run_in_threadpool`, de modo que el I/O bloqueante no detiene el event loop. La comprobación de BD del `lifespan` se ejecuta en el threadpool por el mismo motivo.
- **Degradación intacta**: `_persistir_prediccion()` sigue capturando sus excepciones y devolviendo `None`, así que si la BD falla la respuesta se sirve con `id_prediccion = null` y sin excepción (el `INSERT` va dentro de la transacción del context manager, con `commit()` explícito).
- La creación del engine se protege con un `threading.Lock` (doble comprobación), porque ahora puede ocurrir desde varios hilos del threadpool.

### Consecuencias

- (+) El arranque del contenedor es **determinista**: nunca se queda colgado esperando a la base de datos.
- (+) El servicio atiende peticiones concurrentes mientras la persistencia espera a la BD.
- (+) El contrato de degradación del ADR-002 no cambia: HTTP 200 con `id_prediccion: null` cuando la BD no está disponible.
- (−) `statement_timeout` y `connect_timeout` son valores por defecto **del prototipo**: en producción deben ajustarse al presupuesto de latencia y monitorizarse (`pg_stat_activity`, alertas por `pool` saturado).

---

## 15. ADR-013 — `.dockerignore` en la raíz del contexto de build

> Registro Lemoria: `ba79adb3-b619-4a79-be05-174cf94785bb` ("backend/.dockerignore efectivo: el .dockerignore debe vivir en la raíz del contexto de build").

### Contexto

Docker busca el `.dockerignore` **en la raíz del contexto de build**. `docker-compose.yml` declara `build: ./backend` para el servicio `api`, luego **el contexto de build es `backend/`**: el `.dockerignore` de la raíz del repositorio **no se aplica** a esa imagen (era inerte para el servicio `api`, aunque siga siendo útil para un `docker build .` con contexto raíz). Sin un `.dockerignore` en `backend/`, la imagen se construía enviando las cachés de Python y otros artefactos locales.

### Decisión

- Se crea **`backend/.dockerignore`** (el que sí aplica al servicio `api`), que excluye `__pycache__/`, `*.pyc`, `*.pyo`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.venv/`, `venv/`, `.env`, `.env.*` (con `!.env.example`), `.git/`, `.gitignore`, `.DS_Store`, `.vscode/`, `.idea/`.
- `tests/` **no** se excluye a propósito: el README documenta ejecutar los tests dentro del contenedor.
- El `.dockerignore` de la raíz del repositorio se conserva (sigue siendo válido para builds con contexto raíz) y su cabecera replica la razón de su inercia para el servicio `api`.

### Consecuencias

- (+) El contexto de build del API es mínimo y reproducible: no se filtran cachés ni secretos al daemon de Docker.
- (+) La imagen es idéntica entre máquinas (el `.pytest_cache/` local ya no la contaminaba).
- (−) Hay **dos** ficheros `.dockerignore` en el repo: hay que recordar que el que cuenta es el de la raíz del contexto de build. Ambos lo documentan en su cabecera para evitar la confusión.


---

## 16. Observabilidad: qué mirar

- **API**: `GET http://localhost:8000/metrics` → `http_requests_total`, `http_request_duration_seconds_bucket`, `predict_requests_total{riesgo="Bajo|Medio|Alto"}`. El label `endpoint` distingue `/predict`, `/health`, `/metrics`, `/docs`, `/redoc`, `/openapi.json`, `static` y `unmatched` (ver **ADR-010**).
- **Prometheus** (http://localhost:9090 → *Status → Targets*): job `api` debe estar `up` scrapeando `api:8000/metrics` cada 10 s.
- **Grafana** (http://localhost:3000, `admin`/`admin`): dashboard **"Prevención Incendios Forestales - Valparaíso"** (carpeta *Incendios Forestales*), auto-provisionado desde `monitoring/grafana/dashboards/incendios.json`. Paneles: latencia promedio HTTP, RPS y errores 5xx.
- El provisioning de Grafana es **declarativo**: datasource (`editable: false`) y dashboard se reimportan en cada arranque; `allowUiUpdates: true` permite ajustes temporales desde la UI que se sobreescribirán con el siguiente `up`.

---

## 17. Hoja de ruta: de la heurística a un modelo CNN/LSTM real

Objetivo: sustituir `_score_riesgo()` por un modelo entrenado **manteniendo intacto el contrato de la API** (`PrediccionRequest`/`PrediccionResponse`) y la degradación del ADR-002.

### 12.1 Fuentes de datos (Chile)

| Fuente | Aporte | Uso en el modelo |
|--------|--------|------------------|
| **Dirección Meteorológica de Chile (DMC)** | Series horarias y diarias de temperatura, humedad, viento y radiación en estaciones de la región | Secuencia temporal de entrada de la **LSTM** |
| **CONAF** | Historial de incendios (área quemada, fechas, comuna), riesgo de incendios y focos de calor | **Etiquetas** y validación de eventos |
| **INIA** | Cobertura vegetal, uso y cambio de suelo, características agroclimáticas | Capas estáticas de contexto |
| **Imágenes satelitales** (Sentinel-2 / Landsat, NDVI/NDWI, temperatura de brillo) | Mapas de vegetación y estrés hídrico | **CNN** (canal(es) de imagen por fecha) |
| **Cartografía base** (catastro DGA, curvas de nivel, interfaz urbano-forestal) | Predictores geográficos estáticos | Capas CNN + features de proximidad |
| **Índices de clima de incendio** (FWI/FFDI computados a partir de la meteorología) | Referencia de dominio | Baseline comparativo |

> Los datos oficiales se obtienen de los portales institucionales de CONAF (conaf.cl), SENAPRED (senapred.cl), la Dirección Meteorológica de Chile (meteorologia.cl) e INIA (inia.cl); revisar licencias y condiciones de uso antes de redistribuir.

### 12.2 Ingeniería de características

- **Rama LSTM**: secuencia de N días (p. ej. 7–14) × variables meteorológicas (T máx/mín/media, HR, viento, lluvia acumulada, FWI). Normalización por estándar con estadísticas de entrenamiento; imputación explícita de faltantes.
- **Rama CNN**: recorte geoespacial alrededor del punto (p. ej. 64×64 píxeles con SRID 4326 → reproyección a una métrica equidistante para evitar distorsión) con canales NDVI/NDWI/Tbrillo + capa estática de cobertura del suelo.
- **Fusión**: concatenación de la salida de la CNN con la secuencia LSTM (y features tabulares estáticos) → capas densas → salida softmax de 3 clases (Bajo/Medio/Alto) o regresión de probabilidad calibrada.
- **Sustitución en el código**: nuevo paquete `backend/model/` (carga del artefacto ONNX/TorchScript en el `lifespan`), guardia de versión y `modelo: "cnn-lstm-v1"` en la respuesta. La heurística actual queda como **fallback** si el artefacto no está cargado (misma filosofía de degradación del ADR-002).

### 12.3 Entrenamiento y validación

- **Split temporal** (no aleatorio): entrenar con años anteriores y validar con el último año/campaña estival, evitando fuga de información.
- **Clases desbalanceadas** (los incendios grandes son pocos): usar *class weighting*, focal loss o sobremuestreo controlado de eventos.
- **Métricas**: F1 macro y PR-AUC por clase, matriz de confusión por comuna/estación, Brier/ECE si se produce probabilidad calibrada, y **backtesting** contra incendios históricos (¿señaló alto riesgo los días previos?).
- **Reproducibilidad**: semillas fijas, datasets versionados (DVC o bucket), experimentos registrados (MLflow o similar), ficha del dataset y de la licencia.

### 12.4 Integración y despliegue

1. **Modo sombra (shadow mode)**: el modelo real corre junto a la heurística sin responder al usuario; se comparan distribuciones de score/nivel y se detectan regresiones.
2. **Serving**: exportar a **ONNX** o servir con TorchServe/Triton; en el API, inferencia bajo umbral de latencia (p. ej. < 200 ms) con *batching* opcional.
3. **Contrato estable**: se conservan `nivel_riesgo`, `score`, `factores[]` (ahora SHAP/permutación o atenciones en lugar de las reglas fijas) e `id_prediccion`; se documenta el cambio en el campo `modelo`.
4. **Operación**: GPU para entrenamiento (no imprescindible para inferencia ONNX ligera), recalibración estacional, detección de *data drift* en las entradas (PSI/KS) y alerta cuando la distribución se aleje del entrenamiento.

---

## 18. Limitaciones y riesgos del prototipo

- **El riesgo mostrado no es una alerta oficial** ni un pronóstico: es una demostración técnica.
- La heurística ignora sequía acumulada, combustible vivo/muerto, pendiente, infraestructura de respuesta y comportamiento humano (la mayoría de los incendios de Valparaíso son de origen antrópico).
- El factor de "vegetación" es determinista-sintético; no representa cobertura real.
- Sin modelo calibrado, el umbral 70/40 es arbitrario (sirve para demostrar la clasificación, no para priorizar recursos reales).
- Seguridad mínima (API Key estática, sin TLS): nunca desplegar expuesto a internet en su estado actual.
- Un solo nodo Compose: sin HA, sin backups automáticos del volumen `pgdata`, sin retención configurable de Prometheus más allá de su default.

---

## 19. Convenciones del repositorio

- Los nombres de campos de la API están en **español** (`temperatura_c`, `humedad_relativa`, `nivel_riesgo`, `id_prediccion`), coherentes entre Pydantic, SQL y frontend.
- Los artefactos sensibles/locales (`.env`, `__pycache__`, `.pytest_cache`, `.venv`) están en `.gitignore`; la plantilla versionada es `.env.example`.
- La configuración de monitoreo es **código**: cualquier cambio de panel o scrape se hace en `monitoring/` y se versiona con el resto.
- Las decisiones arquitectónicas de este documento se registran además en Lemoria (tabla de la sección 2) para trazabilidad con el flujo SDD.
