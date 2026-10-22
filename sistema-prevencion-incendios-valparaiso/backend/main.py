"""API de apoyo a la prevención y clasificación de riesgo de incendios forestales.

Dominio: Región de Valparaíso (Chile).

Expone:
  * GET  /health   -> estado del servicio (sin autenticación)
  * POST /predict  -> clasificación de riesgo (requiere header X-API-Key)
  * GET  /metrics  -> métricas Prometheus (sin autenticación)
  * /              -> frontend estático (Leaflet) servido desde ../frontend

La puntuación de riesgo se obtiene con una HEURÍSTICA DETERMINISTA que SIMULA
la salida de un modelo CNN/LSTM (ver módulo `_score_riesgo`). No existe
inferencia de red neuronal en esta versión del prototipo.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import secrets
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

import psycopg2
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import Response
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool
from starlette.routing import Route as StarletteRoute
from starlette.types import Scope

# ---------------------------------------------------------------------------
# Configuración y logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("incendios.api")

DEFAULT_API_KEY = "dev-api-key-123"
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg2://incendios:incendios123@localhost:5432/prevencion_incendios"
)

# Timeouts de base de datos: un host que filtra (no rechaza) el tráfico dejaría
# el `SELECT 1` de arranque colgado indefinidamente sin `connect_timeout`.
DB_CONNECT_TIMEOUT_SECONDS = 5
DB_STATEMENT_TIMEOUT_MS = 10_000

# Labels de métricas con cardinalidad acotada (nunca el path pedido por el
# cliente, que es controlado por el usuario).
ENDPOINT_STATIC = "static"
ENDPOINT_UNMATCHED = "unmatched"

# Límites geográficos de la zona de Valparaíso (validados además por Pydantic).
# Nota: la ciudad de Valparaíso está en -33.05, por lo que el límite sur se
# fija en -33.5 para incluir el área urbana (Valparaíso, Viña, Quilpué).
LAT_MIN, LAT_MAX = -33.5, -30.0
LON_MIN, LON_MAX = -72.0, -70.0

# Puntos de referencia para el factor de proximidad/vegetación simulado.
_REF_POINTS: tuple[tuple[str, float, float, float], ...] = (
    ("Valparaíso - cerros", -33.0472, -71.6127, 1.00),
    ("Viña del Mar", -33.0153, -71.5500, 0.85),
    ("Quilpué - zona agrícola", -33.0400, -71.4400, 0.70),
)
# Distancia (km) a la cual se considera que el punto queda fuera del área
# de Valparaíso y se aplica la penalización.
AREA_KM = 60.0

# ---------------------------------------------------------------------------
# Métricas Prometheus
# ---------------------------------------------------------------------------

HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "Duración (segundos) de cada petición HTTP atendida por la API.",
    labelnames=["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total de peticiones HTTP.",
    labelnames=["method", "endpoint", "status"],
)
PREDICT_REQUESTS = Counter(
    "predict_requests_total",
    "Total de predicciones de riesgo generadas, por nivel de riesgo.",
    labelnames=["riesgo"],
)

# ---------------------------------------------------------------------------
# Modelos de datos (Pydantic)
# ---------------------------------------------------------------------------


class PrediccionRequest(BaseModel):
    """Entrada de /predict: condiciones meteorológicas y ubicación."""

    temperatura_c: float = Field(
        ..., ge=-10.0, le=55.0, description="Temperatura en grados Celsius."
    )
    humedad_relativa: float = Field(
        ..., ge=0.0, le=100.0, description="Humedad relativa en porcentaje."
    )
    velocidad_viento_kmh: float = Field(
        ..., ge=0.0, le=200.0, description="Velocidad del viento en km/h."
    )
    latitud: float = Field(
        ...,
        ge=LAT_MIN,
        le=LAT_MAX,
        description="Latitud en grados decimales (zona de Valparaíso, -33.5 a -30).",
    )
    longitud: float = Field(
        ..., ge=LON_MIN, le=LON_MAX, description="Longitud (zona de Valparaíso)."
    )


class FactorRiesgo(BaseModel):
    """Factor explicativo que aporta puntos al score final."""

    nombre: str
    valor: float
    aporte_puntos: float
    descripcion: str


class PrediccionResponse(BaseModel):
    """Salida de /predict: clasificación de riesgo explicada."""

    nivel_riesgo: Literal["Bajo", "Medio", "Alto"]
    score: float = Field(..., description="Score de riesgo 0-100.")
    factores: list[FactorRiesgo]
    timestamp: datetime
    latitud: float
    longitud: float
    modelo: str
    id_prediccion: int | None = None


# ---------------------------------------------------------------------------
# Acceso a base de datos (PostgreSQL + PostGIS)
# ---------------------------------------------------------------------------

_engine: Engine | None = None
_engine_lock = threading.Lock()


def _database_url_base() -> str:
    """URL de conexión tal y como la define el entorno (sin timeouts)."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    user = os.getenv("POSTGRES_USER", "incendios")
    password = os.getenv("POSTGRES_PASSWORD", "incendios123")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "prevencion_incendios")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def _database_url() -> str:
    """URL de conexión con timeouts de conexión y de consulta.

    `connect_timeout` acota el establecimiento de la conexión: si el host de la
    base de datos filtra el tráfico en lugar de rechazarlo, el `SELECT 1` de
    arranque se quedaría colgado indefinidamente. `options=-c
    statement_timeout=...` acota además cualquier consulta bloqueada en el
    servidor. Los valores ya presentes en `DATABASE_URL` se respetan.
    """
    url = _database_url_base()
    try:
        parsed = make_url(url)
    except Exception as exc:  # noqa: BLE001 - create_engine reportará el error
        logger.warning("DATABASE_URL no se pudo parsear (%s); se usa tal cual.", exc)
        return url

    query = dict(parsed.query)
    query.setdefault("connect_timeout", str(DB_CONNECT_TIMEOUT_SECONDS))
    query.setdefault("options", f"-c statement_timeout={DB_STATEMENT_TIMEOUT_MS}")
    return parsed.update_query_dict(query).render_as_string(hide_password=False)


def _get_engine() -> Engine | None:
    """Devuelve (creando una sola vez) el engine de SQLAlchemy o None si falla."""
    global _engine
    if _engine is not None:
        return _engine
    # Doble comprobación con lock: la persistencia se ejecuta en un threadpool
    # y varias peticiones concurrentes podrían crear motores duplicados.
    with _engine_lock:
        if _engine is not None:
            return _engine
        try:
            _engine = create_engine(_database_url(), pool_pre_ping=True, pool_size=5)
            return _engine
        except Exception as exc:  # noqa: BLE001 - el prototipo no debe caerse por la DB
            logger.warning("No se pudo crear el motor de base de datos: %s", exc)
            return None


def _persistir_prediccion(
    temperatura_c: float,
    humedad_relativa: float,
    velocidad_viento_kmh: float,
    nivel_riesgo: str,
    score: float,
    latitud: float,
    longitud: float,
) -> int | None:
    """Inserta la predicción en la tabla `prediction` con geometría PostGIS.

    Devuelve el id insertado o None. Cualquier fallo de base de datos se
    registra como warning y NO interrumpe la respuesta al cliente.
    """
    engine = _get_engine()
    if engine is None:
        logger.warning("Predicción no persistida: motor de base de datos no disponible.")
        return None

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    """
                    INSERT INTO prediction
                        (temperatura_c, humedad, viento, riesgo, score, geom, created_at)
                    VALUES
                        (:temperatura, :humedad, :viento, :riesgo, :score,
                         ST_SetSRID(ST_MakePoint(:longitud, :latitud), 4326), now())
                    RETURNING id
                    """
                ),
                {
                    "temperatura": temperatura_c,
                    "humedad": humedad_relativa,
                    "viento": velocidad_viento_kmh,
                    "riesgo": nivel_riesgo,
                    "score": score,
                    "latitud": latitud,
                    "longitud": longitud,
                },
            )
            prediction_id: int | None = result.scalar_one_or_none()
            connection.commit()
        logger.info(
            "Predicción persistida id=%s riesgo=%s score=%.1f",
            prediction_id,
            nivel_riesgo,
            score,
        )
        return prediction_id
    except SQLAlchemyError as exc:
        if isinstance(exc.__cause__, psycopg2.Error):
            logger.warning(
                "PostgreSQL no disponible, la predicción se responde sin persistir: %s",
                exc.__cause__,
            )
        else:
            logger.warning("Error de SQLAlchemy al persistir predicción: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 - degradación controlada del prototipo
        logger.warning("Error inesperado al persistir predicción: %s", exc)
        return None


def _comprobar_base_de_datos() -> None:
    """Registra el estado de la conexión a PostgreSQL al arrancar (sin abortar)."""
    engine = _get_engine()
    if engine is None:
        return
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Conexión a PostgreSQL verificada correctamente.")
    except Exception as exc:  # noqa: BLE001 - solo informativo
        logger.warning(
            "PostgreSQL no responde al inicio (%s). El servicio continúa en modo degradado.",
            exc,
        )


# ---------------------------------------------------------------------------
# Autenticación por API Key
# ---------------------------------------------------------------------------


def _api_key_esperada() -> str:
    """API key esperada: variable de entorno o valor por defecto.

    Se usa `os.getenv("API_KEY") or DEFAULT_API_KEY` (y no `getenv(..., default)`)
    porque una variable de entorno presente pero vacía (`API_KEY=`) devolvía ""
    y `compare_digest("", "")` daba True: la API aceptaba un header vacío.
    """
    return os.getenv("API_KEY") or DEFAULT_API_KEY


def _validar_api_key_de_arranque() -> str:
    """Valida la configuración de la API key al arrancar.

    Si la clave efectiva quedara vacía se aborta el arranque con un error
    claro: continuar dejaría el endpoint /predict sin protección real.
    """
    api_key = _api_key_esperada()
    if not api_key.strip():
        raise RuntimeError(
            "API_KEY está vacía: define la variable de entorno API_KEY con un "
            "valor no vacío antes de arrancar el servicio."
        )
    return api_key


def _coincide_api_key(recibida: str, esperada: str) -> bool:
    """Compara dos API keys en tiempo constante usando bytes UTF-8.

    `secrets.compare_digest` sólo admite cadenas ASCII: con una clave no-ASCII
    lanzaba `TypeError` (HTTP 500 ante entrada del cliente). Comparando los
    bytes UTF-8 se mantiene la comparación en tiempo constante y nunca se
    produce una excepción por el contenido del header.
    """
    try:
        recibido_bytes = recibida.encode("utf-8")
        esperado_bytes = esperada.encode("utf-8")
    except UnicodeEncodeError:  # pragma: no cover - defensivo
        return False
    return secrets.compare_digest(recibido_bytes, esperado_bytes)


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Valida el header X-API-Key contra la variable de entorno API_KEY."""
    expected = _api_key_esperada()
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Falta el header X-API-Key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    if not _coincide_api_key(x_api_key, expected):
        raise HTTPException(status_code=403, detail="API key inválida.")


# ---------------------------------------------------------------------------
# Heurística de riesgo (SIMULA un modelo CNN/LSTM)
# ---------------------------------------------------------------------------


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _distancia_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia aproximada en km (proyección equirrectangular)."""
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon1 - lon2) * 111.32 * math.cos(mean_lat)
    dy = (lat1 - lat2) * 110.57
    return math.hypot(dx, dy)


def _vegetacion_simulada(latitud: float, longitud: float) -> float:
    """Índice de cobertura vegetal simulado, determinista a partir del punto."""
    digest = hashlib.sha256(f"{latitud:.4f},{longitud:.4f}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF


def _factor_proximidad_vegetacion(latitud: float, longitud: float) -> tuple[float, str]:
    """Factor 0-1 que combina proximidad a focos urbanos/vegetación de la zona.

    Combina:
      * proximidad a puntos de referencia de Valparaíso (gaussiana),
      * un índice de vegetación simulado derivado de las coordenadas,
      * penalización si el punto cae fuera del área de Valparaíso.
    """
    mejor_proximidad = 0.0
    distancia_min = float("inf")
    for _nombre, ref_lat, ref_lon, peso in _REF_POINTS:
        distancia = _distancia_km(latitud, longitud, ref_lat, ref_lon)
        distancia_min = min(distancia_min, distancia)
        gaussiana = math.exp(-((distancia / 45.0) ** 2))
        mejor_proximidad = max(mejor_proximidad, peso * gaussiana)

    vegetacion = _vegetacion_simulada(latitud, longitud)
    factor = 0.55 * mejor_proximidad + 0.45 * vegetacion

    if distancia_min > AREA_KM:
        factor *= 0.4
        detalle = (
            f"Fuera del área de Valparaíso (distancia mínima {distancia_min:.1f} km): "
            "factor de proximidad/vegetación penalizado 60%."
        )
    else:
        detalle = (
            f"Dentro del área de Valparaíso ({distancia_min:.1f} km del punto de "
            "referencia más cercano); vegetación simulada = "
            f"{vegetacion:.2f}."
        )
    return _clamp(factor, 0.0, 1.0), detalle


def _score_riesgo(request: PrediccionRequest) -> tuple[float, list[FactorRiesgo]]:
    """Calcula el score 0-100 y los factores explicativos.

    SIMULACIÓN: esta función sustituye a la inferencia de un modelo CNN/LSTM
    (imágenes satelitales + series meteorológicas). La lógica es determinista,
    explicable y pondera cuatro bloques que suman máximo 100 puntos.
    """
    temperatura_puntos = _clamp((request.temperatura_c - 15.0) / 30.0, 0.0, 1.0) * 30.0
    humedad_puntos = _clamp((100.0 - request.humedad_relativa) / 100.0, 0.0, 1.0) * 30.0
    viento_puntos = _clamp(request.velocidad_viento_kmh / 80.0, 0.0, 1.0) * 25.0

    proximidad, detalle_proximidad = _factor_proximidad_vegetacion(
        request.latitud, request.longitud
    )
    proximidad_puntos = proximidad * 15.0

    factores = [
        FactorRiesgo(
            nombre="temperatura",
            valor=round(request.temperatura_c, 2),
            aporte_puntos=round(temperatura_puntos, 2),
            descripcion="Aporte térmico: escala creciente entre 15 °C y 45 °C.",
        ),
        FactorRiesgo(
            nombre="humedad_relativa",
            valor=round(request.humedad_relativa, 2),
            aporte_puntos=round(humedad_puntos, 2),
            descripcion="Aporte por sequedad: inverso de la humedad relativa (0-100%).",
        ),
        FactorRiesgo(
            nombre="velocidad_viento",
            valor=round(request.velocidad_viento_kmh, 2),
            aporte_puntos=round(viento_puntos, 2),
            descripcion="Aporte de dispersión: escala creciente hasta 80 km/h.",
        ),
        FactorRiesgo(
            nombre="proximidad_vegetacion",
            valor=round(proximidad, 4),
            aporte_puntos=round(proximidad_puntos, 2),
            descripcion=detalle_proximidad,
        ),
    ]

    score = round(_clamp(temperatura_puntos + humedad_puntos + viento_puntos + proximidad_puntos, 0.0, 100.0), 1)
    return score, factores


def _clasificar(score: float) -> Literal["Bajo", "Medio", "Alto"]:
    if score >= 70.0:
        return "Alto"
    if score >= 40.0:
        return "Medio"
    return "Bajo"


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Ciclo de vida de la aplicación (arranque verificando la base de datos)."""
    _validar_api_key_de_arranque()
    # La comprobación va en un threadpool: es I/O bloqueante y no debe
    # bloquear el event loop (queda acotada por el connect_timeout).
    await run_in_threadpool(_comprobar_base_de_datos)
    yield


app = FastAPI(
    title="API Riesgo Incendios Forestales - Valparaíso",
    version="1.0.0",
    description=(
        "Prototipo de apoyo a la prevención y clasificación de riesgo de "
        "incendios forestales en la Región de Valparaíso."
    ),
    lifespan=lifespan,
)


# Caché del índice de rutas Starlette puras, indexado por id(app) (el valor
# guarda la propia app para que un id reutilizado no devuelva el índice de
# otra aplicación).
_indices_rutas: dict[int, tuple[FastAPI, int, dict[Any, str]]] = {}


def _indice_rutas_starlette(app: FastAPI) -> dict[Any, str]:
    """Índice `endpoint -> path` de las rutas Starlette puras (/docs, /openapi.json).

    Sólo contiene rutas definidas por la aplicación, por lo que la cardinalidad
    del label `endpoint` sigue siendo acotada. El resultado se cachea por app y
    se recalcula si se registra alguna ruta nueva.
    """
    entrada = _indices_rutas.get(id(app))
    if entrada is not None and entrada[0] is app and entrada[1] == len(app.routes):
        return entrada[2]

    indice: dict[Any, str] = {
        route.endpoint: route.path
        for route in app.routes
        if isinstance(route, StarletteRoute) and not isinstance(route, APIRoute)
    }
    _indices_rutas[id(app)] = (app, len(app.routes), indice)
    return indice


def _derive_endpoint(scope: Scope, status_code: int, app: FastAPI) -> str:
    """Deriva el label `endpoint` de las métricas con cardinalidad ACOTADA.

    NUNCA se usa `request.url.path` como fallback: el path lo elige el cliente
    y generaría un número de series ilimitado (ataque de cardinalidad).

    Reglas de etiquetado:
      * Ruta de FastAPI (`APIRoute`)             -> su path ("/predict", ...).
      * Rutas Starlette puras (/docs, /redoc, ...) -> su path.
      * Mont de ficheros estáticos (StaticFiles)   -> "static".
      * Resto (404/405 sin ruta de API)            -> "unmatched".
    """
    route = scope.get("route")
    if isinstance(route, APIRoute):
        return route.path

    endpoint = scope.get("endpoint")
    if isinstance(endpoint, StaticFiles):
        # El mount estático sirve el frontend; un 404 aquí es un fichero
        # inexistente, no tráfico de la API, así que se etiqueta "unmatched".
        return ENDPOINT_STATIC if status_code < 400 else ENDPOINT_UNMATCHED

    if endpoint is not None:
        path = _indice_rutas_starlette(app).get(endpoint)
        if path is not None:
            return path

    return ENDPOINT_UNMATCHED


@app.middleware("http")
async def metrics_middleware(request: Request, call_next: Any) -> Response:
    """Mide duración y cuenta peticiones HTTP para Prometheus."""
    method = request.method
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start
        endpoint = _derive_endpoint(request.scope, status_code, request.app)
        HTTP_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        HTTP_REQUESTS.labels(
            method=method, endpoint=endpoint, status=str(status_code)
        ).inc()


@app.get("/health", tags=["salud"])
async def health() -> dict[str, str]:
    """Comprobación de salud del servicio (sin autenticación)."""
    return {"status": "ok"}


@app.get("/metrics", tags=["observabilidad"])
async def metrics() -> Response:
    """Expone las métricas Prometheus (sin autenticación)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/predict",
    response_model=PrediccionResponse,
    tags=["predicción"],
    dependencies=[Depends(require_api_key)],
)
async def predict(payload: PrediccionRequest) -> PrediccionResponse:
    """Clasifica el riesgo de incendio forestal para una ubicación y condiciones."""
    score, factores = _score_riesgo(payload)
    nivel_riesgo = _clasificar(score)

    # La persistencia es I/O bloqueante (psycopg2 síncrono): se ejecuta en un
    # threadpool para no bloquear el event loop en cada /predict. El
    # comportamiento degradado se mantiene: si la base de datos falla, la
    # función devuelve None y la respuesta se sirve con id_prediccion = null.
    prediction_id = await run_in_threadpool(
        _persistir_prediccion,
        temperatura_c=payload.temperatura_c,
        humedad_relativa=payload.humedad_relativa,
        velocidad_viento_kmh=payload.velocidad_viento_kmh,
        nivel_riesgo=nivel_riesgo,
        score=score,
        latitud=payload.latitud,
        longitud=payload.longitud,
    )

    PREDICT_REQUESTS.labels(riesgo=nivel_riesgo).inc()
    logger.info(
        "predicción riesgo=%s score=%.1f lat=%.4f lon=%.4f id=%s",
        nivel_riesgo,
        score,
        payload.latitud,
        payload.longitud,
        prediction_id,
    )

    return PrediccionResponse(
        nivel_riesgo=nivel_riesgo,
        score=score,
        factores=factores,
        timestamp=datetime.now(timezone.utc),
        latitud=payload.latitud,
        longitud=payload.longitud,
        modelo="heuristica-simulada-cnn-lstm",
        id_prediccion=prediction_id,
    )


# ---------------------------------------------------------------------------
# Frontend estático (Leaflet) servido en /
# ---------------------------------------------------------------------------


def _resolver_directorio_frontend() -> str:
    """Ruta absoluta del frontend, válida en local y dentro del contenedor."""
    directorio_backend = os.path.dirname(os.path.abspath(__file__))
    candidatas = [
        os.getenv("STATIC_DIR", ""),
        os.path.join(os.path.dirname(directorio_backend), "frontend"),
        os.path.join(directorio_backend, "frontend"),
    ]
    for candidata in candidatas:
        if candidata and os.path.isdir(candidata):
            return candidata
    logger.warning(
        "No se encontró el directorio del frontend; se usa %s (puede no existir).",
        candidatas[1],
    )
    return candidatas[1]


FRONTEND_DIR = _resolver_directorio_frontend()
app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True, check_dir=False),
    name="frontend",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
