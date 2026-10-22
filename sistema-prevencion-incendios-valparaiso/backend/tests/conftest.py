"""Configuración de fixtures para los tests del backend de incendios forestales.

* Añade `backend/` al `sys.path` para poder hacer `import main`.
* Fija `API_KEY` conocida para los tests de autenticación.
* Apunta `DATABASE_URL` a un puerto inválido (127.0.0.1:1) para que los tests
  sean herméticos: no tocan ninguna base de datos real y ejercitan la ruta
  degradada del prototipo (la predicción responde igualmente con
  `id_prediccion = None` cuando PostgreSQL no está disponible).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# backend/ debe estar en el path antes de importar `main` (que vive en
# backend/main.py, no en un paquete instalable).
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Entorno de test determinista (se fija ANTES de importar main).
os.environ["API_KEY"] = "dev-api-key-123"
os.environ["DATABASE_URL"] = "postgresql+psycopg2://test:test@127.0.0.1:1/testdb"

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

API_KEY = os.environ["API_KEY"]

# Payload válido dentro de la zona de Valparaíso (cerros de Valparaíso).
VALID_PAYLOAD: dict[str, float] = {
    "temperatura_c": 25.0,
    "humedad_relativa": 35.0,
    "velocidad_viento_kmh": 20.0,
    "latitud": -33.0472,
    "longitud": -71.6127,
}


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Cliente de prueba de FastAPI (ejecuta el hook de startup una vez)."""
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def api_key() -> str:
    """API key válida para el entorno de test."""
    return API_KEY


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers con autenticación válida para /predict."""
    return {"X-API-Key": API_KEY}


@pytest.fixture
def valid_payload() -> dict[str, float]:
    """Copia independiente del payload válido por defecto."""
    return dict(VALID_PAYLOAD)
