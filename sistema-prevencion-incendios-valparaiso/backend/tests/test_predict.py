"""Tests del endpoint /predict y endpoints relacionados (health, metrics).

Cobertura:
  1. GET /health sin autenticación.
  2. POST /predict sin header X-API-Key.
  3. POST /predict con X-API-Key incorrecta.
  4. POST /predict con payload válido y key válida (esquema de respuesta).
  5. Casificación Alto / Medio / Bajo con condiciones de ejemplo.
  6. Validaciones 422 (temperatura y latitud fuera de rango).
  7. GET /metrics sin autenticación.
  8. GET /metrics expone el histograma de duración y el contador de predicciones.
  9. Regresión: X-API-Key no-ASCII responde 403 (nunca 5xx).
"""

from __future__ import annotations

from typing import Any


def _post_predict(
    client: Any,
    payload: dict[str, float],
    headers: dict[str, str | bytes] | None = None,
) -> Any:
    """Envía POST /predict con los headers indicados ({} si no hay)."""
    return client.post("/predict", json=payload, headers=headers or {})


# ---------------------------------------------------------------------------
# 1. Health
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_200_without_authentication(client: Any) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# 2-3. Autenticación
# ---------------------------------------------------------------------------


def test_predict_without_api_key_header_is_rejected(
    client: Any, valid_payload: dict[str, float]
) -> None:
    response = _post_predict(client, valid_payload, headers={})

    assert response.status_code == 401


def test_predict_with_incorrect_api_key_is_rejected(
    client: Any, valid_payload: dict[str, float]
) -> None:
    response = _post_predict(
        client, valid_payload, headers={"X-API-Key": "clave-incorrecta-999"}
    )

    assert response.status_code == 403


def test_predict_with_non_ascii_api_key_returns_403_and_never_5xx(
    client: Any, valid_payload: dict[str, float]
) -> None:
    """Regresión: una X-API-Key no-ASCII debe dar 403, nunca un error 5xx.

    `secrets.compare_digest` sólo admite cadenas ASCII, así que comparar en
    bytes UTF-8 es lo que evita el TypeError (HTTP 500) ante entrada del
    cliente.
    """
    # httpx sólo admite headers ASCII: se envía el byte latin-1 0xE9 en crudo,
    # igual que `curl -H $'X-API-Key: cl\xe9-mala'`.
    response = _post_predict(
        client, valid_payload, headers={"X-API-Key": b"cl\xe9-mala"}
    )

    assert response.status_code == 403
    assert response.status_code < 500


# ---------------------------------------------------------------------------
# 4. Respuesta válida con key válida
# ---------------------------------------------------------------------------


def test_predict_with_valid_payload_and_key_returns_expected_schema(
    client: Any, valid_payload: dict[str, float], auth_headers: dict[str, str]
) -> None:
    response = _post_predict(client, valid_payload, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()

    # Campos obligatorios de la respuesta.
    assert "nivel_riesgo" in data
    assert "score" in data
    assert "factores" in data
    assert "timestamp" in data
    assert "latitud" in data
    assert "longitud" in data
    assert "modelo" in data
    assert "id_prediccion" in data

    # Nivel de riesgo dentro del dominio permitido.
    assert data["nivel_riesgo"] in {"Alto", "Medio", "Bajo"}

    # Score numérico dentro de [0, 100].
    assert isinstance(data["score"], (int, float))
    assert 0.0 <= data["score"] <= 100.0

    # Coherencia entre score y nivel clasificado (heurística del backend:
    # score >= 70 Alto, >= 40 Medio, resto Bajo).
    score = float(data["score"])
    expected_level = "Alto" if score >= 70.0 else "Medio" if score >= 40.0 else "Bajo"
    assert data["nivel_riesgo"] == expected_level

    # La base de datos de test está deliberadamente caída: el prototipo
    # debe responder 200 igualmente con id_prediccion=None.
    assert data["id_prediccion"] is None


# ---------------------------------------------------------------------------
# 5. Clasificación Alto / Medio / Bajo
# ---------------------------------------------------------------------------


def test_extreme_heat_and_dry_windy_conditions_are_classified_as_alto(
    client: Any, auth_headers: dict[str, str]
) -> None:
    payload = {
        "temperatura_c": 42.0,
        "humedad_relativa": 8.0,
        "velocidad_viento_kmh": 60.0,
        "latitud": -33.0472,
        "longitud": -71.6127,
    }

    response = _post_predict(client, payload, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["nivel_riesgo"] == "Alto"
    assert float(data["score"]) >= 70.0


def test_warm_and_dry_conditions_are_classified_as_medio(
    client: Any, auth_headers: dict[str, str]
) -> None:
    payload = {
        "temperatura_c": 28.0,
        "humedad_relativa": 40.0,
        "velocidad_viento_kmh": 25.0,
        "latitud": -33.0472,
        "longitud": -71.6127,
    }

    response = _post_predict(client, payload, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["nivel_riesgo"] == "Medio"
    assert 40.0 <= float(data["score"]) < 70.0


def test_cold_and_humid_conditions_are_classified_as_bajo(
    client: Any, auth_headers: dict[str, str]
) -> None:
    payload = {
        "temperatura_c": 12.0,
        "humedad_relativa": 80.0,
        "velocidad_viento_kmh": 8.0,
        "latitud": -33.0472,
        "longitud": -71.6127,
    }

    response = _post_predict(client, payload, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["nivel_riesgo"] == "Bajo"
    assert float(data["score"]) < 40.0


# ---------------------------------------------------------------------------
# 6. Validaciones (422)
# ---------------------------------------------------------------------------


def test_predict_with_temperature_out_of_range_returns_422(
    client: Any, valid_payload: dict[str, float], auth_headers: dict[str, str]
) -> None:
    valid_payload["temperatura_c"] = 99.0

    response = _post_predict(client, valid_payload, headers=auth_headers)

    assert response.status_code == 422


def test_predict_with_latitude_out_of_range_returns_422(
    client: Any, valid_payload: dict[str, float], auth_headers: dict[str, str]
) -> None:
    valid_payload["latitud"] = 0.0

    response = _post_predict(client, valid_payload, headers=auth_headers)

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 7. Métricas Prometheus
# ---------------------------------------------------------------------------


def test_metrics_endpoint_exposes_http_requests_total_without_authentication(
    client: Any,
) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_metrics_endpoint_exposes_duration_histogram_and_predict_counter(
    client: Any, valid_payload: dict[str, float], auth_headers: dict[str, str]
) -> None:
    # Se genera al menos una predicción para garantizar que el contador
    # `predict_requests_total` tenga series con labels.
    predict_response = _post_predict(client, valid_payload, headers=auth_headers)
    assert predict_response.status_code == 200

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_request_duration_seconds_bucket" in response.text
    assert "predict_requests_total" in response.text
