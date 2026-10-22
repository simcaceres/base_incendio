-- Inicialización de la base de datos del prototipo de prevención de incendios
-- forestales en Valparaíso (PostgreSQL 16 + PostGIS).

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS prediction (
    id            SERIAL PRIMARY KEY,
    temperatura_c DOUBLE PRECISION NOT NULL,
    humedad       DOUBLE PRECISION NOT NULL,
    viento        DOUBLE PRECISION NOT NULL,
    riesgo        TEXT NOT NULL CHECK (riesgo IN ('Bajo', 'Medio', 'Alto')),
    score         DOUBLE PRECISION NOT NULL,
    geom          geometry(Point, 4326),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Índice espacial para consultas geográficas (mapa del frontend).
CREATE INDEX IF NOT EXISTS idx_prediction_geom
    ON prediction USING GIST (geom);

-- Índice temporal para el histórico reciente.
CREATE INDEX IF NOT EXISTS idx_prediction_created_at
    ON prediction (created_at DESC);

COMMENT ON TABLE prediction IS
    'Predicciones de riesgo de incendio forestal registradas por la API.';
COMMENT ON COLUMN prediction.geom IS
    'Punto de la predicción en WGS84 (SRID 4326).';
