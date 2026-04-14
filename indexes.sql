-- =============================================================
-- indexes.sql
-- All 16 B-tree indexes for medical_analytics database
-- Run AFTER data is loaded and BEFORE running benchmark queries
--
-- Usage:
--   \i indexes.sql
-- =============================================================

-- =============================================================
-- JOIN PERFORMANCE INDEXES (11)
-- These speed up queries that join tables on patient/encounter/
-- provider columns. Without these, every join does a full scan.
-- =============================================================

-- encounters
CREATE INDEX IF NOT EXISTS idx_encounters_patient
    ON encounters (patient);

CREATE INDEX IF NOT EXISTS idx_encounters_provider
    ON encounters (provider);

-- conditions
CREATE INDEX IF NOT EXISTS idx_conditions_patient
    ON conditions (patient);

CREATE INDEX IF NOT EXISTS idx_conditions_encounter
    ON conditions (encounter);

-- procedures
CREATE INDEX IF NOT EXISTS idx_procedures_patient
    ON procedures (patient);

CREATE INDEX IF NOT EXISTS idx_procedures_encounter
    ON procedures (encounter);

-- medications
CREATE INDEX IF NOT EXISTS idx_medications_patient
    ON medications (patient);

-- observations
CREATE INDEX IF NOT EXISTS idx_observations_patient
    ON observations (patient);

CREATE INDEX IF NOT EXISTS idx_observations_encounter
    ON observations (encounter);

-- imaging_studies
CREATE INDEX IF NOT EXISTS idx_imaging_patient
    ON imaging_studies (patient);

CREATE INDEX IF NOT EXISTS idx_imaging_encounter
    ON imaging_studies (encounter);

-- =============================================================
-- DATE RANGE INDEXES (5)
-- These speed up time-range queries which are common in
-- healthcare analytics (e.g. "find all visits in 2021")
-- =============================================================

CREATE INDEX IF NOT EXISTS idx_encounters_start
    ON encounters (start);

CREATE INDEX IF NOT EXISTS idx_conditions_start
    ON conditions (start);

CREATE INDEX IF NOT EXISTS idx_procedures_start
    ON procedures (start);

CREATE INDEX IF NOT EXISTS idx_medications_start
    ON medications (start);

CREATE INDEX IF NOT EXISTS idx_observations_date
    ON observations (date);

-- =============================================================
-- Update planner statistics after creating indexes
-- This is REQUIRED — without it PostgreSQL won't use the indexes
-- efficiently because it doesn't know the data distribution yet
-- =============================================================
ANALYZE;

SELECT 'All 16 indexes created and ANALYZE complete' AS status;
