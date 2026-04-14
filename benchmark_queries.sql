-- =============================================================
-- benchmark_queries.sql
-- 20 benchmark queries used to collect training data
-- Each query should be run with EXPLAIN ANALYZE in pgAdmin
--
-- For each query:
--   1. Open pgAdmin Query Tool
--   2. Paste the query (with EXPLAIN ANALYZE prefix)
--   3. Press F5
--   4. Check the Messages tab for:
--      Planning Time: X ms
--      Execution Time: X ms
--
-- Usage in psql:
--   \i benchmark_queries.sql
-- =============================================================

-- ── Q1: Simple patient-encounter join ─────────────────────────────────
EXPLAIN ANALYZE
SELECT p.first, p.last, e.start, e.description
FROM patients p
JOIN encounters e ON p.id = e.patient
LIMIT 100;

-- ── Q2: Time range filter 2020-2022 ───────────────────────────────────
EXPLAIN ANALYZE
SELECT e.start, e.description, c.description AS condition
FROM encounters e
JOIN conditions c ON e.id = c.encounter
WHERE e.start BETWEEN '2020-01-01' AND '2022-12-31';

-- ── Q3: Aggregation — count encounters per patient ────────────────────
EXPLAIN ANALYZE
SELECT e.patient, COUNT(e.id) AS total_encounters
FROM encounters e
GROUP BY e.patient
ORDER BY total_encounters DESC
LIMIT 100;

-- ── Q4: Deep 4-table join (fastest due to all indexes) ────────────────
EXPLAIN ANALYZE
SELECT p.first, p.last, e.start, pr.description AS procedure, c.description AS condition
FROM patients p
JOIN encounters e ON p.id = e.patient
JOIN procedures pr ON e.id = pr.encounter
JOIN conditions c ON e.id = c.encounter
WHERE e.start >= '2021-01-01'
LIMIT 500;

-- ── Q5: Vital signs observation lookup (slowest — no index on category)
EXPLAIN ANALYZE
SELECT o.patient, o.description, o.value, o.units
FROM observations o
JOIN encounters e ON o.encounter = e.id
WHERE o.category = 'vital-signs'
AND e.start BETWEEN '2019-01-01' AND '2021-01-01';

-- ── Q6: Provider encounter count ──────────────────────────────────────
EXPLAIN ANALYZE
SELECT pr.name, COUNT(e.id) AS total_encounters
FROM providers pr
JOIN encounters e ON pr.id = e.provider
GROUP BY pr.name
ORDER BY total_encounters DESC
LIMIT 50;

-- ── Q7: Patient conditions list ───────────────────────────────────────
EXPLAIN ANALYZE
SELECT p.first, p.last, c.description, c.start
FROM patients p
JOIN encounters e ON p.id = e.patient
JOIN conditions c ON e.id = c.encounter
ORDER BY c.start DESC
LIMIT 200;

-- ── Q8: Medication lookup per patient ─────────────────────────────────
EXPLAIN ANALYZE
SELECT p.first, p.last, m.description, m.start
FROM patients p
JOIN encounters e ON p.id = e.patient
JOIN medications m ON e.id = m.encounter
LIMIT 300;

-- ── Q9: Procedures by date range ──────────────────────────────────────
EXPLAIN ANALYZE
SELECT pr.description, pr.start, pr.base_cost
FROM procedures pr
WHERE pr.start BETWEEN '2019-01-01' AND '2021-12-31'
LIMIT 500;

-- ── Q10: Count conditions per patient ────────────────────────────────
EXPLAIN ANALYZE
SELECT e.patient, COUNT(DISTINCT c.code) AS unique_conditions
FROM encounters e
JOIN conditions c ON e.id = c.encounter
GROUP BY e.patient
ORDER BY unique_conditions DESC
LIMIT 100;

-- ── Q11: Immunizations by date ────────────────────────────────────────
EXPLAIN ANALYZE
SELECT i.patient, i.description, i.date
FROM immunizations i
WHERE i.date BETWEEN '2020-01-01' AND '2022-12-31'
LIMIT 500;

-- ── Q12: Observations by category ────────────────────────────────────
EXPLAIN ANALYZE
SELECT o.patient, o.description, o.value, o.units
FROM observations o
WHERE o.category = 'laboratory'
LIMIT 1000;

-- ── Q13: Medications with date filter ────────────────────────────────
EXPLAIN ANALYZE
SELECT p.first, p.last, e.start, m.description
FROM patients p
JOIN encounters e ON p.id = e.patient
JOIN medications m ON e.id = m.encounter
WHERE e.start BETWEEN '2020-01-01' AND '2023-01-01'
LIMIT 300;

-- ── Q14: Encounter class aggregation ─────────────────────────────────
EXPLAIN ANALYZE
SELECT e.encounterclass, COUNT(*) AS total
FROM encounters e
GROUP BY e.encounterclass
ORDER BY total DESC;

-- ── Q15: Most common conditions ───────────────────────────────────────
EXPLAIN ANALYZE
SELECT c.description, COUNT(*) AS total
FROM conditions c
GROUP BY c.description
ORDER BY total DESC
LIMIT 50;

-- ── Q16: Imaging studies by date range ───────────────────────────────
EXPLAIN ANALYZE
SELECT i.patient, i.modality_description, i.date
FROM imaging_studies i
WHERE i.date BETWEEN '2018-01-01' AND '2022-01-01'
LIMIT 500;

-- ── Q17: 5-table join ─────────────────────────────────────────────────
EXPLAIN ANALYZE
SELECT p.first, p.last, e.start, c.description, m.description, pr.description
FROM patients p
JOIN encounters e ON p.id = e.patient
JOIN conditions c ON e.id = c.encounter
JOIN medications m ON e.id = m.encounter
JOIN procedures pr ON e.id = pr.encounter
WHERE e.start >= '2020-01-01'
LIMIT 200;

-- ── Q18: Allergy lookup per patient ──────────────────────────────────
EXPLAIN ANALYZE
SELECT p.first, p.last, a.description, a.category
FROM patients p
JOIN encounters e ON p.id = e.patient
JOIN allergies a ON e.id = a.encounter
LIMIT 200;

-- ── Q19: Supplies by date range ───────────────────────────────────────
EXPLAIN ANALYZE
SELECT s.patient, s.description, s.quantity, s.date
FROM supplies s
WHERE s.date BETWEEN '2019-01-01' AND '2023-01-01'
LIMIT 300;

-- ── Q20: Full patient profile ─────────────────────────────────────────
EXPLAIN ANALYZE
SELECT p.first, p.last,
       COUNT(DISTINCT e.id) AS visits,
       COUNT(DISTINCT c.code) AS conditions,
       COUNT(DISTINCT m.code) AS medications
FROM patients p
JOIN encounters e ON p.id = e.patient
JOIN conditions c ON e.id = c.encounter
JOIN medications m ON e.id = m.encounter
GROUP BY p.first, p.last
ORDER BY visits DESC
LIMIT 50;
