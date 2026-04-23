-- This script runs once on first Postgres container startup.
-- It creates the 'retail' database alongside the 'airflow' metadata database.
SELECT 'CREATE DATABASE retail'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'retail'
)\gexec
