-- One-time Snowflake setup for retail-media-lab.
-- Run as a role that can create warehouses, databases and roles (for example ACCOUNTADMIN on a
-- trial account). Then:
--   pip install "retail-media-lab[snowflake]"
--   export SNOWFLAKE_ACCOUNT=... SNOWFLAKE_USER=... SNOWFLAKE_PASSWORD=...
--   SNOWFLAKE_ROLE=RML_LOADER SNOWFLAKE_WAREHOUSE=RML_WH rml load
-- and point the API, the agent and Dataiku at the read-only RML_ANALYST role:
--   RML_BACKEND=snowflake SNOWFLAKE_ROLE=RML_ANALYST rml ask "..."

CREATE WAREHOUSE IF NOT EXISTS RML_WH
    WAREHOUSE_SIZE = XSMALL AUTO_SUSPEND = 60 AUTO_RESUME = TRUE INITIALLY_SUSPENDED = TRUE;
CREATE DATABASE IF NOT EXISTS RML;
CREATE SCHEMA IF NOT EXISTS RML.AMC;

-- Loader: creates and replaces the tables (used by `rml load`)
CREATE ROLE IF NOT EXISTS RML_LOADER;
GRANT USAGE ON WAREHOUSE RML_WH TO ROLE RML_LOADER;
GRANT USAGE ON DATABASE RML TO ROLE RML_LOADER;
GRANT USAGE, CREATE TABLE, CREATE STAGE, CREATE FILE FORMAT ON SCHEMA RML.AMC TO ROLE RML_LOADER;

-- Analyst: read-only. Use it for the API, the agent and Dataiku, so model-written SQL can
-- never change data even if it got past the SQL guard.
CREATE ROLE IF NOT EXISTS RML_ANALYST;
GRANT USAGE ON WAREHOUSE RML_WH TO ROLE RML_ANALYST;
GRANT USAGE ON DATABASE RML TO ROLE RML_ANALYST;
GRANT USAGE ON SCHEMA RML.AMC TO ROLE RML_ANALYST;
GRANT SELECT ON ALL TABLES IN SCHEMA RML.AMC TO ROLE RML_ANALYST;
GRANT SELECT ON FUTURE TABLES IN SCHEMA RML.AMC TO ROLE RML_ANALYST;

-- Replace <your_user> with your Snowflake user name
GRANT ROLE RML_LOADER TO USER <your_user>;
GRANT ROLE RML_ANALYST TO USER <your_user>;

-- Optional, after `rml load` (Enterprise Edition): an AMC-style aggregation policy. Under the
-- analyst role, queries on the shopper tables must aggregate, and groups smaller than 100 rows
-- are folded into a remainder, so not even a query that slipped past the SQL guard can list
-- individual shoppers. The loader role is exempt. Not run in CI.
-- CREATE AGGREGATION POLICY IF NOT EXISTS RML.AMC.MIN_GROUP_100
--     AS () RETURNS AGGREGATION_CONSTRAINT ->
--         CASE
--             WHEN CURRENT_ROLE() = 'RML_ANALYST' THEN AGGREGATION_CONSTRAINT(MIN_GROUP_SIZE => 100)
--             ELSE NO_AGGREGATION_CONSTRAINT()
--         END;
-- ALTER TABLE RML.AMC.CONVERSIONS SET AGGREGATION POLICY RML.AMC.MIN_GROUP_100;
-- ALTER TABLE RML.AMC.DSP_IMPRESSIONS SET AGGREGATION POLICY RML.AMC.MIN_GROUP_100;
-- ALTER TABLE RML.AMC.SPONSORED_ADS_TRAFFIC SET AGGREGATION POLICY RML.AMC.MIN_GROUP_100;
