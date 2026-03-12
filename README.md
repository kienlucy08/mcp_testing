# FieldSync MCP Analytics Server

A conversational analytics interface that connects Claude AI to a PostgreSQL database of tower/telecommunications infrastructure. Ask questions in plain English — the server translates them into safe, read-only database queries and returns formatted results.

---

## What It Does

The server exposes seven tools to Claude, covering:

- **Schema exploration** — understand the database structure before querying
- **Custom SQL** — write ad-hoc SELECT queries (org-scoped, read-only enforced)
- **Inspection compliance** — find towers overdue for inspection by tower type
- **Deficiency search** — query deficiency records by keyword, severity, date, or site
- **Geographic queries** — find sites by coordinates or radius
- **Weather** — live conditions and 5-day forecast for any site location

All queries are automatically scoped to a single organization defined in `.env`. No write operations are possible.

---

## Current Configuration

| Setting | Value |
|---|---|
| **Database** | Production RDS (`fieldsync-postgres-app-...`) |
| **DB User** | `postgres_readonly` — a read-only database account |
| **Organization** | EverestInfrastructure (`073de7c7-eaa1-4f83-af72-26116b4caba8`) |

> The database user `postgres_readonly` has no write privileges at the account level. Combined with the application-level `default_transaction_read_only=on` session flag, this gives two independent layers of write protection.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | `python --version` to check |
| PostgreSQL access | RDS endpoint, read-only username, password |
| Anthropic API key | [console.anthropic.com](https://console.anthropic.com) |
| OpenWeatherMap API key | [openweathermap.org/api](https://openweathermap.org/api) — free tier is fine |

---

## Step-by-Step Setup

### 1. Clone or download the project

```bash
git clone <repo-url>
cd mcp_testing
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **Mac/Linux:** `source .venv/bin/activate`

You should see `(.venv)` at the start of your terminal prompt.

### 3. Install dependencies

```bash
pip install anthropic mcp flask python-dotenv psycopg2-binary requests
```

### 4. Configure `.env`

Create a file named `.env` in the project root (same folder as `mcp_server.py`). Copy the template below and fill in your values:

```
# Anthropic
API_KEY="sk-ant-..."

# PostgreSQL database — use a read-only account where possible
url="your-rds-endpoint.rds.amazonaws.com"
postgres_user="your_readonly_db_user"
postgres_pass="your_db_password"
port=5432

# Weather (OpenWeatherMap)
weather_api_key="your_openweathermap_key"
weather_api_endpoint="api.openweathermap.org"

# Demo scope — all queries are locked to this organization
DEMO_ORG_ID="the-org-uuid-from-your-database"
DEMO_ORG_NAME="Organization Name"
```

> **Finding `DEMO_ORG_ID`:** This is the UUID of the organization row in your database's `organization` table. Ask your database administrator if you don't have it.

> **Database user:** Use a dedicated read-only database account (`GRANT SELECT` only) for an extra layer of protection beyond the application-level safeguards.

### 5. Run the server

```bash
python mcp_client.py
```

Open your browser and go to **http://localhost:5000**

The first startup takes 1–3 minutes while the server caches the database schema. Subsequent tool calls are fast.

---

## Switching to a Different Organization

To run the demo for a different client organization:

1. Open `.env`
2. Update `DEMO_ORG_ID` to the new organization's UUID
3. Update `DEMO_ORG_NAME` to the new organization's display name
4. **Restart the server** (`Ctrl+C`, then `python mcp_client.py` again)

The restart is required — the org ID is loaded once at startup and baked into the system prompt and all query filters. There is no risk of data from the previous org leaking after a restart because every query layer enforces the new `DEMO_ORG_ID` independently:

| Layer | How it enforces the org scope |
|---|---|
| System prompt | Claude is instructed to only query the named org |
| Structured tools | `organization_id` is hard-overridden to `DEMO_ORG_ID` in code, ignoring anything Claude passes |
| `execute_query` tool | Rejected at the server if `DEMO_ORG_ID` is not present in the query parameters |
| `get_weather_for_site` | Pre-flight DB check confirms the site belongs to `DEMO_ORG_ID` before any data is returned |
| PostgreSQL session | `default_transaction_read_only=on` — the database itself rejects any write attempt |
| Database account | `postgres_readonly` has no write privileges at the account level |

---

## Available Tools

| Tool | What to ask | Example |
|---|---|---|
| `schema` | Questions about database structure | *"What tables are related to inspections?"* |
| `execute_query` | Custom data questions | *"How many sites have no address?"* |
| `explore_data_relationships` | Understanding how tables connect | *"How does a deficiency link back to a site?"* |
| `query_sites_by_location` | Geography / proximity | *"Find all sites within 30 miles of this location"* |
| `query_inspection_data` | Safety climbs, deficiencies by keyword/severity/date | *"Show me all severity-1 deficiencies in the last year"* |
| `get_sites_needing_inspection` | Compliance reporting | *"Which guyed towers are overdue for inspection?"* |
| `get_weather_for_site` | Live weather at a site | *"What are the current conditions at site X?"* |

---

## Example Demo Questions

**Inspection compliance:**
- *Which towers need TIA inspection?*
- *How many towers have never been inspected?*
- *Show me all guyed towers that are overdue for inspection.*
- *Which sites have the most overdue structures?*

**Deficiency analysis:**
- *What are the most common deficiencies?*
- *Find all deficiencies related to rust.*
- *What severity-1 deficiencies were logged in the last 6 months?*
- *Which sites have the highest number of open deficiencies?*
- *Compare deficiency rates across different tower types (monopole, guyed, and self-support).*

**Tower inventory:**
- *How many total towers does EverestInfrastructure have?*
- *Show me the top 10 tallest structures.*
- *What is the breakdown of tower types (guyed vs monopole vs self-support)?*
- *Which sites have structures over 500 feet tall?*

**Geographic:**
- *Find all sites within 25 miles of [city/coordinates].*
- *What sites are in [state]?*

**Weather:**
- *What is the weather at [site name]? Any extreme conditions?*
- *Show the 5-day forecast for [site name].*

---

## Security Safeguards Summary

This server is configured for safe, read-only use against the production database. Six independent layers of protection are in place:

1. **Read-only database account** — `postgres_readonly` has `SELECT` privileges only at the PostgreSQL account level. Even a direct database connection with these credentials cannot write data.
2. **Read-only session flag** — every connection sets `default_transaction_read_only=on` at the driver level. The database rejects any write or DDL statement even if one were to bypass application checks.
3. **Statement timeout** — queries are killed after 30 seconds to prevent resource exhaustion on the production database.
4. **SQL keyword blocking** — custom queries are scanned (after stripping comments) for forbidden keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `TRUNCATE`, `ALTER`, `CREATE`, `GRANT`, `REVOKE`, `COPY`, `EXECUTE`, `CALL`, `VACUUM`, `DO`.
5. **Parameterized queries** — no user input is ever interpolated directly into SQL strings, preventing SQL injection.
6. **Org ID enforcement** — every query path enforces `DEMO_ORG_ID`; the value cannot be overridden by the user or by Claude. Custom queries are rejected server-side if the org ID is not present in the parameters.
7. **Query length cap** — custom SQL is limited to 5,000 characters.
