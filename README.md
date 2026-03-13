# FieldSync MCP Analytics Server

A conversational analytics interface that connects Claude AI to a PostgreSQL database of tower/telecommunications infrastructure. Ask questions in plain English — the server translates them into safe, read-only database queries and returns formatted results.

---

## What It Does

The server exposes tools to Claude covering:

- **Schema exploration** — understand the database structure before querying
- **Custom SQL** — write ad-hoc SELECT queries (org-scoped, read-only enforced)
- **Inspection compliance** — find towers overdue for inspection by tower type
- **Deficiency search** — query deficiency records by keyword, severity, date, or site
- **Site facilities** — pre-built queries for generators, shelters, and lighting observations
- **Geographic queries** — find sites by US state or radius from coordinates
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

The **Server Script Path** field is pre-populated automatically based on where you cloned the repo — no manual path editing required. Just click **Connect**.

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
| `query_sites_by_location` | Geography — US state lookup or radius search | *"Which sites are in Wisconsin?"* |
| `query_inspection_data` | Safety climbs, deficiencies by keyword/severity/date | *"Show me all severity-1 deficiencies in the last year"* |
| `get_sites_needing_inspection` | Compliance — applies correct 3yr/5yr rules by tower type | *"Which guyed towers are overdue for inspection?"* |
| `query_site_facilities` | Generators, shelters, lighting observations | *"How many sites have generators?"* |
| `get_weather_for_site` | Live weather at a site | *"What are the current conditions at site X?"* |

---

## Demo Questions

These are the questions used in the live demo, organized by category. All are clickable in the UI.

### Inspection Compliance
- *Which towers need TIA inspection?*
- *How many towers have never been inspected?*
- *Which sites have the most overdue structures?*

> **Note:** Inspection overdue status is determined by tower type — guyed towers every 3 years, monopole/self-support every 5 years. The compliance tool (`get_sites_needing_inspection`) applies these thresholds correctly. Do not rely on raw "days since visit" numbers alone.

### Deficiency Analysis
- *What are our top trending deficiencies?*
- *Show a breakdown of deficiency severity across all sites.*
- *How many deficiencies are there by state?*
- *What are the most common deficiencies?*
- *Find all deficiencies related to rust.*
- *What severity-1 deficiencies were logged in the last 6 months?*
- *Compare deficiency rates across different tower types (monopole, guyed, and self-support).*

### Site Equipment & Observations
- *Which sites have shelters? How many sites have no shelter?*
- *How many sites have generators? Show a count of sites with and without generators.*
- *Which sites have lightning rod observations? Are there any compliance issues?*
- *Show me all lighting compliance observations across sites.*

### Tower Inventory
- *How many total towers does the organization have?*
- *Show me the top 10 tallest structures.*
- *What is the breakdown of tower types (guyed vs monopole vs self-support)?*
- *Which sites have structures over 500 feet tall?*

### Geographic
- *Find all sites within 50 miles of Syracuse NY.*
- *Tell me about the sites in Wisconsin.*
- *Give me a full equipment breakdown for the Montana site with structure owner id Mt. Baldy*

> **Note on Mt. Baldy:** This is the organization's only Montana site. It is a self-support tower and its last inspection was approximately 1.5 years ago — well within the 5-year compliance window. It should not appear in overdue reports.

### Equipment Inventory
- *What types of equipment have been catalogued across all sites?*
- *How many antenna equipment records exist across the organization?*
- *Show me a count of appurtenances by site.*
- *Which sites have the most equipment inventory records?*
- *Show me all guy wire and guy attachment records across the organization.*

---

## Key Demo Talking Points

### Live AI-to-database pipeline
Every question goes through Claude in real time — there is no pre-cached answer set. Claude selects tools, writes SQL, reads results, and synthesizes a natural-language response on the fly. The tool call log shown in the UI is the actual sequence of database operations that produced the answer.

### Correct inspection thresholds, always enforced
Guyed towers require inspection every **3 years**. Monopole and self-support towers require inspection every **5 years**. This logic is enforced in the pre-built compliance tool and is baked into Claude's system prompt — so ad-hoc queries follow the same rules. A self-support tower last visited 1.5 years ago is correctly identified as compliant, not overdue.

### Pre-built tools for common questions
Generators, shelters, and lighting observations each have dedicated pre-built tools that answer in one database round-trip without schema exploration. This keeps responses fast and avoids multi-step fallback paths.

### Automatic token and tool-call management
The client caps tool calls at 30 per query, trims message history to the last 15 messages, and truncates large tool results to 3,000 characters. This prevents runaway queries during a live demo.

### No UUIDs shown to users
Internal database UUIDs are never displayed in responses. The system prompt explicitly forbids it, and the pre-built tool formatters strip UUIDs from all output. Users only see site names, site codes, and addresses.

### Current date awareness
The system prompt is rebuilt on every query with today's date injected. Date-relative questions ("in the last 6 months", "overdue for 3 years") always calculate from the actual current date, not a stale startup value.

### Geographic queries by state
"Which sites are in Montana?" or "Tell me about sites in Wisconsin" routes directly to `query_sites_by_location` with a built-in US state bounding-box lookup — no schema exploration needed, single tool call. All 50 states are supported.

---

## Security Safeguards Summary

This server is configured for safe, read-only use against the production database. Seven independent layers of protection are in place:

1. **Read-only database account** — `postgres_readonly` has `SELECT` privileges only at the PostgreSQL account level. Even a direct database connection with these credentials cannot write data.
2. **Read-only session flag** — every connection sets `default_transaction_read_only=on` at the driver level. The database rejects any write or DDL statement even if one were to bypass application checks.
3. **Statement timeout** — queries are killed after 30 seconds to prevent resource exhaustion on the production database.
4. **SQL keyword blocking** — custom queries are scanned (after stripping comments) for forbidden keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `TRUNCATE`, `ALTER`, `CREATE`, `GRANT`, `REVOKE`, `COPY`, `EXECUTE`, `CALL`, `VACUUM`, `DO`.
5. **Parameterized queries** — no user input is ever interpolated directly into SQL strings, preventing SQL injection.
6. **Org ID enforcement** — every query path enforces `DEMO_ORG_ID`; the value cannot be overridden by the user or by Claude. Custom queries are rejected server-side if the org ID is not present in the parameters.
7. **Query length cap** — custom SQL is limited to 5,000 characters.

---

## Architecture Notes

### How tool routing works
To reduce API token usage, the client does not send all available tools to Claude on every query. `select_relevant_tools()` in `mcp_client.py` pattern-matches the user's question and sends only the relevant subset. Core tools (`schema`, `execute_query`) are always included. Specialized tools are added based on keywords:

| Keyword triggers | Tool added |
|---|---|
| rust, deficiency, deficiencies, issue, fault | `query_inspection_data` |
| overdue, most overdue, never inspected, compliance | `get_sites_needing_inspection` |
| generator, shelter, lighting, FAA, lightning | `query_site_facilities` |
| state, Montana, Wisconsin, near, radius, miles | `query_sites_by_location` |
| weather, forecast, wind, temperature | `get_weather_for_site` |

### System prompt routing rules
Claude's system prompt includes explicit routing instructions that tell it which tool to use for specific question types, even before it decides to write a query. These prevent Claude from defaulting to `execute_query` + schema exploration for questions that have faster dedicated tools.

### SQL correctness rules in the system prompt
Claude's system prompt includes explicit SQL rules that govern every `execute_query` call:

- **LEFT JOIN subquery rule** — Filtering a LEFT JOIN's right-side table inside the ON clause silently returns zero counts. All filtered LEFT JOINs must use a subquery. This is the single most common source of incorrect zero results in this codebase.
- **camelCase quoting** — All camelCase identifiers must be double-quoted (`"siteId"`, `"organizationId"`, `"siteVisitDate"`, etc.). PostgreSQL is case-sensitive.
- **Schema verification** — Claude calls the `schema` tool before querying any table it hasn't used in the session.
- **Suspicious zero counts** — A COUNT returning 0 for something plausible must be treated as a broken query, not a confirmed finding. Claude runs a sanity check before reporting.
- **TIA inspection CASE** — A specific CASE structure is provided for determining inspection status, applying 3-year and 5-year thresholds correctly based on tower type.

### Database join paths for key tables
Some tables have non-obvious join paths that caused incorrect zero results before dedicated tools were built:

| Facility | Join path |
|---|---|
| Generators | `site → compound → compound_general → generator` (via `"compoundGeneralId"`) |
| Shelters (bphocs) | `site → compound → bphocs` (via `"compoundId"`) |
| Lighting observations | `other_appurtenance` joined to `site` via `"siteId"` |

These are handled by `query_site_facilities` so Claude does not need to discover them at query time.
