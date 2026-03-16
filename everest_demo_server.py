import asyncio
import sys
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from dotenv import load_dotenv
import os

load_dotenv()

# Logging — WARNING level only so errors surface without cluttering the console
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# ============================================================
# DEMO CONSTANTS — hardcoded for single-org, single-site demos
# ============================================================

DEMO_ORG_ID   = os.getenv("DEMO_ORG_ID")
DEMO_ORG_NAME = os.getenv("DEMO_ORG_NAME", "EverestInfrastructure")
MONTANA_SITE_ID = "cb86562b-b588-4d22-a897-abfd7603d205"

if not DEMO_ORG_ID:
    raise RuntimeError("DEMO_ORG_ID must be set in .env before starting the server.")

# Database configuration
DB_CONFIG = {
    "type": "postgres",
    "host": os.getenv("url"),
    "port": 5432,
    "database": "postgres",
    "user": os.getenv("postgres_user"),
    "password": os.getenv("postgres_pass"),
    "sslmode": "require"
}

# Create server instance
app = Server("everest-demo-server")


class DatabaseManager:
    """Manages database connections for the Everest demo server."""

    def __init__(self, config: dict):
        self.config = config
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            self.connection_class = psycopg2
            self.cursor_factory = RealDictCursor
        except ImportError:
            raise Exception("psycopg2 not installed. Run: pip install psycopg2-binary")

    def get_connection(self):
        """Get a database connection.

        DEMO SAFEGUARDS applied to every session:
          - default_transaction_read_only=on  — PostgreSQL will reject any write
            statement (INSERT/UPDATE/DELETE/DDL) even if one slips through.
          - statement_timeout=30000           — kills any query running longer
            than 30 seconds to prevent resource exhaustion on the production DB.
        """
        conn_params = {
            "host": self.config["host"],
            "port": self.config.get("port", 5432),
            "database": self.config["database"],
            "user": self.config["user"],
            "password": self.config["password"],
            "cursor_factory": self.cursor_factory,
            # Enforce read-only sessions and query time limits at the driver level
            "options": (
                "-c default_transaction_read_only=on "
                "-c statement_timeout=30000"
            ),
        }

        if "sslmode" in self.config:
            conn_params["sslmode"] = self.config["sslmode"]

        return self.connection_class.connect(**conn_params)

    def execute_query(self, query: str, params: tuple = ()) -> list:
        """Execute a SQL query and return results as list of dicts."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if params and len(params) > 0:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if cursor.description:
                results = [dict(row) for row in cursor.fetchall()]
            else:
                results = []

            conn.close()
            return results
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            raise Exception(f"Database error: {str(e)}")


db = DatabaseManager(DB_CONFIG)

# ============================================================
# SCHEMA CACHE — fetched once at startup, served instantly
# ============================================================

_SCHEMA_QUERY = """
SELECT t.table_name,
       STRING_AGG(c.column_name || ' (' || c.data_type || ')',
                  ', ' ORDER BY c.ordinal_position) AS columns
FROM information_schema.tables t
JOIN information_schema.columns c
     ON c.table_name = t.table_name AND c.table_schema = t.table_schema
WHERE t.table_schema = 'public'
  AND t.table_type  = 'BASE TABLE'
GROUP BY t.table_name
ORDER BY t.table_name
"""

CACHED_SCHEMA = ""
try:
    _rows = db.execute_query(_SCHEMA_QUERY)
    CACHED_SCHEMA = "\n".join(
        f"{r['table_name']}: {r['columns']}" for r in _rows
    )
    logger.warning(f"Schema cached: {len(_rows)} tables.")
except Exception as _e:
    logger.warning(f"Schema cache failed at startup: {_e}")

# ============================================================
# APPURTENANCE FK MAP — which tables reference appurtenance.id
# ============================================================

_APPURTENANCE_FK_QUERY = """
SELECT
    kcu.table_name          AS referencing_table,
    kcu.column_name         AS referencing_column,
    ccu.table_name          AS referenced_table,
    ccu.column_name         AS referenced_column
FROM information_schema.table_constraints   tc
JOIN information_schema.key_column_usage    kcu
     ON kcu.constraint_name = tc.constraint_name
     AND kcu.table_schema   = tc.table_schema
JOIN information_schema.constraint_column_usage ccu
     ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
  AND ccu.table_name     = 'appurtenance'
  AND tc.table_schema    = 'public'
ORDER BY kcu.table_name
"""

CACHED_APPURTENANCE_REFS = ""
try:
    _fk_rows = db.execute_query(_APPURTENANCE_FK_QUERY)
    if _fk_rows:
        lines = [
            f"  {r['referencing_table']}.{r['referencing_column']} → appurtenance.{r['referenced_column']}"
            for r in _fk_rows
        ]
        CACHED_APPURTENANCE_REFS = "\n".join(lines)
        logger.warning(
            f"Tables referencing appurtenance ({len(_fk_rows)}):\n" + CACHED_APPURTENANCE_REFS
        )
    else:
        CACHED_APPURTENANCE_REFS = "  (no foreign keys found referencing appurtenance)"
        logger.warning("No FK references to appurtenance found.")
except Exception as _e:
    logger.warning(f"Appurtenance FK map failed at startup: {_e}")


# ============================================================
# TOOL DEFINITIONS
# ============================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="tia_inspection_compliance",
            description=(
                "Which towers need TIA-222 structural inspection? Which structures are overdue for inspection? "
                "Use ONLY for questions about TIA-222 structural inspection schedules and overdue inspections — "
                "NOT for questions about lighting fixtures, FAA lights, beacons, or lightning rods. "
                "Applies 3-year rule for guyed towers and 5-year rule for monopole/self-support towers."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="top_trending_deficiencies",
            description=(
                "What are our top trending deficiencies? "
                "Returns the top 15 most frequently occurring deficiency names across all sites, "
                "with occurrence counts and average severity."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="deficiency_severity_breakdown",
            description=(
                "Show a breakdown of deficiency severity across all sites. "
                "Returns deficiency counts grouped by severity level, plus the top 10 sites "
                "by total deficiency count."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="deficiencies_by_state",
            description=(
                "How many deficiencies are there by state? "
                "Returns deficiency counts and number of affected sites grouped by US state, "
                "extracted from site addresses."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="recent_severity1_deficiencies",
            description=(
                "What severity-1 deficiencies were logged in the last 6 months? "
                "Returns the most critical (severity=1) deficiencies recorded in the past 6 months, "
                "ordered by date descending."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="shelter_summary",
            description=(
                "Which sites have shelters? How many sites have no shelter? "
                "Returns a per-site count of shelters/buildings and a summary of sites with vs. without shelters."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="generator_summary",
            description=(
                "How many sites have generators? Show a count of sites with and without generators. "
                "Returns a per-site generator count and a summary of sites with vs. without generators."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="lightning_protection_summary",
            description=(
                "Which sites have lightning rods? Show lightning rod observations. "
                "Use for questions about lightning protection systems, lightning rods, grounding, "
                "apex heights — NOT about FAA lights, beacons, or tower lighting fixtures. "
                "Returns lightning rod type, location, apex height, and centerline height per structure."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="tower_lighting_compliance",
            description=(
                "Which towers have FAA lighting or beacons? Show lighting observations across towers. "
                "Use for questions about tower lights, top beacons, light fixtures, light assemblies, "
                "towers without lights, or towers over 200 feet that need lighting — "
                "NOT for lightning rods or TIA structural inspections. "
                "Returns lighting present flag per tower and all light assembly fixture records."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="montana_equipment_breakdown",
            description=(
                "Give me a full equipment breakdown for the Montana site with structure owner id Mt. Baldy. "
                "Returns site info, structures, antenna equipment from mount_centerline records, "
                "and other equipment/observations for the hardcoded Montana site."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="equipment_types_summary",
            description=(
                "What types of equipment have been catalogued across all sites? "
                "Returns a full equipment breakdown: appurtenance groups, mount centerlines by mount type, "
                "antenna equipment by antenna type, lightning protection, light assemblies, "
                "AM skirts, and other named equipment records."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="antenna_equipment_count",
            description=(
                "How many antenna equipment records exist across the organization? "
                "Returns antenna records from mount_centerline grouped by antenna type, "
                "with total record count, total antenna count, and number of sites."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="appurtenances_by_site",
            description=(
                "Show me a count of appurtenances by site. What equipment exists at each site? "
                "Returns each appurtenance by name and structure, categorized by what hangs off it: "
                "mount centerlines, antenna equipment, lightning protection, light assemblies, "
                "AM skirts, and other appurtenances. Includes org-wide totals per equipment category."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="experiment_farm_equipment_breakdown",
            description=(
                "Give me a full equipment breakdown for the Experiment Farm site. "
                "Returns all equipment at 1450 Experiment Farm Rd split by category: "
                "structures, appurtenance groups, mount centerlines by mount type, "
                "antenna equipment by type, lightning protection, light assemblies, "
                "AM skirts, and other equipment — with totals per category."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="wisconsin_sites",
            description=(
                "Tell me about the sites in Wisconsin. "
                "Returns all sites located in Wisconsin with their structures, last inspection date, "
                "deficiency counts, and key details."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_schema_cache",
            description=(
                "Return the pre-cached database schema (tables and columns). "
                "This is fetched once at server startup and returned instantly — no DB round-trip. "
                "Use this if you need to understand the data model."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


# ============================================================
# TOOL HANDLERS
# ============================================================

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        # ------------------------------------------------------------------
        # 1. tia_inspection_compliance
        # ------------------------------------------------------------------
        if name == "tia_inspection_compliance":
            # Minimal SQL — raw data only; all TIA logic is computed in Python
            # LEFT JOIN site so structures with NULL siteId are still included and flagged
            q = """
SELECT
    st.name                AS structure_name,
    st.type                AS structure_type,
    st."constructedHeight" AS height,
    st."internalId"        AS structure_code,
    s.name                 AS site_name,
    s."internalId"         AS site_code,
    sv.last_visit
FROM structure st
LEFT JOIN site s ON s.id = st."siteId"
LEFT JOIN (
    SELECT "siteId", MAX("siteVisitDate") AS last_visit
    FROM site_visit
    GROUP BY "siteId"
) sv ON sv."siteId" = s.id
WHERE st."organizationId" = %s
ORDER BY sv.last_visit ASC NULLS FIRST, s.name NULLS LAST
"""
            rows = db.execute_query(q, (DEMO_ORG_ID,))

            # Python-side TIA classification
            from datetime import date as _date
            today = _date.today()

            def _classify(stype):
                t = (stype or "").lower()
                if "guy" in t:
                    return "Guyed", 3
                if "monopole" in t:
                    return "Monopole", 5
                if "self" in t and "support" in t:
                    return "Self-Support", 5
                return "Other", 3

            never, past_due, compliant = [], [], []
            summary_counts: dict = {}

            for r in rows:
                cat, req_yrs = _classify(r.get("structure_type"))
                lv = r.get("last_visit")

                # Normalise to Python date
                if lv is not None and hasattr(lv, "date"):
                    lv = lv.date()

                if lv is None:
                    is_overdue = True
                    years_since = None
                else:
                    years_since = round((today - lv).days / 365.25, 1)
                    is_overdue = years_since > req_yrs

                entry = dict(r)
                entry["tower_category"] = cat
                entry["required_years"] = req_yrs
                entry["years_since"] = years_since

                sc = summary_counts.setdefault(cat, {"total": 0, "overdue": 0, "req_yrs": req_yrs})
                sc["total"] += 1
                if is_overdue:
                    sc["overdue"] += 1
                    if lv is None:
                        never.append(entry)
                    else:
                        past_due.append(entry)
                else:
                    compliant.append(entry)

            total_structures = len(rows)
            total_overdue    = len(never) + len(past_due)
            total_compliant  = len(compliant)

            resp  = f"TIA INSPECTION COMPLIANCE — {DEMO_ORG_NAME}\n"
            resp += "=" * 60 + "\n\n"
            resp += f"PORTFOLIO SUMMARY ({total_structures} total structures):\n"
            resp += "-" * 50 + "\n"
            resp += f"  Compliant:        {total_compliant}\n"
            resp += f"  Overdue / Never:  {total_overdue}\n\n"

            if summary_counts:
                resp += "BY TOWER TYPE:\n"
                resp += "-" * 50 + "\n"
                rule_label = {"Guyed": "3-yr rule", "Monopole": "5-yr rule",
                              "Self-Support": "5-yr rule", "Other": "3-yr rule"}
                for cat in sorted(summary_counts):
                    sc = summary_counts[cat]
                    rl = rule_label.get(cat, "")
                    comp = sc["total"] - sc["overdue"]
                    resp += (f"  {cat} ({rl}): {sc['total']} total — "
                             f"{comp} compliant, {sc['overdue']} overdue\n")
                resp += "\n"

            if never:
                resp += f"NEVER INSPECTED ({len(never)} structure(s)):\n"
                resp += "-" * 50 + "\n"
                for r in never:
                    h_str    = f", {r['height']} ft" if r.get("height") else ""
                    site_label = r.get("site_name") or "[No site assigned]"
                    resp += (f"  • {site_label} — "
                             f"{r.get('structure_name','unnamed')} "
                             f"({r.get('tower_category','?')}{h_str})\n")
                resp += "\n"

            if past_due:
                resp += f"PAST DUE ({len(past_due)} structure(s)):\n"
                resp += "-" * 50 + "\n"
                for r in past_due:
                    h_str      = f", {r['height']} ft" if r.get("height") else ""
                    yrs        = r.get("years_since")
                    req        = r.get("required_years")
                    yrs_str    = f" — {yrs} yrs since last visit ({req}-yr rule)" if yrs else ""
                    site_label = r.get("site_name") or "[No site assigned]"
                    resp += (f"  • {site_label} — "
                             f"{r.get('structure_name','unnamed')} "
                             f"({r.get('tower_category','?')}{h_str}){yrs_str}\n")

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 2. top_trending_deficiencies
        # ------------------------------------------------------------------
        elif name == "top_trending_deficiencies":
            # Query 1: trending — exact query from confirmed full-server run
            q_trends = """
SELECT
    name,
    "deficiencyCode",
    COUNT(*) as total_frequency,
    COUNT(CASE WHEN "createdAt" >= CURRENT_DATE - INTERVAL '90 days' THEN 1 END) as recent_90_days,
    COUNT(CASE WHEN "createdAt" >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as recent_30_days,
    COUNT(CASE WHEN severity = '1' THEN 1 END) as severe_count,
    COUNT(CASE WHEN severity = '2' THEN 1 END) as moderate_count,
    COUNT(CASE WHEN severity = '3' THEN 1 END) as minor_count,
    CAST(
        (COUNT(CASE WHEN "createdAt" >= CURRENT_DATE - INTERVAL '90 days' THEN 1 END)::float /
         GREATEST(COUNT(*), 1)) * 100 AS DECIMAL(5,1)
    ) as recent_trend_pct,
    MIN("createdAt") as first_seen,
    MAX("createdAt") as most_recent
FROM deficiency
WHERE "organizationId" = %s
GROUP BY name, "deficiencyCode"
HAVING COUNT(*) >= 3
ORDER BY recent_90_days DESC, total_frequency DESC
LIMIT 15
"""
            # Query 2: overall + recent-90-day summary (UNION ALL, 2 params)
            q_summary = """
SELECT
    'Overall' as period,
    COUNT(*) as total_deficiencies,
    COUNT(CASE WHEN "createdAt" >= CURRENT_DATE - INTERVAL '90 days' THEN 1 END) as recent_90_days,
    COUNT(CASE WHEN "createdAt" >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as recent_30_days,
    COUNT(CASE WHEN severity = '1' THEN 1 END) as total_severe,
    COUNT(CASE WHEN severity = '2' THEN 1 END) as total_moderate,
    COUNT(CASE WHEN severity = '3' THEN 1 END) as total_minor
FROM deficiency
WHERE "organizationId" = %s

UNION ALL

SELECT
    'Last 90 days' as period,
    COUNT(*) as total_deficiencies,
    COUNT(*) as recent_90_days,
    COUNT(CASE WHEN "createdAt" >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) as recent_30_days,
    COUNT(CASE WHEN severity = '1' THEN 1 END) as total_severe,
    COUNT(CASE WHEN severity = '2' THEN 1 END) as total_moderate,
    COUNT(CASE WHEN severity = '3' THEN 1 END) as total_minor
FROM deficiency
WHERE "organizationId" = %s
  AND "createdAt" >= CURRENT_DATE - INTERVAL '90 days'
"""
            trends  = db.execute_query(q_trends,  (DEMO_ORG_ID,))
            summary = db.execute_query(q_summary, (DEMO_ORG_ID, DEMO_ORG_ID))

            resp  = f"TOP TRENDING DEFICIENCIES — {DEMO_ORG_NAME}\n"
            resp += "=" * 60 + "\n\n"

            if not trends:
                resp += "No deficiency data found.\n"
                return [TextContent(type="text", text=resp)]

            # Summary header
            overall = next((r for r in summary if r.get("period") == "Overall"), None)
            recent  = next((r for r in summary if r.get("period") == "Last 90 days"), None)
            if overall:
                resp += f"PORTFOLIO TOTALS:\n"
                resp += f"  Total deficiencies (all time): {overall.get('total_deficiencies', 0)}\n"
                resp += f"  Last 90 days: {overall.get('recent_90_days', 0)}  |  Last 30 days: {overall.get('recent_30_days', 0)}\n"
                resp += (f"  Severity breakdown: {overall.get('total_severe',0)} critical  "
                         f"/ {overall.get('total_moderate',0)} moderate  "
                         f"/ {overall.get('total_minor',0)} minor\n")
            if recent:
                resp += (f"  Last-90-day severity: {recent.get('total_severe',0)} critical  "
                         f"/ {recent.get('total_moderate',0)} moderate  "
                         f"/ {recent.get('total_minor',0)} minor\n")
            resp += "\n"

            resp += "TOP TRENDING DEFICIENCY TYPES (min 3 occurrences, sorted by recent activity):\n"
            resp += "-" * 60 + "\n"
            labels = {"1": "Critical", "2": "Moderate", "3": "Minor"}
            for i, r in enumerate(trends, 1):
                name_val   = r.get("name") or "(unnamed)"
                code       = f" ({r['deficiencyCode']})" if r.get("deficiencyCode") else ""
                total      = r.get("total_frequency", 0)
                r90        = r.get("recent_90_days", 0)
                r30        = r.get("recent_30_days", 0)
                trend_pct  = r.get("recent_trend_pct") or 0
                sev1       = r.get("severe_count", 0)
                sev2       = r.get("moderate_count", 0)
                sev3       = r.get("minor_count", 0)
                most_rec   = r.get("most_recent")
                rec_str    = most_rec.strftime("%b %d, %Y") if most_rec else ""
                sev_detail = []
                if sev1: sev_detail.append(f"{sev1} critical")
                if sev2: sev_detail.append(f"{sev2} moderate")
                if sev3: sev_detail.append(f"{sev3} minor")
                sev_str = f"  [{', '.join(sev_detail)}]" if sev_detail else ""
                resp += (f"  {i:2}. {name_val}{code}\n"
                         f"      Total: {total}  |  Last 90 days: {r90}  |  Last 30 days: {r30}  "
                         f"({trend_pct}% recent){sev_str}"
                         f"{f'  Last seen: {rec_str}' if rec_str else ''}\n")

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 3. deficiency_severity_breakdown
        # ------------------------------------------------------------------
        elif name == "deficiency_severity_breakdown":
            # Severity distribution — direct org filter, severity stored as string
            query_sev = """
SELECT
    severity,
    COUNT(*) as deficiency_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM deficiency
WHERE "organizationId" = %s
GROUP BY severity
ORDER BY
    CASE severity
        WHEN '1' THEN 1
        WHEN '2' THEN 2
        WHEN '3' THEN 3
        ELSE 4
    END
"""
            # Top sites by deficiency count — join through site_visit (confirmed path)
            query_sites = """
SELECT s.name as site_name, s."internalId" as site_code, COUNT(d.id) as deficiency_count
FROM deficiency d
JOIN survey sv ON d."surveyId" = sv.id
JOIN site_visit svi ON sv."siteVisitId" = svi.id
JOIN site s ON svi."siteId" = s.id
WHERE d."organizationId" = %s
GROUP BY s.id, s.name, s."internalId"
ORDER BY deficiency_count DESC
LIMIT 10
"""
            sev_results  = db.execute_query(query_sev,   (DEMO_ORG_ID,))
            site_results = db.execute_query(query_sites, (DEMO_ORG_ID,))

            resp  = f"DEFICIENCY SEVERITY BREAKDOWN — {DEMO_ORG_NAME}\n"
            resp += "=" * 60 + "\n\n"

            if sev_results:
                resp += "BY SEVERITY LEVEL:\n"
                resp += "-" * 40 + "\n"
                sev_labels = {"1": "Critical", "2": "Moderate", "3": "Minor"}
                total = sum(r.get("deficiency_count", 0) for r in sev_results)
                for r in sev_results:
                    sev   = str(r.get("severity") or "")
                    count = r.get("deficiency_count", 0)
                    pct   = r.get("percentage") or 0
                    label = sev_labels.get(sev, "Other")
                    resp += f"  Severity {sev} ({label}): {count} deficiencies ({pct}%)\n"
                resp += f"\n  Total deficiencies: {total}\n\n"
            else:
                resp += "No deficiency data found.\n\n"

            if site_results:
                resp += "TOP 10 SITES BY DEFICIENCY COUNT:\n"
                resp += "-" * 40 + "\n"
                for i, r in enumerate(site_results, 1):
                    sname = r.get("site_name") or "Unknown"
                    code = f" [{r['site_code']}]" if r.get("site_code") else ""
                    count = r.get("deficiency_count", 0)
                    resp += f"  {i:2}. {sname}{code} — {count} deficiencies\n"

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 4. deficiencies_by_state
        # ------------------------------------------------------------------
        elif name == "deficiencies_by_state":
            # Coordinate-based state determination — confirmed working query from full server
            # site.address fields are sparse; PostGIS location is the reliable source
            q_state = """
WITH site_coordinates AS (
    SELECT
        s.id  AS site_id,
        ST_Y(s.location) AS latitude,
        ST_X(s.location) AS longitude
    FROM site s
    WHERE s."organizationId" = %s
      AND s.location IS NOT NULL
),
site_states AS (
    SELECT
        site_id,
        CASE
            WHEN latitude BETWEEN 44.3  AND 49.0  AND longitude BETWEEN -116.1 AND -104.0 THEN 'Montana'
            WHEN latitude BETWEEN 33.8  AND 36.6  AND longitude BETWEEN -84.3  AND -75.4  THEN 'North Carolina'
            WHEN latitude BETWEEN 37.8  AND 41.8  AND longitude BETWEEN -88.1  AND -84.8  THEN 'Indiana'
            WHEN latitude BETWEEN 30.2  AND 35.0  AND longitude BETWEEN -88.5  AND -84.9  THEN 'Alabama'
            WHEN latitude BETWEEN 43.5  AND 49.4  AND longitude BETWEEN -97.2  AND -89.5  THEN 'Minnesota'
            WHEN latitude BETWEEN 42.5  AND 47.3  AND longitude BETWEEN -92.9  AND -86.8  THEN 'Wisconsin'
            WHEN latitude BETWEEN 36.9  AND 42.5  AND longitude BETWEEN -91.5  AND -87.0  THEN 'Illinois'
            WHEN latitude BETWEEN 38.4  AND 42.0  AND longitude BETWEEN -84.8  AND -80.5  THEN 'Ohio'
            WHEN latitude BETWEEN 41.7  AND 48.3  AND longitude BETWEEN -90.4  AND -82.1  THEN 'Michigan'
            WHEN latitude BETWEEN 25.8  AND 36.5  AND longitude BETWEEN -106.6 AND -93.5  THEN 'Texas'
            WHEN latitude BETWEEN 40.4  AND 43.5  AND longitude BETWEEN -96.6  AND -90.1  THEN 'Iowa'
            WHEN latitude BETWEEN 40.5  AND 45.0  AND longitude BETWEEN -79.8  AND -71.9  THEN 'New York'
            WHEN latitude BETWEEN 41.9  AND 46.3  AND longitude BETWEEN -124.6 AND -116.5 THEN 'Oregon'
            WHEN latitude BETWEEN 39.7  AND 42.5  AND longitude BETWEEN -80.5  AND -74.7  THEN 'Pennsylvania'
            WHEN latitude BETWEEN 42.7  AND 45.0  AND longitude BETWEEN -73.4  AND -71.5  THEN 'Vermont'
            WHEN latitude BETWEEN 42.7  AND 45.3  AND longitude BETWEEN -72.6  AND -70.6  THEN 'New Hampshire'
            WHEN latitude BETWEEN 41.2  AND 42.9  AND longitude BETWEEN -73.5  AND -69.9  THEN 'Massachusetts'
            WHEN latitude BETWEEN 40.9  AND 42.1  AND longitude BETWEEN -73.7  AND -71.8  THEN 'Connecticut'
            ELSE 'Other/Unknown'
        END AS state
    FROM site_coordinates
)
SELECT
    ss.state,
    COUNT(d.id) AS deficiency_count,
    ROUND(COUNT(d.id) * 100.0 / SUM(COUNT(d.id)) OVER(), 1) AS percentage,
    COUNT(CASE WHEN d.severity = '1' THEN 1 END) AS critical_count,
    COUNT(CASE WHEN d.severity = '2' THEN 1 END) AS moderate_count,
    COUNT(DISTINCT ss.site_id) AS sites_affected
FROM deficiency d
INNER JOIN survey su  ON d."surveyId"      = su.id  AND su."organizationId"  = %s
INNER JOIN site_visit sv ON su."siteVisitId" = sv.id AND sv."organizationId" = %s
INNER JOIN site_states ss ON sv."siteId" = ss.site_id
WHERE d."organizationId" = %s
GROUP BY ss.state
ORDER BY deficiency_count DESC, ss.state
"""
            state_rows = db.execute_query(q_state, (DEMO_ORG_ID,) * 4)

            resp  = f"DEFICIENCIES BY STATE — {DEMO_ORG_NAME}\n"
            resp += "=" * 60 + "\n\n"

            if not state_rows:
                resp += "No deficiency data found.\n"
                return [TextContent(type="text", text=resp)]

            total = sum(r.get("deficiency_count", 0) for r in state_rows)
            known = [r for r in state_rows if r.get("state") != "Other/Unknown"]
            resp += f"Total: {total} deficiencies across {len(known)} identified state(s)\n\n"

            for r in state_rows:
                state = r.get("state") or "Unknown"
                count = r.get("deficiency_count", 0)
                pct   = r.get("percentage") or 0
                sites = r.get("sites_affected", 0)
                crit  = r.get("critical_count", 0)
                mod   = r.get("moderate_count", 0)
                sev_str = f"  [{crit} critical, {mod} moderate]" if (crit or mod) else ""
                resp += f"  {state}: {count} deficiencies ({pct}%) across {sites} site(s){sev_str}\n"

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 5. recent_severity1_deficiencies
        # ------------------------------------------------------------------
        elif name == "recent_severity1_deficiencies":
            # Query 1: recent severity-1 with correct join path (survey → site_visit → site)
            q_recent = """
SELECT
    d.name                  AS deficiency_name,
    d."deficiencyCode"      AS code,
    d.description,
    d."deficiencyLocation"  AS location,
    d."createdAt"           AS date_logged,
    s.name                  AS site_name,
    s."internalId"          AS site_code
FROM deficiency d
LEFT JOIN survey su      ON d."surveyId"       = su.id
LEFT JOIN site_visit svi ON su."siteVisitId"   = svi.id
LEFT JOIN site s         ON svi."siteId"       = s.id
WHERE d."organizationId" = %s
  AND d.severity = '1'
  AND d."createdAt" >= CURRENT_DATE - INTERVAL '6 months'
ORDER BY d."createdAt" DESC
"""
            # Query 2: all-time most affected sites (same join path)
            q_context = """
SELECT
    s.name          AS site_name,
    s."internalId"  AS site_code,
    COUNT(d.id)     AS sev1_count,
    MAX(d."createdAt") AS last_sev1
FROM deficiency d
LEFT JOIN survey sv      ON d."surveyId"       = sv.id
LEFT JOIN site_visit svi ON sv."siteVisitId"   = svi.id
LEFT JOIN site s         ON svi."siteId"       = s.id
WHERE d."organizationId" = %s
  AND d.severity = '1'
GROUP BY s.id, s.name, s."internalId"
ORDER BY sev1_count DESC
"""
            # Query 3: most common types — direct org filter, no join needed
            q_types = """
SELECT 
    "deficiencyCode",
    name, 
    COUNT(id) as count,
    ROUND(COUNT(id) * 100.0 / SUM(COUNT(id)) OVER(), 1) as percentage
FROM deficiency
WHERE "organizationId" = %s
  AND severity = '1'
  AND name IS NOT NULL
GROUP BY "deficiencyCode", name
ORDER BY count DESC
LIMIT 10
"""
            recent = db.execute_query(q_recent,  (DEMO_ORG_ID,))
            sites  = db.execute_query(q_context, (DEMO_ORG_ID,))
            types  = db.execute_query(q_types,   (DEMO_ORG_ID,))

            total_sev1 = sum(r.get("sev1_count", 0) for r in sites)

            resp  = f"SEVERITY-1 (CRITICAL) DEFICIENCIES — {DEMO_ORG_NAME}\n"
            resp += "=" * 60 + "\n\n"
            resp += f"All-time total severity-1 deficiencies: {total_sev1}\n"
            resp += f"Found in last 6 months: {len(recent)}\n\n"

            if not recent:
                resp += "No severity-1 deficiencies logged in the last 6 months.\n\n"
            else:
                resp += "RECENT SEVERITY-1 DEFICIENCIES (last 6 months):\n"
                resp += "-" * 50 + "\n"
                for r in recent:
                    dname    = r.get("deficiency_name") or "(unnamed)"
                    code     = f" ({r['code']})" if r.get("code") else ""
                    sname    = r.get("site_name") or "Unknown"
                    sc       = f" [{r['site_code']}]" if r.get("site_code") else ""
                    dv       = r.get("date_logged")
                    date_str = dv.strftime("%b %d, %Y") if dv else ""
                    desc     = r.get("description") or ""
                    desc_str = f"\n    {desc[:120]}" if desc else ""
                    resp += f"  • [{date_str}] {dname}{code} — {sname}{sc}{desc_str}\n"
                resp += "\n"

            if types:
                resp += "MOST COMMON SEVERITY-1 TYPES (all time):\n"
                resp += "-" * 50 + "\n"
                for i, r in enumerate(types, 1):
                    resp += f"  {i:2}. {r.get('name','?')} — {r['count']} occurrence(s)\n"
                resp += "\n"

            if sites:
                resp += "SITES WITH MOST SEVERITY-1 DEFICIENCIES:\n"
                resp += "-" * 50 + "\n"
                for r in sites:
                    sc = f" [{r['site_code']}]" if r.get("site_code") else ""
                    lv = r.get("last_sev1")
                    lv_str = f", last: {lv.strftime('%b %Y')}" if lv else ""
                    resp += f"  • {r.get('site_name','?')}{sc} — {r['sev1_count']} critical deficiencies{lv_str}\n"

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 6. shelter_summary
        # ------------------------------------------------------------------
        elif name == "shelter_summary":
            query = """
SELECT s.name as site_name, s."internalId" as site_code, s.address, COUNT(b.id) as shelter_count
FROM site s
LEFT JOIN compound c ON c."siteId" = s.id
LEFT JOIN bphocs b ON b."compoundId" = c.id
WHERE s."organizationId" = %s
GROUP BY s.id, s.name, s."internalId", s.address
ORDER BY shelter_count DESC, s.name
"""
            results = db.execute_query(query, (DEMO_ORG_ID,))

            resp  = f"SHELTER SUMMARY — {DEMO_ORG_NAME}\n"
            resp += "=" * 60 + "\n\n"

            if not results:
                resp += "No site data found.\n"
                return [TextContent(type="text", text=resp)]

            with_shelter = [r for r in results if r.get("shelter_count", 0) > 0]
            without_shelter = [r for r in results if r.get("shelter_count", 0) == 0]

            resp += f"Sites WITH shelters:    {len(with_shelter)}\n"
            resp += f"Sites WITHOUT shelters: {len(without_shelter)}\n"
            resp += f"Total sites:            {len(results)}\n\n"

            if with_shelter:
                resp += "SITES WITH SHELTERS:\n"
                resp += "-" * 40 + "\n"
                for r in with_shelter:
                    sname = r.get("site_name") or "Unknown"
                    code = f" [{r['site_code']}]" if r.get("site_code") else ""
                    count = r.get("shelter_count", 0)
                    resp += f"  {sname}{code} — {count} shelter(s)\n"
                resp += "\n"

            if without_shelter:
                resp += "SITES WITHOUT SHELTERS:\n"
                resp += "-" * 40 + "\n"
                for r in without_shelter[:25]:
                    sname = r.get("site_name") or "Unknown"
                    code = f" [{r['site_code']}]" if r.get("site_code") else ""
                    resp += f"  {sname}{code}\n"
                if len(without_shelter) > 25:
                    resp += f"  ... and {len(without_shelter) - 25} more\n"

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 7. generator_summary
        # ------------------------------------------------------------------
        elif name == "generator_summary":
            query = """
SELECT s.name as site_name, s."internalId" as site_code, s.address, COUNT(g.id) as generator_count
FROM site s
LEFT JOIN compound c ON c."siteId" = s.id
LEFT JOIN compound_general cg ON cg."compoundId" = c.id
LEFT JOIN generator g ON g."compoundGeneralId" = cg.id
WHERE s."organizationId" = %s
GROUP BY s.id, s.name, s."internalId", s.address
ORDER BY generator_count DESC, s.name
"""
            results = db.execute_query(query, (DEMO_ORG_ID,))

            resp  = f"GENERATOR SUMMARY — {DEMO_ORG_NAME}\n"
            resp += "=" * 60 + "\n\n"

            if not results:
                resp += "No site data found.\n"
                return [TextContent(type="text", text=resp)]

            with_gen = [r for r in results if r.get("generator_count", 0) > 0]
            without_gen = [r for r in results if r.get("generator_count", 0) == 0]

            resp += f"Sites WITH generators:    {len(with_gen)}\n"
            resp += f"Sites WITHOUT generators: {len(without_gen)}\n"
            resp += f"Total sites:              {len(results)}\n\n"

            if with_gen:
                resp += "SITES WITH GENERATORS:\n"
                resp += "-" * 40 + "\n"
                for r in with_gen:
                    sname = r.get("site_name") or "Unknown"
                    code = f" [{r['site_code']}]" if r.get("site_code") else ""
                    count = r.get("generator_count", 0)
                    resp += f"  {sname}{code} — {count} generator(s)\n"
                resp += "\n"

            if without_gen:
                resp += "SITES WITHOUT GENERATORS:\n"
                resp += "-" * 40 + "\n"
                for r in without_gen[:25]:
                    sname = r.get("site_name") or "Unknown"
                    code = f" [{r['site_code']}]" if r.get("site_code") else ""
                    resp += f"  {sname}{code}\n"
                if len(without_gen) > 25:
                    resp += f"  ... and {len(without_gen) - 25} more\n"

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 8a. lightning_protection_summary
        # ------------------------------------------------------------------
        elif name == "lightning_protection_summary":
            q_lp = """
SELECT
    s.name                                   AS site_name,
    s."internalId"                           AS site_code,
    st.name                                  AS structure_name,
    st."constructedHeight"                   AS height,
    lp."lightningProtectionType"             AS protection_type,
    lp."lightningProtectionLocation"         AS protection_location,
    lp."lightningRodApex"                    AS apex_height,
    lp."lightningProtectionCenterlineHeight" AS centerline_height,
    lp.elevation,
    lp.notes
FROM lightning_protection lp
JOIN appurtenance a ON lp."appurtenanceId" = a.id
JOIN structure st   ON a."structureId"     = st.id
JOIN site s         ON st."siteId"         = s.id
WHERE lp."organizationId" = %s
  AND s."organizationId"  = %s
ORDER BY s.name, st.name
"""
            rows = db.execute_query(q_lp, (DEMO_ORG_ID, DEMO_ORG_ID))

            resp  = f"LIGHTNING PROTECTION — {DEMO_ORG_NAME}\n"
            resp += "=" * 60 + "\n\n"

            if not rows:
                resp += "No lightning protection records found.\n"
                return [TextContent(type="text", text=resp)]

            with_rod     = [r for r in rows if r.get("protection_type") and r.get("protection_type").lower() != "none"]
            no_rod       = [r for r in rows if not r.get("protection_type") or r.get("protection_type").lower() == "none"]
            missing_apex = [r for r in rows if r.get("apex_height") is None and r in with_rod]
            sites_count  = len({r.get("site_name") for r in rows})

            resp += f"Total records:              {len(rows)} across {sites_count} site(s)\n"
            resp += f"With active protection:     {len(with_rod)}\n"
            resp += f"Listed as none/absent:      {len(no_rod)}\n"
            resp += f"Missing apex measurement:   {len(missing_apex)}\n\n"

            if with_rod:
                resp += "ACTIVE LIGHTNING PROTECTION:\n"
                resp += "-" * 50 + "\n"
                for r in with_rod:
                    sname  = r.get("site_name") or "Unknown"
                    sc     = f" [{r['site_code']}]" if r.get("site_code") else ""
                    stname = r.get("structure_name") or ""
                    h      = r.get("height")
                    apex   = r.get("apex_height")
                    cl     = r.get("centerline_height")
                    ptype  = r.get("protection_type") or ""
                    ploc   = r.get("protection_location") or ""
                    notes  = r.get("notes") or ""
                    h_str    = f", {h} ft" if h else ""
                    apex_str = f", rod apex: {apex} ft" if apex else " (no apex recorded)"
                    cl_str   = f", centerline: {cl} ft" if cl else ""
                    loc_str  = f", location: {ploc}" if ploc else ""
                    note_str = f"\n    Note: {notes[:120]}" if notes else ""
                    resp += f"  • {sname}{sc} — {stname}{h_str}: {ptype}{loc_str}{apex_str}{cl_str}{note_str}\n"
                resp += "\n"

            if no_rod:
                resp += "NO LIGHTNING PROTECTION (recorded as none):\n"
                resp += "-" * 50 + "\n"
                for r in no_rod:
                    sname  = r.get("site_name") or "Unknown"
                    sc     = f" [{r['site_code']}]" if r.get("site_code") else ""
                    stname = r.get("structure_name") or ""
                    h      = r.get("height")
                    h_str  = f", {h} ft" if h else ""
                    resp += f"  • {sname}{sc} — {stname}{h_str}\n"

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 8b. tower_lighting_compliance
        # ------------------------------------------------------------------
        elif name == "tower_lighting_compliance":
            # Query 1: all structures ≥200 ft — FAA lighting required at 200 ft+
            q_towers = """
SELECT
    s.name              AS site_name,
    s."internalId"      AS site_code,
    st.name             AS structure_name,
    st.type             AS structure_type,
    st."constructedHeight" AS height,
    st."lightingPresent"   AS lighting_present,
    st."topFlashHeadPresent" AS top_flash_head
FROM structure st
JOIN site s ON s.id = st."siteId"
WHERE st."organizationId" = %s
  AND s."organizationId"  = %s
  AND st."constructedHeight" >= 200
ORDER BY st."constructedHeight" DESC, s.name
"""
            # Query 2: actual light assembly fixtures
            q_la = """
SELECT
    s.name                 AS site_name,
    s."internalId"         AS site_code,
    st.name                AS structure_name,
    st."constructedHeight" AS height,
    la.name                AS light_name,
    la.elevation,
    la."numberOfLights"    AS num_lights,
    la.location,
    la."iceShield"         AS ice_shield,
    la.notes
FROM light_assembly la
JOIN appurtenance a ON la."appurtenanceId" = a.id
JOIN structure st   ON a."structureId"     = st.id
JOIN site s         ON st."siteId"         = s.id
WHERE s."organizationId" = %s
ORDER BY st."constructedHeight" DESC, s.name, la.elevation DESC
"""
            towers = db.execute_query(q_towers, (DEMO_ORG_ID, DEMO_ORG_ID))
            la_rows = db.execute_query(q_la, (DEMO_ORG_ID,))

            resp  = f"TOWER LIGHTING COMPLIANCE — {DEMO_ORG_NAME}\n"
            resp += "=" * 60 + "\n\n"

            compliant     = [r for r in towers if r.get("lighting_present")]
            non_compliant = [r for r in towers if not r.get("lighting_present")]

            resp += f"Towers ≥200 ft total:      {len(towers)}\n"
            resp += f"  Lighting present:        {len(compliant)}\n"
            resp += f"  NO lighting (⚠ flag):   {len(non_compliant)}\n"
            resp += f"Light assembly fixtures:   {len(la_rows)}\n\n"

            if non_compliant:
                resp += "⚠ NON-COMPLIANT — TOWERS ≥200 FT WITHOUT LIGHTING:\n"
                resp += "-" * 50 + "\n"
                for r in non_compliant:
                    sname  = r.get("site_name") or "Unknown"
                    sc     = f" [{r['site_code']}]" if r.get("site_code") else ""
                    stname = r.get("structure_name") or ""
                    h      = r.get("height")
                    stype  = r.get("structure_type") or ""
                    tfh    = r.get("top_flash_head")
                    tfh_str = ", top flash head present" if tfh else ", NO top flash head"
                    resp += f"  • {sname}{sc} — {stname} ({stype}, {h} ft){tfh_str}\n"
                resp += "\n"

            if compliant:
                resp += "COMPLIANT — TOWERS ≥200 FT WITH LIGHTING:\n"
                resp += "-" * 50 + "\n"
                for r in compliant:
                    sname  = r.get("site_name") or "Unknown"
                    sc     = f" [{r['site_code']}]" if r.get("site_code") else ""
                    stname = r.get("structure_name") or ""
                    h      = r.get("height")
                    tfh    = r.get("top_flash_head")
                    tfh_str = ", top flash head ✓" if tfh else ""
                    resp += f"  • {sname}{sc} — {stname} ({h} ft){tfh_str}\n"
                resp += "\n"

            if la_rows:
                resp += "LIGHT ASSEMBLY FIXTURES (all towers):\n"
                resp += "-" * 50 + "\n"
                for r in la_rows:
                    sname   = r.get("site_name") or "Unknown"
                    sc      = f" [{r['site_code']}]" if r.get("site_code") else ""
                    stname  = r.get("structure_name") or ""
                    lname   = r.get("light_name") or "(unnamed)"
                    elev    = r.get("elevation")
                    nlights = r.get("num_lights")
                    h       = r.get("height")
                    elev_str = f" @ {elev} ft" if elev else ""
                    n_str    = f", {nlights} fixture(s)" if nlights else ""
                    h_str    = f", tower {h} ft" if h else ""
                    resp += f"  • {sname}{sc} — {stname}{h_str}: {lname}{elev_str}{n_str}\n"
            else:
                resp += "LIGHT ASSEMBLY FIXTURES: none recorded.\n"

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 9. montana_equipment_breakdown
        # ------------------------------------------------------------------
        elif name == "montana_equipment_breakdown":
            # Query A — Structures
            query_a = """
SELECT st.name as structure_name, st.type, st."constructedHeight" as height,
       st."internalId" as structure_code, st."fccAsrNumber" as fcc
FROM structure st
WHERE st."siteId" = %s
ORDER BY st.name
"""
            # Query B — Appurtenances with mount_centerline antenna data
            query_b = """
SELECT
    a.name as appurtenance_name,
    mc."antennaType", mc."antennaTypeCount", mc."antennaTypeDescription",
    mc."mountType", mc.elevation, mc.owner as tenant
FROM appurtenance a
JOIN structure st ON a."structureId" = st.id
LEFT JOIN (
    SELECT "appurtenanceId", "antennaType", "antennaTypeCount",
           "antennaTypeDescription", "mountType", elevation, owner
    FROM mount_centerline
    WHERE "organizationId" = %s
) mc ON mc."appurtenanceId" = a.id
WHERE st."siteId" = %s
  AND a."organizationId" = %s
ORDER BY mc.elevation DESC NULLS LAST, a.name
"""
            # Query C — Other appurtenances (joined through structure to filter by site)
            query_c = """
SELECT oa.name, oa.description
FROM other_appurtenance oa
JOIN structure st ON oa."structureId" = st.id
WHERE st."siteId" = %s
  AND oa."organizationId" = %s
ORDER BY oa.name
LIMIT 50
"""
            structures = db.execute_query(query_a, (MONTANA_SITE_ID,))
            appurtenances = db.execute_query(query_b, (DEMO_ORG_ID, MONTANA_SITE_ID, DEMO_ORG_ID))
            try:
                other_equip = db.execute_query(query_c, (MONTANA_SITE_ID, DEMO_ORG_ID))
            except Exception:
                other_equip = []

            resp  = f"EQUIPMENT BREAKDOWN — Montana / Mt. Baldy Site\n"
            resp += "=" * 60 + "\n\n"

            resp += "STRUCTURES:\n"
            resp += "-" * 40 + "\n"
            if structures:
                for st in structures:
                    sname = st.get("structure_name") or "unnamed"
                    stype = st.get("type") or "Unknown"
                    height = st.get("height")
                    code = st.get("structure_code") or ""
                    fcc = st.get("fcc") or ""
                    h_str = f", {height} ft" if height else ""
                    code_str = f" [{code}]" if code else ""
                    fcc_str = f", FCC ASR: {fcc}" if fcc else ""
                    resp += f"  • {sname}{code_str} — {stype}{h_str}{fcc_str}\n"
            else:
                resp += "  No structures found.\n"
            resp += "\n"

            # Separate into those with antenna data and those without
            antenna_records = [a for a in appurtenances if a.get("antennaType")]
            other_appurt = [a for a in appurtenances if not a.get("antennaType")]

            resp += "ANTENNA EQUIPMENT (mount centerline records):\n"
            resp += "-" * 40 + "\n"
            if antenna_records:
                for a in antenna_records:
                    aname = a.get("appurtenance_name") or "unnamed"
                    atype = a.get("antennaType") or ""
                    acount = a.get("antennaTypeCount")
                    adesc = a.get("antennaTypeDescription") or ""
                    mount = a.get("mountType") or ""
                    elev = a.get("elevation")
                    tenant = a.get("tenant") or ""
                    count_str = f" x{acount}" if acount else ""
                    elev_str = f" @ {elev} ft" if elev else ""
                    mount_str = f", mount: {mount}" if mount else ""
                    tenant_str = f", tenant: {tenant}" if tenant else ""
                    adesc_str = f" ({adesc})" if adesc else ""
                    resp += f"  • {aname} — {atype}{count_str}{adesc_str}{elev_str}{mount_str}{tenant_str}\n"
            else:
                resp += "  No antenna records found.\n"
            resp += "\n"

            if other_appurt:
                resp += "OTHER APPURTENANCES:\n"
                resp += "-" * 40 + "\n"
                for a in other_appurt:
                    aname = a.get("appurtenance_name") or "unnamed"
                    atype = a.get("appurtenance_type") or ""
                    type_str = f" ({atype})" if atype else ""
                    resp += f"  • {aname}{type_str}\n"
                resp += "\n"

            resp += "OTHER EQUIPMENT / OBSERVATIONS:\n"
            resp += "-" * 40 + "\n"
            if other_equip:
                for o in other_equip:
                    oname = o.get("name") or "(unnamed)"
                    desc = o.get("description") or ""
                    desc_str = f" — {desc[:100]}" if desc else ""
                    resp += f"  • {oname}{desc_str}\n"
            else:
                resp += "  No other equipment records found.\n"

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 10. equipment_types_summary
        # ------------------------------------------------------------------
        elif name == "equipment_types_summary":
            # Appurtenance parent records by name
            query_appurt = """
SELECT a.name AS equipment_type,
       COUNT(a.id) AS total_count,
       COUNT(DISTINCT st."siteId") AS sites_count
FROM appurtenance a
JOIN structure st ON a."structureId" = st.id
JOIN site s       ON st."siteId"     = s.id
WHERE a."organizationId" = %s AND s."organizationId" = %s
  AND a.name IS NOT NULL
GROUP BY a.name ORDER BY total_count DESC
"""
            # Mount centerlines by mount type
            query_mounts = """
SELECT mc."mountType" AS mount_type,
       COUNT(mc.id) AS total_count,
       COUNT(DISTINCT st."siteId") AS sites_count
FROM mount_centerline mc
JOIN appurtenance a ON mc."appurtenanceId" = a.id
JOIN structure st   ON a."structureId"     = st.id
JOIN site s         ON st."siteId"         = s.id
WHERE mc."organizationId" = %s AND s."organizationId" = %s
GROUP BY mc."mountType" ORDER BY total_count DESC
"""
            # Antenna equipment by type
            query_antenna = """
SELECT mc."antennaType" AS antenna_type,
       COUNT(mc.id) AS total_records,
       SUM(mc."antennaTypeCount") AS total_antennas,
       COUNT(DISTINCT st."siteId") AS sites_count
FROM mount_centerline mc
JOIN appurtenance a ON mc."appurtenanceId" = a.id
JOIN structure st   ON a."structureId"     = st.id
JOIN site s         ON st."siteId"         = s.id
WHERE mc."organizationId" = %s AND s."organizationId" = %s
  AND mc."antennaType" IS NOT NULL
GROUP BY mc."antennaType" ORDER BY total_records DESC
"""
            # Lightning protection count
            query_lp = """
SELECT COUNT(lp.id) AS total,
       COUNT(DISTINCT st."siteId") AS sites_count
FROM lightning_protection lp
JOIN appurtenance a ON lp."appurtenanceId" = a.id
JOIN structure st   ON a."structureId"     = st.id
JOIN site s         ON st."siteId"         = s.id
WHERE lp."organizationId" = %s AND s."organizationId" = %s
"""
            # Light assembly count
            query_la = """
SELECT COUNT(la.id) AS total,
       COUNT(DISTINCT st."siteId") AS sites_count
FROM light_assembly la
JOIN appurtenance a ON la."appurtenanceId" = a.id
JOIN structure st   ON a."structureId"     = st.id
JOIN site s         ON st."siteId"         = s.id
WHERE s."organizationId" = %s
"""
            # AM skirt count
            query_sk = """
SELECT COUNT(sk.id) AS total,
       COUNT(DISTINCT st."siteId") AS sites_count
FROM am_skirt sk
JOIN appurtenance a ON sk."appurtenanceId" = a.id
JOIN structure st   ON a."structureId"     = st.id
JOIN site s         ON st."siteId"         = s.id
WHERE s."organizationId" = %s
"""
            # Other appurtenance by name
            query_other = """
SELECT oa.name, COUNT(oa.id) AS count
FROM other_appurtenance oa
WHERE oa."organizationId" = %s AND oa.name IS NOT NULL
GROUP BY oa.name ORDER BY count DESC LIMIT 15
"""
            appurt_rows  = db.execute_query(query_appurt,  (DEMO_ORG_ID, DEMO_ORG_ID))
            mount_rows   = db.execute_query(query_mounts,  (DEMO_ORG_ID, DEMO_ORG_ID))
            antenna_rows = db.execute_query(query_antenna, (DEMO_ORG_ID, DEMO_ORG_ID))
            lp_rows      = db.execute_query(query_lp,      (DEMO_ORG_ID, DEMO_ORG_ID))
            la_rows      = db.execute_query(query_la,      (DEMO_ORG_ID,))
            try:
                sk_rows  = db.execute_query(query_sk,      (DEMO_ORG_ID,))
            except Exception:
                sk_rows  = []
            other_rows   = db.execute_query(query_other,   (DEMO_ORG_ID,))

            resp  = f"EQUIPMENT CATALOGUE — {DEMO_ORG_NAME}\n"
            resp += "=" * 60 + "\n\n"

            # --- Appurtenance groups ---
            if appurt_rows:
                total = sum(r.get("total_count", 0) for r in appurt_rows)
                resp += f"APPURTENANCE GROUPS ({total} total, {len(appurt_rows)} categories):\n"
                resp += "-" * 40 + "\n"
                for r in appurt_rows:
                    resp += f"  • {r.get('equipment_type','?')}: {r['total_count']} record(s) across {r['sites_count']} site(s)\n"
                resp += "\n"

            # --- Mount centerlines ---
            if mount_rows:
                total = sum(r.get("total_count", 0) for r in mount_rows)
                resp += f"MOUNT CENTERLINES ({total} total):\n"
                resp += "-" * 40 + "\n"
                for r in mount_rows:
                    mtype = r.get("mount_type") or "(unspecified)"
                    resp += f"  • {mtype}: {r['total_count']} mount(s) across {r['sites_count']} site(s)\n"
                resp += "\n"

            # --- Antenna equipment ---
            if antenna_rows:
                total_rec = sum(r.get("total_records", 0) for r in antenna_rows)
                total_ant = sum(r.get("total_antennas") or 0 for r in antenna_rows)
                resp += f"ANTENNA EQUIPMENT ({total_rec} records, {total_ant} total antennas, {len(antenna_rows)} type(s)):\n"
                resp += "-" * 40 + "\n"
                for r in antenna_rows:
                    atype = r.get("antenna_type") or "(unspecified)"
                    ants  = r.get("total_antennas") or 0
                    resp += f"  • {atype}: {r['total_records']} record(s), {ants} antenna(s) across {r['sites_count']} site(s)\n"
                resp += "\n"

            # --- Lightning protection ---
            lp_total = lp_rows[0].get("total", 0) if lp_rows else 0
            lp_sites = lp_rows[0].get("sites_count", 0) if lp_rows else 0
            resp += f"LIGHTNING PROTECTION: {lp_total} record(s) across {lp_sites} site(s)\n"

            # --- Light assemblies ---
            la_total = la_rows[0].get("total", 0) if la_rows else 0
            la_sites = la_rows[0].get("sites_count", 0) if la_rows else 0
            resp += f"LIGHT ASSEMBLIES:     {la_total} record(s) across {la_sites} site(s)\n"

            # --- AM skirts ---
            sk_total = sk_rows[0].get("total", 0) if sk_rows else 0
            sk_sites = sk_rows[0].get("sites_count", 0) if sk_rows else 0
            resp += f"AM SKIRTS:            {sk_total} record(s) across {sk_sites} site(s)\n\n"

            # --- Other appurtenances ---
            if other_rows:
                total_oa = sum(r.get("count", 0) for r in other_rows)
                resp += f"OTHER EQUIPMENT ({total_oa} total named records):\n"
                resp += "-" * 40 + "\n"
                for r in other_rows:
                    resp += f"  • {r.get('name','?')}: {r['count']} record(s)\n"

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 11. antenna_equipment_count
        # ------------------------------------------------------------------
        elif name == "antenna_equipment_count":
            query = """
SELECT
    mc."antennaType",
    COUNT(mc.id) as total_records,
    SUM(mc."antennaTypeCount") as total_antennas,
    COUNT(DISTINCT st."siteId") as sites
FROM mount_centerline mc
JOIN appurtenance a  ON mc."appurtenanceId" = a.id
JOIN structure st    ON a."structureId"     = st.id
JOIN site s          ON st."siteId"         = s.id
WHERE mc."organizationId" = %s
  AND s."organizationId"  = %s
  AND mc."antennaType" IS NOT NULL
GROUP BY mc."antennaType"
ORDER BY total_records DESC
"""
            results = db.execute_query(query, (DEMO_ORG_ID, DEMO_ORG_ID))

            resp  = f"ANTENNA EQUIPMENT COUNT — {DEMO_ORG_NAME}\n"
            resp += "=" * 60 + "\n\n"

            if not results:
                resp += "No antenna equipment records found.\n"
                return [TextContent(type="text", text=resp)]

            total_records = sum(r.get("total_records", 0) for r in results)
            total_antennas = sum(r.get("total_antennas") or 0 for r in results)
            resp += f"Total antenna records: {total_records}\n"
            resp += f"Total antenna count:   {total_antennas}\n"
            resp += f"Antenna types:         {len(results)}\n\n"

            resp += "BY ANTENNA TYPE:\n"
            resp += "-" * 40 + "\n"
            for r in results:
                atype = r.get("antennaType") or "Unknown"
                records = r.get("total_records", 0)
                antennas = r.get("total_antennas") or 0
                sites = r.get("sites", 0)
                resp += f"  • {atype}: {records} record(s), {antennas} antenna(s) across {sites} site(s)\n"

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 12. appurtenances_by_site
        # ------------------------------------------------------------------
        elif name == "appurtenances_by_site":
            # One row per appurtenance, with child-equipment counts derived
            # from the specialized tables that hang off it.
            # Chain: appurtenance → mount_centerline → antenna_equipment
            #        appurtenance → lightning_protection
            #        appurtenance → light_assembly
            #        appurtenance → am_skirt
            #        appurtenance → other_appurtenance
            query = """
SELECT
    s.name                          AS site_name,
    s."internalId"                  AS site_code,
    st.name                         AS structure_name,
    a.name                          AS appurtenance_name,
    COUNT(DISTINCT mc.id)           AS mount_count,
    COUNT(DISTINCT ae.id)           AS antenna_count,
    COUNT(DISTINCT lp.id)           AS lightning_count,
    COUNT(DISTINCT la.id)           AS light_assembly_count,
    COUNT(DISTINCT sk.id)           AS am_skirt_count,
    COUNT(DISTINCT oa.id)           AS other_count
FROM appurtenance a
JOIN structure st               ON st.id            = a."structureId"
JOIN site s                     ON s.id             = st."siteId"
LEFT JOIN mount_centerline mc   ON mc."appurtenanceId" = a.id
LEFT JOIN antenna_equipment ae  ON ae."mountCenterlineId" = mc.id
LEFT JOIN lightning_protection lp ON lp."appurtenanceId" = a.id
LEFT JOIN light_assembly la     ON la."appurtenanceId" = a.id
LEFT JOIN am_skirt sk            ON sk."appurtenanceId" = a.id
LEFT JOIN other_appurtenance oa  ON oa."appurtenanceId" = a.id
WHERE a."organizationId" = %s
  AND s."organizationId" = %s
GROUP BY s.id, s.name, s."internalId", st.name, a.id, a.name
ORDER BY s.name, st.name, a.name
"""
            results = db.execute_query(query, (DEMO_ORG_ID, DEMO_ORG_ID))

            resp  = f"APPURTENANCES BY SITE — {DEMO_ORG_NAME}\n"
            resp += "=" * 60 + "\n\n"

            if not results:
                resp += "No appurtenance data found.\n"
                return [TextContent(type="text", text=resp)]

            # Group rows by site for readable output
            from collections import defaultdict as _dd
            by_site = _dd(list)
            for r in results:
                by_site[r.get("site_name") or "Unknown"].append(r)

            total_appurt   = len(results)
            total_antennas = sum(r.get("antenna_count", 0) for r in results)
            total_mounts   = sum(r.get("mount_count", 0) for r in results)
            total_lp       = sum(r.get("lightning_count", 0) for r in results)
            total_la       = sum(r.get("light_assembly_count", 0) for r in results)
            total_sk       = sum(r.get("am_skirt_count", 0) for r in results)
            total_oa       = sum(r.get("other_count", 0) for r in results)

            resp += f"Total appurtenances:   {total_appurt} across {len(by_site)} site(s)\n"
            resp += f"  Mount centerlines:   {total_mounts}\n"
            resp += f"  Antenna equipment:   {total_antennas}\n"
            resp += f"  Lightning protection:{total_lp}\n"
            resp += f"  Light assemblies:    {total_la}\n"
            resp += f"  AM skirts:           {total_sk}\n"
            resp += f"  Other appurtenances: {total_oa}\n\n"

            for site_name, rows in sorted(by_site.items()):
                site_code = rows[0].get("site_code") or ""
                code_str  = f" [{site_code}]" if site_code else ""
                resp += f"SITE: {site_name}{code_str}\n"
                resp += "-" * 50 + "\n"

                for r in rows:
                    aname   = r.get("appurtenance_name") or "(unnamed)"
                    stname  = r.get("structure_name") or ""
                    struct_str = f" on {stname}" if stname else ""

                    # Determine category from which child tables have records
                    cats = []
                    if r.get("mount_count", 0):
                        cats.append(f"Mount×{r['mount_count']}")
                    if r.get("antenna_count", 0):
                        cats.append(f"Antenna×{r['antenna_count']}")
                    if r.get("lightning_count", 0):
                        cats.append(f"Lightning Prot.×{r['lightning_count']}")
                    if r.get("light_assembly_count", 0):
                        cats.append(f"Light Assembly×{r['light_assembly_count']}")
                    if r.get("am_skirt_count", 0):
                        cats.append(f"AM Skirt×{r['am_skirt_count']}")
                    if r.get("other_count", 0):
                        cats.append(f"Other×{r['other_count']}")

                    cat_str = " [" + ", ".join(cats) + "]" if cats else " [no child records]"
                    resp += f"  • {aname}{struct_str}{cat_str}\n"

                resp += "\n"

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 13. experiment_farm_equipment_breakdown
        # ------------------------------------------------------------------
        elif name == "experiment_farm_equipment_breakdown":
            # Resolve the Experiment Farm site ID by name
            q_site = """
SELECT s.id AS site_id, s.name AS site_name, s."internalId" AS site_code, s.address
FROM site s
WHERE s."organizationId" = %s
  AND s.name = %s
LIMIT 1
"""
            site_rows = db.execute_query(q_site, (DEMO_ORG_ID, "1450 Experiment Farm Rd"))
            if not site_rows:
                return [TextContent(type="text", text="Experiment Farm site not found.")]
            site_id   = site_rows[0]["site_id"]
            site_name = site_rows[0]["site_name"]
            site_code = site_rows[0].get("site_code") or ""
            address   = site_rows[0].get("address") or ""

            # Structures
            q_structs = """
SELECT st.name, st.type, st."constructedHeight" AS height,
       st."internalId" AS code, st."fccAsrNumber" AS fcc,
       st."lightingPresent" AS lighting, st."topFlashHeadPresent" AS top_flash
FROM structure st WHERE st."siteId" = %s ORDER BY st.name
"""
            # Appurtenance groups
            q_appurt = """
SELECT a.name, COUNT(a.id) AS count
FROM appurtenance a
JOIN structure st ON a."structureId" = st.id
WHERE st."siteId" = %s AND a."organizationId" = %s
GROUP BY a.name ORDER BY count DESC
"""
            # Mount centerlines by type
            q_mounts = """
SELECT mc."mountType" AS mount_type, COUNT(mc.id) AS count
FROM mount_centerline mc
JOIN appurtenance a ON mc."appurtenanceId" = a.id
JOIN structure st   ON a."structureId"     = st.id
WHERE st."siteId" = %s AND mc."organizationId" = %s
GROUP BY mc."mountType" ORDER BY count DESC
"""
            # Antenna equipment by type
            q_antenna = """
SELECT mc."antennaType" AS antenna_type,
       COUNT(mc.id) AS records,
       SUM(mc."antennaTypeCount") AS total_antennas
FROM mount_centerline mc
JOIN appurtenance a ON mc."appurtenanceId" = a.id
JOIN structure st   ON a."structureId"     = st.id
WHERE st."siteId" = %s AND mc."organizationId" = %s
  AND mc."antennaType" IS NOT NULL
GROUP BY mc."antennaType" ORDER BY records DESC
"""
            # Lightning protection
            q_lp = """
SELECT lp."lightningProtectionType" AS ptype,
       lp."lightningRodApex" AS apex, lp.notes,
       st.name AS structure_name
FROM lightning_protection lp
JOIN appurtenance a ON lp."appurtenanceId" = a.id
JOIN structure st   ON a."structureId"     = st.id
WHERE st."siteId" = %s AND lp."organizationId" = %s
ORDER BY st.name
"""
            # Light assemblies
            q_la = """
SELECT la.name AS light_name, la.elevation,
       la."numberOfLights" AS num_lights, la.location,
       st.name AS structure_name
FROM light_assembly la
JOIN appurtenance a ON la."appurtenanceId" = a.id
JOIN structure st   ON a."structureId"     = st.id
WHERE st."siteId" = %s
ORDER BY la.elevation DESC
"""
            # AM skirts
            q_sk = """
SELECT sk.id, st.name AS structure_name
FROM am_skirt sk
JOIN appurtenance a ON sk."appurtenanceId" = a.id
JOIN structure st   ON a."structureId"     = st.id
WHERE st."siteId" = %s
"""
            # Other appurtenances
            q_other = """
SELECT oa.name, oa.description
FROM other_appurtenance oa
WHERE oa."organizationId" = %s
  AND oa.name IS NOT NULL
ORDER BY oa.name LIMIT 30
"""

            structs  = db.execute_query(q_structs, (site_id,))
            appurts  = db.execute_query(q_appurt,  (site_id, DEMO_ORG_ID))
            mounts   = db.execute_query(q_mounts,  (site_id, DEMO_ORG_ID))
            antennas = db.execute_query(q_antenna, (site_id, DEMO_ORG_ID))
            lp_recs  = db.execute_query(q_lp,      (site_id, DEMO_ORG_ID))
            la_recs  = db.execute_query(q_la,      (site_id,))
            try:
                sk_recs = db.execute_query(q_sk,   (site_id,))
            except Exception:
                sk_recs = []
            other    = db.execute_query(q_other,   (DEMO_ORG_ID,))

            code_str = f" [{site_code}]" if site_code else ""
            addr_str = f"\n  Address: {address}" if address else ""
            resp  = f"EQUIPMENT BREAKDOWN — {site_name}{code_str}{addr_str}\n"
            resp += "=" * 60 + "\n\n"

            # Totals banner
            total_mounts   = sum(r.get("count", 0) for r in mounts)
            total_ant_recs = sum(r.get("records", 0) for r in antennas)
            total_ants     = sum(r.get("total_antennas") or 0 for r in antennas)
            resp += f"TOTALS:\n"
            resp += f"  Structures:          {len(structs)}\n"
            resp += f"  Appurtenance groups: {sum(r.get('count',0) for r in appurts)}\n"
            resp += f"  Mount centerlines:   {total_mounts}\n"
            resp += f"  Antenna records:     {total_ant_recs} ({total_ants} antennas)\n"
            resp += f"  Lightning protection:{len(lp_recs)}\n"
            resp += f"  Light assemblies:    {len(la_recs)}\n"
            resp += f"  AM skirts:           {len(sk_recs)}\n"
            resp += f"  Other equipment:     {len(other)}\n\n"

            # Structures
            resp += "STRUCTURES:\n"
            resp += "-" * 40 + "\n"
            for r in structs:
                h     = r.get("height")
                stype = r.get("type") or ""
                code  = r.get("code") or ""
                fcc   = r.get("fcc") or ""
                lit   = " [lighting ✓]" if r.get("lighting") else ""
                h_str   = f", {h} ft" if h else ""
                c_str   = f" [{code}]" if code else ""
                fcc_str = f", FCC: {fcc}" if fcc else ""
                resp += f"  • {r.get('name','?')}{c_str} — {stype}{h_str}{fcc_str}{lit}\n"
            resp += "\n"

            # Appurtenance groups
            if appurts:
                resp += "APPURTENANCE GROUPS:\n"
                resp += "-" * 40 + "\n"
                for r in appurts:
                    resp += f"  • {r.get('name','?')}: {r['count']} record(s)\n"
                resp += "\n"

            # Mount centerlines
            if mounts:
                resp += f"MOUNT CENTERLINES ({total_mounts} total):\n"
                resp += "-" * 40 + "\n"
                for r in mounts:
                    mtype = r.get("mount_type") or "(unspecified)"
                    resp += f"  • {mtype}: {r['count']} mount(s)\n"
                resp += "\n"

            # Antenna equipment
            if antennas:
                resp += f"ANTENNA EQUIPMENT ({total_ant_recs} records, {total_ants} antennas):\n"
                resp += "-" * 40 + "\n"
                for r in antennas:
                    atype = r.get("antenna_type") or "(unspecified)"
                    ants  = r.get("total_antennas") or 0
                    resp += f"  • {atype}: {r['records']} record(s), {ants} antenna(s)\n"
                resp += "\n"

            # Lightning protection
            if lp_recs:
                resp += f"LIGHTNING PROTECTION ({len(lp_recs)} record(s)):\n"
                resp += "-" * 40 + "\n"
                for r in lp_recs:
                    ptype  = r.get("ptype") or "none"
                    apex   = r.get("apex")
                    stname = r.get("structure_name") or ""
                    notes  = r.get("notes") or ""
                    apex_s = f", apex: {apex} ft" if apex else ""
                    note_s = f" — {notes[:80]}" if notes else ""
                    resp += f"  • {stname}: {ptype}{apex_s}{note_s}\n"
                resp += "\n"

            # Light assemblies
            if la_recs:
                resp += f"LIGHT ASSEMBLIES ({len(la_recs)} record(s)):\n"
                resp += "-" * 40 + "\n"
                for r in la_recs:
                    lname  = r.get("light_name") or "(unnamed)"
                    elev   = r.get("elevation")
                    nlights= r.get("num_lights")
                    stname = r.get("structure_name") or ""
                    e_str  = f" @ {elev} ft" if elev else ""
                    n_str  = f", {nlights} fixture(s)" if nlights else ""
                    resp += f"  • {stname}: {lname}{e_str}{n_str}\n"
                resp += "\n"

            # AM skirts
            if sk_recs:
                resp += f"AM SKIRTS ({len(sk_recs)} record(s)):\n"
                resp += "-" * 40 + "\n"
                for r in sk_recs:
                    resp += f"  • {r.get('structure_name','?')}\n"
                resp += "\n"

            # Other equipment
            if other:
                resp += f"OTHER EQUIPMENT ({len(other)} record(s)):\n"
                resp += "-" * 40 + "\n"
                for r in other:
                    oname = r.get("name") or "(unnamed)"
                    desc  = r.get("description") or ""
                    d_str = f" — {desc[:80]}" if desc else ""
                    resp += f"  • {oname}{d_str}\n"

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 14. wisconsin_sites
        # ------------------------------------------------------------------
        elif name == "wisconsin_sites":
            # Wisconsin bounding box: lat 42.5–47.3, lon -92.9 to -86.8
            q_sites = """
WITH wi_sites AS (
    SELECT s.id, s.name, s."internalId" AS site_code, s.address,
           ST_Y(s.location) AS latitude, ST_X(s.location) AS longitude
    FROM site s
    WHERE s."organizationId" = %s
      AND s.location IS NOT NULL
      AND ST_Y(s.location) BETWEEN 42.5 AND 47.3
      AND ST_X(s.location) BETWEEN -92.9 AND -86.8
)
SELECT
    ws.name AS site_name,
    ws.site_code,
    ws.address,
    ROUND(ws.latitude::numeric, 4)  AS latitude,
    ROUND(ws.longitude::numeric, 4) AS longitude,
    COUNT(DISTINCT st.id)           AS structure_count,
    MAX(sv."siteVisitDate")         AS last_visit,
    COUNT(DISTINCT d.id)            AS total_deficiencies,
    COUNT(DISTINCT CASE WHEN d.severity = '1' THEN d.id END) AS sev1_count
FROM wi_sites ws
LEFT JOIN structure st   ON st."siteId"       = ws.id
LEFT JOIN site_visit sv  ON sv."siteId"       = ws.id
LEFT JOIN survey su      ON su."siteVisitId"  = sv.id
LEFT JOIN deficiency d   ON d."surveyId"      = su.id
                         AND d."organizationId" = %s
GROUP BY ws.id, ws.name, ws.site_code, ws.address, ws.latitude, ws.longitude
ORDER BY ws.name
"""
            wi_rows = db.execute_query(q_sites, (DEMO_ORG_ID, DEMO_ORG_ID))

            resp  = f"WISCONSIN SITES — {DEMO_ORG_NAME}\n"
            resp += "=" * 60 + "\n\n"

            if not wi_rows:
                resp += "No sites found in Wisconsin.\n"
                return [TextContent(type="text", text=resp)]

            total_structs = sum(r.get("structure_count", 0) for r in wi_rows)
            total_defs    = sum(r.get("total_deficiencies", 0) for r in wi_rows)
            total_sev1    = sum(r.get("sev1_count", 0) for r in wi_rows)

            resp += f"Total sites in Wisconsin: {len(wi_rows)}\n"
            resp += f"Total structures:         {total_structs}\n"
            resp += f"Total deficiencies:       {total_defs} ({total_sev1} severity-1)\n\n"

            resp += "SITE DETAILS:\n"
            resp += "-" * 50 + "\n"
            for r in wi_rows:
                sname    = r.get("site_name") or "Unknown"
                code     = f" [{r['site_code']}]" if r.get("site_code") else ""
                addr     = r.get("address") or ""
                lat      = r.get("latitude")
                lon      = r.get("longitude")
                structs  = r.get("structure_count", 0)
                defs     = r.get("total_deficiencies", 0)
                sev1     = r.get("sev1_count", 0)
                lv       = r.get("last_visit")
                lv_str   = lv.strftime("%b %d, %Y") if lv else "Never inspected"
                coord_str = f" ({lat}, {lon})" if lat else ""
                addr_str  = f"\n    Address: {addr}" if addr else ""
                sev1_str  = f" ({sev1} critical)" if sev1 else ""
                resp += (
                    f"\n  {sname}{code}{coord_str}{addr_str}\n"
                    f"    Structures: {structs} | Last visit: {lv_str}\n"
                    f"    Deficiencies: {defs}{sev1_str}\n"
                )

            return [TextContent(type="text", text=resp)]

        # ------------------------------------------------------------------
        # 14. get_schema_cache  (instant — no DB round-trip)
        # ------------------------------------------------------------------
        elif name == "get_schema_cache":
            if CACHED_SCHEMA:
                text = "DATABASE SCHEMA:\n" + CACHED_SCHEMA
                if CACHED_APPURTENANCE_REFS:
                    text += (
                        "\n\nTABLES WITH FK → appurtenance.id:\n"
                        + CACHED_APPURTENANCE_REFS
                    )
                return [TextContent(type="text", text=text)]
            return [TextContent(type="text", text="Schema not available (cache empty at startup).")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error in {name}: {str(e)}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# ============================================================
# MAIN
# ============================================================

async def main():
    logger.warning("Everest Demo MCP Server starting...")

    # Test DB connection
    try:
        logger.warning("Testing database connection...")
        db.execute_query("SELECT 1 as test")
        logger.warning("Database connection OK.")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        return

    logger.warning("Starting MCP stdio interface...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
