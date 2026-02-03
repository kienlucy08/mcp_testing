import asyncio
import json
import sys
import logging
from typing import Any, Optional
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from dotenv import load_dotenv
import os

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Token limit constants - adjust these to stay within API limits
MAX_RESPONSE_CHARS = 4000  # Max characters per tool response
MAX_ROWS_DEFAULT = 5       # Default row limit for samples/queries
MAX_VALUE_LENGTH = 100     # Truncate individual cell values to this length
MAX_COLUMNS_SAMPLE = 10    # Max columns to show in table samples

load_dotenv()

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

logger.info("Environment loaded")
logger.info(f"  Host: {os.getenv('url')}")
logger.info(f"  User: {os.getenv('postgres_user')}")

# Create server instance
app = Server("dynamic-database-server")

class DatabaseManager:
    """Manages database connections with schema awareness"""
    
    def __init__(self, config: dict):
        self.config = config
        self.db_type = config.get("type", "postgres").lower()
        self.schema_cache = None
        
        # Import appropriate database driver
        if self.db_type in ["postgres", "postgresql"]:
            try:
                import psycopg2
                from psycopg2.extras import RealDictCursor
                self.connection_class = psycopg2
                self.cursor_factory = RealDictCursor
            except ImportError:
                raise Exception("psycopg2 not installed. Run: pip install psycopg2-binary")
        else:
            raise Exception(f"Unsupported database type: {self.db_type}")
    
    def get_connection(self):
        """Get a database connection"""
        conn_params = {
            "host": self.config["host"],
            "port": self.config.get("port", 5432),
            "database": self.config["database"],
            "user": self.config["user"],
            "password": self.config["password"],
            "cursor_factory": self.cursor_factory
        }
        
        if "sslmode" in self.config:
            conn_params["sslmode"] = self.config["sslmode"]
        
        return self.connection_class.connect(**conn_params)
    
    def execute_query(self, query: str, params: tuple = ()) -> list[dict]:
        """Execute a SQL query and return results as list of dicts"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            # Handle SELECT vs non-SELECT queries
            if cursor.description:
                results = [dict(row) for row in cursor.fetchall()]
            else:
                results = []
            
            conn.close()
            return results
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            raise Exception(f"Database error: {str(e)}")
    
    def get_schema(self, force_refresh: bool = False, table_names: list = None) -> dict:
        """Get database schema with optional filtering"""
        if self.schema_cache and not force_refresh and not table_names:
            return self.schema_cache
        
        schema = {}
        
        # Build table filter condition
        if table_names:
            table_filter = f"AND table_name IN ({','.join(['%s']*len(table_names))})"
            tables_query = f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                {table_filter}
                ORDER BY table_name
            """
            tables = self.execute_query(tables_query, tuple(table_names))
        else:
            tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """
            tables = self.execute_query(tables_query)
        
        for table in tables:
            table_name = table['table_name']
            
            # Get columns for this table
            columns_query = """
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position
            """
            columns = self.execute_query(columns_query, (table_name,))
            
            # Get foreign key relationships
            fk_query = """
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_name = %s
            """
            foreign_keys = self.execute_query(fk_query, (table_name,))
            
            schema[table_name] = {
                'columns': columns,
                'foreign_keys': foreign_keys
            }
        if not table_names:
            self.schema_cache = schema
        return schema
    
    def search_schema(self, search_term: str) -> dict:
        """Search schema for tables/columns matching a term"""
        schema = self.get_schema()
        matches = {
            'tables': [],
            'columns': []
        }
        
        search_lower = search_term.lower()
        
        for table_name, table_info in schema.items():
            # Check if table name matches
            if search_lower in table_name.lower():
                matches['tables'].append(table_name)
            
            # Check columns
            for col in table_info['columns']:
                col_name = col['column_name']
                if search_lower in col_name.lower():
                    matches['columns'].append({
                        'table': table_name,
                        'column': col_name,
                        'type': col['data_type']
                    })
        
        return matches

db = DatabaseManager(DB_CONFIG)

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="get_database_schema",
            description="Get database schema for tables, columns, and relationships. Optional: specify table_name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Optional: Get schema for specific table only"
                    }
                }
            }
        ),
        Tool(
            name="search_schema",
            description="""Search database schema for tables or columns matching a term.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Term to search for in table and column names"
                    }
                },
                "required": ["search_term"]
            }
        ),
        Tool(
            name="execute_query",
            description="""Execute a SELECT query. Use after understanding schema.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT query to execute (use %s for parameters in PostgreSQL)"
                    },
                    "params": {
                        "type": "array",
                        "items": {"type": ["string", "number", "null"]},
                        "description": "Parameters for the query (optional)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of what this query does"
                    }
                },
                "required": ["query", "description"]
            }
        ),
        Tool(
            name="get_table_sample",
            description="Get a sample of rows from a table to understand the data",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to sample"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of rows to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["table_name"]
            }
        ),
        Tool(
            name="get_survey_coordinates",
            description="Extract coordinates from survey payloads.",
            inputSchema={
                "type": "object",
                "properties": {
                    "survey_id": {"type": "string", "description": "Survey UUID"},
                    "organization_id": {"type": "string", "description": "Organization UUID"},
                    "site_id": {"type": "string", "description": "Site UUID"},
                    "limit": {"type": "integer", "description": "Max results", "default": 100}
                }
            }
        ),
        Tool(
            name="query_survey_payload",
            description="Query specific fields in surveyPayload JSON (e.g., 'result.globalId').",
            inputSchema={
                "type": "object",
                "properties": {
                    "json_path": {"type": "string", "description": "JSON path to extract"},
                    "survey_id": {"type": "string"},
                    "organization_id": {"type": "string"},
                    "site_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 100}
                },
                "required": ["json_path"]
            }
        ),
        Tool(
            name="get_surveys_near_location",
            description="Find surveys within radius of coordinates or another survey.",
            inputSchema={
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "survey_id": {"type": "string"},
                    "radius_miles": {"type": "number", "default": 10},
                    "organization_id": {"type": "string"},
                    "limit": {"type": "integer", "default": 50}
                }
            }
        ),
        Tool(
            name="get_weather_for_survey",
            description="Get current weather and extreme weather events for a survey location. Extracts coordinates from surveyPayload and uses web search for weather data.",
            inputSchema={
                "type": "object",
                "properties": {
                    "survey_id": {
                        "type": "string",
                        "description": "Survey UUID to get weather for"
                    },
                    "site_id": {
                        "type": "string",
                        "description": "Alternative: Get weather for all surveys at a site"
                    },
                    "include_forecast": {
                        "type": "boolean",
                        "description": "Include weather forecast (default: false)",
                        "default": False
                    },
                    "include_extreme_events": {
                        "type": "boolean",
                        "description": "Search for recent extreme weather events in the area (default: true)",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="find_related_data",
            description="Find what tables are related to a given table through foreign keys",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Table to find relationships for"
                    }
                },
                "required": ["table_name"]
            }
        ),
        Tool(
            name="explore_data_relationships",
            description="Comprehensively explore how data flows between tables. Use this to understand the full data model for concepts like 'organization towers', 'site visits', 'surveys', etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "starting_table": {
                        "type": "string",
                        "description": "Starting point table (e.g., 'organization', 'site', 'structure', 'survey', 'deficencies', 'compound', 'site_visit')"
                    },
                    "concept": {
                        "type": "string",
                        "description": "What you're trying to find (e.g., 'towers', 'surveys', 'visits', 'appurtenance', 'deficiency', 'safety climb')"
                    }
                },
                "required": ["starting_table", "concept"]
            }
        ),
        Tool(
            name="count_records_in_relationship",
            description="Count records across related tables to verify data exists. Useful for finding where data actually lives.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_table": {
                        "type": "string",
                        "description": "Starting table"
                    },
                    "from_id": {
                        "type": "string",
                        "description": "ID value to trace"
                    },
                    "related_tables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of potentially related tables to check"
                    }
                },
                "required": ["from_table", "from_id"]
            }
        ),       
        Tool(
            name="get_overdue_inspections",
            description="Find towers needing TIA inspection based on tower type and last inspection date. Guyed towers: 3 years, Self-support/Monopole: 5 years.",
            inputSchema={
                "type": "object",
                "properties": {
                    "organization_id": {
                        "type": "string",
                        "description": "Filter by organization UUID"
                    },
                    "inspection_type": {
                        "type": "string",
                        "description": "Type of inspection (e.g., 'TIA', 'safety_climb')",
                        "default": "TIA"
                    },
                    "include_details": {
                        "type": "boolean",
                        "description": "Include full site and structure details",
                        "default": True
                    }
                }
            }
        )
]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    
    try:
        if name == "get_database_schema":
            table_name = arguments.get("table_name")
            schema = db.get_schema()
            
            if table_name:
                if table_name not in schema:
                    return [TextContent(
                        type="text",
                        text=f"Table '{table_name}' not found in schema"
                    )]
                
                table_info = schema[table_name]
                response = f"Schema for table: {table_name}\n\n"
                response += "Columns:\n"
                for col in table_info['columns']:
                    nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                    response += f"  - {col['column_name']} ({col['data_type']}) {nullable}\n"
                
                if table_info['foreign_keys']:
                    response += "\nForeign Keys:\n"
                    for fk in table_info['foreign_keys']:
                        response += f"  - {fk['column_name']} -> {fk['foreign_table_name']}.{fk['foreign_column_name']}\n"
            else:
                response = "Database Schema:\n\n"
                response += f"Total tables: {len(schema)}\n\n"
                
                for table, info in schema.items():
                    response += f"📋 {table}\n"
                    response += f"   Columns: {len(info['columns'])}\n"
                    if info['foreign_keys']:
                        response += f"   Foreign keys: {len(info['foreign_keys'])}\n"
                    response += "\n"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "search_schema":
            search_term = arguments["search_term"]
            matches = db.search_schema(search_term)
            
            response = f"Schema search results for '{search_term}':\n\n"
            
            if matches['tables']:
                response += "Matching Tables:\n"
                for table in matches['tables']:
                    response += f"  - {table}\n"
                response += "\n"
            
            if matches['columns']:
                response += "Matching Columns:\n"
                for col in matches['columns']:
                    response += f"  - {col['table']}.{col['column']} ({col['type']})\n"
            
            if not matches['tables'] and not matches['columns']:
                response += "No matches found."
            
            return [TextContent(type="text", text=response)]
        
        elif name == "execute_query":
            query = arguments["query"]
            params = tuple(arguments.get("params", []))
            description = arguments["description"]
            
            # Security check - only allow SELECT queries
            query_upper = query.strip().upper()
            if not query_upper.startswith("SELECT") and not query_upper.startswith("WITH"):
                return [TextContent(
                    type="text",
                    text="Error: Only SELECT and WITH (CTE) queries are allowed for security reasons"
                )]
            
            logger.info(f"Executing query: {description}")
            logger.info(f"SQL: {query}")
            logger.info(f"Params: {params}")
            
            results = db.execute_query(query, params)
            
            response = f"Query: {description}\n\n"
            response += f"Results: {len(results)} rows\n\n"
            
            if results:
                max_results_to_show = 10
                max_chars_per_result = 500

                truncated_results = []
                for i, result in enumerate(results[:max_results_to_show]):
                    result_str = json.dump(result, default=str)
                    if len(result_str) > max_chars_per_result:
                        # Truncate individual fields
                        truncated_result = {}
                        for key, value in result.items():
                            value_str = str(value)
                            if len(value_str) > 100:
                                truncated_result[key] = value_str[:100] + "..."
                            else:
                                truncated_result[key] = value
                        truncated_results.append(truncated_result)
                    else:
                        truncated_results.append(result)
                
                response += json.dumps(truncated_results, indent=2, default=str)
                
                if len(results) > max_results_to_show:
                    response += f"\n\n... and {len(results) - max_results_to_show} more rows (truncated to save tokens)"
            else:
                response += "No results returned."
            
            return [TextContent(type="text", text=response)]
        
        elif name == "get_table_sample":
            table_name = arguments["table_name"]
            limit = arguments.get("limit", 5)
            
            # Verify table exists
            schema = db.get_schema()
            if table_name not in schema:
                return [TextContent(
                    type="text",
                    text=f"Table '{table_name}' not found"
                )]
            
            query = f'SELECT * FROM "{table_name}" LIMIT %s'
            results = db.execute_query(query, (limit,))
            
            response = f"Sample data from {table_name} ({len(results)} rows):\n\n"
            response += json.dumps(results, indent=2, default=str)
            
            return [TextContent(type="text", text=response)]
        
        elif name == "find_related_data":
            table_name = arguments["table_name"]
            schema = db.get_schema()
            
            if table_name not in schema:
                return [TextContent(
                    type="text",
                    text=f"Table '{table_name}' not found"
                )]
            
            table_info = schema[table_name]
            response = f"Relationships for table: {table_name}\n\n"
            
            # Outgoing relationships (this table references others)
            if table_info['foreign_keys']:
                response += "This table references:\n"
                for fk in table_info['foreign_keys']:
                    response += f"  - {fk['foreign_table_name']} (via {fk['column_name']})\n"
                response += "\n"
            
            # Incoming relationships (other tables reference this one)
            incoming = []
            for other_table, other_info in schema.items():
                if other_table == table_name:
                    continue
                for fk in other_info['foreign_keys']:
                    if fk['foreign_table_name'] == table_name:
                        incoming.append({
                            'table': other_table,
                            'column': fk['column_name']
                        })
            
            if incoming:
                response += "Referenced by:\n"
                for rel in incoming:
                    response += f"  - {rel['table']} (via {rel['column']})\n"
            
            if not table_info['foreign_keys'] and not incoming:
                response += "No foreign key relationships found."
            
            return [TextContent(type="text", text=response)]
        
        elif name == "explore_data_relationships":
            starting_table = arguments["starting_table"]
            concept = arguments["concept"]
            schema = db.get_schema()
            
            if starting_table not in schema:
                return [TextContent(
                    type="text",
                    text=f"Table '{starting_table}' not found"
                )]
            
            response = f"Exploring data relationships from '{starting_table}' for concept '{concept}':\n\n"
            
            # Search for tables related to the concept
            concept_matches = db.search_schema(concept)
            
            response += f"Tables matching '{concept}':\n"
            if concept_matches['tables']:
                for table in concept_matches['tables']:
                    response += f"  📋 {table}\n"
            else:
                response += "  (none found)\n"
            response += "\n"
            
            response += f"Columns matching '{concept}':\n"
            if concept_matches['columns']:
                for col in concept_matches['columns']:
                    response += f"  - {col['table']}.{col['column']} ({col['type']})\n"
            else:
                response += "  (none found)\n"
            response += "\n"
            
            # Build relationship chain from starting table
            response += f"Relationship chain from {starting_table}:\n"
            visited = set()
            
            def explore_relations(table, depth=0, max_depth=2):
                if depth > max_depth or table in visited:
                    return ""
                
                visited.add(table)
                result = ""
                indent = "  " * depth
                
                if table not in schema:
                    return result
                
                table_info = schema[table]
                
                # Outgoing relationships
                for fk in table_info['foreign_keys']:
                    related_table = fk['foreign_table_name']
                    result += f"{indent}→ {related_table} (via {fk['column_name']})\n"
                    result += explore_relations(related_table, depth + 1, max_depth)
                
                # Incoming relationships
                for other_table, other_info in schema.items():
                    if other_table in visited:
                        continue
                    for fk in other_info['foreign_keys']:
                        if fk['foreign_table_name'] == table:
                            result += f"{indent}← {other_table} (via {fk['column_name']})\n"
                            result += explore_relations(other_table, depth + 1, max_depth)
                
                return result
            
            response += explore_relations(starting_table)
            
            return [TextContent(type="text", text=response)]
        
        elif name == "count_records_in_relationship":
            from_table = arguments["from_table"]
            from_id = arguments["from_id"]
            related_tables = arguments.get("related_tables", [])
            
            schema = db.get_schema()
            
            if from_table not in schema:
                return [TextContent(
                    type="text",
                    text=f"Table '{from_table}' not found"
                )]
            
            response = f"Counting records related to {from_table} (id: {from_id}):\n\n"
            
            # If no related tables specified, find them automatically
            if not related_tables:
                related_tables = []
                # Find tables that reference this table
                for table, info in schema.items():
                    for fk in info['foreign_keys']:
                        if fk['foreign_table_name'] == from_table:
                            related_tables.append(table)
            
            # Check each related table
            for related_table in related_tables:
                if related_table not in schema:
                    response += f"  ⚠️  {related_table}: Table not found\n"
                    continue
                
                table_info = schema[related_table]
                
                # Find the foreign key column that points to from_table
                fk_column = None
                for fk in table_info['foreign_keys']:
                    if fk['foreign_table_name'] == from_table:
                        fk_column = fk['column_name']
                        break
                
                if not fk_column:
                    # Try common column name patterns
                    possible_columns = [
                        f"{from_table}Id",
                        f"{from_table}_id",
                        f'"{from_table}Id"',
                        "organizationId",
                        "siteId",
                        "structureId"
                    ]
                    
                    # Check which column exists
                    table_columns = [col['column_name'] for col in table_info['columns']]
                    for possible in possible_columns:
                        clean_col = possible.strip('"')
                        if clean_col in table_columns:
                            fk_column = clean_col
                            break
                
                if fk_column:
                    try:
                        # Handle quoted column names
                        if any(c.isupper() for c in fk_column):
                            fk_column_quoted = f'"{fk_column}"'
                        else:
                            fk_column_quoted = fk_column
                        
                        query = f"SELECT COUNT(*) as count FROM {related_table} WHERE {fk_column_quoted} = %s"
                        result = db.execute_query(query, (from_id,))
                        count = result[0]['count'] if result else 0
                        response += f"  ✓ {related_table}: {count} records (via {fk_column})\n"
                    except Exception as e:
                        response += f"  ⚠️  {related_table}: Error - {str(e)}\n"
                else:
                    response += f"  ⚠️  {related_table}: No clear foreign key found\n"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "get_survey_coordinates":
            where_clauses = []
            params = []
            
            if arguments.get("survey_id"):
                where_clauses.append("survey.id = %s")
                params.append(arguments["survey_id"])
            
            if arguments.get("organization_id"):
                where_clauses.append('survey."organizationId" = %s')
                params.append(arguments["organization_id"])
            
            if arguments.get("site_id"):
                where_clauses.append('survey."siteId" = %s')
                params.append(arguments["site_id"])
            
            limit = arguments.get("limit", 100)
            
            # Extract coordinates from JSON payload
            query = """
                    SELECT 
                        survey.id as survey_id,
                        survey.name as survey_name,
                        site_visit."siteId" as site_id,
                        site.name as site_name,
                        (survey."surveyPayload"::jsonb->'geometry'->>'y')::float as latitude,
                        (survey."surveyPayload"::jsonb->'geometry'->>'x')::float as longitude,
                        survey."surveyPayload"::jsonb->'geometry'->'spatialReference'->>'wkid' as wkid,
                        survey."surveyPayload"::jsonb->'result'->>'globalId' as global_id,
                        survey."surveyPayload"::jsonb->'result'->>'objectId' as object_id
                    FROM survey
                    LEFT JOIN site_visit ON survey."siteVisitId" = site_visit.id
                    LEFT JOIN site ON site_visit."siteId" = site.id
                """
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " LIMIT %s"
            params.append(limit)
            
            results = db.execute_query(query, tuple(params))
            
            response = f"Survey Coordinates: {len(results)} record(s)\n\n"
            response += json.dumps(results, indent=2, default=str)
            
            return [TextContent(type="text", text=response)]

        elif name == "query_survey_payload":
            where_clauses = []
            params = []
            
            if arguments.get("survey_id"):
                where_clauses.append("survey.id = %s")
                params.append(arguments["survey_id"])
            
            if arguments.get("organization_id"):
                where_clauses.append('survey."organizationId" = %s')
                params.append(arguments["organization_id"])
            
            if arguments.get("site_id"):
                where_clauses.append('survey."siteId" = %s')
                params.append(arguments["site_id"])
            
            json_path = arguments["json_path"]
            limit = arguments.get("limit", 100)
            
            # Convert dot notation to PostgreSQL JSON path
            # e.g., "result.globalId" -> "surveyPayload"::jsonb->'result'->>'globalId'
            path_parts = json_path.split('.')
            json_accessor = 'survey."surveyPayload"::jsonb'
            
            for i, part in enumerate(path_parts):
                if i == len(path_parts) - 1:
                    # Last element, use ->> to get text value
                    json_accessor += f"->>{repr(part)}"
                else:
                    json_accessor += f"->{repr(part)}"
            
            query = f"""
                SELECT 
                    survey.id as survey_id,
                    survey.name as survey_name,
                    survey."siteId" as site_id,
                    {json_accessor} as extracted_value
                FROM survey
                LEFT JOIN site_visit ON survey."siteVisitId" = site_visit.id
            """
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " LIMIT %s"
            params.append(limit)
            
            results = db.execute_query(query, tuple(params))
            
            response = f"Survey Payload Query (path: {json_path}): {len(results)} record(s)\n\n"
            response += json.dumps(results, indent=2, default=str)
            
            return [TextContent(type="text", text=response)]

        elif name == "get_surveys_near_location":
            # First, determine the center coordinates
            center_lat = arguments.get("latitude")
            center_lon = arguments.get("longitude")
            survey_id = arguments.get("survey_id")
            radius_miles = arguments.get("radius_miles", 10)
            organization_id = arguments.get("organization_id")
            limit = arguments.get("limit", 50)
            
            # If survey_id provided, get its coordinates first
            if survey_id and not (center_lat and center_lon):
                coord_query = """
                    SELECT 
                        (survey."surveyPayload"::jsonb->'geometry'->>'y')::float as latitude,
                        (survey."surveyPayload"::jsonb->'geometry'->>'x')::float as longitude
                    FROM survey
                    WHERE survey.id = %s
                """
                coord_result = db.execute_query(coord_query, (survey_id,))
                
                if not coord_result or not coord_result[0].get('latitude'):
                    return [TextContent(
                        type="text",
                        text=f"No coordinates found for survey_id: {survey_id}"
                    )]
                
                center_lat = coord_result[0]['latitude']
                center_lon = coord_result[0]['longitude']
            
            if not (center_lat and center_lon):
                return [TextContent(
                    type="text",
                    text="Error: Either latitude/longitude or survey_id must be provided"
                )]
            
            # Build WHERE clause
            where_clauses = [
                'survey."surveyPayload"::jsonb->\'geometry\' IS NOT NULL',
                'survey."surveyPayload"::jsonb->\'geometry\'->>\'y\' IS NOT NULL',
                'survey."surveyPayload"::jsonb->\'geometry\'->>\'x\' IS NOT NULL'
            ]
            params = [center_lat, center_lon, center_lat, radius_miles]
            
            if organization_id:
                where_clauses.append('survey."organizationId" = %s')
                params.append(organization_id)
            
            if survey_id:
                where_clauses.append('survey.id != %s')
                params.append(survey_id)

            params.append(limit)
            
            # Haversine distance calculation using JSON extracted coordinates
            query = f"""
                SELECT 
                    survey.id as survey_id,
                    survey.name as survey_name,
                    site_visit."siteId" as site_id,
                    site.name as site_name,
                    (survey."surveyPayload"::jsonb->'geometry'->>'y')::float as latitude,
                    (survey."surveyPayload"::jsonb->'geometry'->>'x')::float as longitude,
                    (3959 * acos(
                        cos(radians(%s)) * cos(radians((survey."surveyPayload"::jsonb->'geometry'->>'y')::float)) * 
                        cos(radians((survey."surveyPayload"::jsonb->'geometry'->>'x')::float) - radians(%s)) + 
                        sin(radians(%s)) * sin(radians((survey."surveyPayload"::jsonb->'geometry'->>'y')::float))
                    )) AS distance_miles
                FROM survey
                LEFT JOIN site_visit ON survey."siteVisitId" = site_visit.id
                LEFT JOIN site ON site_visit."siteId" = site.id
                WHERE {' AND '.join(where_clauses)}
                HAVING distance_miles <= %s
                ORDER BY distance_miles
                LIMIT %s
            """
            
            results = db.execute_query(query, tuple(params))
            
            response = f"Surveys within {radius_miles} miles of ({center_lat}, {center_lon}): {len(results)} record(s)\n\n"
            # Simplify output for token efficiency
            if len(results) > 10:
                summary = []
                for r in results:
                    summary.append({
                        'survey_name': r['survey_name'],
                        'site_name': r['site_name'],
                        'distance_miles': round(r['distance_miles'], 2)
                    })
                response += json.dumps(summary, indent=2, default=str)
            else:
                response += json.dumps(results, indent=2, default=str)
            
            return [TextContent(type="text", text=response)]

        elif name == "get_weather_for_survey":
            survey_id = arguments.get("survey_id")
            site_id = arguments.get("site_id")
            include_forecast = arguments.get("include_forecast", False)
            include_extreme_events = arguments.get("include_extreme_events", True)
            
            if not survey_id and not site_id:
                return [TextContent(
                    type="text",
                    text="Error: Either survey_id or site_id must be provided"
                )]
            
            # Get coordinates from survey(s)
            if survey_id:
                coord_query = """
                    SELECT 
                        survey.id as survey_id,
                        survey.name as survey_name,
                        site.name as site_name,
                        site.city,
                        site.state,
                        (survey."surveyPayload"::jsonb->'geometry'->>'y')::float as latitude,
                        (survey."surveyPayload"::jsonb->'geometry'->>'x')::float as longitude
                    FROM survey
                    LEFT JOIN site_visit ON survey."siteVisitId" = site_visit.id
                    LEFT JOIN site ON site_visit."siteId" = site.id
                    WHERE survey.id = %s
                """
                params = (survey_id,)
            else:
                coord_query = """
                    SELECT 
                        survey.id as survey_id,
                        survey.name as survey_name,
                        site.name as site_name,
                        site.city,
                        site.state,
                        (survey."surveyPayload"::jsonb->'geometry'->>'y')::float as latitude,
                        (survey."surveyPayload"::jsonb->'geometry'->>'x')::float as longitude
                    FROM survey
                    LEFT JOIN site_visit ON survey."siteVisitId" = site_visit.id
                    LEFT JOIN site ON site_visit."siteId" = site.id
                    WHERE site.id = %s
                    LIMIT 1
                """
                params = (site_id,)
            
            survey_results = db.execute_query(coord_query, params)
            
            if not survey_results or not survey_results[0].get('latitude'):
                return [TextContent(
                    type="text",
                    text=f"No coordinates found for {'survey' if survey_id else 'site'}"
                )]
            
            survey_data = survey_results[0]
            lat = survey_data['latitude']
            lon = survey_data['longitude']
            location_name = survey_data.get('city', '') + (', ' + survey_data.get('state', '') if survey_data.get('state') else '')
            
            response = f"Weather for Survey: {survey_data['survey_name']}\n"
            response += f"Site: {survey_data['site_name']}\n"
            response += f"Location: {location_name} ({lat}, {lon})\n\n"
            response += f"Coordinates: {lat}°N, {lon}°W\n\n"
            # Use web_search to get current weather
            # Note: This would require the web_search tool to be called from within this function
            # or we return instructions for Claude to use web_search
            
            response += "To get live weather data, use web_search with:\n"
            response += f"  Query: 'current weather {lat} {lon}'\n"
            
            if include_forecast:
                response += f"  Forecast query: 'weather forecast {location_name}'\n"
            
            if include_extreme_events:
                response += f"  Extreme events query: 'extreme weather events {location_name} recent'\n"
            
            response += f"\nCoordinates ready for weather API calls: {lat}, {lon}"
            
            return [TextContent(type="text", text=response)]
        
        elif name == "get_overdue_inspections":
            organization_id = arguments.get("organization_id")
            inspection_type = arguments.get("inspection_type", "TIA")
            include_details = arguments.get("include_details", True)
            
            # Build WHERE clause for organization filter
            org_filter = ""
            params = []
            if organization_id:
                org_filter = 'AND survey."organizationId" = %s'
                params.append(organization_id)
            
            # Query to find the most recent survey for each site and check if overdue
            # Guyed towers: 3 years, Self-support/Monopole: 5 years
            query = f"""
                WITH latest_surveys AS (
                    SELECT 
                        survey."siteId",
                        s.name as site_name,
                        s.address,
                        s.city,
                        s.state,
                        survey.id as survey_id,
                        survey.name as survey_name,
                        survey."createdAt" as last_inspection_date,
                        survey."surveyPayload"::jsonb->'structureType'->>'type' as structure_type,
                        ROW_NUMBER() OVER (PARTITION BY survey."siteId" ORDER BY survey."createdAt" DESC) as rn
                    FROM survey
                    LEFT JOIN site s ON survey."siteId" = s.id
                    WHERE survey."surveyPayload" IS NOT NULL
                    {org_filter}
                ),
                categorized_surveys AS (
                    SELECT 
                        *,
                        CASE 
                            WHEN LOWER(structure_type) LIKE '%guyed%' THEN 3
                            WHEN LOWER(structure_type) LIKE '%monopole%' THEN 5
                            WHEN LOWER(structure_type) LIKE '%self%support%' THEN 5
                            WHEN LOWER(structure_type) LIKE '%self-support%' THEN 5
                            ELSE 5  -- Default to 5 years for unknown types
                        END as years_required,
                        CASE 
                            WHEN LOWER(structure_type) LIKE '%guyed%' THEN 'Guyed'
                            WHEN LOWER(structure_type) LIKE '%monopole%' THEN 'Monopole'
                            WHEN LOWER(structure_type) LIKE '%self%support%' THEN 'Self-Support'
                            WHEN LOWER(structure_type) LIKE '%self-support%' THEN 'Self-Support'
                            ELSE 'Unknown'
                        END as tower_category
                    FROM latest_surveys
                    WHERE rn = 1
                )
                SELECT 
                    "siteId",
                    site_name,
                    address,
                    city,
                    state,
                    survey_id,
                    survey_name,
                    last_inspection_date,
                    structure_type,
                    tower_category,
                    years_required,
                    EXTRACT(YEAR FROM AGE(CURRENT_DATE, last_inspection_date::date)) as years_since_inspection,
                    EXTRACT(YEAR FROM AGE(CURRENT_DATE, last_inspection_date::date)) 
                        + (EXTRACT(MONTH FROM AGE(CURRENT_DATE, last_inspection_date::date)) / 12.0) as exact_years_since,
                    CASE 
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, last_inspection_date::date)) >= years_required 
                        THEN 'OVERDUE'
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, last_inspection_date::date)) >= (years_required - 1)
                        THEN 'DUE SOON'
                        ELSE 'CURRENT'
                    END as inspection_status
                FROM categorized_surveys
                WHERE EXTRACT(YEAR FROM AGE(CURRENT_DATE, last_inspection_date::date)) >= years_required
                ORDER BY exact_years_since DESC
            """
            
            try:
                results = db.execute_query(query, tuple(params))
                
                if not results:
                    response = "No overdue inspections found!\n\n"
                    if organization_id:
                        response += f"All sites for organization {organization_id} have current inspections."
                    else:
                        response += "All sites have current inspections."
                    return [TextContent(type="text", text=response)]
                
                # Build response
                response = f"OVERDUE INSPECTIONS REPORT\n"
                response += f"{'='*60}\n\n"
                response += f"Total Overdue Sites: {len(results)}\n\n"
                
                # Group by tower type
                guyed_count = sum(1 for r in results if r['tower_category'] == 'Guyed')
                monopole_count = sum(1 for r in results if r['tower_category'] == 'Monopole')
                self_support_count = sum(1 for r in results if r['tower_category'] == 'Self-Support')
                unknown_count = sum(1 for r in results if r['tower_category'] == 'Unknown')
                
                response += "Breakdown by Tower Type:\n"
                response += f"  • Guyed (>3 years): {guyed_count}\n"
                response += f"  • Monopole (>5 years): {monopole_count}\n"
                response += f"  • Self-Support (>5 years): {self_support_count}\n"
                if unknown_count > 0:
                    response += f"  • Unknown Type (>5 years): {unknown_count}\n"
                response += "\n"
                
                if include_details:
                    response += f"{'='*60}\n"
                    response += "DETAILED OVERDUE SITES:\n"
                    response += f"{'='*60}\n\n"
                    
                    for idx, site in enumerate(results, 1):
                        response += f"{idx}. {site['site_name']}\n"
                        response += f"   Location: {site['address']}, {site['city']}, {site['state']}\n"
                        response += f"   Tower Type: {site['tower_category']} ({site['structure_type']})\n"
                        response += f"   Last Inspection: {site['last_inspection_date']}\n"
                        response += f"   Years Since: {site['exact_years_since']:.1f} years\n"
                        response += f"   Required Interval: {site['years_required']} years\n"
                        response += f"   Status: {site['inspection_status']}\n"
                        response += f"   Overdue By: {site['exact_years_since'] - site['years_required']:.1f} years\n"
                        response += f"   Site ID: {site['siteId']}\n"
                        response += f"   Survey ID: {site['survey_id']}\n"
                        response += "\n"
                else:
                    # Just show summary table
                    response += "\nSummary (Site Name | Type | Years Since | Status):\n"
                    response += "-" * 80 + "\n"
                    for site in results:
                        response += f"{site['site_name'][:30]:30} | "
                        response += f"{site['tower_category']:15} | "
                        response += f"{site['exact_years_since']:5.1f} years | "
                        response += f"{site['inspection_status']}\n"
                
                return [TextContent(type="text", text=response)]
                
            except Exception as e:
                logger.error(f"Error in get_overdue_inspections: {str(e)}", exc_info=True)
                return [TextContent(
                    type="text",
                    text=f"Error getting overdue inspections: {str(e)}"
                )]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        logger.error(f"Error in {name}: {str(e)}", exc_info=True)
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]

async def main():
    logger.info("=" * 60)
    logger.info("Dynamic Database MCP Server Starting...")
    logger.info("=" * 60)
    
    logger.info(f"Database Type: {DB_CONFIG.get('type')}")
    logger.info(f"Database Host: {DB_CONFIG.get('host')}")
    logger.info(f"Database Name: {DB_CONFIG.get('database')}")
    logger.info(f"Database User: {DB_CONFIG.get('user')}")
    
    # Test connection and cache schema
    try:
        logger.info("Testing database connection...")
        test_results = db.execute_query("SELECT 1 as test")
        logger.info("✓ Successfully connected to database")
        
        logger.info("Caching database schema...")
        schema = db.get_schema()
        logger.info(f"✓ Cached schema with {len(schema)} tables")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        return
    
    logger.info("Starting MCP server stdio interface...")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())