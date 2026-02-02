#!/usr/bin/env python3
"""
Dynamic Database Query MCP Server
Provides intelligent database querying with schema awareness
"""

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
            description="Get the complete database schema including all tables, columns, and relationships.  The 'site' table contains tower information (best for counting towers). The 'structure' table contains structural details including heights. 'safety_climb' table contains safety climb information.",
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
            description="""Search the database schema for tables or columns matching a term.
            IMPORTANT TABLE USAGE:
            - 'site' table: Contains tower records. Use for counting towers, listing towers, tower locations.
            - 'structure' table: Contains structural details and heights. Use for height queries, structural specifications.
            - For complete tower information, JOIN site with structure on siteId.
            """,
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
            description="""Execute a custom SQL SELECT query. Use this after understanding the schema.
            CRITICAL: For tower-related queries:
            - COUNT towers: Query 'site' table (each site = one tower)
            - Get heights: Query 'structure' table  
            - Complete info: JOIN site and structure tables
            
            Always use parameterized queries for safety.""",
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
                # Format results as a table-like structure
                if len(results) <= 10:
                    response += json.dumps(results, indent=2, default=str)
                else:
                    # Show first 10 and summary
                    response += json.dumps(results[:10], indent=2, default=str)
                    response += f"\n\n... and {len(results) - 10} more rows"
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