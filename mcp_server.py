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
import requests
from datetime import datetime, timedelta

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Create log filename with timestamp
log_filename = LOGS_DIR / f"mcp_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging to both file and stderr
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        # File handler - all logs go here
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        # Stream handler - also show in console/stderr
        logging.StreamHandler(sys.stderr)
    ]
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
    
    @staticmethod
    def fetch_weather_data(lat: float, lon: float, api_key: str = None) -> dict:
        """Fetch current weather and forecast data from OpenWeatherMap API"""
        if not api_key:
            api_key = os.getenv("weather_api_key")
        
        if not api_key:
            return {"error": "OpenWeatherMap API key not configured"}
        
        try:
            # Current weather
            current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=imperial"
            current_response = requests.get(current_url, timeout=10)
            current_response.raise_for_status()
            current_data = current_response.json()
            
            # 5-day forecast
            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=imperial"
            forecast_response = requests.get(forecast_url, timeout=10)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
            
            return {
                "current": current_data,
                "forecast": forecast_data,
                "success": True
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API error: {str(e)}")
            return {"error": f"Failed to fetch weather data: {str(e)}"}

    @staticmethod
    def format_weather_data(weather_response: dict, site_name: str, location: str) -> str:
        """Format weather data into readable text"""
        if weather_response.get("error"):
            return f"Error: {weather_response['error']}"
        
        current = weather_response.get("current", {})
        forecast = weather_response.get("forecast", {})
        
        response = f"WEATHER REPORT for {site_name}\n"
        response += f"Location: {location}\n"
        response += f"{'='*60}\n\n"
        
        # Current weather
        if current:
            main = current.get("main", {})
            weather = current.get("weather", [{}])[0]
            wind = current.get("wind", {})
            
            response += "CURRENT CONDITIONS:\n"
            response += f"  Temperature: {main.get('temp', 'N/A')}°F (Feels like: {main.get('feels_like', 'N/A')}°F)\n"
            response += f"  Conditions: {weather.get('description', 'N/A').title()}\n"
            response += f"  Humidity: {main.get('humidity', 'N/A')}%\n"
            response += f"  Pressure: {main.get('pressure', 'N/A')} hPa\n"
            response += f"  Wind: {wind.get('speed', 'N/A')} mph at {wind.get('deg', 'N/A')}°\n"
            response += f"  Visibility: {current.get('visibility', 0) / 1609.34:.2f} miles\n"
            
            if 'clouds' in current:
                response += f"  Cloud Cover: {current['clouds'].get('all', 'N/A')}%\n"
            
            if 'rain' in current:
                response += f"  Rain (last 1h): {current['rain'].get('1h', 0)} mm\n"
            
            if 'snow' in current:
                response += f"  Snow (last 1h): {current['snow'].get('1h', 0)} mm\n"
            
            response += "\n"
        
        # Forecast summary
        if forecast and 'list' in forecast:
            response += "5-DAY FORECAST SUMMARY:\n"
            
            # Group forecast by day
            daily_forecasts = {}
            for item in forecast['list']:
                dt = datetime.fromtimestamp(item['dt'])
                date_key = dt.strftime('%Y-%m-%d')
                
                if date_key not in daily_forecasts:
                    daily_forecasts[date_key] = {
                        'temps': [],
                        'conditions': [],
                        'rain': 0,
                        'wind_speeds': []
                    }
                
                daily_forecasts[date_key]['temps'].append(item['main']['temp'])
                daily_forecasts[date_key]['conditions'].append(item['weather'][0]['description'])
                daily_forecasts[date_key]['wind_speeds'].append(item['wind']['speed'])
                
                if 'rain' in item:
                    daily_forecasts[date_key]['rain'] += item['rain'].get('3h', 0)
            
            # Show next 5 days
            for date_key in sorted(daily_forecasts.keys())[:5]:
                day_data = daily_forecasts[date_key]
                dt = datetime.strptime(date_key, '%Y-%m-%d')
                
                temps = day_data['temps']
                high = max(temps)
                low = min(temps)
                avg_wind = sum(day_data['wind_speeds']) / len(day_data['wind_speeds'])
                
                # Most common condition
                conditions = day_data['conditions']
                common_condition = max(set(conditions), key=conditions.count)
                
                response += f"  {dt.strftime('%A, %b %d')}:\n"
                response += f"    High: {high:.1f}°F | Low: {low:.1f}°F\n"
                response += f"    Conditions: {common_condition.title()}\n"
                response += f"    Wind: {avg_wind:.1f} mph avg\n"
                
                if day_data['rain'] > 0:
                    response += f"    Rain: {day_data['rain']:.2f} mm\n"
                
                response += "\n"
        
        return response

    @staticmethod
    def check_extreme_weather(weather_data: dict) -> str:
        """Analyze weather data for extreme conditions"""
        if weather_data.get("error"):
            return ""
        
        current = weather_data.get("current", {})
        forecast = weather_data.get("forecast", {})
        
        alerts = []
        
        # Current extreme conditions
        main = current.get("main", {})
        wind = current.get("wind", {})
        
        temp = main.get('temp', 70)
        wind_speed = wind.get('speed', 0)
        
        if temp > 95:
            alerts.append(f"EXTREME HEAT: Current temperature {temp}°F")
        elif temp < 20:
            alerts.append(f"EXTREME COLD: Current temperature {temp}°F")
        
        if wind_speed > 35:
            alerts.append(f"HIGH WINDS: Current wind speed {wind_speed} mph")
        
        # Check forecast for extreme conditions
        if forecast and 'list' in forecast:
            for item in forecast['list'][:8]:  # Next 24 hours
                f_temp = item['main']['temp']
                f_wind = item['wind']['speed']
                f_dt = datetime.fromtimestamp(item['dt'])
                
                if f_temp > 100:
                    alerts.append(f"EXTREME HEAT FORECAST: {f_temp}°F at {f_dt.strftime('%I:%M %p %m/%d')}")
                elif f_temp < 10:
                    alerts.append(f"EXTREME COLD FORECAST: {f_temp}°F at {f_dt.strftime('%I:%M %p %m/%d')}")
                
                if f_wind > 40:
                    alerts.append(f"HIGH WINDS FORECAST: {f_wind} mph at {f_dt.strftime('%I:%M %p %m/%d')}")
        
        if alerts:
            return "\nEXTREME WEATHER ALERTS:\n" + "\n".join(alerts) + "\n"
        else:
            return "\nNo extreme weather conditions detected.\n"

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
            name="schema",
            description="Get database schema information. Can retrieve full schema, specific table details, or search for tables/columns by name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Get schema for a specific table only"
                    },
                    "search_term": {
                        "type": "string",
                        "description": "Search for tables/columns matching this term"
                    }
                }
            }
        ),
        Tool(
            name="execute_query",
            description="""Execute a custom SQL SELECT query. Use this for any data retrieval after understanding the schema.
            """,
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
            name="explore_data_relationships",
            description="Explore how data flows between tables. Use this to understand the data model, find foreign key relationships, and trace connections between concepts like organizations, sites, structures, visits, etc.",
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
            name="query_inspection_data",
            description="Query safety climb, deficiency, appurtenance data with flexible filtering. Unfiied tool for all inspection-related queries",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_type": {
                        "type": "string",
                        "enum": ["safety_climb", "deficiency", "appurtenance", "antenna_equipment", "guy_attachment", "guy_wire", "generator", "fuel_tank"],
                        "description": "Type of inspection data to query"
                    },
                    "organization_id": {
                        "type": "string",
                        "description": "Filter by organization UUID"
                    },
                    "safety_climb_id": {
                        "type": "string",
                        "description": "Filter deficiencies by specific safety climb UUID"
                    },
                    "severity": {
                        "type": "string",
                        "description": "Filter deficiencies by severity level. 1 is the worst severity, 3 is the least."
                    },
                    "survey_id": {
                        "type": "string",
                        "description": "Filter by survey UUID"
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start date"
                    }
                }
            }
        ),
        Tool(
            name="query_sites_by_location",
            description="""Query sites by location. Can get coordinates for a single site,
            or find all sites within a radius of coordinates or another site.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "site_id": {
                        "type": "string",
                        "description": "Site UUID - if provided alone, returns that site's coordinates. If with radius finds nearby sites."
                    },
                    "latitude": {
                        "type": "number",
                        "description": "Center latitude for radius search"
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Center longitude for radius search"
                    },
                    "radius_miles": {
                        "type": "number",
                        "description": "Search radius in miles (triggers nearby search mode)",
                        "default": 10
                    },
                    "organization_id": {
                        "type": "string",
                        "description": "Filter by organization UUID"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results for radius search (default: 50)",
                        "default": 50
                    }
                }
            }
        ),
        Tool(
            name="get_weather_for_site",
            description="""Get current weather for forecast for a site location.
            Includes temperature, conditions, wind, and extreme weather alerts.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "site_id": {
                        "type": "string",
                        "description": "Site UUID"
                    },
                    "include_forecast": {
                        "type": "boolean",
                        "description": "Include 5-day forecast info (default: true)",
                        "default": True
                    },
                    "include_extreme_events": {
                        "type": "boolean",
                        "description": "Search for extreme weather (default: true)",
                        "default": True
                    }
                },
                "required": ["site_id"]
            }
        ),
        Tool(
            name="get_sites_needing_inspection",
            description="""Find towers overdue for inspection based on tower type requiements.
            Guyed towers require inspection every 3 years, monopole/self-support every 5 years.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "organization_id": {
                        "type": "string",
                        "description": "Filter by organization UUID (optional)"
                    },
                    "years_threshold": {
                        "type": "number",
                        "description": "Years since last visit to consider overdue (default: 3)",
                        "default": 3
                    },
                    "include_never_visited": {
                        "type": "boolean",
                        "description": "Include sites that have never been visited",
                        "default": True
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 1000)",
                        "default": 1000
                    }
                }
            }
        )
]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    
    try:
        if name == "schema":
            table_name = arguments.get("table_name")
            search_term = arguments.get("search_term")
            schema = db.get_schema()

            # Search mode!
            if search_term:
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
                        response += f"  - {col['tables']}.{col['column']} ({col['type']})\n"
                if not matches['tables'] and not matches['columns']:
                    response += "No matches found."

                return [TextContent(type="text", text=response)]
            
            # Specific table mode!!
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
                # full schema overview
                response = "Database Schema:\n\n"
                response += f"Total tables: {len(schema)}\n\n"
                
                for table, info in schema.items():
                    response += f"{table}\n"
                    response += f"   Columns: {len(info['columns'])}\n"
                    if info['foreign_keys']:
                        response += f"   Foreign keys: {len(info['foreign_keys'])}\n"
                    response += "\n"
            
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
                    response += f"  {table}\n"
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
        
        elif name == "get_sites_by_location":
            site_id = arguments.get("site_id")
            center_lat = arguments.get("latitude")
            center_long = arguments.get("longitude")
            radius_miles = arguments.get("radius_miles")
            organization_id = arguments.get("organization_id")
            limit = arguments.get("limit", 50)

            # Determine mode: single site lookup vs radius search
            is_radius_search = radius_miles is not None or center_lat is not None or center_long is not None

            # Mode 1 - single site coordniates lookup no radius
            if site_id and not is_radius_search:
                query = """
                    SELECT 
                        id as site_id,
                        name as site_name,
                        address,
                        country,
                        elevation,
                        ST_Y(location) as latitude,
                        ST_X(location) as longitude
                    FROM site
                    WHERE id = %s
                """
                results = db.execute_query(query, (site_id,))
                if not results:
                    return [TextContent(
                        type="text",
                        text=f"Site not found: {site_id}"
                    )]
            
                site_data = results[0]
                
                if not site_data.get('latitude') or not site_data.get('longitude'):
                    return [TextContent(
                        type="text",
                        text=f"No coordinates found for site: {site_data.get('site_name', 'Unknown')}"
                    )]
                
                response = f"Site Location Data\n\n"
                response += f"Site: {site_data['site_name']}\n"
                response += f"Address: {site_data.get('address', 'N/A')}\n"
                response += f"Country: {site_data.get('country', 'N/A')}\n"
                response += f"Elevation: {site_data.get('elevation', 'N/A')} meters\n"
                response += f"Coordinates: {site_data['latitude']}°N, {site_data['longitude']}°W\n"
            
                return [TextContent(type="text", text=response)]

            radius_miles = radius_miles or 10
            reference_name = None

            if site_id and not (center_lat and center_long):
                coord_query = """
                    SELECT
                        name,
                        ST_Y(location) as latitude,
                        ST_X(location) as longitude
                    WHERE id = %s
                """
                coord_result = db.execute_query(coord_query, (site_id,))
                if not coord_result or not coord_result[0].get('latitude'):
                    return [TextContent(type="text", text=f"No coordniates found for site: {site_id}")]
                
                center_lat = coord_result[0]['latitude']
                center_long = coord_result[0]['longitude']
                reference_name = coord_result[0]['name']
            if not (center_lat and center_long):
                return [TextContent(
                    type="text",
                    text="Error: Either latitude/longitude or site_id must be provided"
                )]
            # Build WHERE clause
            where_clauses = ["location IS NOT NULL"]
            params = [center_long, center_lat, radius_miles]
            if organization_id:
                where_clauses.append('"organizationId" = %s')
                params.append(organization_id)
            if site_id:
                where_clauses.append('id != %s')
                params.append(site_id)
            params.append(limit)
            query = f"""
                SELECT
                    id,
                    name,
                    address,
                    ) / 1609.34 AS distance_miles
                FROM site
                WHERE {' AND '.join(where_clauses)}
                  AND ST_Distance(
                        location::geography,
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
                      ) / 1609.34 <= %s
                ORDER BY distance_miles
                LIMIT %s
            """
            # Adjust params for the repeated use of coordinates
            params = [center_long, center_lat, center_long, center_lat, radius_miles]
            if organization_id:
                params.append(organization_id)
            if site_id:
                params.append(site_id)
            params.append(limit)
            results = db.execute_query(query, tuple(params))
            if reference_name:
                response = f"Sites within {radius_miles} miles of '{reference_name}':\n\n"
            else:
                response = f"Sites within {radius_miles} miles of ({center_lat}, {center_long}):\n\n"
            response += f"Found {len(results)} site(s)\n\n"
            if len(results) <= 10:
                for site in results:
                    response += f"{site['name']}\n"
                    response += f"   {site.get('address', 'No address')}, {site.get('country', '')}\n"
                    response += f"   Distance: {site['distance_miles']:.2f} miles\n"
                    response += f"   ID: {site['id']}\n\n"
            else:
                for site in results[:10]:
                    response += f"* {site['name']} - {site['distance_miles']:.2f} mi - {site.get('country', '')}\n"
                response += f"\n... and {len(results) - 10} more sites\n"
            return [TextContent(type="text", text=response)]
        
        elif name == "query_inspection_data":
            data_type = arguments.get("data_type", "both")
            organization_id = arguments.get("organization_id")
            site_id = arguments.get("site_id")
            safety_climb_id = arguments.get("safety_climb_id")
            severity = arguments.get("severity")
            status = arguments.get("status")
            date_from = arguments.get("date_from")
            date_to = arguments.get("date_to")
            limit = arguments.get("limit", 100)
            include_related = arguments.get("include_related", True)
            results_data = {}
            # Query safety climbs
            if data_type in ["safety_climbs", "both"]:
                where_clauses = []
                params = []
                if include_related:
                    base_query = """
                        SELECT
                            sc.id as safety_climb_id,
                            sc.status,
                            sc."createdAt" as created_at,
                            sc."updatedAt" as updated_at,
                            s.id as site_id,
                            s.name as site_name,
                            s.address,
                            o.name as organization_name
                        FROM safety_climb sc
                        LEFT JOIN site s ON sc."siteId" = s.id
                        LEFT JOIN organization o ON s."organizationId" = o.id
                    """
                else:
                    base_query = "SELECT * FROM safety_climb sc"
                if organization_id:
                    where_clauses.append('s."organizationId" = %s')
                    params.append(organization_id)
                if site_id:
                    where_clauses.append('sc."siteId" = %s')
                    params.append(site_id)
                if status:
                    where_clauses.append('sc.status = %s')
                    params.append(status)
                if date_from:
                    where_clauses.append('sc."createdAt" >= %s')
                    params.append(date_from)
                if date_to:
                    where_clauses.append('sc."createdAt" <= %s')
                    params.append(date_to)
                where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                query = f"{base_query} {where_clause} ORDER BY sc.\"createdAt\" DESC LIMIT %s"
                params.append(limit)
                results_data['safety_climbs'] = db.execute_query(query, tuple(params))
            # Query deficiencies
            if data_type in ["deficiencies", "both"]:
                where_clauses = []
                params = []
                if include_related:
                    base_query = """
                        SELECT
                            d.id as deficiency_id,
                            d.severity,
                            d.status,
                            d.description,
                            d."createdAt" as created_at,
                            sc.id as safety_climb_id,
                            s.id as site_id,
                            s.name as site_name,
                            o.name as organization_name
                        FROM deficencies d
                        LEFT JOIN safety_climb sc ON d."safetyClimbId" = sc.id
                        LEFT JOIN site s ON sc."siteId" = s.id
                        LEFT JOIN organization o ON s."organizationId" = o.id
                    """
                else:
                    base_query = "SELECT * FROM deficencies d"
                if organization_id:
                    where_clauses.append('s."organizationId" = %s')
                    params.append(organization_id)
                if site_id:
                    where_clauses.append('s.id = %s')
                    params.append(site_id)
                if safety_climb_id:
                    where_clauses.append('d."safetyClimbId" = %s')
                    params.append(safety_climb_id)
                if severity:
                    where_clauses.append('d.severity = %s')
                    params.append(severity)
                if status:
                    where_clauses.append('d.status = %s')
                    params.append(status)
                if date_from:
                    where_clauses.append('d."createdAt" >= %s')
                    params.append(date_from)
                if date_to:
                    where_clauses.append('d."createdAt" <= %s')
                    params.append(date_to)
                where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
                query = f"{base_query} {where_clause} ORDER BY d.\"createdAt\" DESC LIMIT %s"
                params.append(limit)
                results_data['deficiencies'] = db.execute_query(query, tuple(params))
            # Format response
            response = f"Inspection Data Query Results\n"
            response += f"{'='*60}\n\n"
            if 'safety_climbs' in results_data:
                sc_results = results_data['safety_climbs']
                response += f"SAFETY CLIMBS: {len(sc_results)} records\n"
                response += f"{'-'*40}\n"
                if sc_results:
                    response += json.dumps(sc_results[:10], indent=2, default=str)
                    if len(sc_results) > 10:
                        response += f"\n... and {len(sc_results) - 10} more records\n"
                else:
                    response += "No safety climb records found.\n"
                response += "\n\n"
            if 'deficiencies' in results_data:
                def_results = results_data['deficiencies']
                response += f"DEFICIENCIES: {len(def_results)} records\n"
                response += f"{'-'*40}\n"
                if def_results:
                    response += json.dumps(def_results[:10], indent=2, default=str)
                    if len(def_results) > 10:
                        response += f"\n... and {len(def_results) - 10} more records\n"
                else:
                    response += "No deficiency records found.\n"
            return [TextContent(type="text", text=response)]

        elif name == "get_weather_for_site":
            site_id = arguments["site_id"]
            include_forecast = arguments.get("include_forecast", True)  # Changed default to True
            include_extreme_events = arguments.get("include_extreme_events", True)
            
            # Get site coordinates and location info
            query = """
                SELECT 
                    id,
                    name,
                    address,
                    country,
                    elevation,
                    ST_Y(location) as latitude,
                    ST_X(location) as longitude
                FROM site
                WHERE id = %s
            """
            
            logger.info(f">>> Fetching site data for {site_id}")
            results = db.execute_query(query, (site_id,))
            logger.info(f">>> Query returned {len(results)} results")
            
            if not results:
                logger.warning(f">>> No results found for site {site_id}")
                return [TextContent(
                    type="text",
                    text=f"Site not found: {site_id}"
                )]
            
            site_data = results[0]
            lat = site_data.get('latitude')
            lon = site_data.get('longitude')
            logger.info(f">>> Extracted coordinates: lat={lat}, lon={lon}")
    
            if not lat or not lon:
                logger.warning(f">>> No coordinates found")
                return [TextContent(
                    type="text",
                    text=f"No coordinates found for site: {site_data.get('name', 'Unknown')}"
                )]
            
            # Use address and country for location description
            location_name = f"{site_data.get('address', 'Unknown location')}, {site_data.get('country', '')}"
            
            # Fetch live weather data
            logger.info(f">>> CALLING WEATHER API for {lat}, {lon}")
            weather_data = DatabaseManager.fetch_weather_data(lat, lon)
            logger.info(f">>> Weather API returned: {list(weather_data.keys())}")
            logger.info(f">>> Weather success: {weather_data.get('success')}")
            
            if weather_data.get("error"):
                logger.error(f">>> Weather API error: {weather_data['error']}")
                return [TextContent(
                    type="text",
                    text=f"Error fetching weather: {weather_data['error']}"
                )]
            
            logger.info(f">>> Formatting weather data...")
            # Format weather response
            response = DatabaseManager.format_weather_data(weather_data, site_data['name'], location_name)
            
            # Add extreme weather alerts
            if include_extreme_events:
                logger.info(f">>> Adding extreme weather check...")
                response += DatabaseManager.check_extreme_weather(weather_data)
            
            # Add site details
            response += f"\nSite Details:\n"
            response += f"  Coordinates: {lat}°N, {lon}°W\n"
            response += f"  Elevation: {site_data.get('elevation', 'N/A')} meters\n"
            response += f"  Site ID: {site_id}\n"
            
            logger.info(f">>> WEATHER TOOL COMPLETE - returning {len(response)} chars")
            return [TextContent(type="text", text=response)]

        elif name == "get_sites_needing_inspection":
            organization_id = arguments.get("organization_id")
            years_threshold = arguments.get("years_threshold", 3)
            include_never_visited = arguments.get("include_never_visited", True)
            limit = arguments.get("limit", 1000)
            
            logger.info(f">>> Finding sites needing inspection based on tower type")
            
            # Calculate cutoff dates
            guyed_cutoff = datetime.now() - timedelta(days=int(3 * 365.25))
            monopole_cutoff = datetime.now() - timedelta(days=int(5 * 365.25))
            guyed_cutoff_str = guyed_cutoff.strftime('%Y-%m-%d')
            monopole_cutoff_str = monopole_cutoff.strftime('%Y-%m-%d')
            
            logger.info(f">>> Guyed tower cutoff: {guyed_cutoff_str} (3 years)")
            logger.info(f">>> Monopole/Self-support cutoff: {monopole_cutoff_str} (5 years)")
            
            # Build WHERE clause
            where_clauses = []
            params = []
            
            if organization_id:
                where_clauses.append('s."organizationId" = %s')
                params.append(organization_id)
            
            where_clause = f"AND {' AND '.join(where_clauses)}" if where_clauses else ""
            
            # Add cutoff dates to params
            params.extend([guyed_cutoff_str, monopole_cutoff_str, guyed_cutoff_str, monopole_cutoff_str])
            params.append(limit)
            
            # Query with structure type logic - FIXED to use siteVisitDate
            query = f"""
                WITH site_structure_visits AS (
                    SELECT 
                        s.id as site_id,
                        s.name as site_name,
                        s.address,
                        s.country,
                        s."organizationId",
                        s."internalId" as site_internal_id,
                        o.name as organization_name,
                        st.id as structure_id,
                        st.name as structure_name,
                        st.type as tower_type,
                        st."constructedHeight" as tower_height,
                        st."fccAsrNumber" as fcc_asr_number,
                        st."internalId" as structure_internal_id,
                        MAX(sv."siteVisitDate") as last_visit_date,
                        COUNT(CASE WHEN sv.id IS NOT NULL THEN 1 END) as total_visits
                    FROM site s
                    LEFT JOIN organization o ON s."organizationId" = o.id
                    LEFT JOIN structure st ON s.id = st."siteId"
                    LEFT JOIN site_visit sv ON s.id = sv."siteId"
                    WHERE st.id IS NOT NULL
                    {where_clause}
                    GROUP BY s.id, s.name, s.address, s.country, s."organizationId", s."internalId", 
                            o.name, st.id, st.name, st.type, st."constructedHeight", 
                            st."fccAsrNumber", st."internalId"
                ),
                overdue_analysis AS (
                    SELECT 
                        *,
                        CASE 
                            WHEN tower_type ILIKE '%guyed%' THEN 'Guyed'
                            WHEN tower_type ILIKE '%monopole%' THEN 'Monopole'
                            WHEN tower_type ILIKE '%self%support%' OR tower_type ILIKE '%self-support%' THEN 'Self-Support'
                            ELSE 'Other'
                        END as tower_category,
                        CASE 
                            WHEN tower_type ILIKE '%guyed%' THEN 3
                            ELSE 5
                        END as required_inspection_years,
                        CASE 
                            WHEN tower_type ILIKE '%guyed%' THEN %s::date
                            ELSE %s::date
                        END as cutoff_date,
                        CASE 
                            WHEN last_visit_date IS NULL THEN NULL
                            ELSE ROUND(CAST(EXTRACT(EPOCH FROM (CURRENT_DATE - last_visit_date)) / 86400 / 365.25 AS NUMERIC), 2)
                        END as years_since_visit,
                        CASE 
                            WHEN last_visit_date IS NULL THEN true
                            WHEN tower_type ILIKE '%guyed%' AND last_visit_date < %s::date THEN true
                            WHEN (tower_type NOT ILIKE '%guyed%') AND last_visit_date < %s::date THEN true
                            ELSE false
                        END as is_overdue
                    FROM site_structure_visits
                )
                SELECT 
                    site_id,
                    site_name,
                    site_internal_id,
                    address,
                    country,
                    organization_name,
                    structure_id,
                    structure_name,
                    structure_internal_id,
                    tower_type,
                    tower_category,
                    tower_height,
                    fcc_asr_number,
                    last_visit_date,
                    years_since_visit,
                    total_visits,
                    required_inspection_years,
                    CASE 
                        WHEN last_visit_date IS NULL THEN 'CRITICAL - Never Inspected'
                        WHEN years_since_visit >= 10 THEN 'CRITICAL - 10+ Years Overdue'
                        WHEN years_since_visit >= 7 THEN 'HIGH - 7+ Years Overdue'
                        WHEN years_since_visit >= 5 THEN 'HIGH - 5+ Years Overdue'
                        WHEN years_since_visit >= 4 THEN 'MEDIUM - 4+ Years Overdue'
                        ELSE 'MEDIUM - 3+ Years Overdue'
                    END as priority_level
                FROM overdue_analysis
                WHERE is_overdue = true
                ORDER BY 
                    CASE WHEN last_visit_date IS NULL THEN 1 ELSE 2 END,
                    last_visit_date ASC NULLS FIRST,
                    tower_height DESC,
                    site_name,
                    structure_name
                LIMIT %s
            """
            
            logger.info(f">>> Executing query...")
            results = db.execute_query(query, tuple(params))
            logger.info(f">>> Found {len(results)} structures needing inspection")
            
            # Format response
            response = f"🔍 TOWER INSPECTION COMPLIANCE REPORT\n"
            response += f"{'='*80}\n\n"
            response += f"Inspection Requirements by Tower Type:\n"
            response += f"  • Guyed Towers: Inspection required every 3 years\n"
            response += f"  • Monopole/Self-Support: Inspection required every 5 years\n\n"
            response += f"Cutoff Dates:\n"
            response += f"  • Guyed: {guyed_cutoff_str}\n"
            response += f"  • Monopole/Self-Support: {monopole_cutoff_str}\n\n"
            response += f"Total towers found: {len(results)}\n\n"
            
            if not results:
                response += "✅ All towers are compliant with inspection schedules!\n"
                return [TextContent(type="text", text=response)]
            
            # Categorize by priority and tower type
            never_inspected = [r for r in results if r['last_visit_date'] is None]
            critical_10plus = [r for r in results if r['last_visit_date'] and r['years_since_visit'] and r['years_since_visit'] >= 10]
            high_7plus = [r for r in results if r['last_visit_date'] and r['years_since_visit'] and 7 <= r['years_since_visit'] < 10]
            high_5plus = [r for r in results if r['last_visit_date'] and r['years_since_visit'] and 5 <= r['years_since_visit'] < 7]
            medium_overdue = [r for r in results if r['last_visit_date'] and r['years_since_visit'] and 3 <= r['years_since_visit'] < 5]
            
            # Group by tower type
            def group_by_type(tower_list):
                guyed = [t for t in tower_list if t['tower_category'] == 'Guyed']
                monopole = [t for t in tower_list if t['tower_category'] == 'Monopole']
                self_support = [t for t in tower_list if t['tower_category'] == 'Self-Support']
                other = [t for t in tower_list if t['tower_category'] == 'Other']
                return guyed, monopole, self_support, other
            
            # Never inspected towers (CRITICAL)
            if never_inspected:
                response += f"🔴 CRITICAL PRIORITY - NEVER INSPECTED ({len(never_inspected)} towers):\n"
                response += f"{'-'*80}\n"
                
                guyed, monopole, self_support, other = group_by_type(never_inspected)
                
                if guyed:
                    response += f"\n  Guyed Towers ({len(guyed)}) - REQUIRED EVERY 3 YEARS:\n"
                    for i, tower in enumerate(guyed[:10], 1):
                        response += f"  {i}. {tower['structure_name'] or tower['site_name']}\n"
                        response += f"     Site: {tower['site_name']}\n"
                        response += f"     Type: {tower['tower_type']}\n"
                        response += f"     Height: {tower['tower_height']} ft\n"
                        response += f"     FCC ASR: {tower['fcc_asr_number'] or 'N/A'}\n"
                        response += f"     Location: {tower.get('address', 'N/A')}\n"
                        response += f"     Status: Never visited\n"
                        response += f"     Site ID: {tower['site_id']}\n\n"
                    if len(guyed) > 10:
                        response += f"     ... and {len(guyed) - 10} more guyed towers\n\n"
                
                if monopole or self_support:
                    mono_self = monopole + self_support
                    response += f"\n  Monopole/Self-Support Towers ({len(mono_self)}) - REQUIRED EVERY 5 YEARS:\n"
                    for i, tower in enumerate(mono_self[:10], 1):
                        response += f"  {i}. {tower['structure_name'] or tower['site_name']}\n"
                        response += f"     Type: {tower['tower_type']}\n"
                        response += f"     Height: {tower['tower_height']} ft\n"
                        response += f"     FCC ASR: {tower['fcc_asr_number'] or 'N/A'}\n"
                        response += f"     Status: Never visited\n\n"
                    if len(mono_self) > 10:
                        response += f"     ... and {len(mono_self) - 10} more monopole/self-support towers\n\n"
                
                if other:
                    response += f"\n  Other Tower Types ({len(other)}):\n"
                    for tower in other[:5]:
                        response += f"  • {tower['structure_name']} - {tower['tower_type']}\n"
            
            # 10+ years overdue (CRITICAL)
            if critical_10plus:
                response += f"\n🔴 CRITICAL - 10+ YEARS OVERDUE ({len(critical_10plus)} towers):\n"
                response += f"{'-'*80}\n"
                for i, tower in enumerate(critical_10plus[:15], 1):
                    response += f"{i}. {tower['structure_name'] or tower['site_name']}\n"
                    response += f"   Type: {tower['tower_type']} ({tower['tower_category']}) - Required every {tower['required_inspection_years']} years\n"
                    response += f"   Last visited: {tower['last_visit_date'].strftime('%Y-%m-%d')} ({tower['years_since_visit']:.1f} years ago)\n"
                    response += f"   Height: {tower['tower_height']} ft | FCC ASR: {tower['fcc_asr_number'] or 'N/A'}\n"
                    response += f"   Site: {tower['site_name']}\n\n"
                if len(critical_10plus) > 15:
                    response += f"   ... and {len(critical_10plus) - 15} more\n\n"
            
            # 7-10 years overdue (HIGH)
            if high_7plus:
                response += f"\n⚠️  HIGH PRIORITY - 7-10 YEARS OVERDUE ({len(high_7plus)} towers):\n"
                response += f"{'-'*80}\n"
                for i, tower in enumerate(high_7plus[:10], 1):
                    response += f"{i}. {tower['structure_name'] or tower['site_name']}\n"
                    response += f"   Type: {tower['tower_category']} - Last visit: {tower['last_visit_date'].strftime('%Y-%m-%d')} ({tower['years_since_visit']:.1f} years ago)\n"
                if len(high_7plus) > 10:
                    response += f"   ... and {len(high_7plus) - 10} more\n\n"
            
            # 5-7 years overdue (HIGH)
            if high_5plus:
                response += f"\n⚠️  HIGH PRIORITY - 5-7 YEARS OVERDUE ({len(high_5plus)} towers):\n"
                response += f"{'-'*80}\n"
                for i, tower in enumerate(high_5plus[:10], 1):
                    response += f"{i}. {tower['structure_name'] or tower['site_name']}\n"
                    response += f"   Type: {tower['tower_category']} - Last visit: {tower['last_visit_date'].strftime('%Y-%m-%d')} ({tower['years_since_visit']:.1f} years ago)\n"
                if len(high_5plus) > 10:
                    response += f"   ... and {len(high_5plus) - 10} more\n\n"
            
            # 3-5 years overdue (MEDIUM)
            if medium_overdue:
                response += f"\n⚠️  MEDIUM PRIORITY - 3-5 YEARS OVERDUE ({len(medium_overdue)} towers):\n"
                response += f"{'-'*80}\n"
                guyed_medium = [t for t in medium_overdue if t['tower_category'] == 'Guyed']
                if guyed_medium:
                    response += f"  Note: These are Guyed towers (3-year requirement)\n"
                for i, tower in enumerate(medium_overdue[:10], 1):
                    response += f"{i}. {tower['structure_name'] or tower['site_name']}\n"
                    response += f"   Type: {tower['tower_category']} - Last visit: {tower['last_visit_date'].strftime('%Y-%m-%d')} ({tower['years_since_visit']:.1f} years ago)\n"
                if len(medium_overdue) > 10:
                    response += f"   ... and {len(medium_overdue) - 10} more\n\n"
            
            # Summary statistics
            response += f"\n{'='*80}\n"
            response += f"📊 SUMMARY BY TOWER TYPE:\n"
            response += f"{'='*80}\n"
            
            all_guyed = [r for r in results if r['tower_category'] == 'Guyed']
            all_monopole = [r for r in results if r['tower_category'] == 'Monopole']
            all_self_support = [r for r in results if r['tower_category'] == 'Self-Support']
            all_other = [r for r in results if r['tower_category'] == 'Other']
            
            response += f"\nGuyed Towers (3-year inspection cycle):\n"
            response += f"  Total overdue: {len(all_guyed)}\n"
            response += f"  Never inspected: {len([r for r in all_guyed if r['last_visit_date'] is None])}\n"
            
            response += f"\nMonopole Towers (5-year inspection cycle):\n"
            response += f"  Total overdue: {len(all_monopole)}\n"
            response += f"  Never inspected: {len([r for r in all_monopole if r['last_visit_date'] is None])}\n"
            
            response += f"\nSelf-Support Towers (5-year inspection cycle):\n"
            response += f"  Total overdue: {len(all_self_support)}\n"
            response += f"  Never inspected: {len([r for r in all_self_support if r['last_visit_date'] is None])}\n"
            
            if all_other:
                response += f"\nOther Tower Types:\n"
                response += f"  Total overdue: {len(all_other)}\n"
            
            response += f"\n{'='*80}\n"
            response += f"OVERALL SUMMARY:\n"
            response += f"{'='*80}\n"
            response += f"  🔴 Never inspected:              {len(never_inspected):>6} towers\n"
            response += f"  🔴 Critically overdue (10+ yrs): {len(critical_10plus):>6} towers\n"
            response += f"  ⚠️  High priority (7-10 yrs):     {len(high_7plus):>6} towers\n"
            response += f"  ⚠️  High priority (5-7 yrs):      {len(high_5plus):>6} towers\n"
            response += f"  ⚠️  Medium priority (3-5 yrs):    {len(medium_overdue):>6} towers\n"
            response += f"  {'─'*80}\n"
            response += f"  📋 TOTAL NEEDING INSPECTION:     {len(results):>6} towers\n"
            response += f"{'='*80}\n"
            
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