#!/usr/bin/env python3
"""
Simple MCP Server for testing
This server provides basic tools for demonstration purposes
"""

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Create server instance
app = Server("demo-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="get_greeting",
            description="Get a personalized greeting message",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the person to greet"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="calculate",
            description="Perform basic mathematical calculations",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The mathematical operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        ),
        Tool(
            name="get_time_info",
            description="Get current date and time information",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["full", "date", "time"],
                        "description": "Format of the time information"
                    }
                },
                "required": ["format"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    
    if name == "get_greeting":
        person_name = arguments.get("name", "there")
        greeting = f"Hello, {person_name}! Welcome to the MCP demo server. 👋"
        return [TextContent(type="text", text=greeting)]
    
    elif name == "calculate":
        operation = arguments.get("operation")
        a = arguments.get("a")
        b = arguments.get("b")
        
        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return [TextContent(type="text", text="Error: Division by zero")]
                result = a / b
            else:
                return [TextContent(type="text", text=f"Unknown operation: {operation}")]
            
            return [TextContent(
                type="text",
                text=f"Result: {a} {operation} {b} = {result}"
            )]
        except Exception as e:
            return [TextContent(type="text", text=f"Calculation error: {str(e)}")]
    
    elif name == "get_time_info":
        from datetime import datetime
        now = datetime.now()
        
        format_type = arguments.get("format", "full")
        
        if format_type == "full":
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        elif format_type == "date":
            time_str = now.strftime("%Y-%m-%d")
        elif format_type == "time":
            time_str = now.strftime("%H:%M:%S")
        else:
            time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        
        return [TextContent(
            type="text",
            text=f"Current {format_type}: {time_str}"
        )]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Main entry point for the server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())