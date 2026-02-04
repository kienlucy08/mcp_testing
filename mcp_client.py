import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from threading import Thread
import os
import re

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# Token management constants
MAX_TOOL_RESULT_CHARS = 3000
MAX_MESSAGES_HISTORY = 15
MAX_TOTAL_TOOL_CALLS = 30

app = Flask(__name__)

def truncate_tool_result(result: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Truncate tool result to prevent token explosion"""
    if len(result) > max_chars:
        return result[:max_chars] + f"\n[TRUNCATED - {len(result) - max_chars} chars omitted]"
    return result

def select_relevant_tools(query: str, all_tools: list) -> list:
    """
    Intelligently select only relevant tools based on the query.
    This dramatically reduces token usage by not sending all tools every time.
    """
    query_lower = query.lower()
    
    # Always include these core tools
    core_tools = ["schema", "execute_query"]
    
    # Tool selection patterns
    tool_patterns = {
        "explore_data_relationships": [
            "related", "relationship", "foreign key", "joins", "how does",
            "data model", "connected", "link between"
        ],
        "query_inspection_data": [
            "safety climb", "climb", "deficiency", "deficiencies",
            "issue", "problem", "fault", "inspection data"
        ],
        "get_sites_needing_inspection": [
            "overdue", "due", "tia", "inspection needed", "needs inspection",
            "checkup", "compliance", "never inspected"
        ],
        "query_sites_by_location": [
            "coordinate", "location", "where is", "address", "near",
            "within", "nearby", "around", "radius", "miles", "find sites"
        ],
        "get_weather_for_site": [
            "weather", "temperature", "forecast", "wind", "conditions"
        ],
    }
    
    selected_tools = set(core_tools)
    
    # Check query against patterns
    for tool_name, patterns in tool_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            selected_tools.add(tool_name)
    
    # Special case: if asking about weather, also include location tool
    if "weather" in query_lower:
        selected_tools.add("get_weather_for_site")
        selected_tools.add("query_sites_by_location")
    # Special case: inspection queries might need relationships
    if "inspection" in query_lower:
        selected_tools.add("explore_data_relationships")
    
    # Filter the actual tool definitions
    relevant_tools = [
        tool for tool in all_tools 
        if tool["name"] in selected_tools
    ]
    
    return relevant_tools

class MCPClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic(api_key=os.getenv("API_KEY"))
        self.connected = False
        self.tools = []
        self.all_tools = []  # Store all available tools

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        path = Path(server_script_path).resolve()
        
        if is_python:
            server_params = StdioServerParameters(
                command="python",
                args=[str(path)],
                env=None,
            )
        else:
            server_params = StdioServerParameters(
                command="node", 
                args=[server_script_path], 
                env=None
            )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()
        
        response = await self.session.list_tools()
        self.tools = response.tools
        self.connected = True
        
        # Store formatted tools for dynamic selection
        self.all_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in response.tools
        ]
        
        return [tool.name for tool in self.tools]

    async def process_query(self, query: str) -> dict:
        """Process a query using Claude and available tools"""
        if not self.connected:
            return {"error": "Not connected to MCP server"}

        messages = [{"role": "user", "content": query}]

        # DYNAMIC TOOL SELECTION - Only send relevant tools
        relevant_tools = select_relevant_tools(query, self.all_tools)
        
        print(f"\n📊 Total available tools: {len(self.all_tools)}")
        print(f"📊 Selected relevant tools: {len(relevant_tools)}")
        print(f"📊 Tools selected: {[t['name'] for t in relevant_tools]}")

        # Initial Claude API call with only relevant tools
        response = self.anthropic.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4000,
            messages=messages,
            tools=relevant_tools  # Only send relevant tools!
        )

        # Initialize token tracking
        total_input_tokens = response.usage.input_tokens
        total_output_tokens = response.usage.output_tokens

        print(f"📊 Initial input tokens: {response.usage.input_tokens}")
        print(f"📊 Initial output tokens: {response.usage.output_tokens}")

        # Track conversation and tool calls
        tools_used = []
        tool_call_count = 0

        # Process response with safeguards
        while response.stop_reason == "tool_use":
            tool_call_count += 1

            # Safety: stop if too many tool calls
            if tool_call_count > MAX_TOTAL_TOOL_CALLS:
                print(f"⚠️  Stopping after {MAX_TOTAL_TOOL_CALLS} tool calls")
                # Add a message explaining we hit the limit
                final_text = "I've reached the maximum number of tool calls for this query. "
                final_text += f"Based on {tool_call_count} tool executions, here's what I found: "
                final_text += "\n\n[Results would be summarized here]"
                
                return {
                    "response": final_text,
                    "tools_used": tools_used,
                    "success": True,
                    "warning": "Hit max tool call limit"
                }

            # Add assistant's response to messages
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Process each tool use
            tool_results = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_name = content_block.name
                    tool_args = content_block.input

                    tools_used.append({
                        "name": tool_name,
                        "arguments": tool_args
                    })

                    print(f"\n🔧 Calling tool #{tool_call_count}: {tool_name}")

                    try:
                        # Execute tool via MCP
                        result = await self.session.call_tool(tool_name, tool_args)

                        # Extract text from result
                        result_text = ""
                        for item in result.content:
                            if hasattr(item, 'text'):
                                result_text += item.text

                        # Truncate tool results
                        original_length = len(result_text)
                        result_text = truncate_tool_result(result_text)
                        
                        if len(result_text) < original_length:
                            print(f"   ✂️  Truncated result from {original_length} to {len(result_text)} chars")

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": result_text
                        })
                    except Exception as e:
                        print(f"   ❌ Tool execution failed: {str(e)}")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Error executing tool: {str(e)}"
                        })

            # Add tool results to messages
            messages.append({
                "role": "user",
                "content": tool_results
            })

            # Trim message history if too long
            if len(messages) > MAX_MESSAGES_HISTORY:
                messages = [messages[0]] + messages[-(MAX_MESSAGES_HISTORY-1):]
                print(f"✂️  Trimmed message history to {len(messages)} messages")

            # Get next response from Claude (still with only relevant tools)
            response = self.anthropic.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=4000,
                messages=messages,
                tools=relevant_tools  # Keep using same relevant tools
            )

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            print(f"📊 Cumulative tokens - Input: {total_input_tokens}, Output: {total_output_tokens}")

        # Extract final text response
        final_text = ""
        for content_block in response.content:
            if content_block.type == "text":
                final_text += content_block.text

        print(f"\n✅ COMPLETE - Total Input: {total_input_tokens}, Output: {total_output_tokens}")
        print(f"✅ Tool calls made: {tool_call_count}")

        return {
            "response": final_text,
            "tools_used": tools_used,
            "success": True,
            "token_usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "tool_calls": tool_call_count
            }
        }

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

# Global client instance
mcp_client = MCPClient()
event_loop = None

def run_async_loop():
    """Run the async event loop in a separate thread"""
    global event_loop
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    event_loop.run_forever()

# Start async loop in background thread
thread = Thread(target=run_async_loop, daemon=True)
thread.start()

def run_async(coro):
    """Helper to run async functions from sync context"""
    future = asyncio.run_coroutine_threadsafe(coro, event_loop)
    return future.result(timeout=1800)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/connect', methods=['POST'])
def connect():
    try:
        data = request.json
        server_path = data.get('server_path')
        
        if not server_path:
            return jsonify({'error': 'Server path is required'}), 400
        
        if not Path(server_path).exists():
            return jsonify({'error': f'File not found: {server_path}'}), 400
        
        tools = run_async(mcp_client.connect_to_server(server_path))
        
        return jsonify({
            'success': True,
            'message': f'Connected successfully! Found {len(tools)} tools.',
            'tools': tools
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Connection error: {error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        user_query = data.get('query')
        
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400
        
        if not mcp_client.connected:
            return jsonify({'error': 'Not connected to MCP server'}), 400
        
        result = run_async(mcp_client.process_query(user_query))
        
        if result.get('error'):
            return jsonify({'error': result['error']}), 500
        
        return jsonify(result)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Query error: {error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500

@app.route('/status')
def status():
    return jsonify({
        'connected': mcp_client.connected,
        'tools': [tool.name for tool in mcp_client.tools] if mcp_client.connected else []
    })

@app.route('/disconnect', methods=['POST'])
def disconnect():
    try:
        run_async(mcp_client.cleanup())
        mcp_client.connected = False
        mcp_client.tools = []
        mcp_client.all_tools = []
        return jsonify({'success': True, 'message': 'Disconnected'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 MCP Web Interface Starting...")
    print("="*60)
    print("\n📍 Open your browser to: http://localhost:5000")
    print("\n💡 Features:")
    print("  - Dynamic tool selection (reduced token usage)")
    print("  - Automatic result truncation")
    print("  - Token usage monitoring")
    print("  - Tool call limits\n")
    app.run(debug=True, port=5000, use_reloader=False)