import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from threading import Thread
import os

from anthropic import Anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

app = Flask(__name__)

class MCPClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic(api_key=os.getenv("API_KEY"))
        self.connected = False
        self.tools = []

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        path = Path(server_script_path).resolve()
        
        if is_python:
            # Try using python directly instead of uv
            server_params = StdioServerParameters(
                command="python",  # or "python3" on some systems
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
        
        return [tool.name for tool in self.tools]

    async def process_query(self, query: str) -> dict:
        """Process a query using Claude and available tools"""
        if not self.connected:
            return {"error": "Not connected to MCP server"}

        messages = [{"role": "user", "content": query}]

        # Get available tools
        response = await self.session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in response.tools
        ]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,
            messages=messages,
            tools=available_tools
        )

        # Track conversation and tool calls
        conversation_parts = []
        tools_used = []

        # Process response
        while response.stop_reason == "tool_use":
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

                    # Execute tool via MCP
                    result = await self.session.call_tool(tool_name, tool_args)
                    
                    # Extract text from result
                    result_text = ""
                    for item in result.content:
                        if hasattr(item, 'text'):
                            result_text += item.text
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": result_text
                    })

            # Add tool results to messages
            messages.append({
                "role": "user",
                "content": tool_results
            })

            # Get next response from Claude
            response = self.anthropic.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=2000,
                messages=messages,
                tools=available_tools
            )

        # Extract final text response
        final_text = ""
        for content_block in response.content:
            if content_block.type == "text":
                final_text += content_block.text

        return {
            "response": final_text,
            "tools_used": tools_used,
            "success": True
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
    return future.result(timeout=1800)  # Add timeout

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
        
        # Validate file exists
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
        return jsonify({'success': True, 'message': 'Disconnected'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 MCP Web Interface Starting...")
    print("="*60)
    print("\n📍 Open your browser to: http://localhost:5000")
    print("\n💡 Tips:")
    print("  - Enter the full path to your mcp_server.py")
    print("  - Make sure your database is configured correctly")
    print("  - Check the console for error messages\n")
    app.run(debug=True, port=5000, use_reloader=False)