import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from threading import Thread
import queue
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

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        if is_python:
            path = Path(server_script_path).resolve()
            server_params = StdioServerParameters(
                command="uv",
                args=["--directory", str(path.parent), "run", path.name],
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

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        if not self.connected:
            return "Error: Not connected to MCP server"

        messages = [{"role": "user", "content": query}]

        response = await self.session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in response.tools
        ]

        response = self.anthropic.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        final_text = []

        for content in response.content:
            if content.type == "text":
                final_text.append(content.text)
            elif content.type == "tool_use":
                tool_name = content.name
                tool_args = content.input

                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"🔧 Called tool: {tool_name}")

                if hasattr(content, "text") and content.text:
                    messages.append({"role": "assistant", "content": content.text})
                messages.append({"role": "user", "content": result.content})

                response = self.anthropic.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=1000,
                    messages=messages,
                )

                final_text.append(response.content[0].text)

        return "\n\n".join(final_text)

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
    return asyncio.run_coroutine_threadsafe(coro, event_loop).result()

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
        
        tools = run_async(mcp_client.connect_to_server(server_path))
        
        return jsonify({
            'success': True,
            'message': 'Connected successfully',
            'tools': tools
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        user_query = data.get('query')
        
        if not user_query:
            return jsonify({'error': 'Query is required'}), 400
        
        response = run_async(mcp_client.process_query(user_query))
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        'connected': mcp_client.connected,
        'tools': [tool.name for tool in mcp_client.tools] if mcp_client.connected else []
    })

if __name__ == '__main__':
    print("\n🚀 MCP Web Interface Starting...")
    print("📍 Open your browser to: http://localhost:5000")
    app.run(debug=True, port=5000, use_reloader=False)