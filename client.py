import asyncio
import json
import os
import sys
from typing import Optional
from contextlib import AsyncExitStack
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import AsyncGroq

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    filename="llm_output.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self):
        # initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = "llama-3.1-8b-instant"
        self.openai = AsyncGroq(api_key=self.api_key, base_url="https://api.groq.com/")

    # methods will go here
    async def connect_to_server(self, server_script_path: str):
        """
        Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError('Server script path must end with .py or .js')

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # list available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\n Connected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """
        Process a query using Groq and available tools
        """
        messages = [
            {"role": "user", "content": query}
        ]
        logger.info("USER QUERY: %s", query)

        response = await self.session.list_tools()
        functions = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
            for tool in response.tools
        ]

        # Initial Groq API call
        groq_resp = await self.openai.chat.completions.create(
            model = self.model,
            messages = messages,
            max_tokens = 500,
            functions = functions,
            function_call="auto"
        )

        # Process response and handle tool calls
        assistant_msg = groq_resp.choices[0].message
        logger.info("RAW MODEL RESPONSE: %s", assistant_msg)

        fn_call = getattr(assistant_msg, "function_call", None)
        # Se c'è una funzione da chiamare
        if fn_call:
            # assumiamo un solo tool_call per semplicità
            fn_name = fn_call.name
            fn_args = fn_call.arguments
            logger.info("FUNCTION CALLED: %s", fn_name)
            logger.info("ARGS: %s", fn_args)

            # Parse degli argomenti
            if isinstance(fn_args, str):
                try:
                    args = json.loads(fn_args) if fn_args else {}
                except Exception:
                    args = {}
            elif isinstance(fn_args, dict):
                args = fn_args
            else:
                args = {}

            # Chiamo lo strumento
            tool_result = await self.session.call_tool(fn_name, args)

            # Aggiungo il messaggio “function” alla cronologia
            messages.append({
                "role": "function",
                "name": fn_name,
                "content": str(tool_result.content or "")
            })

            # Poi faccio una seconda chiamata a Groq, includendo il risultato
            followup = await self.openai.chat.completions.create(
                messages=messages,
                model=self.model,
                functions=functions,
                function_call="none"
            )
            logger.info("FOLLOWUP RESPONSE: %s", followup.choices[0].message)
            return followup.choices[0].message.content
        else:
            # Se non ci sono tool da chiamare, restituisco direttamente
            return assistant_msg.content or "I didn't use any tools."

    async def chat_loop(self):
        """
        Run an interactive chat loop
        """
        print("\n MCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\n Query: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\n Error: {str(e)}")

    async def cleanup(self):
        """ Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        logger.info("-------- NEW CONVERSATION --------")
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
