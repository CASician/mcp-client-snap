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
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        self.openai = AsyncGroq(base_url="https://api.groq.com/")
        self.messages = [{"role": "system", "content": ""}]

    async def connect_to_server(self, server_script_path: str):
        """
        Connect to an MCP server
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

        # Fetch and store available tools, resources, and prompts
        tools_resp = await self.session.list_tools()
        self.tools = tools_resp.tools

        res_resp = await self.session.list_resources()
        self.resources = res_resp.resources

        try:
            prompt_resp = await self.session.list_prompts()
            self.prompts = prompt_resp.prompts
        except Exception:
            self.prompts = []

        print("\nConnected to server with tools:", [t.name for t in self.tools])
        print("Connected to server with resources:", [r.name for r in self.resources])
        print("Connected to server with prompts:", [p.name for p in self.prompts])

    async def process_query(self, query: str) -> str:
        """
        Process a query using Groq and available tools/resources/prompts
        """
        self.messages.append({"role": "user", "content": query})
        logger.info("USER QUERY: %s", query)

        # Merge tools, resources, and prompts into a single callable schema
        functions = []

        # Tools as functions
        for tool in self.tools:
            functions.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            })

        # Resources as "get_resource" functions
        for resource in self.resources:
            functions.append({
                "name": f"get_resource_{resource.name}",
                "description": f"Access resource: {resource.name}. {resource.description}",
                "parameters": {"type": "object", "properties": {}}
            })

        # Prompts as "use_prompt" functions
        for prompt in self.prompts:
            functions.append({
                "name": f"use_prompt_{prompt.name}",
                "description": f"Use predefined prompt: {prompt.name}. {prompt.description}",
                "parameters": {"type": "object", "properties": {}}
            })

        # Initial Groq call
        groq_resp = await self.openai.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=500,
            functions=functions,
            function_call="auto"
        )

        assistant_msg = groq_resp.choices[0].message
        logger.info("RAW MODEL RESPONSE: %s", assistant_msg)

        fn_call = getattr(assistant_msg, "function_call", None)

        if fn_call:
            fn_name = fn_call.name
            fn_args = fn_call.arguments
            logger.info("FUNCTION CALLED: %s", fn_name)
            logger.info("ARGS: %s", fn_args)

            if isinstance(fn_args, str):
                try:
                    args = json.loads(fn_args) if fn_args else {}
                except Exception:
                    args = {}
            elif isinstance(fn_args, dict):
                args = fn_args
            else:
                args = {}

            # Handle 3 possible categories: tool / resource / prompt
            result_content = None

            if fn_name.startswith("get_resource_"):
                res_name = fn_name.replace("get_resource_", "")
                resource = next((r for r in self.resources if r.name == res_name), None)
                if not resource:
                    result_content = f"Resource '{res_name}' not found."
                else:
                    result = await self.session.read_resource(uri=resource.uri)
                    if result.contents and len(result.contents) > 0:
                        result_content = result.contents[0].text
                    else:
                        result_content = f"Resource '{res_name}' is empty or unreadable."

            elif fn_name.startswith("use_prompt_"):
                prompt_name = fn_name.replace("use_prompt_", "")
                result = await self.session.get_prompt(prompt_name)
                result_content = result.prompt.text if hasattr(result.prompt, "text") else str(result)

            else:
                result = await self.session.call_tool(fn_name, args)
                result_content = result.content or ""
            # ---------------- END ADDITION ------------------

            self.messages.append({
                "role": "function",
                "name": fn_name,
                "content": str(result_content)
            })

            followup = await self.openai.chat.completions.create(
                messages=self.messages,
                model=self.model,
                functions=functions,
                function_call="none"
            )

            logger.info("FOLLOWUP RESPONSE: %s", followup.choices[0].message)
            return followup.choices[0].message.content

        else:
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
