import asyncio
import json
import sys
from typing import Optional
from contextlib import AsyncExitStack
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from llama4.lab_llm import LabLLM 
from tool_schema_builder import build_system_tools

logging.basicConfig(
    filename="logs/llm_output.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# COLORS FOR BASH
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
NC = '\033[0m'

# Load the system message
with open("system_message.txt", "r", encoding="utf-8") as f:
    SYSTEM_MESSAGE = f.read()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        #self.openai = AsyncGroq(base_url="https://api.groq.com/")
        self.lab_llm = LabLLM()
        self.messages = []

    async def connect_to_server(self, server_script_path: str):
        """
        Connect to an MCP server
        """
        is_python = server_script_path.endswith('.py')
        if not is_python:
            raise ValueError('Server script path must end with .py')

        command = "python" 
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
        try:
            tools_resp = await self.session.list_tools()
            self.tools = tools_resp.tools
        except Exception:
            self.tools = []

        try:
            res_resp = await self.session.list_resources()
            self.resources = res_resp.resources
        except Exception:
            self.resources = []

        try:
            prompt_resp = await self.session.list_prompts()
            self.prompts = prompt_resp.prompts
        except Exception:
            self.prompts = []

        self.messages.append({"role": "system", "content": SYSTEM_MESSAGE + build_system_tools(self.tools)})
        # TODO ADD LOGIC FOR RESOURCES AND PROMPTS HERE?

        print(f"\n{BLUE}Connected to server with: {NC}")
        print(f"\n{BLUE}TOOLS: {NC}\n", [t.name for t in self.tools])
        print(f"\n{BLUE}RESOURCES: {NC}\n", [r.name for r in self.resources])
        print(f"\n{BLUE}PROMPTS: {NC}\n", [p.name for p in self.prompts])

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

        # TODO for FELIX, have fun! 
        raw_resp = self.lab_llm.chat_completion(
            messages=self.messages,
            functions=functions,
            function_call="auto",
        )

        first_msg = raw_resp["choices"][0]["message"]
        self.messages.append(first_msg)
        logger.info("FIRST RESPONSE: %s", json.dumps(first_msg, indent=4))

        fn_call = first_msg.get("function_call")

        if fn_call:
            fn_name = fn_call.get("name")
            fn_args = fn_call.get("arguments", {})
            logger.info("FUNCTION CALLED: %s", fn_name)

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
            # end of handling categories

            self.messages.append({
                "role": "function",
                "name": fn_name,
                "content": str(result_content) + f"Show these results in natural language. State that nothing has been retrieved if that is the case."
            })

            # TODO for FELIX, have fun again ehehe
            followup = self.lab_llm.chat_completion(
                messages=self.messages,
                function_call="none",
            )
            
            followup_msg = followup["choices"][0]["message"]
            self.messages.append(followup_msg)
            logger.info("FOLLOWUP RESPONSE: %s", json.dumps(followup_msg, indent=4))
            return followup_msg.get("content")
        else:
            return first_msg.get("content", "I didn't use any tools.")

    async def chat_loop(self):
        """
        Run an interactive chat loop
        """
        print(f"\n{RED}MCP Client Started!{NC}")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input(f"\n {GREEN}Query: {NC}").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\n Error: {str(e)}")
                print(traceback.format_exc())

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
