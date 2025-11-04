# MCP Client 

This repository contains an asynchronous **MCP client** that connects to an MCP-compatible server (Python) and integrates with **Llama model** to process user queries.  
It automatically detects and uses tools provided by the server via MCP, allowing LLMs to call them dynamically.


## Features

- Connects to any MCP-compliant server (`.py`) 
- Lists and uses server-available tools dynamically
- Supports function calling for tool execution
- Maintains interactive chat loop
- Logs all interactions to `logs/llm_output.log`



##  How to use
If you read this from the VM, skip to nr. 3

### 1. Prepare a Server

You need an MCP-compatible server. It will run in background. 
See [mcp-server-snap](https://github.com/casician/mcp-server-snap.git) for more info.


### 2. Connect to DISIT LAB LLM. 
Create a new file called `user_credentials.json`. 

It is Used by the client to authenticate to Snap4City and obtain an access token. You must use a Snap4City account that has been granted with the necessary API rules and usage limits.

```json
{
  "username": "<SNAP4CITY_USERNAME>",
  "password": "<PASSWORD>"
}
```


### 3. Run the Client

```bash
chat
```
This is an alias for `./chat.sh` command.

If successful, you’ll see the token-handling and an output like:

```
Connected to server with:
TOOLS: [...]
RESOURCES: [...]
PROMPTS: [...]
```


### 4. Start Chatting

Once connected, you can enter queries interactively:

```
MCP Client Started!
Type your queries or 'quit' to exit.

Query: list all the available tpl agencies and their websites. 
```

If the model decides to use a tool, you’ll see it in the log and terminal output.
Otherwise, the model will answer directly.


##  Logging

All queries, raw model responses, and tool calls are saved in:

```
log/llm_output.log
```

Example excerpt:

```
2025-10-02 09:23:17 USER QUERY: calculate the area of a circle with radius 5
2025-10-02 09:23:19 RAW MODEL RESPONSE: {'role': 'assistant', 'content': None, 'function_call': {'name': 'tool_name', 'arguments': {...}}}
2025-10-02 09:23:19 FUNCTION CALLED: math_tool
2025-10-02 09:23:19 ARGS: {"operation": "area_circle", "radius": 5}
2025-10-02 09:23:20 FOLLOWUP RESPONSE: The area is approximately 78.54.
```


## Cleanup

When exiting (typing `quit`), the client automatically closes all asynchronous resources.


## ️ Troubleshooting

* **Error: "Usage: python client.py <path_to_server_script>"**
  → You didn't pass the path to the MCP server file.

* **Virtual environment issues**
  → Make sure the correct Python interpreter is active.

* **Connection to DISIT resources**
  → Make sure that the credentials are valid and passed correctly as in `How to use: 2. ` 


## Notes

* The model used is:

  ```
  llama-4-inference
  ```


* Tool calls follow the OpenAI-style `function_call` format, so the client should work with any LLM that supports it.




