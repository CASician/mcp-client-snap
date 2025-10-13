
# MCP Client (Groq + MCP Tools)

This repository contains an asynchronous **MCP client** that connects to an MCP-compatible server (Python or Node.js) and integrates with **Groq’s Llama model** to process user queries.  
It automatically detects and uses tools provided by the server via MCP, allowing LLMs to call them dynamically.

---

## Features

- Connects to any MCP-compliant server (`.py` or `.js`)
- Lists and uses server-available tools dynamically
- Integrates with [Groq API](https://console.groq.com)
- Supports function calling for tool execution
- Maintains interactive chat loop
- Logs all interactions to `llm_output.log`

---



##  How to use

### 1. Prepare a Server

You need an MCP-compatible server.
See casician/mcp-server-snap
```bash
python server.py
```

---

### 2. Run the Client

```bash
python client.py <path_to_server_script>
```

If successful, you’ll see output like:

```
Connected to server with tools: ['get_agencies', 'get_bus_lines', ...]
```

---

### 3. Start Chatting

Once connected, you can enter queries interactively:

```
MCP Client Started!
Type your queries or 'quit' to exit.

Query: list all the available tpl agencies and their websites. 
```

If the model decides to use a tool, you’ll see it in the log and terminal output.
Otherwise, the model will answer directly.

---

## 🧾 Logging

All queries, raw model responses, and tool calls are saved in:

```
llm_output.log
```

Example excerpt:

```
2025-10-02 09:23:17 USER QUERY: calculate the area of a circle with radius 5
2025-10-02 09:23:19 FUNCTION CALLED: math_tool
2025-10-02 09:23:19 ARGS: {"operation": "area_circle", "radius": 5}
2025-10-02 09:23:20 FOLLOWUP RESPONSE: The area is approximately 78.54.
```

---

## Cleanup

When exiting (typing `quit`), the client automatically closes all asynchronous resources.

---

## ️ Troubleshooting

* **Error: "Usage: python client.py <path_to_server_script>"**
  → You forgot to pass the path to the MCP server file.

* **Error: “Server script path must end with .py or .js”**
  → The server file must be a valid Python or Node.js script.

* **Model or API issues**
  → Check that your `.env` file contains a valid `GROQ_API_KEY`.

* **Virtual environment issues**
  → Make sure the correct Python interpreter is active.

---

## Notes

* The model used is:

  ```
  meta-llama/llama-4-scout-17b-16e-instruct
  ```

  You can change it by editing the `self.model` field in `MCPClient`.

* Tool calls follow the OpenAI-style `function_call` format, so the client should work with any LLM that supports it.

---

## License

MIT License © 2025

