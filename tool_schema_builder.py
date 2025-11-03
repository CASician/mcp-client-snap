def build_system_tools(tools):
    """
    Builds the tool-specific section of the system message,
    listing the tools available to the LLM with argument schemas.
    """

    def format_parameters(params):
        if not params:
            return "This tool takes no parameters."
        formatted = []
        for p in params:
            param_type = getattr(p, "type", "string")
            desc = getattr(p, "description", "")
            formatted.append(f'  - "{p.name}" ({param_type}): {desc}')
        return "\n".join(formatted)

    tool_descriptions = "\n\n".join([
        f"""
Tool name: "{t.name}"
Description: {t.description or 'No description provided.'}
Parameters:
{format_parameters(getattr(t, "parameters", []))}
        """
        for t in tools
    ])

    return f"""
⚙️ Rules:
1. Output *only* valid JSON when calling a function — no markdown or extra text.
2. Always use double quotes for keys and string values.
3. If you can answer directly, respond with plain text.
4. Always choose the most appropriate tool.
5. NO EXTRA TEXT IF YOU MAKE A TOOL CALL
6. If you see that a tool call has already been made, analyze it and show the results to use user. Do not execute it again. 

Below is the list of tools you can call. 
When using a tool, you MUST respond in the following JSON format and ONLY with this:

{{
  "function_call": {{
    "name": "<tool_name>",
    "arguments": {{
      "<arg1>": "<value1>",
      "<arg2>": "<value2>"
    }}
  }}
}}

TOOLS AVAILABLE:
{tool_descriptions}

"""
