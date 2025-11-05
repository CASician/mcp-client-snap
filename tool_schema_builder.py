def build_system_tools(tools, primitive_name):
    """
    Builds the tool-specific section of the system message,
    listing the tools available to the LLM without argument schemas because it is already present in docstrings.
    """

    tool_descriptions = "\n\n".join([
        f"""
        {primitive_name} NAME: "{t.name}"
        DESCRIPTION: {t.description or 'No description provided.'}
        """
        for t in tools
    ])

    return f"""
    {primitive_name}S AVAILABLE:
    {tool_descriptions}
    """
