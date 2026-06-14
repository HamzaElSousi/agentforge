"""AgentForge — build multi-agent AI pipelines from a YAML file.

Bring your own model, your own provider, your own tools. The runtime runs a
ReAct loop, executes tools in a tiered sandbox, asks for human approval on
side-effecting tools, enforces a hard per-run USD cap, handles sequential
handoffs, and writes a full ``trace.json``.
"""

__version__ = "0.1.0"
