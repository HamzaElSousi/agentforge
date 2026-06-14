"""Tool registry: the ``@tool`` decorator + auto-generated JSON schema + lookup.

Decorate any typed Python function with :func:`tool` and it becomes callable by
an LLM agent — the argument schema is generated from the signature and the
description from the docstring, in under 10 lines and zero framework edits.

Tools receive a :class:`ToolContext` as their first positional argument
(injected by the executor, never by the model) so they can reach the run
workspace and sandbox without the model controlling those values.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_args, get_origin, get_type_hints

from agentforge.messages import ToolSpec

#: Risk tiers used by the permission layer to decide default gating.
RiskLevel = str  # "read_only" | "side_effecting" | "dangerous"


@dataclass
class ToolContext:
    """Injected runtime context handed to every tool at call time.

    The model never sees or controls these fields — they are bound by the
    executor. Tools use them to resolve workspace paths, run code in the
    configured sandbox, and emit structured notes.
    """

    workspace: Any = None  # agentforge.tools.files.Workspace (late-bound)
    sandbox: Any = None  # agentforge.sandbox.base.Sandbox (late-bound)
    network: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """A registered tool: the callable plus its generated schema and metadata."""

    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any]  # JSON schema for the model-visible args
    risk: RiskLevel = "side_effecting"
    needs_context: bool = True
    needs_network: bool = False

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


# --- JSON-schema generation from type hints -------------------------------- #

_PRIMITIVE_SCHEMA: dict[Any, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
}


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    """Best-effort JSON-schema for a parameter annotation."""
    if annotation in _PRIMITIVE_SCHEMA:
        return dict(_PRIMITIVE_SCHEMA[annotation])

    origin = get_origin(annotation)
    if origin is list:
        (item_type,) = (get_args(annotation) or (str,))
        return {"type": "array", "items": _annotation_to_schema(item_type)}
    if origin is dict:
        return {"type": "object"}
    # Optional[X] / Union[X, None] -> schema of X
    if origin is not None:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _annotation_to_schema(args[0])
    return {"type": "string"}


def _build_schema(func: Callable[..., Any]) -> tuple[dict[str, Any], bool]:
    """Return (json_schema, needs_context) derived from the signature.

    The first parameter named ``ctx`` (or annotated ToolContext) is treated as
    injected and excluded from the model-visible schema.
    """
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    needs_context = False

    params = list(sig.parameters.values())
    for i, param in enumerate(params):
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if i == 0 and (param.name in ("ctx", "context") or hints.get(param.name) is ToolContext):
            needs_context = True
            continue

        annotation = hints.get(param.name, str)
        schema = _annotation_to_schema(annotation)
        properties[param.name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(param.name)

    json_schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        json_schema["required"] = required
    return json_schema, needs_context


class ToolRegistry:
    """A named collection of tools. The global default registry lives below."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, t: Tool) -> Tool:
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def subset(self, names: list[str]) -> list[Tool]:
        """Return tools for ``names`` in order; raise on unknown tool."""
        out: list[Tool] = []
        for n in names:
            t = self._tools.get(n)
            if t is None:
                raise KeyError(f"Unknown tool: {n!r}. Registered: {sorted(self._tools)}")
            out.append(t)
        return out


#: Process-wide default registry that ``@tool`` writes into.
REGISTRY = ToolRegistry()


def tool(
    _func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    risk: RiskLevel = "side_effecting",
    needs_network: bool = False,
    registry: Optional[ToolRegistry] = None,
) -> Callable[..., Any]:
    """Register a typed Python function as an LLM-callable tool.

    Example::

        @tool(risk="read_only")
        def add(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        schema, needs_context = _build_schema(func)
        desc = description or (inspect.getdoc(func) or "").strip() or func.__name__
        t = Tool(
            name=name or func.__name__,
            description=desc,
            func=func,
            parameters=schema,
            risk=risk,
            needs_context=needs_context,
            needs_network=needs_network,
        )
        (registry or REGISTRY).register(t)
        return t

    if _func is not None:  # used as bare @tool
        return decorator(_func)
    return decorator
