"""Small helpers for building pipeline YAML files in tests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

_MINIMAL = """\
name: {name}
llm:
  provider: openrouter
  model: {model}
  api_key_env: OPENROUTER_API_KEY
budget:
  max_usd_per_run: {cap}
  max_total_iterations: {total_iters}{max_parallel_line}
sandbox:
  backend: subprocess
permissions:
  mode: {mode}
  auto_approve: {auto_approve}
  require_approval: {require_approval}
  deny: {deny}
  non_interactive: {non_interactive}
agents:
{agents}
start: {start}
"""


def write_pipeline(
    tmp_path: Path,
    *,
    name: str = "t",
    model: str = "fake/model",
    cap: float = 0.25,
    total_iters: int = 30,
    mode: str = "auto",
    auto_approve=None,
    require_approval=None,
    deny=None,
    non_interactive: str = "deny",
    agents: str,
    start: str,
    max_parallel: Optional[int] = None,
) -> Path:
    # Inject max_parallel into the budget block only when explicitly provided.
    # When None, omit the line entirely so the config default of 4 applies.
    if max_parallel is not None:
        max_parallel_line = f"\n  max_parallel: {max_parallel}"
    else:
        max_parallel_line = ""

    text = _MINIMAL.format(
        name=name,
        model=model,
        cap=cap,
        total_iters=total_iters,
        max_parallel_line=max_parallel_line,
        mode=mode,
        auto_approve=auto_approve or [],
        require_approval=require_approval or [],
        deny=deny or [],
        non_interactive=non_interactive,
        agents=agents,
        start=start,
    )
    p = tmp_path / "pipeline.yaml"
    p.write_text(text)
    return p
