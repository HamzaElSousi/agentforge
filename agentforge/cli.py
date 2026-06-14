"""AgentForge command-line interface (Typer).

Commands
--------
- ``run``      run a pipeline against a goal
- ``models``   live-probe OpenRouter's catalog for tool-capable models + pricing
- ``validate`` validate a pipeline YAML without running it
- ``estimate`` dry-run prompt sizes and print projected cost

The heavier commands (``run``/``models``/``estimate``) are wired to their
implementations as later phases land; ``validate`` is fully functional now.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from agentforge import __version__

app = typer.Typer(
    name="agentforge",
    help="Build multi-agent AI pipelines from a YAML file. Bring your own model.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"agentforge {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """AgentForge — YAML-defined multi-agent pipelines."""


@app.command()
def validate(
    pipeline: Path = typer.Argument(..., help="Path to pipeline YAML."),
) -> None:
    """Validate a pipeline file and report any configuration errors."""
    from agentforge.config import ConfigError, load_pipeline

    try:
        cfg = load_pipeline(pipeline)
    except ConfigError as e:
        err_console.print(f"[bold red]✗ invalid[/bold red]\n{e}")
        raise typer.Exit(code=1)

    console.print(f"[bold green]✓ valid[/bold green] — pipeline [bold]{cfg.name}[/bold]")
    console.print(f"  agents: {', '.join(cfg.agents)}")
    console.print(f"  start:  {cfg.start}")
    console.print(f"  provider/model: {cfg.llm.provider.value} / {cfg.llm.model}")
    console.print(f"  budget: ${cfg.budget.max_usd_per_run}/run, "
                  f"{cfg.budget.max_total_iterations} iters")


@app.command()
def models(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Filter slug substring."),
    tools_only: bool = typer.Option(True, help="Only show tool-capable models."),
) -> None:
    """List tool-capable OpenRouter models with live pricing."""
    from agentforge.cost import print_models_table

    print_models_table(console, query=query, tools_only=tools_only)


@app.command()
def run(
    pipeline: Path = typer.Argument(..., help="Path to pipeline YAML."),
    goal: str = typer.Option(..., "--goal", "-g", help="The objective for the pipeline."),
    trace: Path = typer.Option(Path("trace.json"), "--trace", help="Where to write the trace."),
    estimate_only: bool = typer.Option(
        False, "--estimate", help="Estimate cost without calling the model."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Auto-approve gated tools (non-interactive auto mode)."
    ),
) -> None:
    """Run a pipeline end-to-end against GOAL and write a trace."""
    from agentforge.orchestrator import run_pipeline

    run_pipeline(
        pipeline_path=pipeline,
        goal=goal,
        trace_path=trace,
        estimate_only=estimate_only,
        assume_yes=yes,
        console=console,
    )


@app.command()
def estimate(
    pipeline: Path = typer.Argument(..., help="Path to pipeline YAML."),
    goal: str = typer.Option(..., "--goal", "-g", help="The objective for the pipeline."),
) -> None:
    """Project the cost of a run without spending anything (alias for run --estimate)."""
    from agentforge.orchestrator import run_pipeline

    run_pipeline(
        pipeline_path=pipeline,
        goal=goal,
        trace_path=None,
        estimate_only=True,
        assume_yes=True,
        console=console,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
