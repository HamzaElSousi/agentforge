"""FastAPI app: serve the dashboard and stream pipeline runs as NDJSON.

The run executes in a worker thread; its emitted events are pushed onto a
thread-safe queue that the streaming response drains to the browser. This keeps
the synchronous orchestrator unchanged while giving the UI a live feed.
"""

from __future__ import annotations

import json
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

_STATIC = Path(__file__).parent / "static"
_SENTINEL = object()


def _list_pipelines(examples_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if examples_dir.is_dir():
        for p in sorted(examples_dir.glob("*.yaml")):
            try:
                out.append({"name": p.name, "yaml": p.read_text(encoding="utf-8")})
            except Exception:
                continue
    return out


def create_app(examples_dir: Optional[Path] = None):
    """Build the FastAPI app. Imports FastAPI lazily so the dependency is only
    needed when actually serving."""
    try:
        from fastapi import Body, FastAPI
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "the web UI needs extra deps — install with:  pip install \"agentforge[web]\""
        ) from e

    examples_dir = examples_dir or (Path.cwd() / "examples")
    app = FastAPI(title="AgentForge", docs_url=None, redoc_url=None)

    @app.get("/")
    def index():  # pragma: no cover - static file
        return FileResponse(_STATIC / "index.html")

    @app.get("/api/pipelines")
    def pipelines():
        return JSONResponse({"pipelines": _list_pipelines(examples_dir)})

    @app.get("/api/models")
    def models():
        """Live tool-capable models + pricing (best-effort; never fatal)."""
        from agentforge.cost import fetch_openrouter_models

        try:
            data = fetch_openrouter_models()
        except Exception as e:
            return JSONResponse({"error": str(e), "models": []})
        rows = []
        for m in data:
            params = m.get("supported_parameters") or []
            if "tools" not in params:
                continue
            pr = m.get("pricing", {})
            rows.append({
                "id": m.get("id"),
                "in": float(pr.get("prompt") or 0) * 1e6,
                "out": float(pr.get("completion") or 0) * 1e6,
                "context": m.get("context_length"),
            })
        rows.sort(key=lambda r: r["in"])
        return JSONResponse({"models": rows[:60]})

    @app.post("/api/run")
    def run(payload: dict = Body(...)):
        goal = (payload.get("goal") or "").strip()
        pipeline_yaml = payload.get("yaml") or ""
        assume_yes = bool(payload.get("assume_yes", True))
        if not goal or not pipeline_yaml.strip():
            return JSONResponse({"error": "both 'goal' and 'yaml' are required"}, status_code=400)

        events: "queue.Queue[Any]" = queue.Queue()

        def emit(event: dict) -> None:
            events.put(event)

        def worker() -> None:
            from agentforge.orchestrator import run_pipeline

            tmpdir = Path(tempfile.mkdtemp(prefix="agentforge-web-"))
            ppath = tmpdir / "pipeline.yaml"
            ppath.write_text(pipeline_yaml, encoding="utf-8")
            try:
                run_pipeline(
                    pipeline_path=ppath,
                    goal=goal,
                    trace_path=tmpdir / "trace.json",
                    assume_yes=assume_yes,
                    console=Console(quiet=True),
                    emit=emit,
                )
            except SystemExit as e:
                events.put({"type": "error", "message": f"run aborted (exit {e.code}) — "
                            "check that the provider API key is set in your environment / .env"})
            except Exception as e:  # surface any failure to the UI
                events.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                events.put(_SENTINEL)

        threading.Thread(target=worker, daemon=True).start()

        def stream():
            while True:
                event = events.get()
                if event is _SENTINEL:
                    break
                yield json.dumps(event) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return app


def serve(host: str = "127.0.0.1", port: int = 8000, examples_dir: Optional[Path] = None) -> None:
    """Launch the dashboard with uvicorn."""
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "the web UI needs extra deps — install with:  pip install \"agentforge[web]\""
        ) from e
    uvicorn.run(create_app(examples_dir), host=host, port=port, log_level="warning")
