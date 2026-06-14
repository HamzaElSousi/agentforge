"""Tiered sandbox abstraction: subprocess (default), Docker, E2B."""

from __future__ import annotations

from agentforge.sandbox.base import ExecResult, Sandbox


def make_sandbox(cfg, workdir: str) -> Sandbox:
    """Construct the appropriate :class:`Sandbox` from a :class:`SandboxConfig`.

    Concrete backend classes are imported *lazily* inside each branch so that
    optional dependencies (``docker``, ``e2b_code_interpreter``) are not
    imported at module load time — only when the corresponding backend is
    actually requested.

    Parameters
    ----------
    cfg:
        A :class:`agentforge.config.SandboxConfig` instance (typed as ``Any``
        here to avoid a circular import; validated by Pydantic at load time).
    workdir:
        Absolute path to the per-run workspace directory handed to the sandbox.

    Returns
    -------
    Sandbox
        A concrete sandbox instance ready to accept :meth:`Sandbox.run_python`
        calls.

    Raises
    ------
    RuntimeError
        If the selected backend's optional dependency is not installed.
    ValueError
        If ``cfg.backend`` is not a recognised :class:`SandboxBackend` value.
    """
    common = dict(
        network=cfg.network,
        timeout_s=cfg.timeout_s,
        cpu_seconds=cfg.cpu_seconds,
        memory_mb=cfg.memory_mb,
    )

    backend_value = cfg.backend.value if hasattr(cfg.backend, "value") else str(cfg.backend)

    if backend_value == "subprocess":
        from agentforge.sandbox.subprocess import SubprocessSandbox
        return SubprocessSandbox(workdir, **common)

    if backend_value == "docker":
        from agentforge.sandbox.docker import DockerSandbox
        return DockerSandbox(workdir, **common)

    if backend_value == "e2b":
        from agentforge.sandbox.e2b import E2BSandbox
        return E2BSandbox(workdir, **common)

    raise ValueError(
        f"Unknown sandbox backend: {backend_value!r}. "
        "Valid choices are: 'subprocess', 'docker', 'e2b'."
    )


__all__ = ["ExecResult", "Sandbox", "make_sandbox"]
