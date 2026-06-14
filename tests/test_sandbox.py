"""Tests for agentforge/sandbox/subprocess.py — SubprocessSandbox.

All tests use short timeouts and tmp_path as workdir. No test raises — all
failures must be reported in ExecResult fields per the sandbox contract.

Skipped on Windows: process-group kill (os.killpg) and rlimits are POSIX-only.
The sandbox degrades gracefully on Windows but the timeout enforcement and
resource limits are not guaranteed, making several tests unreliable there.
"""

from __future__ import annotations

import sys

import pytest

from agentforge.sandbox.base import ExecResult
from agentforge.sandbox.subprocess import SubprocessSandbox


# Skip the entire module on Windows where POSIX guarantees don't hold.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="SubprocessSandbox POSIX guarantees (killpg, rlimits) not available on Windows",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sandbox(tmp_path):
    """Default sandbox with generous limits for non-timeout tests."""
    return SubprocessSandbox(str(tmp_path), timeout_s=10, cpu_seconds=5, memory_mb=256)


# ---------------------------------------------------------------------------
# Basic execution — happy path
# ---------------------------------------------------------------------------


class TestBasicExecution:
    def test_simple_print_succeeds(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("print('hello')")
        assert result.ok, f"Expected ok=True, got: {result.summary()}"
        assert "hello" in result.stdout

    def test_stdout_captured(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("print('foo'); print('bar')")
        assert "foo" in result.stdout
        assert "bar" in result.stdout

    def test_exit_code_zero_on_success(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("x = 1 + 1")
        assert result.exit_code == 0

    def test_not_timed_out_on_fast_code(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("pass")
        assert result.timed_out is False

    def test_duration_ms_is_non_negative(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("pass")
        assert result.duration_ms >= 0

    def test_run_python_returns_exec_result_type(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("pass")
        assert isinstance(result, ExecResult)

    def test_multiline_code_executes(self, sandbox: SubprocessSandbox):
        code = """
x = 10
y = 20
print(x + y)
"""
        result = sandbox.run_python(code)
        assert result.ok
        assert "30" in result.stdout


# ---------------------------------------------------------------------------
# Timeout handling — timed_out, ok=False, no exception
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_infinite_loop_times_out(self, tmp_path):
        """Infinite loop must time out, set timed_out=True, and NOT raise."""
        sb = SubprocessSandbox(str(tmp_path), timeout_s=1, cpu_seconds=2, memory_mb=128)
        result = sb.run_python("while True: pass")
        assert result.timed_out is True, "Infinite loop must set timed_out=True"

    def test_timed_out_result_is_not_ok(self, tmp_path):
        sb = SubprocessSandbox(str(tmp_path), timeout_s=1, cpu_seconds=2, memory_mb=128)
        result = sb.run_python("while True: pass")
        assert result.ok is False, "A timed-out result must not be ok"

    def test_timeout_does_not_raise_exception(self, tmp_path):
        """The sandbox contract: timeouts are NEVER raised, always returned in ExecResult."""
        sb = SubprocessSandbox(str(tmp_path), timeout_s=1, cpu_seconds=2, memory_mb=128)
        # If this raised instead of returning, the test would fail with an exception
        result = sb.run_python("while True: pass")
        assert isinstance(result, ExecResult), "Timeout must return ExecResult, not raise"

    def test_timeout_summary_mentions_timed_out(self, tmp_path):
        sb = SubprocessSandbox(str(tmp_path), timeout_s=1, cpu_seconds=2, memory_mb=128)
        result = sb.run_python("while True: pass")
        summary = result.summary()
        assert "timed out" in summary.lower() or "timeout" in summary.lower()


# ---------------------------------------------------------------------------
# Non-zero exit code — captured, not raised
# ---------------------------------------------------------------------------


class TestNonZeroExit:
    def test_sys_exit_3_captured_in_exit_code(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("import sys; sys.exit(3)")
        assert result.exit_code == 3, f"Expected exit_code=3, got {result.exit_code}"

    def test_sys_exit_3_result_is_not_ok(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("import sys; sys.exit(3)")
        assert result.ok is False

    def test_sys_exit_3_does_not_raise(self, sandbox: SubprocessSandbox):
        """Non-zero exit must be returned in ExecResult, never raised."""
        result = sandbox.run_python("import sys; sys.exit(3)")
        assert isinstance(result, ExecResult)

    def test_raise_system_exit_captured(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("raise SystemExit(42)")
        assert result.exit_code == 42

    def test_summary_for_nonzero_exit_contains_exit_code(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("import sys; sys.exit(1)")
        summary = result.summary()
        assert "1" in summary or "exit" in summary.lower()


# ---------------------------------------------------------------------------
# Runtime / syntax errors — captured in stderr/summary, not raised
# ---------------------------------------------------------------------------


class TestErrorCapture:
    def test_syntax_error_captured_in_stderr(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("def foo(:")
        assert len(result.stderr) > 0, "SyntaxError must appear in stderr"
        assert result.ok is False

    def test_runtime_error_captured_in_stderr(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("raise ValueError('something broke')")
        assert "ValueError" in result.stderr or "ValueError" in result.summary()
        assert result.ok is False

    def test_name_error_captured(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("print(undefined_variable)")
        assert "NameError" in result.stderr or "undefined_variable" in result.stderr
        assert result.ok is False

    def test_error_does_not_raise_exception_in_caller(self, sandbox: SubprocessSandbox):
        """Runtime errors inside the sandbox must never propagate as Python exceptions."""
        result = sandbox.run_python("1/0")
        assert isinstance(result, ExecResult)

    def test_summary_for_error_is_non_empty(self, sandbox: SubprocessSandbox):
        result = sandbox.run_python("raise RuntimeError('oops')")
        assert result.summary().strip(), "Error summary must not be empty"


# ---------------------------------------------------------------------------
# ExecResult.ok property
# ---------------------------------------------------------------------------


class TestExecResultOk:
    def test_ok_true_when_exit_zero_not_timed_out(self):
        r = ExecResult(stdout="hi", stderr="", exit_code=0, timed_out=False)
        assert r.ok is True

    def test_ok_false_when_exit_nonzero(self):
        r = ExecResult(stdout="", stderr="err", exit_code=1, timed_out=False)
        assert r.ok is False

    def test_ok_false_when_timed_out_even_exit_zero(self):
        r = ExecResult(stdout="", stderr="", exit_code=0, timed_out=True)
        assert r.ok is False


# ---------------------------------------------------------------------------
# Network block preamble
# ---------------------------------------------------------------------------


def test_network_blocked_by_default(tmp_path):
    """Without network=True, a socket.socket() call should raise OSError."""
    sb = SubprocessSandbox(str(tmp_path), network=False, timeout_s=5)
    code = """
import socket
try:
    s = socket.socket()
    print("SHOULD NOT REACH HERE")
except OSError as e:
    print("BLOCKED:", e)
"""
    result = sb.run_python(code)
    assert "BLOCKED" in result.stdout, (
        "Network socket creation must be blocked when network=False"
    )
    assert "SHOULD NOT REACH HERE" not in result.stdout
