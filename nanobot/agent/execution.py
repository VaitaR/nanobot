"""Subprocess execution for delegated CLI-based agent runs (ACPX).

This is the **canonical** implementation used by both:
- ``SubagentManager`` (core) for explicit ``spawn()`` calls with CLI executors
- ``HeartbeatTaskRunner`` (workspace) for auto-pickup delegated tasks

Consumers should import from here, never duplicate the subprocess logic.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

_SIGNAL_NAMES: dict[int, str] = {
    1: "SIGHUP",
    2: "SIGINT",
    3: "SIGQUIT",
    6: "SIGABRT",
    9: "SIGKILL",
    11: "SIGSEGV",
    13: "SIGPIPE",
    15: "SIGTERM",
}


def _signal_label(rc: int) -> str | None:
    """Return 'SIGNAME (signal N)' for exit codes > 128, else None."""
    if rc > 128:
        signum = rc - 128
        name = _SIGNAL_NAMES.get(signum, f"signal {signum}")
        return f"{name} (signal {signum})"
    return None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Normalized result of a delegated subprocess execution.

    Legacy type for backward compatibility. Use DelegatedResult for new code
    that needs structured ACPX output (tool_calls, duration, etc.).
    """

    success: bool
    summary: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error_type: str = ""  # config | timeout | exit_nonzero | exception


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call made by the agent during execution."""

    name: str
    arguments: dict[str, Any]
    status: str  # "ok" | "error" | "running"
    result: str | None = None
    duration_ms: int | None = None


@dataclass(frozen=True, slots=True)
class DelegatedResult:
    """Structured result from ACPX delegation with tool call metadata.

    Replaces raw text output with structured data for better completion
    reporting and failure classification.
    """

    success: bool
    final_message: str  # The agent's final response text
    tool_calls: tuple[ToolCall, ...] = ()  # Tool calls made during execution
    total_duration: float = 0.0  # Total wall-clock time in seconds
    error: str | None = None  # Error message if any
    exit_code: int | None = None  # Process exit code
    stdout: str = ""  # Raw stdout (for fallback/debugging)
    stderr: str = ""  # Raw stderr (for fallback/debugging)
    error_type: str = ""  # config | timeout | exit_nonzero | exception
    usage: dict[str, int] = None  # Token usage when provided by ACPX/underlying CLI
    spawn_id: str | None = None  # Runtime subagent id, when invoked from spawn()
    task_id: str | None = None  # Linked nanobot task id, when available
    _has_valid_json: bool = False  # Internal: whether valid JSON-RPC was found

    def __post_init__(self) -> None:
        if self.usage is None:
            object.__setattr__(self, "usage", {})

    @property
    def summary(self) -> str:
        """Backward compatibility: summary from final_message or error or stdout."""
        return self.final_message or self.error or self.stdout[:400]

    def to_execution_result(self) -> ExecutionResult:
        """Convert to legacy ExecutionResult for backward compatibility."""
        summary = self.final_message or self.error or self.stdout[:400]
        return ExecutionResult(
            success=self.success,
            summary=summary,
            exit_code=self.exit_code,
            stdout=self.stdout,
            stderr=self.stderr,
            error_type=self.error_type,
        )


# ---------------------------------------------------------------------------
# ACPX executor
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_S = 600


def get_default_timeout() -> int:
    """Return the default ACPX timeout in seconds."""
    return _DEFAULT_TIMEOUT_S


# Backward compat alias
ACPX_TIMEOUT_S: int = _DEFAULT_TIMEOUT_S


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def get_executor_status(workspace: Path) -> dict[str, str]:
    """Return CLI executor health from acpx telemetry.

    Scans ``memory/improvement-log.jsonl`` for recent acpx session entries
    and returns the *last known status* per agent.

    Returns
    -------
    dict[str, str]
        Mapping ``"codex" | "claude"`` → ``"ok" | "unauthorized" | "rate_limited" | "error"``.
    """
    log_path = workspace / "memory" / "improvement-log.jsonl"
    if not log_path.exists():
        return {}

    cutoff = datetime.now(UTC) - timedelta(hours=1)

    try:
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    except OSError:
        return {}

    status: dict[str, str] = {}
    for line in reversed(lines):
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if obj.get("session") != "acpx":
            continue
        model = obj.get("model", "")
        if model not in ("codex", "claude"):
            continue
        if model in status:
            continue  # already have latest
        # Parse timestamp; skip stale entries older than 1 hour
        ts_str = obj.get("ts") or obj.get("checked_at") or obj.get("timestamp") or obj.get("created_at")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts < cutoff:
                    continue
            except (ValueError, TypeError):
                pass
        error_cat = obj.get("error_category")
        if error_cat == "unauthorized":
            status[model] = "unauthorized"
        elif error_cat == "rate_limited":
            status[model] = "rate_limited"
        elif obj.get("stop_reason") == "completed" and not error_cat:
            status[model] = "ok"
        else:
            status[model] = "error"

    return status


def _detect_acpx_error_type(stdout: str, stderr: str, returncode: int) -> str:
    """Classify ACPX execution errors into semantic types for fallback routing.

    Returns one of: 'unauthorized', 'rate_limited', 'config', 'timeout', 'exit_nonzero', '' (success).
    """
    # Check for known auth error patterns in JSON-RPC error responses
    combined = stdout + "\n" + stderr
    combined_lower = combined.lower()

    rate_limit_markers = [
        "usage limit",
        "rate limit",
        "rate limited",
        "too many requests",
        "quota exceeded",
        "hit your usage limit",
    ]
    for marker in rate_limit_markers:
        if marker in combined_lower:
            return "rate_limited"

    auth_markers = [
        "refresh token was already used",
        "access token could not be refreshed",
        "unauthorized",
        "authentication failed",
        "not authenticated",
        "login required",
        "sign in again",
        "codex_error_info",
    ]
    for marker in auth_markers:
        if marker.lower() in combined_lower:
            return "unauthorized"

    config_markers = [
        "acpx_claude.sh not found",
        "unknown agent:",
        "no acpx_agent mapping",
    ]
    for marker in config_markers:
        if marker.lower() in combined_lower:
            return "config"

    if returncode != 0:
        return "exit_nonzero"
    return ""


def _normalize_usage(raw: Any) -> dict[str, int]:
    """Normalize token usage payloads from ACPX/CLI events."""
    if not isinstance(raw, dict):
        return {}
    prompt_tokens = int(raw.get("prompt_tokens", raw.get("input_tokens", 0)) or 0)
    completion_tokens = int(raw.get("completion_tokens", raw.get("output_tokens", 0)) or 0)
    total_tokens = int(raw.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _parse_acpx_json_output(stdout: str, stderr: str, duration: float) -> DelegatedResult:
    """Parse ACPX JSON-RPC output and extract structured data.

    Args:
        stdout: Raw stdout from ACPX (JSON-RPC NDJSON messages)
        stderr: Raw stderr from ACPX
        duration: Total execution time in seconds

    Returns:
        DelegatedResult with parsed tool_calls, final_message, and error info.
    """
    final_message = ""
    tool_calls: dict[str, ToolCall] = {}
    error: str | None = None
    stop_reason = ""
    usage: dict[str, int] = {}
    has_valid_json = False  # Track if we found any valid JSON-RPC messages

    try:
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            # ACPX may emit JSON arrays or non-dict values; skip them
            if not isinstance(msg, dict):
                continue

            msg_type = msg.get("type", "")

            # Handle Codex CLI format: {"type": "error", "message": "..."}
            if msg_type == "error":
                has_valid_json = True
                error_message = msg.get("message", "")
                if error_message:
                    error = error_message[:500]
                continue

            # Handle Codex CLI format: {"type": "turn.failed", "error": {...}}
            if msg_type == "turn.failed":
                has_valid_json = True
                err_obj = msg.get("error", {})
                if isinstance(err_obj, dict):
                    error_message = err_obj.get("message", "")
                    if error_message:
                        error = error_message[:500]
                stop_reason = "turn_failed"
                continue

            # Handle Codex CLI format: {"type": "turn.completed", ...}
            if msg_type == "turn.completed":
                has_valid_json = True
                stop_reason = "end_turn"
                continue

            # Handle Codex CLI format: {"type": "thread.started", ...}
            if msg_type == "thread.started":
                has_valid_json = True
                continue

            # Parse JSON-RPC 2.0 messages (ACPX format)
            if msg.get("jsonrpc") == "2.0":
                has_valid_json = True

                # Extract final result (session/prompt response)
                if "result" in msg and msg.get("id") is not None:
                    result = msg["result"]
                    if isinstance(result, dict):
                        stop_reason = result.get("stopReason", "")
                        result_usage = _normalize_usage(result.get("usage"))
                        if result_usage.get("total_tokens", 0) > 0:
                            usage = result_usage

                # Extract agent message chunks (build final message)
                if msg.get("method") == "session/update":
                    params = msg.get("params", {})
                    if isinstance(params, dict):
                        update = params.get("update", {})
                        if isinstance(update, dict):
                            update_usage = _normalize_usage(update.get("usage"))
                            if update_usage.get("total_tokens", 0) > 0:
                                usage = update_usage
                            content = update.get("content", {})
                            if isinstance(content, dict) and content.get("type") == "text":
                                final_message += content.get("text", "")
                            if content.get("type") == "tool_call":
                                call_id = str(content.get("id") or content.get("toolCallId") or "")
                                name = str(content.get("name") or content.get("toolName") or "")
                                arguments = content.get("arguments") or {}
                                if isinstance(arguments, str):
                                    try:
                                        arguments = json.loads(arguments)
                                    except json.JSONDecodeError:
                                        arguments = {"raw": arguments}
                                status = str(content.get("status") or "running").lower()
                                if status == "in_progress":
                                    status = "running"
                                if call_id or name:
                                    tool_calls[call_id or name] = ToolCall(
                                        name=name or call_id,
                                        arguments=arguments if isinstance(arguments, dict) else {},
                                        status=status,
                                    )
                            elif content.get("type") in {"tool_result", "tool_call_result"}:
                                call_id = str(content.get("toolCallId") or content.get("id") or "")
                                prior = tool_calls.get(call_id)
                                raw_status = str(content.get("status") or "").lower()
                                is_error = bool(content.get("isError") or content.get("error"))
                                if raw_status in {"completed", "failed", "error"}:
                                    status = raw_status
                                else:
                                    status = "error" if is_error else "completed"
                                result_text = (
                                    content.get("result")
                                    or content.get("text")
                                    or content.get("output")
                                )
                                duration_ms = content.get("durationMs")
                                if prior is not None:
                                    tool_calls[call_id] = ToolCall(
                                        name=prior.name,
                                        arguments=prior.arguments,
                                        status=status,
                                        result=str(result_text)
                                        if result_text is not None
                                        else None,
                                        duration_ms=int(duration_ms)
                                        if duration_ms is not None
                                        else prior.duration_ms,
                                    )
                                elif call_id:
                                    fallback_name = str(
                                        content.get("name") or content.get("toolName") or call_id
                                    )
                                    tool_calls[call_id] = ToolCall(
                                        name=fallback_name,
                                        arguments={},
                                        status=status,
                                        result=str(result_text)
                                        if result_text is not None
                                        else None,
                                        duration_ms=int(duration_ms)
                                        if duration_ms is not None
                                        else None,
                                    )

    except Exception as e:
        logger.warning("acpx.json_parse_failed", error=str(e))
        error = f"JSON parse error: {e}"

    # Determine success based on stop_reason
    success = stop_reason == "end_turn"
    if not success and stop_reason and not error:
        error = f"Agent stopped: {stop_reason}"

    # Check for errors in stderr
    if stderr and not error:
        error_lines = stderr.strip().splitlines()
        if error_lines:
            error = error_lines[-1][:200]  # Last line, truncated

    return DelegatedResult(
        success=success,
        final_message=final_message,
        tool_calls=tuple(tool_calls.values()),
        total_duration=duration,
        error=error,
        stdout=stdout,
        stderr=stderr,
        error_type="",  # Will be set by execute_acpx
        usage=usage,
        _has_valid_json=has_valid_json,  # Internal flag for fallback detection
    )


async def execute_acpx(
    agent: str,
    prompt: str,
    workspace: Path,
    *,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    spawn_id: str | None = None,
    task_id: str | None = None,
) -> DelegatedResult:
    """Execute a task via ACPX subprocess with structured JSON output.

    Parameters
    ----------
    agent:
        ``"codex"`` or ``"claude"``.
    prompt:
        The task description / prompt to send to the agent.
    workspace:
        Working directory for the subprocess.
    timeout_s:
        Hard wall-clock timeout for the subprocess.

    Returns
    -------
    DelegatedResult:
        Structured result with tool_calls, final_message, duration, and error info.
        Falls back to raw text if JSON parsing fails.

    Telemetry is written to ``memory/improvement-log.jsonl`` for every
    invocation, ensuring all ACPX delegations (heartbeat, chat-spawn, etc.)
    are visible in the dashboard.
    """
    result = await _execute_acpx_impl(agent, prompt, workspace, timeout_s=timeout_s)
    result = DelegatedResult(
        success=result.success,
        final_message=result.final_message,
        tool_calls=result.tool_calls,
        total_duration=result.total_duration,
        error=result.error,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        error_type=result.error_type,
        usage=result.usage,
        spawn_id=spawn_id,
        task_id=task_id,
        _has_valid_json=result._has_valid_json,
    )
    _write_acpx_telemetry(workspace, agent, result)
    return result


async def _execute_acpx_impl(
    agent: str,
    prompt: str,
    workspace: Path,
    *,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> DelegatedResult:
    """Internal implementation — see :func:`execute_acpx` for public API."""
    import tempfile as _tf

    acpx_claude_sh = workspace / "acpx_claude.sh"

    # Use /root as cwd so subagents can access both runtime and workspace repos.
    runtime_cwd = Path("/root")

    # Write prompt to a temp file — avoids CLI arg length limits and
    # prevents content starting with dashes from being parsed as flags.
    with _tf.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix="acpx_prompt_",
        delete=False,
        encoding="utf-8",
    ) as tmpf:
        tmpf.write(prompt)
        temp_prompt_path = tmpf.name

    # Build command with --format json for structured output
    if agent == "codex":
        cmd = [
            "npx",
            "-y",
            "acpx@latest",
            "--cwd",
            str(runtime_cwd),
            "--format",
            "json",
            "--approve-all",
            "codex",
            "exec",
            "--file",
            temp_prompt_path,
        ]
    elif agent == "claude":
        if not acpx_claude_sh.exists():
            try:
                os.unlink(temp_prompt_path)
            except OSError:
                pass
            return DelegatedResult(
                success=False,
                final_message=f"acpx_claude.sh not found at {acpx_claude_sh}",
                error=f"acpx_claude.sh not found at {acpx_claude_sh}",
                error_type="config",
                _has_valid_json=False,
            )
        cmd = [str(acpx_claude_sh), "--format", "json", "exec", "--file", temp_prompt_path]
    else:
        try:
            os.unlink(temp_prompt_path)
        except OSError:
            pass
        return DelegatedResult(
            success=False,
            final_message=f"Unknown agent: {agent}",
            error=f"Unknown agent: {agent}",
            error_type="config",
            _has_valid_json=False,
        )

    logger.info("acpx.start", agent=agent)
    start_time = time.time()

    try:
        env = {**os.environ, "ACPX_CWD": str(runtime_cwd)} if agent == "claude" else None
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(runtime_cwd),
            env=env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_s,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            duration = time.time() - start_time
            return DelegatedResult(
                success=False,
                final_message=f"Timed out after {timeout_s}s",
                error=f"Timed out after {timeout_s}s",
                total_duration=duration,
                error_type="timeout",
                _has_valid_json=False,
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        duration = time.time() - start_time

        # Persist raw output for post-mortem debugging
        _write_acpx_raw_log(workspace, agent, stdout, stderr, duration)

        # Try to parse JSON output
        try:
            parsed = _parse_acpx_json_output(stdout, stderr, duration)

            # If no valid JSON-RPC messages were found, fall back to raw text
            if not parsed._has_valid_json:
                raise ValueError("No valid JSON-RPC messages found")

            # Detect auth/config errors from JSON-RPC error responses
            error_type = _detect_acpx_error_type(stdout, stderr, process.returncode)

            result = DelegatedResult(
                success=parsed.success and (process.returncode == 0),
                final_message=parsed.final_message,
                tool_calls=parsed.tool_calls,
                total_duration=duration,
                error=parsed.error,
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                error_type=error_type,
                _has_valid_json=True,
            )

            if result.success:
                logger.info("acpx.ok", agent=agent, duration=duration)
            else:
                _rc = process.returncode
                logger.warning(
                    "acpx.fail",
                    agent=agent,
                    rc=_rc,
                    signal=_signal_label(_rc),
                    error=result.error,
                    stderr_tail=stderr[-500:] if stderr else "",
                )

            return result

        except Exception as json_exc:
            logger.warning("acpx.json_fallback", error=str(json_exc))
            # Fallback to raw text parsing if JSON parsing fails
            if process.returncode == 0:
                return DelegatedResult(
                    success=True,
                    final_message=stdout[:400],
                    total_duration=duration,
                    exit_code=0,
                    stdout=stdout,
                    stderr=stderr,
                    _has_valid_json=False,
                )
            else:
                summary = f"Exit {process.returncode}: {stderr or stdout}"
                return DelegatedResult(
                    success=False,
                    final_message=summary,
                    error=summary,
                    total_duration=duration,
                    exit_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    error_type="exit_nonzero",
                    _has_valid_json=False,
                )

    except Exception as exc:
        duration = time.time() - start_time
        logger.exception("acpx.error", agent=agent)
        return DelegatedResult(
            success=False,
            final_message=str(exc),
            error=str(exc),
            total_duration=duration,
            error_type="exception",
        )
    finally:
        if temp_prompt_path is not None:
            try:
                os.unlink(temp_prompt_path)
            except OSError:
                pass


def _write_acpx_raw_log(
    workspace: Path,
    agent: str,
    stdout: str,
    stderr: str,
    duration: float,
) -> None:
    """Append raw ACPX stdout+stderr to a per-day log for post-mortem inspection.

    Written to ``sessions/acpx_<date>.log`` with a separator between invocations.
    """
    sessions_dir = workspace / "sessions"
    try:
        sessions_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("acpx.raw_log_mkdir_failed", path=str(sessions_dir))
        return

    date_stamp = datetime.now(UTC).strftime("%Y%m%d")
    log_path = sessions_dir / f"acpx_{date_stamp}.log"

    duration_ms = int(duration * 1000)
    separator = (
        f"\n{'=' * 60}\n"
        f"[{datetime.now(UTC).isoformat()}] agent={agent} duration={duration_ms}ms\n"
        f"{'=' * 60}\n"
    )

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(separator)
            f.write("--- STDOUT ---\n")
            f.write(stdout)
            if not stdout.endswith("\n"):
                f.write("\n")
            if stderr.strip():
                f.write("--- STDERR ---\n")
                f.write(stderr)
                if not stderr.endswith("\n"):
                    f.write("\n")
    except OSError:
        logger.warning("acpx.raw_log_write_failed", path=str(log_path))


def _write_acpx_telemetry(
    workspace: Path,
    agent: str,
    result: DelegatedResult,
) -> None:
    """Append delegation telemetry to ``memory/improvement-log.jsonl``.

    Called automatically by :func:`execute_acpx` for every invocation,
    ensuring all ACPX delegations (heartbeat, chat-spawn, etc.) are tracked
    alongside main-agent LLM turns already logged by TelemetryCollector.
    """
    memory_dir = workspace / "memory"
    try:
        memory_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("acpx.telemetry_mkdir_failed", path=str(memory_dir))
        return

    target = memory_dir / "improvement-log.jsonl"
    duration_ms = int(result.total_duration * 1000) if result.total_duration else 0
    stop_reason = "completed" if result.success else "error"
    tools = [tc.name for tc in result.tool_calls] if result.tool_calls else []
    tool_statuses = {tc.name: tc.status for tc in result.tool_calls} if result.tool_calls else {}
    usage = _normalize_usage(result.usage)

    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "session": "acpx",
        "channel": "acpx",
        "chat_id": None,
        "model": agent,
        "request_id": None,
        "spawn_id": result.spawn_id,
        "task_id": result.task_id,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        },
        "total_tokens": usage.get("total_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "duration_ms": duration_ms,
        "stop_reason": stop_reason,
        "error": result.error,
        "error_category": result.error_type or None,
        "tools": tools,
        "tool_statuses": tool_statuses,
        "skills": [],
        "files_touched": [],
    }

    try:
        with open(target, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug(
            "acpx.telemetry_written", agent=agent, duration_ms=duration_ms, stop=stop_reason
        )
    except OSError:
        logger.warning("acpx.telemetry_write_failed", agent=agent)
