"""Shared execution loop for tool-using agents."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nanobot.agent.cost_guard import CostGuard
from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.checkpoint.loop_detector import LoopDetector
from nanobot.providers.base import LLMProvider, ToolCallRequest
from nanobot.react.permission_engine import PermissionEngine
from nanobot.utils.helpers import build_assistant_message


def _is_checkpoint_hook(hook: AgentHook) -> bool:
    """Return True if *hook* is a :class:`CheckpointHook` (lazy import)."""
    try:
        from nanobot.checkpoint.hook import CheckpointHook
        return isinstance(hook, CheckpointHook)
    except ImportError:
        return False

_DEFAULT_MAX_ITERATIONS_MESSAGE = (
    "I reached the maximum number of tool call iterations ({max_iterations}) "
    "without completing the task. You can try breaking the task into smaller steps."
)
_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict[str, Any]]
    tools: ToolRegistry
    model: str
    max_iterations: int
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    hook: AgentHook | None = None
    error_message: str | None = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: str | None = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False
    wall_clock_cap: int | None = None
    cost_guard: CostGuard | None = None
    permission_engine: PermissionEngine | None = None


@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: str | None
    messages: list[dict[str, Any]]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: str | None = None
    tool_events: list[dict[str, str]] = field(default_factory=list)


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages = list(spec.initial_messages)
        final_content: str | None = None
        tools_used: list[str] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        error: str | None = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []

        # Proactive loop detector — observes tool calls and can intervene
        loop_detector = LoopDetector(window=8)

        # Compute initial effective max (may grow via CheckpointHook)
        eff_max = self._effective_max(hook, spec)

        execution_elapsed = 0.0
        iteration = 0

        while iteration < eff_max:
            # --- Hard cap check (execution time only — pause time excluded) ---
            if spec.wall_clock_cap is not None and execution_elapsed >= spec.wall_clock_cap:
                stop_reason = "hard_cap"
                break

            # --- Checkpoint pause (blocks here if requested) ---
            if _is_checkpoint_hook(hook) and hook.pause_requested:
                action, _extra = await hook.do_pause()
                eff_max = self._effective_max(hook, spec)
                if hook.finalize_requested:
                    messages.append({"role": "user", "content": hook.finalize_prompt})
                elif hook.should_stop:
                    stop_reason = "checkpoint_stop"
                    break
                # CONTINUE: fall through to normal iteration body

            iter_start = time.monotonic()

            # --- Cost guard check (before LLM call) ---
            if spec.cost_guard is not None:
                check = spec.cost_guard.check_before_call()
                if not check.allowed:
                    stop_reason = "cost_cap"
                    final_content = check.reason or "Cost budget exceeded."
                    execution_elapsed += time.monotonic() - iter_start
                    break

            context = AgentHookContext(iteration=iteration, messages=messages)
            await hook.before_iteration(context)
            kwargs: dict[str, Any] = {
                "messages": messages,
                "tools": spec.tools.get_definitions(),
                "model": spec.model,
            }
            if spec.temperature is not None:
                kwargs["temperature"] = spec.temperature
            if spec.max_tokens is not None:
                kwargs["max_tokens"] = spec.max_tokens
            if spec.reasoning_effort is not None:
                kwargs["reasoning_effort"] = spec.reasoning_effort

            if hook.wants_streaming():
                async def _stream(delta: str) -> None:
                    await hook.on_stream(context, delta)

                response = await self.provider.chat_stream_with_retry(
                    **kwargs,
                    on_content_delta=_stream,
                )
            else:
                response = await self.provider.chat_with_retry(**kwargs)

            raw_usage = response.usage or {}
            usage["prompt_tokens"] += int(raw_usage.get("prompt_tokens", 0) or 0)
            usage["completion_tokens"] += int(raw_usage.get("completion_tokens", 0) or 0)
            if spec.cost_guard is not None:
                spec.cost_guard.record_usage(usage, model=spec.model)
            context.response = response
            context.usage = usage
            context.tool_calls = list(response.tool_calls)

            if response.has_tool_calls:
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)

                messages.append(build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                ))
                tools_used.extend(tc.name for tc in response.tool_calls)

                await hook.before_execute_tools(context)

                results, new_events, fatal_error = await self._execute_tools(spec, response.tool_calls)
                tool_events.extend(new_events)
                context.tool_results = list(results)
                context.tool_events = list(new_events)

                # Proactive loop detection: observe + check
                for ev in new_events:
                    loop_detector.observe(
                        ev.get("name", "?"),
                        ev.get("detail", ""),
                        iteration,
                    )
                loop_signal = loop_detector.detect()
                if loop_signal is not None:
                    logger.warning(
                        "Loop detected (proactive): {} — {}",
                        loop_signal.pattern,
                        loop_signal.evidence,
                    )
                    # Inject a system message forcing approach change
                    messages.append({
                        "role": "user",
                        "content": (
                            "[SYSTEM] Repetitive tool-call loop detected: "
                            f"{loop_signal.evidence}. "
                            "You are repeating the same actions. "
                            "Change your approach or conclude your response."
                        ),
                    })
                    loop_detector.reset()

                if fatal_error is not None:
                    error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                    stop_reason = "tool_error"
                    context.error = error
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    execution_elapsed += time.monotonic() - iter_start
                    break
                # If all tool calls were blocked by PermissionEngine, stop.
                if new_events and all(e.get("status") == "denied" for e in new_events):
                    final_content = "All requested tool calls were denied by permission policy."
                    await hook.after_iteration(context)
                    execution_elapsed += time.monotonic() - iter_start
                    break
                for tool_call, result in zip(response.tool_calls, results):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": result,
                    })
                await hook.after_iteration(context)
                execution_elapsed += time.monotonic() - iter_start
                iteration += 1
                continue

            if hook.wants_streaming():
                await hook.on_stream_end(context, resuming=False)

            clean = hook.finalize_content(context, response.content)
            if response.finish_reason == "error":
                final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                stop_reason = "error"
                error = final_content
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                execution_elapsed += time.monotonic() - iter_start
                break

            messages.append(build_assistant_message(
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            ))
            final_content = clean
            context.final_content = final_content
            context.stop_reason = stop_reason
            await hook.after_iteration(context)
            execution_elapsed += time.monotonic() - iter_start
            break
        else:
            stop_reason = "max_iterations"
            template = spec.max_iterations_message or _DEFAULT_MAX_ITERATIONS_MESSAGE
            final_content = template.format(max_iterations=spec.max_iterations)

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
        )

    @staticmethod
    def _effective_max(hook: AgentHook, spec: AgentRunSpec) -> int:
        """Return the current effective maximum iterations."""
        if _is_checkpoint_hook(hook):
            return hook.effective_max
        return spec.max_iterations

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
    ) -> tuple[list[Any], list[dict[str, str]], BaseException | None]:
        # --- Permission gate: filter blocked tool calls ---
        effective_calls = tool_calls
        if spec.permission_engine is not None:
            call_tuples = [
                (tc.name, dict(tc.arguments))
                for tc in tool_calls
            ]
            allowed, blocked = spec.permission_engine.filter_calls(call_tuples)
            if blocked:
                allowed_names = {name for name, _ in allowed}
                blocked_names = [name for name, _ in call_tuples if name not in allowed_names]
                logger.info(
                    "PermissionEngine blocked {} tool(s): {}",
                    len(blocked), ", ".join(blocked_names),
                )
            if len(allowed) < len(tool_calls):
                # Rebuild the tool_calls list with only allowed calls.
                allowed_set = {(name, tuple(sorted(args.items()))) for name, args in allowed}
                effective_calls = [
                    tc for tc in tool_calls
                    if (tc.name, tuple(sorted(dict(tc.arguments).items()))) in allowed_set
                ]
            if not effective_calls:
                # All calls blocked — return synthetic denial results.
                results: list[Any] = []
                events: list[dict[str, str]] = []
                for _, tc in zip(blocked, tool_calls[: len(blocked)]):
                    results.append(f"Permission denied: {blocked[len(results)].reason}")
                    events.append({
                        "name": tc.name,
                        "status": "denied",
                        "detail": blocked[len(events)].reason,
                        "arguments": dict(tc.arguments),
                    })
                return results, events, None

        if spec.concurrent_tools:
            tool_results = await asyncio.gather(*(
                self._run_tool(spec, tool_call)
                for tool_call in effective_calls
            ))
        else:
            tool_results = [
                await self._run_tool(spec, tool_call)
                for tool_call in effective_calls
            ]

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: BaseException | None = None
        for result, event, error in tool_results:
            results.append(result)
            events.append(event)
            if error is not None and fatal_error is None:
                fatal_error = error
        return results, events, fatal_error

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
    ) -> tuple[Any, dict[str, str], BaseException | None]:
        try:
            result = await spec.tools.execute(tool_call.name, tool_call.arguments)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
                "arguments": dict(tool_call.arguments),
            }
            if spec.fail_on_tool_error:
                return f"Error: {type(exc).__name__}: {exc}", event, exc
            return f"Error: {type(exc).__name__}: {exc}", event, None

        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        return result, {
            "name": tool_call.name,
            "status": "error" if isinstance(result, str) and result.startswith("Error") else "ok",
            "detail": detail,
            "arguments": dict(tool_call.arguments),
        }, None
