"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import AsyncExitStack, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.cost_guard import CostGuard
from nanobot_workspace.observability import (
    CostTracker,
    bind_correlation_id,
    generate_correlation_id,
    get_correlation_id,
    unbind_correlation_id,
)
from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.telemetry import TelemetryCollector
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.notify import NotifyTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.restart import RestartGatewayTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.spawn_status import SpawnStatusTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.providers.base import LLMProvider
from nanobot.react.signal_detector import SignalDetector
from nanobot.session.manager import Session, SessionManager
from nanobot.session.resume_state import persist_last_active_session
from nanobot.utils.helpers import estimate_prompt_tokens_chain

if TYPE_CHECKING:
    from nanobot.config.schema import (
        ChannelsConfig,
        CompactionConfig,
        ConfirmationRuleConfig,
        CostPolicy,
        ExecToolConfig,
        WebSearchConfig,
    )
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 4_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        system_prompt_max_tokens: int = 0,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        timezone: str | None = None,
        cost_policy: CostPolicy | None = None,
        confirmation_rules: list[ConfirmationRuleConfig] | None = None,
        permission_policy: dict[str, str] | None = None,
        compaction_config: CompactionConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._cost_guard: CostGuard | None = (
            CostGuard.from_config(cost_policy) if cost_policy else None
        )
        self._cost_tracker = CostTracker()

        self.context = ContextBuilder(
            workspace,
            timezone=timezone,
            system_prompt_max_tokens=system_prompt_max_tokens,
        )
        self.sessions = session_manager or SessionManager(workspace)

        # Build confirmation policy from config rules
        from nanobot.agent.tools.confirmation import ConfirmationPolicy

        if confirmation_rules:
            policy = ConfirmationPolicy.from_config([r.model_dump() for r in confirmation_rules])
        else:
            policy = None  # No confirmation rules → no gating
        self.tools = ToolRegistry(confirmation_policy=policy)
        self.runner = AgentRunner(provider)
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._active_task_ts: dict[int, float] = {}  # id(task) -> monotonic timestamp
        self._STALE_TASK_THRESHOLD_S = 120.0  # tasks older than this are considered stale
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
        )
        # --- Workspace: schedule one-time memory maintenance ---
        self.memory_consolidator.schedule_maintenance()
        self.telemetry = TelemetryCollector(workspace)
        # --- SignalDetector: graceful degradation ---
        try:
            self.signal_detector = SignalDetector()
            logger.debug("SignalDetector initialised in AgentLoop")
        except Exception as exc:
            logger.warning("SignalDetector init failed, skipping signal detection: {}", exc)
            self.signal_detector = None
        # --- PermissionEngine: graceful degradation ---
        try:
            from nanobot.react.permission_engine import Permission, PermissionEngine

            custom_policy: dict[str, Permission] = {}
            if permission_policy:
                for tool_name, level_str in permission_policy.items():
                    try:
                        custom_policy[tool_name] = Permission(level_str.lower())
                    except ValueError:
                        logger.warning(
                            "Unknown permission level '{}' for tool '{}', ignoring",
                            level_str,
                            tool_name,
                        )
            self.permission_engine = PermissionEngine(policy=custom_policy)
            logger.debug(
                "PermissionEngine initialised with {} override(s)",
                len(custom_policy),
            )
        except Exception as exc:
            logger.warning("PermissionEngine init failed, defaulting to ALLOW-all: {}", exc)
            self.permission_engine = None
        self.heartbeat_service: Any | None = None  # set after construction by gateway
        # --- Workspace modules: graceful degradation ---
        self._evolution_hook_factory: Any | None = None
        try:
            from nanobot_workspace.agent.loop_hooks import create_evolution_hook  # noqa: WPS433

            self._evolution_hook_factory = create_evolution_hook
            logger.debug("Workspace evolution hook factory available")
        except Exception as exc:
            logger.debug("Workspace evolution hook unavailable: {}", exc)

        self._diagnose_failure: Any | None = None
        try:
            from nanobot_workspace.observability.feedback_loop import (
                diagnose_failure as _diag,  # noqa: WPS433
            )

            self._diagnose_failure = _diag
            logger.debug("Workspace feedback loop available")
        except Exception as exc:
            logger.debug("Workspace feedback loop unavailable: {}", exc)

        self._rotate_sessions: Any | None = None
        try:
            from nanobot_workspace.memory.sessions import rotate_sessions as _rs  # noqa: WPS433

            self._rotate_sessions = _rs
            logger.debug("Workspace session rotation available")
        except Exception as exc:
            logger.debug("Workspace session rotation unavailable: {}", exc)

        self._ws_notify: Any | None = None
        try:
            from nanobot_workspace.proactive import notify as _wn  # noqa: WPS433

            self._ws_notify = _wn
            logger.debug("Workspace notify bridge available")
        except Exception as exc:
            logger.debug("Workspace notify bridge unavailable: {}", exc)

        # --- Compaction: graceful degradation ---
        self._compact_fn: Any | None = None
        self._compaction_policy: Any | None = None
        self._compaction_enabled = False
        _comp_cfg = compaction_config
        if _comp_cfg is not None and not _comp_cfg.enabled:
            logger.debug("Compaction disabled by config")
        elif _comp_cfg is not None or True:  # default: enabled
            try:
                from nanobot_workspace.memory.compaction import (
                    CompactionPolicy,
                )
                from nanobot_workspace.memory.compaction import (  # noqa: WPS433
                    compact_messages as _compact_fn,
                )

                _budget = (
                    _comp_cfg.token_budget
                    if _comp_cfg and _comp_cfg.token_budget > 0
                    else context_window_tokens
                )
                self._compaction_policy = CompactionPolicy(
                    token_budget=_budget,
                    safety_margin=_comp_cfg.safety_margin if _comp_cfg else 8_000,
                    keep_recent=_comp_cfg.keep_recent if _comp_cfg else 10,
                    compaction_threshold=_comp_cfg.compaction_threshold if _comp_cfg else 0.75,
                )
                self._compact_fn = _compact_fn
                self._compaction_enabled = True
                logger.debug(
                    "Compaction wired: budget={}, threshold={:.2f}, keep_recent={}",
                    _budget,
                    self._compaction_policy.compaction_threshold,
                    self._compaction_policy.keep_recent,
                )
            except Exception as exc:
                logger.warning("Workspace compaction unavailable: {}", exc)

        self._register_default_tools()
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(
            ReadFileTool(
                workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read
            )
        )
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(
                ExecTool(
                    working_dir=str(self.workspace),
                    timeout=self.exec_config.timeout,
                    restrict_to_workspace=self.restrict_to_workspace,
                    path_append=self.exec_config.path_append,
                    deny_patterns=self.exec_config.deny_patterns or None,
                    max_tier=3,
                    exec_context="direct",
                )
            )
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(NotifyTool())
        spawn_tool = SpawnTool(manager=self.subagents)
        spawn_tool.set_bus(self.bus)
        self.tools.register(spawn_tool)
        self.tools.register(SpawnStatusTool(get_active_tasks=self.get_active_tasks))
        self.tools.register(
            RestartGatewayTool(
                workspace=self.workspace,
                subagent_manager=self.subagents,
                bus=self.bus,
            )
        )
        if self.cron_service:
            self.tools.register(
                CronTool(self.cron_service, default_timezone=self.context.timezone or "UTC")
            )

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers

        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(
        self,
        channel: str,
        chat_id: str,
        message_id: str | None = None,
        message_thread_id: str | int | None = None,
    ) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    extra: list[Any] = []
                    if name == "message" and message_id is not None:
                        extra.append(message_id)
                    if name == "spawn" and message_thread_id is not None:
                        extra.append(message_thread_id)
                    tool.set_context(channel, chat_id, *extra)

    def _record_last_active_session(
        self,
        *,
        inbound_channel: str,
        sender_id: str,
        channel: str,
        chat_id: str,
        message_thread_id: str | int | None,
    ) -> None:
        """Persist the latest user-facing session for restart recovery."""
        if inbound_channel == "system" and sender_id != "subagent":
            return
        try:
            persist_last_active_session(
                self.workspace,
                channel=channel,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
            )
        except Exception as exc:
            logger.debug("Failed to persist last active session: {}", exc)

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        from nanobot.utils.helpers import strip_think

        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""

        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'

        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
        message_thread_id: str | int | None = None,
    ) -> tuple[str | None, list[str], list[dict], dict, str, str | None, list[dict]]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.
        """
        loop_self = self

        class _LoopHook(AgentHook):
            def __init__(self) -> None:
                self._stream_buf = ""

            def wants_streaming(self) -> bool:
                return on_stream is not None

            async def on_stream(self, context: AgentHookContext, delta: str) -> None:
                from nanobot.utils.helpers import strip_think

                prev_clean = strip_think(self._stream_buf)
                self._stream_buf += delta
                new_clean = strip_think(self._stream_buf)
                incremental = new_clean[len(prev_clean) :]
                if incremental and on_stream:
                    await on_stream(incremental)

            async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
                if on_stream_end:
                    await on_stream_end(resuming=resuming)
                self._stream_buf = ""

            async def before_execute_tools(self, context: AgentHookContext) -> None:
                if on_progress:
                    if not on_stream:
                        thought = loop_self._strip_think(
                            context.response.content if context.response else None
                        )
                        if thought:
                            await on_progress(thought)
                    tool_hint = loop_self._strip_think(loop_self._tool_hint(context.tool_calls))
                    await on_progress(tool_hint, tool_hint=True)
                for tc in context.tool_calls:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tc.name, args_str[:200])
                loop_self._set_tool_context(channel, chat_id, message_id, message_thread_id)

            def finalize_content(
                self, context: AgentHookContext, content: str | None
            ) -> str | None:
                return loop_self._strip_think(content)

        _base_hook = _LoopHook()
        # --- Workspace evolution hook: wrap base hook ---
        hook = _base_hook
        if self._evolution_hook_factory is not None:
            try:
                _evo_hook = self._evolution_hook_factory(
                    base_hook=_base_hook,
                    provider=self.provider,
                    model=self.model,
                    cooldown_seconds=300.0,
                )
                if _evo_hook is not None:
                    hook = _evo_hook
            except Exception as exc:
                logger.warning("Failed to create workspace evolution hook: {}", exc)

        result = await self.runner.run(
            AgentRunSpec(
                initial_messages=initial_messages,
                tools=self.tools,
                model=self.model,
                max_iterations=self.max_iterations,
                hook=hook,
                error_message="Sorry, I encountered an error calling the AI model.",
                concurrent_tools=True,
                cost_guard=self._cost_guard,
                permission_engine=self.permission_engine,
            )
        )
        self._last_usage = result.usage
        if result.stop_reason == "max_iterations":
            logger.warning("Max iterations ({}) reached", self.max_iterations)
        elif result.stop_reason == "error":
            logger.error("LLM returned error: {}", (result.final_content or "")[:200])
        return (
            result.final_content,
            result.tools_used,
            result.messages,
            result.usage,
            result.stop_reason,
            result.error,
            result.tool_events,
        )

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")
        # --- Workspace: background maintenance on startup ---
        if self._rotate_sessions is not None:

            async def _ws_session_rotation() -> None:
                try:
                    self._rotate_sessions()
                    logger.info("Workspace session rotation complete")
                except Exception as exc:
                    logger.warning("Workspace session rotation failed: {}", exc)

            self._schedule_background(_ws_session_rotation())

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            self._active_task_ts[id(task)] = time.monotonic()
            logger.info("_active_tasks DIAG: added task to session={}, now {} tasks total, session has {}",
                         msg.session_key, sum(len(v) for v in self._active_tasks.values()), len(self._active_tasks[msg.session_key]))
            task.add_done_callback(
                lambda t, k=msg.session_key: (
                    (lambda: (
                        logger.info("_active_tasks DIAG: done_callback fired for session={}, task done={}, cancelled={}",
                                     k, t.done(), t.cancelled()),
                        self._active_task_ts.pop(id(t), None),
                        (self._active_tasks.get(k, []) and self._active_tasks[k].remove(t)
                         if t in self._active_tasks.get(k, [])
                         else logger.info("_active_tasks DIAG: task NOT in list for session={}, list_len={}",
                                           k, len(self._active_tasks.get(k, []))))
                    ))()  # IIFE to ensure all expressions execute
                )
            )

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        async with lock, gate:
            previous_correlation_id = get_correlation_id()
            correlation_id = str(
                (msg.metadata or {}).get("correlation_id")
                or msg.request_id
                or generate_correlation_id()
            )
            bind_correlation_id(correlation_id)
            if "correlation_id" not in msg.metadata:
                msg.metadata["correlation_id"] = correlation_id
            if msg.request_id is None:
                msg.request_id = correlation_id
            try:
                on_stream = on_stream_end = None
                if msg.metadata.get("_wants_stream"):
                    # Split one answer into distinct stream segments.
                    stream_base_id = f"{msg.session_key}:{time.time_ns()}"
                    stream_segment = 0

                    def _current_stream_id() -> str:
                        return f"{stream_base_id}:{stream_segment}"

                    async def on_stream(delta: str) -> None:
                        await self.bus.publish_outbound(
                            OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content=delta,
                                metadata={
                                    "_stream_delta": True,
                                    "_stream_id": _current_stream_id(),
                                    "message_thread_id": msg.metadata.get("message_thread_id"),
                                },
                            )
                        )

                    async def on_stream_end(*, resuming: bool = False) -> None:
                        nonlocal stream_segment
                        await self.bus.publish_outbound(
                            OutboundMessage(
                                channel=msg.channel,
                                chat_id=msg.chat_id,
                                content="",
                                metadata={
                                    "_stream_end": True,
                                    "_resuming": resuming,
                                    "_stream_id": _current_stream_id(),
                                    "message_thread_id": msg.metadata.get("message_thread_id"),
                                },
                            )
                        )
                        stream_segment += 1

                try:
                    response = await asyncio.wait_for(
                        self._process_message(
                            msg,
                            on_stream=on_stream,
                            on_stream_end=on_stream_end,
                        ),
                        timeout=300,  # 5-minute hard cap per message
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Message processing timed out after 300s for session {}",
                        msg.session_key,
                    )
                    self.telemetry.record_turn(
                        ts=datetime.now(timezone.utc).isoformat(),
                        session=msg.session_key,
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        model=self.model,
                        usage={},
                        duration_ms=300_000,
                        stop_reason="timeout",
                        error="Message processing timed out after 300s",
                        tools_used=[],
                        skills=[],
                        files_touched=[],
                    )
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content="Request timed out (5 min limit). Try a shorter or simpler request.",
                            metadata=msg.metadata or {},
                        )
                    )
                    response = None
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content="",
                            metadata=msg.metadata or {},
                        )
                    )
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception as exc:
                logger.exception("Error processing message for session {}", msg.session_key)
                error_detail = f"{type(exc).__name__}: {exc}"
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"System error: {error_detail[:400]}",
                        metadata=msg.metadata or {},
                    )
                )
            finally:
                if previous_correlation_id:
                    bind_correlation_id(previous_correlation_id)
                else:
                    unbind_correlation_id()

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    def get_active_tasks(self) -> list[dict[str, Any]]:
        """Return lightweight metadata for currently active foreground tasks."""
        now = time.monotonic()
        active: list[dict[str, Any]] = []
        for session_key, tasks in self._active_tasks.items():
            for task in tasks:
                if task.done():
                    continue
                started_at = self._active_task_ts.get(id(task))
                elapsed = max(0.0, now - started_at) if started_at is not None else 0.0
                active.append(
                    {
                        "session_key": session_key,
                        "elapsed_seconds": elapsed,
                    }
                )
        active.sort(key=lambda item: item["elapsed_seconds"], reverse=True)
        return active

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (
                msg.chat_id.split(":", 1) if ":" in msg.chat_id else ("cli", msg.chat_id)
            )
            logger.info("Processing system message from {}", msg.sender_id)
            self._record_last_active_session(
                inbound_channel=msg.channel,
                sender_id=msg.sender_id,
                channel=channel,
                chat_id=chat_id,
                message_thread_id=msg.metadata.get("message_thread_id"),
            )
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(
                channel,
                chat_id,
                msg.metadata.get("message_id"),
                msg.metadata.get("message_thread_id"),
            )
            # Initialise MessageTool so intermediate sends preserve topic routing
            if message_tool := self.tools.get("message"):
                if isinstance(message_tool, MessageTool):
                    message_tool.start_turn()
                    message_tool.set_turn_metadata(msg.metadata or {})
            history = session.get_history(max_messages=0)
            # Subagent results are inputs TO the main agent (incoming messages),
            # not the agent's own replies — always use "user" role so the history
            # stays properly alternating and providers like GLM don't reject it.
            current_role = "user"
            history = self._maybe_compact_history(
                history,
                session=session,
                current_message=msg.content,
                channel=channel,
                chat_id=chat_id,
                current_role=current_role,
            )
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content,
                channel=channel,
                chat_id=chat_id,
                current_role=current_role,
                session=session,
            )
            t0 = time.monotonic()
            (
                final_content,
                sys_tools_used,
                all_msgs,
                usage,
                stop_reason,
                run_error,
                tool_events,
            ) = await self._run_agent_loop(
                messages,
                channel=channel,
                chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
                message_thread_id=msg.metadata.get("message_thread_id"),
            )
            skills, files_touched = self.telemetry.extract_from_events(tool_events)
            self.telemetry.record_turn(
                ts=datetime.now(timezone.utc).isoformat(),
                session=key,
                channel=channel,
                chat_id=chat_id,
                model=self.model,
                usage=usage,
                duration_ms=int((time.monotonic() - t0) * 1000),
                stop_reason=stop_reason,
                error=run_error,
                tools_used=sys_tools_used,
                skills=skills,
                files_touched=files_touched,
            )
            # --- Cost tracking ---
            try:
                if usage.get("prompt_tokens", 0) or usage.get("completion_tokens", 0):
                    self._cost_tracker.record(
                        tokens_in=int(usage.get("prompt_tokens", 0)),
                        tokens_out=int(usage.get("completion_tokens", 0)),
                        model=self.model,
                        task_id=get_correlation_id() or "",
                    )
            except Exception:
                logger.warning("Failed to record main agent cost (system turn)")
            # --- Workspace feedback loop: diagnose failures ---
            if run_error and self._diagnose_failure is not None:
                try:
                    self._diagnose_failure(f"system:{key}", run_error)
                except Exception:
                    pass
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
            return OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=final_content or "Background task completed.",
                metadata=msg.metadata or {},
            )

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        self._record_last_active_session(
            inbound_channel=msg.channel,
            sender_id=msg.sender_id,
            channel=msg.channel,
            chat_id=msg.chat_id,
            message_thread_id=msg.metadata.get("message_thread_id"),
        )
        session = self.sessions.get_or_create(key)

        # Slash commands (priority first, then regular dispatch)
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if self.commands.is_priority(raw):
            result = await self.commands.dispatch_priority(ctx)
        else:
            result = await self.commands.dispatch(ctx)
        if result:
            return result

        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(
            msg.channel,
            msg.chat_id,
            msg.metadata.get("message_id"),
            msg.metadata.get("message_thread_id"),
        )
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()
                message_tool.set_turn_metadata(msg.metadata or {})

        history = session.get_history(max_messages=0)
        history = self._maybe_compact_history(
            history,
            session=session,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
        )
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            session=session,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=meta,
                )
            )

        t0 = time.monotonic()
        (
            final_content,
            tools_used,
            all_msgs,
            usage,
            stop_reason,
            run_error,
            tool_events,
        ) = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress or _bus_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            channel=msg.channel,
            chat_id=msg.chat_id,
            message_id=msg.metadata.get("message_id"),
            message_thread_id=msg.metadata.get("message_thread_id"),
        )

        skills, files_touched = self.telemetry.extract_from_events(tool_events)
        self.telemetry.record_turn(
            ts=datetime.now(timezone.utc).isoformat(),
            session=key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            model=self.model,
            usage=usage,
            duration_ms=int((time.monotonic() - t0) * 1000),
            stop_reason=stop_reason,
            error=run_error,
            tools_used=tools_used,
            skills=skills,
            files_touched=files_touched,
        )
        # --- Cost tracking ---
        try:
            if usage.get("prompt_tokens", 0) or usage.get("completion_tokens", 0):
                self._cost_tracker.record(
                    tokens_in=int(usage.get("prompt_tokens", 0)),
                    tokens_out=int(usage.get("completion_tokens", 0)),
                    model=self.model,
                    task_id=get_correlation_id() or "",
                )
        except Exception:
            logger.warning("Failed to record main agent cost")
        # --- Workspace feedback loop: diagnose failures ---
        if run_error and self._diagnose_failure is not None:
            try:
                self._diagnose_failure(f"session:{key}", run_error)
            except Exception:
                pass
        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        # Prepend fallback marker if the provider switched to backup this turn
        if (
            hasattr(self.provider, "active_provider_info")
            and self.provider.active_provider_info
        ):
            final_content = f"⚡ fallback: {self.provider.active_provider_info}\n{final_content}"

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        # Only mark as streamed when streaming completed normally — error/limit
        # responses are never delivered via the stream and must be sent as a
        # regular message so the user actually sees the failure reason.
        if on_stream is not None and stop_reason == "completed":
            meta["_streamed"] = True
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=meta,
        )

    @staticmethod
    def _image_placeholder(block: dict[str, Any]) -> dict[str, str]:
        """Convert an inline image block into a compact text placeholder."""
        path = (block.get("_meta") or {}).get("path", "")
        return {"type": "text", "text": f"[image: {path}]" if path else "[image]"}

    def _truncate_tool_result(self, content: str) -> str:
        """Keep the head and tail of oversized tool output."""
        if len(content) <= self._TOOL_RESULT_MAX_CHARS:
            return content

        head_size = self._TOOL_RESULT_MAX_CHARS // 2
        tail_size = self._TOOL_RESULT_MAX_CHARS - head_size
        omitted = len(content) - head_size - tail_size
        return (
            content[:head_size]
            + f"\n\n... [{omitted:,} chars omitted] ...\n\n"
            + content[-tail_size:]
        )

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if block.get("type") == "image_url" and block.get("image_url", {}).get(
                "url", ""
            ).startswith("data:image/"):
                filtered.append(self._image_placeholder(block))
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if truncate_text and len(text) > self._TOOL_RESULT_MAX_CHARS:
                    text = self._truncate_tool_result(text)
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _maybe_compact_history(
        self,
        history: list[dict],
        session: Session | None = None,
        *,
        current_message: str = "",
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        current_role: str = "user",
    ) -> list[dict]:
        """Apply compaction to history if token count exceeds threshold.

        Pure heuristic — no LLM calls.  Returns history unchanged if compaction
        is disabled, unavailable, or not needed.
        """
        if (
            not self._compaction_enabled
            or self._compact_fn is None
            or self._compaction_policy is None
        ):
            return history
        if not history:
            return history

        try:
            token_source = "history_fallback"
            try:
                probe_messages = self.context.build_messages(
                    history=history,
                    current_message=current_message,
                    media=media,
                    channel=channel,
                    chat_id=chat_id,
                    current_role=current_role,
                    session=session,
                )
                total_tokens, token_source = estimate_prompt_tokens_chain(
                    self.provider,
                    self.model,
                    probe_messages,
                    self.tools.get_definitions(),
                )
                if total_tokens <= 0:
                    raise ValueError("full prompt token estimate unavailable")
            except Exception:
                # Keep direct unit tests and degraded loop states working even
                # if full prompt assembly is unavailable.
                total_tokens = 0
                for msg in history:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        from nanobot_workspace.memory.compaction import estimate_tokens

                        total_tokens += estimate_tokens(content) + estimate_tokens(
                            msg.get("role", "")
                        )
                    elif isinstance(content, list):
                        from nanobot_workspace.memory.compaction import estimate_tokens

                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                total_tokens += estimate_tokens(block.get("text", ""))
                            else:
                                total_tokens += 200
                        total_tokens += estimate_tokens(msg.get("role", ""))

            if not self._compaction_policy.should_compact(total_tokens):
                return history

            logger.info(
                "Compaction triggered: {} tokens via {} > threshold, {} history messages",
                total_tokens,
                token_source,
                len(history),
            )
            compacted = self._compact_fn(
                history,
                token_budget=self._compaction_policy.effective_budget(),
                keep_recent=self._compaction_policy.keep_recent,
            )
            logger.info(
                "Compaction done: {} -> {} messages",
                len(history),
                len(compacted),
            )
            if session is not None and compacted != history:
                try:
                    preserved_prefix = session.messages[: session.last_consolidated]
                    session.messages = preserved_prefix + compacted
                    session.updated_at = datetime.now()
                    self.sessions.save(session)
                    logger.info(
                        "Compaction persisted: {} -> {} unconsolidated messages",
                        len(history),
                        len(compacted),
                    )
                    if self._ws_notify:
                        self._ws_notify.notify(
                            f"Compacted session: {len(history)} -> {len(compacted)} messages"
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to persist compaction, using in-memory only: {}",
                        exc,
                    )
            return compacted
        except Exception as exc:
            logger.warning("Compaction failed, continuing with full history: {}", exc)
            return history

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime

        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = self._truncate_tool_result(content)
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(
                    ContextBuilder._RUNTIME_CONTEXT_TAG
                ):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        return await self._process_message(
            msg,
            session_key=session_key,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
        )
