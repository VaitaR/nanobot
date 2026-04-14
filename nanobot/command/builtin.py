"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import sys

from nanobot import __version__
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.utils.helpers import build_status_content


async def cmd_stop(ctx: CommandContext):
    """Cancel all active tasks and subagents for the session."""
    loop = ctx.loop
    msg = ctx.msg
    tasks = loop._active_tasks.pop(msg.session_key, [])
    cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
    for t in tasks:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass
    sub_cancelled = await loop.subagents.cancel_by_session(msg.session_key)
    total = cancelled + sub_cancelled
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return ctx.outbound(content)


async def cmd_restart(ctx: CommandContext):
    """Restart the process in-place via os.execv."""
    import json

    msg = ctx.msg
    resume_prompt = (
        "[System] Gateway restart complete. Continue from the previous conversation "
        "and finish any interrupted work."
    )

    async def _do_restart():
        await asyncio.sleep(1)
        # Write a marker so the new process can confirm the restart to the user.
        try:
            if ctx.loop and hasattr(ctx.loop, "workspace"):
                marker = ctx.loop.workspace / "restart_pending.json"
                marker.write_text(
                    json.dumps(
                        {
                            "channel": msg.channel,
                            "chat_id": msg.chat_id,
                            "resume_prompt": resume_prompt,
                        }
                    )
                )
        except Exception:
            pass
        os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return ctx.outbound("Restarting...")


async def cmd_status(ctx: CommandContext):
    """Build an outbound status message for a session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    ctx_est = 0
    try:
        ctx_est, _ = loop.memory_consolidator.estimate_session_prompt_tokens(session)
    except Exception:
        pass
    if ctx_est <= 0:
        ctx_est = loop._last_usage.get("prompt_tokens", 0)

    usage_snapshot: str | None = None
    try:
        from nanobot_workspace.observability.usage_tracker import (
            format_snapshot,
            load_latest_snapshot,
        )

        results = await asyncio.to_thread(load_latest_snapshot)
        if results:
            usage_snapshot = format_snapshot(results)
    except Exception:
        pass

    return ctx.outbound(
        build_status_content(
            version=__version__, model=loop.model,
            start_time=loop._start_time, last_usage=loop._last_usage,
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
            usage_snapshot=usage_snapshot,
        ),
        render_as="text",
    )


async def cmd_new(ctx: CommandContext):
    """Start a fresh session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    snapshot = session.messages[session.last_consolidated:]
    session.clear()
    loop.sessions.save(session)
    loop.sessions.invalidate(session.key)
    if snapshot:
        loop._schedule_background(loop.memory_consolidator.archive_messages(snapshot))
    return ctx.outbound("New session started.")


async def cmd_help(ctx: CommandContext):
    """Return available slash commands."""
    lines = [
        "🐈 nanobot commands:",
        "/new — Start a new conversation",
        "/stop — Stop the current task",
        "/restart — Restart the bot",
        "/status — Show bot status",
        "/tasks [filter] — List tasks (open|in_progress|blocked|review|done|all)",
        "/heartbeat — Trigger heartbeat check",
        "/help — Show available commands",
    ]
    return ctx.outbound("\n".join(lines), render_as="text")


async def cmd_tasks(ctx: CommandContext):
    """List workspace tasks, optionally filtered by status."""
    status_map = {
        "open": "📋", "in_progress": "🔄",
        "blocked": "🚫", "review": "👁️", "done": "✅",
    }
    arg = ctx.args.strip().lower()
    if arg in ("open", "in_progress", "blocked", "review", "done"):
        status_filter = arg
    elif arg == "all":
        status_filter = "all"
    else:
        status_filter = "active"

    try:
        from nanobot_workspace.tasks import TaskStore
        tasks = TaskStore().list_tasks(status_filter)
    except Exception as exc:
        return ctx.outbound(f"Error loading tasks: {exc}")

    if not tasks:
        return ctx.outbound("No tasks found.")

    lines = [f"📋 Tasks ({status_filter}):"]
    for t in tasks:
        emoji = status_map.get(t.status, "❓")
        short_id = t.id[:16]
        lines.append(f"  {emoji} {short_id} — {t.title}")
    return ctx.outbound("\n".join(lines), render_as="text")


async def cmd_heartbeat(ctx: CommandContext):
    """Manually trigger the heartbeat check and return the result."""
    hb = ctx.loop.heartbeat_service
    if hb is None:
        return ctx.outbound("Heartbeat service is not available.")
    try:
        result = await hb.trigger_now()
        action = result.get("action", "skip")
        tasks = result.get("tasks", "")
        response = result.get("result", "")
        if action == "skip":
            return ctx.outbound(f"💓 Heartbeat: skip — no active tasks.\n{tasks}".strip())
        lines = [f"💓 Heartbeat: run\n\n📋 Tasks: {tasks}"]
        if response:
            lines.append(f"\n✅ Result:\n{response}")
        return ctx.outbound("\n".join(lines))
    except Exception as e:
        return ctx.outbound(f"Heartbeat error: {e}")


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/heartbeat", cmd_heartbeat)
    router.exact("/status", cmd_status)
    router.exact("/tasks", cmd_tasks)
    router.prefix("/tasks ", cmd_tasks)
    router.exact("/help", cmd_help)
