"""Tests for Phase 3 Telegram inline keyboard UX.

Covers: keyboard building, callback data parsing, Telegram channel
integration (reply_markup handling, callback query routing), and
broker → channel message flow.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.checkpoint.broker import CheckpointBroker
from nanobot.checkpoint.policy import ReviewPolicy
from nanobot.checkpoint.snapshot import CheckpointSnapshot, ToolCallSummary
from nanobot.checkpoint.user_action import UserAction


def _make_snapshot(
    total_iterations: int = 8,
    max_iterations: int = 10,
    tool_calls: tuple[ToolCallSummary, ...] = (),
    error_count: int = 0,
    loop_detected: bool = False,
    task_id: str = "152315",
    **kwargs,
) -> CheckpointSnapshot:
    return CheckpointSnapshot(
        total_iterations=total_iterations,
        max_iterations=max_iterations,
        tool_calls=tool_calls,
        files_touched=frozenset(kwargs.get("files_touched", set())),
        last_llm_outputs=tuple(kwargs.get("last_llm_outputs", [])),
        error_count=error_count,
        loop_detected=loop_detected,
        stuck_score=0.0,
        task_id=task_id,
    )


# ───────────────────────────────────────────────────────────────────────
# Keyboard building
# ───────────────────────────────────────────────────────────────────────


class TestBuildKeyboard:
    """CheckpointBroker._build_keyboard produces correct markup dicts."""

    def test_layout_structure(self) -> None:
        kb = CheckpointBroker._build_keyboard("152315", remaining_iterations=5)
        rows = kb["inline_keyboard"]
        assert len(rows) == 2
        # First row: Continue button
        assert len(rows[0]) == 1
        assert rows[0][0]["text"].startswith("▶ Continue")
        # Second row: Done, Stop, Details
        assert len(rows[1]) == 3
        assert rows[1][0]["text"] == "✅ Done"
        assert rows[1][1]["text"] == "⏹ Stop"
        assert rows[1][2]["text"] == "📋 Details"

    def test_continue_button_shows_correct_count(self) -> None:
        kb = CheckpointBroker._build_keyboard("152315", remaining_iterations=20)
        cont_btn = kb["inline_keyboard"][0][0]
        # Should cap at 10
        assert cont_btn["text"] == "▶ Continue +10"
        assert cont_btn["callback_data"] == "chk:152315:continue:10"

    def test_continue_button_defaults_when_zero_remaining(self) -> None:
        kb = CheckpointBroker._build_keyboard("152315", remaining_iterations=0)
        cont_btn = kb["inline_keyboard"][0][0]
        assert cont_btn["text"] == "▶ Continue +5"
        assert cont_btn["callback_data"] == "chk:152315:continue:5"

    def test_callback_data_uses_task_id(self) -> None:
        kb = CheckpointBroker._build_keyboard("abc123", remaining_iterations=3)
        assert "chk:abc123:" in kb["inline_keyboard"][0][0]["callback_data"]
        assert "chk:abc123:done" in kb["inline_keyboard"][1][0]["callback_data"]
        assert "chk:abc123:stop" in kb["inline_keyboard"][1][1]["callback_data"]
        assert "chk:abc123:details" in kb["inline_keyboard"][1][2]["callback_data"]


# ───────────────────────────────────────────────────────────────────────
# Callback data parsing
# ───────────────────────────────────────────────────────────────────────


class TestParseCallbackData:
    """CheckpointBroker.parse_callback_data extracts (task_id, action, param)."""

    def test_continue_with_param(self) -> None:
        result = CheckpointBroker.parse_callback_data("chk:152315:continue:5")
        assert result == ("152315", "continue", 5)

    def test_done_no_param(self) -> None:
        result = CheckpointBroker.parse_callback_data("chk:152315:done")
        assert result == ("152315", "done", None)

    def test_stop_no_param(self) -> None:
        result = CheckpointBroker.parse_callback_data("chk:152315:stop")
        assert result == ("152315", "stop", None)

    def test_details_no_param(self) -> None:
        result = CheckpointBroker.parse_callback_data("chk:152315:details")
        assert result == ("152315", "details", None)

    def test_non_checkpoint_callback_returns_none(self) -> None:
        assert CheckpointBroker.parse_callback_data("other:123:action") is None

    def test_empty_string_returns_none(self) -> None:
        assert CheckpointBroker.parse_callback_data("") is None

    def test_none_input_returns_none(self) -> None:
        assert CheckpointBroker.parse_callback_data(None) is None

    def test_invalid_param_returns_none_param(self) -> None:
        result = CheckpointBroker.parse_callback_data("chk:152315:continue:abc")
        assert result == ("152315", "continue", None)

    def test_insufficient_parts_returns_none(self) -> None:
        assert CheckpointBroker.parse_callback_data("chk:") is None
        assert CheckpointBroker.parse_callback_data("chk:152315") is None


# ───────────────────────────────────────────────────────────────────────
# Broker escalation includes reply_markup
# ───────────────────────────────────────────────────────────────────────


class TestBrokerEscalationKeyboard:
    """Escalation messages from the broker include reply_markup."""

    async def test_escalation_has_reply_markup(self) -> None:
        bus = MessageBus()
        policy = ReviewPolicy(review_timeout=60)
        snapshot = _make_snapshot(
            total_iterations=8,
            max_iterations=10,
            task_id="152315",
            last_llm_outputs=["same"] * 5,
            files_touched=set(),
        )
        broker = CheckpointBroker(policy=policy, bus=bus, origin={"channel": "cli", "chat_id": "c1"})

        pause_task = asyncio.create_task(broker.pause("152315", snapshot))
        await asyncio.sleep(0.05)

        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1)
        assert msg.reply_markup is not None
        assert "inline_keyboard" in msg.reply_markup
        # Keyboard has 2 rows
        assert len(msg.reply_markup["inline_keyboard"]) == 2

        # Clean up
        broker.resolve_checkpoint("152315", UserAction.CONTINUE, 5)
        await asyncio.wait_for(pause_task, timeout=1)

    async def test_details_has_reply_markup(self) -> None:
        bus = MessageBus()
        policy = ReviewPolicy(review_timeout=60)
        snapshot = _make_snapshot(
            total_iterations=8,
            max_iterations=10,
            task_id="test123",
            last_llm_outputs=["same"] * 5,
            files_touched=set(),
        )
        broker = CheckpointBroker(policy=policy, bus=bus, origin={})

        pause_task = asyncio.create_task(broker.pause("test123", snapshot))
        await asyncio.sleep(0.05)
        await bus.consume_outbound()  # drain escalation alert

        # Trigger details sub-flow
        broker.resolve_checkpoint("test123", UserAction.DETAILS)
        await asyncio.sleep(0.05)

        # The details message should have reply_markup
        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1)
        assert msg.reply_markup is not None
        assert "inline_keyboard" in msg.reply_markup

        # Clean up
        broker.resolve_checkpoint("test123", UserAction.STOP)
        await asyncio.wait_for(pause_task, timeout=1)

    async def test_timeout_notification_no_keyboard(self) -> None:
        """Timeout notification should NOT have reply_markup (no buttons to press)."""
        bus = MessageBus()
        policy = ReviewPolicy(review_timeout=0.1)
        snapshot = _make_snapshot(
            task_id="timeout1",
            last_llm_outputs=["same"] * 5,
            files_touched=set(),
        )
        broker = CheckpointBroker(policy=policy, bus=bus, origin={})

        action, extra = await broker.pause("timeout1", snapshot)

        assert action == UserAction.CONTINUE
        # Should have escalation + timeout notification
        msgs: list[OutboundMessage] = []
        while bus.outbound_size > 0:
            msgs.append(await asyncio.wait_for(bus.consume_outbound(), timeout=1))

        # Find the timeout notification
        timeout_msgs = [m for m in msgs if "auto-continued" in m.content.lower()]
        assert len(timeout_msgs) >= 1
        # Timeout notification should not have keyboard
        assert timeout_msgs[0].reply_markup is None


# ───────────────────────────────────────────────────────────────────────
# OutboundMessage new fields
# ───────────────────────────────────────────────────────────────────────


class TestOutboundMessageFields:
    """OutboundMessage supports reply_markup and edit_message_id."""

    def test_default_none(self) -> None:
        msg = OutboundMessage(channel="telegram", chat_id="123", content="hello")
        assert msg.reply_markup is None
        assert msg.edit_message_id is None

    def test_with_reply_markup(self) -> None:
        markup = {"inline_keyboard": [[{"text": "OK", "callback_data": "chk:1:done"}]]}
        msg = OutboundMessage(
            channel="telegram", chat_id="123", content="hello", reply_markup=markup
        )
        assert msg.reply_markup == markup

    def test_with_edit_message_id(self) -> None:
        msg = OutboundMessage(
            channel="telegram", chat_id="123", content="",
            edit_message_id=42,
        )
        assert msg.edit_message_id == 42


# ───────────────────────────────────────────────────────────────────────
# Telegram channel inline keyboard handling
# ───────────────────────────────────────────────────────────────────────


class TestTelegramInlineKeyboard:
    """Telegram channel correctly handles reply_markup in outbound messages."""

    @pytest.mark.asyncio
    async def test_send_with_reply_markup(self) -> None:
        """send() passes reply_markup to send_message via _send_text."""
        try:
            from nanobot.channels.telegram import TelegramChannel, TelegramConfig
            from telegram import InlineKeyboardMarkup
        except ImportError:
            pytest.skip("python-telegram-bot not installed")

        bus = MessageBus()
        config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
        channel = TelegramChannel(config, bus)

        # Minimal fake app
        class _FakeBot:
            def __init__(self):
                self.sent = []

            async def send_message(self, **kwargs):
                self.sent.append(kwargs)
                return SimpleNamespace(message_id=len(self.sent))

        class _FakeApp:
            def __init__(self):
                self.bot = _FakeBot()

        channel._app = _FakeApp()

        markup = {
            "inline_keyboard": [
                [{"text": "✅ Done", "callback_data": "chk:1:done"}],
            ]
        }
        await channel.send(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="checkpoint alert",
            reply_markup=markup,
        ))

        assert len(channel._app.bot.sent) == 1
        sent = channel._app.bot.sent[0]
        assert sent["reply_markup"] is not None
        assert isinstance(sent["reply_markup"], InlineKeyboardMarkup)

    @pytest.mark.asyncio
    async def test_send_without_markup_no_keyboard(self) -> None:
        """Normal messages without reply_markup should not get a keyboard."""
        try:
            from nanobot.channels.telegram import TelegramChannel, TelegramConfig
        except ImportError:
            pytest.skip("python-telegram-bot not installed")

        bus = MessageBus()
        config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
        channel = TelegramChannel(config, bus)

        class _FakeBot:
            def __init__(self):
                self.sent = []

            async def send_message(self, **kwargs):
                self.sent.append(kwargs)
                return SimpleNamespace(message_id=len(self.sent))

        class _FakeApp:
            def __init__(self):
                self.bot = _FakeBot()

        channel._app = _FakeApp()

        await channel.send(OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="normal message",
        ))

        assert len(channel._app.bot.sent) == 1
        sent = channel._app.bot.sent[0]
        assert sent.get("reply_markup") is None

    @pytest.mark.asyncio
    async def test_build_inline_markup_converts_dict(self) -> None:
        """_build_inline_markup correctly converts dict to InlineKeyboardMarkup."""
        try:
            from nanobot.channels.telegram import TelegramChannel
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        except ImportError:
            pytest.skip("python-telegram-bot not installed")

        markup_dict = {
            "inline_keyboard": [
                [{"text": "▶ Continue +5", "callback_data": "chk:152315:continue:5"}],
                [
                    {"text": "✅ Done", "callback_data": "chk:152315:done"},
                    {"text": "⏹ Stop", "callback_data": "chk:152315:stop"},
                ],
            ]
        }

        result = TelegramChannel._build_inline_markup(markup_dict)
        assert isinstance(result, InlineKeyboardMarkup)
        assert len(result.inline_keyboard) == 2
        assert len(result.inline_keyboard[0]) == 1
        assert isinstance(result.inline_keyboard[0][0], InlineKeyboardButton)
        assert result.inline_keyboard[0][0].text == "▶ Continue +5"
        assert result.inline_keyboard[0][0].callback_data == "chk:152315:continue:5"

    @pytest.mark.asyncio
    async def test_build_inline_markup_empty_rows(self) -> None:
        """_build_inline_markup returns None for empty rows."""
        try:
            from nanobot.channels.telegram import TelegramChannel
        except ImportError:
            pytest.skip("python-telegram-bot not installed")

        assert TelegramChannel._build_inline_markup({"inline_keyboard": []}) is None
        assert TelegramChannel._build_inline_markup({}) is None


# ───────────────────────────────────────────────────────────────────────
# Callback query routing
# ───────────────────────────────────────────────────────────────────────


class TestCallbackQueryRouting:
    """_on_callback_query correctly parses and routes checkpoint callbacks."""

    @pytest.mark.asyncio
    async def test_valid_callback_resolves_checkpoint(self) -> None:
        """A valid chk: callback calls the resolver and acknowledges."""
        try:
            from nanobot.channels.telegram import TelegramChannel, TelegramConfig
        except ImportError:
            pytest.skip("python-telegram-bot not installed")

        bus = MessageBus()
        config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
        channel = TelegramChannel(config, bus)
        channel._app = None  # not needed for callback handling

        resolved_calls: list[tuple] = []

        def mock_resolver(task_id: str, action: str, param=None) -> bool:
            resolved_calls.append((task_id, action, param))
            return True

        channel._checkpoint_resolver = mock_resolver

        # Simulate a callback query
        query = SimpleNamespace(
            data="chk:152315:continue:5",
            answer=AsyncMock(),
            edit_message_reply_markup=AsyncMock(),
        )
        update = SimpleNamespace(callback_query=query)

        await channel._on_callback_query(update, None)

        assert len(resolved_calls) == 1
        assert resolved_calls[0] == ("152315", "continue", 5)
        query.answer.assert_awaited_once()
        query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)

    @pytest.mark.asyncio
    async def test_invalid_callback_ignored(self) -> None:
        """Non-chk: callbacks are silently ignored."""
        try:
            from nanobot.channels.telegram import TelegramChannel, TelegramConfig
        except ImportError:
            pytest.skip("python-telegram-bot not installed")

        bus = MessageBus()
        config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
        channel = TelegramChannel(config, bus)

        resolved_calls: list = []
        channel._checkpoint_resolver = lambda *a: resolved_calls.append(a) or True

        query = SimpleNamespace(
            data="other:data:here",
            answer=AsyncMock(),
            edit_message_reply_markup=AsyncMock(),
        )
        update = SimpleNamespace(callback_query=query)

        await channel._on_callback_query(update, None)

        assert len(resolved_calls) == 0

    @pytest.mark.asyncio
    async def test_no_resolver_set_does_nothing(self) -> None:
        """If _checkpoint_resolver is None, callback is ignored."""
        try:
            from nanobot.channels.telegram import TelegramChannel, TelegramConfig
        except ImportError:
            pytest.skip("python-telegram-bot not installed")

        bus = MessageBus()
        config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
        channel = TelegramChannel(config, bus)
        assert channel._checkpoint_resolver is None

        query = SimpleNamespace(
            data="chk:152315:done",
            answer=AsyncMock(),
            edit_message_reply_markup=AsyncMock(),
        )
        update = SimpleNamespace(callback_query=query)

        # Should not raise
        await channel._on_callback_query(update, None)
        query.answer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unresolved_checkpoint_shows_alert(self) -> None:
        """When resolver returns False, answer with alert text."""
        try:
            from nanobot.channels.telegram import TelegramChannel, TelegramConfig
        except ImportError:
            pytest.skip("python-telegram-bot not installed")

        bus = MessageBus()
        config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
        channel = TelegramChannel(config, bus)

        channel._checkpoint_resolver = lambda *a: False  # not resolved

        query = SimpleNamespace(
            data="chk:999:done",
            answer=AsyncMock(),
            edit_message_reply_markup=AsyncMock(),
        )
        update = SimpleNamespace(callback_query=query)

        await channel._on_callback_query(update, None)

        query.answer.assert_awaited_once_with(
            "Checkpoint no longer pending", show_alert=True,
        )
        query.edit_message_reply_markup.assert_not_awaited()


# ───────────────────────────────────────────────────────────────────────
# CheckpointSnapshot task_id field
# ───────────────────────────────────────────────────────────────────────


class TestSnapshotTaskId:
    """CheckpointSnapshot.task_id is populated by CheckpointHook.build_snapshot."""

    def test_default_task_id_empty(self) -> None:
        snapshot = CheckpointSnapshot(total_iterations=5, max_iterations=10)
        assert snapshot.task_id == ""

    def test_custom_task_id(self) -> None:
        snapshot = _make_snapshot(task_id="nb-42")
        assert snapshot.task_id == "nb-42"
