import json
from types import SimpleNamespace

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel


class _DummyChannel(BaseChannel):
    name = "dummy"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, msg: OutboundMessage) -> None:
        return None


def test_is_allowed_requires_exact_match() -> None:
    channel = _DummyChannel(SimpleNamespace(allow_from=["allow@email.com"]), MessageBus())

    assert channel.is_allowed("allow@email.com") is True
    assert channel.is_allowed("attacker|allow@email.com") is False


@pytest.mark.asyncio
async def test_handle_message_persists_last_active_session(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("nanobot.channels.base.get_workspace_path", lambda: tmp_path)
    channel = _DummyChannel(SimpleNamespace(allow_from=["sender"]), MessageBus())

    await channel._handle_message(
        sender_id="sender",
        chat_id="chat-1",
        content="hello",
        metadata={"message_thread_id": "42"},
    )

    payload = json.loads((tmp_path / "data" / "last_active_session.json").read_text())
    assert payload["channel"] == "dummy"
    assert payload["chat_id"] == "chat-1"
    assert payload["message_thread_id"] == "42"
