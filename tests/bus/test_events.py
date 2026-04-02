"""Tests for nanobot.bus.events — OBS-001 correlation IDs."""

from __future__ import annotations

import re

from nanobot.bus.events import InboundMessage, generate_request_id


def test_inbound_message_defaults() -> None:
    """All optional fields have defaults; request_id is None."""
    msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="hi")
    assert msg.request_id is None
    assert msg.session_key_override is None
    assert msg.media == []
    assert msg.metadata == {}


def test_inbound_message_with_request_id() -> None:
    """request_id can be set on InboundMessage."""
    msg = InboundMessage(
        channel="telegram",
        sender_id="u1",
        chat_id="c1",
        content="hi",
        request_id="abc123def456",
    )
    assert msg.request_id == "abc123def456"


def test_generate_request_id_format() -> None:
    """generate_request_id returns a 12-char hex string."""
    rid = generate_request_id()
    assert isinstance(rid, str)
    assert len(rid) == 12
    assert re.fullmatch(r"[0-9a-f]{12}", rid), f"not hex: {rid!r}"


def test_generate_request_id_unique() -> None:
    """Two calls produce different IDs (probabilistic but sufficient)."""
    assert generate_request_id() != generate_request_id()


def test_session_key_property_default() -> None:
    """session_key defaults to channel:chat_id."""
    msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="hi")
    assert msg.session_key == "telegram:c1"


def test_session_key_property_override() -> None:
    """session_key uses override when set."""
    msg = InboundMessage(
        channel="telegram",
        sender_id="u1",
        chat_id="c1",
        content="hi",
        session_key_override="custom:key",
    )
    assert msg.session_key == "custom:key"
