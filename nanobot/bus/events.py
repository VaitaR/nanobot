"""Event types for the message bus."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def generate_request_id() -> str:
    """Generate a short correlation ID for request tracing."""
    return uuid.uuid4().hex[:12]


@dataclass
class InboundMessage:
    """Message received from a chat channel."""

    channel: str  # telegram, discord, slack, whatsapp
    sender_id: str  # User identifier
    chat_id: str  # Chat/channel identifier
    content: str  # Message text
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)  # Media URLs
    metadata: dict[str, Any] = field(default_factory=dict)  # Channel-specific data
    session_key_override: str | None = None  # Optional override for thread-scoped sessions
    request_id: str | None = None  # OBS-001: correlation ID for request tracing

    def __post_init__(self) -> None:
        """Auto-generate request_id if not provided (CRIT-02 fix)."""
        if self.request_id is None:
            self.request_id = generate_request_id()

    @property
    def session_key(self) -> str:
        """Unique key for session identification."""
        return self.session_key_override or f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """Message to send to a chat channel."""

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    reply_markup: dict[str, Any] | None = None  # Inline keyboard markup (Telegram)
    edit_message_id: int | None = None  # Edit an existing message's markup


