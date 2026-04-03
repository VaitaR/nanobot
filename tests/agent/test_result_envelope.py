"""Tests for the structured subagent result envelope."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from nanobot.agent.result_envelope import Artifact, ResultEnvelope, extract_artifacts

# ---------------------------------------------------------------------------
# ResultEnvelope & Artifact creation
# ---------------------------------------------------------------------------


class TestResultEnvelope:
    def test_ok_envelope_defaults(self):
        env = ResultEnvelope(status="ok", summary="Done.")
        assert env.status == "ok"
        assert env.summary == "Done."
        assert env.artifacts == []
        assert env.details == ""
        assert env.stop_reason == "completed"
        assert env.error is None

    def test_error_envelope(self):
        env = ResultEnvelope(
            status="error",
            summary="Something went wrong",
            error="boom",
            stop_reason="tool_error",
        )
        assert env.status == "error"
        assert env.error == "boom"
        assert env.stop_reason == "tool_error"

    def test_partial_envelope_with_artifacts(self):
        arts = [Artifact(path="/tmp/a.py", description="new file", kind="file")]
        env = ResultEnvelope(
            status="partial",
            summary="Half done",
            artifacts=arts,
            details="full output here",
        )
        assert env.status == "partial"
        assert len(env.artifacts) == 1
        assert env.details == "full output here"


class TestArtifact:
    def test_fields(self):
        a = Artifact(path="/foo/bar.py", description="edited file", kind="diff")
        assert a.path == "/foo/bar.py"
        assert a.kind == "diff"


# ---------------------------------------------------------------------------
# extract_artifacts
# ---------------------------------------------------------------------------


class TestExtractArtifacts:
    def test_empty_events(self):
        assert extract_artifacts([]) == []

    def test_skips_error_events(self):
        events = [
            {"name": "write_file", "status": "error", "arguments": {"path": "/tmp/bad.py"}},
        ]
        assert extract_artifacts(events) == []

    def test_write_file_artifact(self):
        events = [
            {
                "name": "write_file",
                "status": "ok",
                "arguments": {"path": "/tmp/new.py"},
            },
        ]
        arts = extract_artifacts(events)
        assert len(arts) == 1
        assert arts[0].path == "/tmp/new.py"
        assert arts[0].kind == "file"

    def test_edit_file_artifact(self):
        events = [
            {
                "name": "edit_file",
                "status": "ok",
                "arguments": {"path": "/tmp/old.py"},
            },
        ]
        arts = extract_artifacts(events)
        assert len(arts) == 1
        assert arts[0].kind == "diff"

    def test_deduplicates_same_path(self):
        events = [
            {"name": "write_file", "status": "ok", "arguments": {"path": "/tmp/a.py"}},
            {"name": "edit_file", "status": "ok", "arguments": {"path": "/tmp/a.py"}},
        ]
        arts = extract_artifacts(events)
        assert len(arts) == 1
        # First (write_file) wins
        assert arts[0].kind == "file"

    def test_ignores_non_file_tools(self):
        events = [
            {"name": "list_dir", "status": "ok", "arguments": {"path": "/tmp"}},
            {"name": "read_file", "status": "ok", "arguments": {"path": "/tmp/x.py"}},
        ]
        arts = extract_artifacts(events)
        assert len(arts) == 0


# ---------------------------------------------------------------------------
# SubagentManager._announce_result routing
# ---------------------------------------------------------------------------


def _make_manager(bus=None):
    """Create a minimal SubagentManager with mocked provider."""
    from nanobot.agent.subagent import SubagentManager
    from nanobot.bus.queue import MessageBus

    bus = bus or MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    mgr = SubagentManager(provider=provider, workspace=MagicMock(), bus=bus)
    return mgr, bus


class TestAnnounceResultRouting:
    """Verify that _announce_result routes short ok results directly
    and complex/error results through the main agent."""

    @pytest.mark.asyncio
    async def test_short_ok_goes_to_outbound(self):
        import os
        import tempfile

        mgr, bus = _make_manager()
        origin = {"channel": "cli", "chat_id": "direct"}
        # Create a real temp file so verify_envelope doesn't downgrade to partial
        tmp = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
        tmp.write("# test\n")
        tmp.close()
        try:
            envelope = ResultEnvelope(
                status="ok",
                summary="Created the new module.",
                artifacts=[
                    Artifact(path=tmp.name, description="new file", kind="file"),
                ],
            )
            await mgr._announce_result("t1", "label", "do thing", envelope, origin)

            # Should have published exactly one outbound message
            out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            assert "[label] Created the new module." in out.content
            assert tmp.name in out.content
            # Nothing in inbound
            assert bus.inbound_size == 0
        finally:
            os.unlink(tmp.name)

    @pytest.mark.asyncio
    async def test_long_ok_goes_to_inbound(self):
        mgr, bus = _make_manager()
        origin = {"channel": "cli", "chat_id": "direct"}
        long_summary = "A" * 200
        envelope = ResultEnvelope(status="ok", summary=long_summary)

        await mgr._announce_result("t2", "lbl", "task", envelope, origin)

        # Should have published to inbound (for main agent), not outbound
        assert bus.outbound_size == 0
        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert "Summarize this naturally" in msg.content
        assert "task" in msg.content

    @pytest.mark.asyncio
    async def test_error_goes_to_inbound(self):
        mgr, bus = _make_manager()
        origin = {"channel": "cli", "chat_id": "direct"}
        envelope = ResultEnvelope(
            status="error",
            summary="Something broke",
            error="division by zero",
        )

        await mgr._announce_result("t3", "lbl", "task", envelope, origin)

        assert bus.outbound_size == 0
        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert "failed" in msg.content
        assert "division by zero" in msg.content

    @pytest.mark.asyncio
    async def test_partial_goes_to_inbound(self):
        mgr, bus = _make_manager()
        origin = {"channel": "cli", "chat_id": "direct"}
        envelope = ResultEnvelope(
            status="partial",
            summary="Half done",
        )

        await mgr._announce_result("t4", "lbl", "task", envelope, origin)

        assert bus.outbound_size == 0
        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert "partial results" in msg.content

    @pytest.mark.asyncio
    async def test_ok_with_error_field_goes_to_inbound(self):
        mgr, bus = _make_manager()
        origin = {"channel": "cli", "chat_id": "direct"}
        # status="ok" but error is set — route through main agent
        envelope = ResultEnvelope(
            status="ok",
            summary="Done but with warning",
            error="non-fatal issue",
        )

        await mgr._announce_result("t5", "lbl", "task", envelope, origin)

        assert bus.outbound_size == 0
        await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_message_thread_id_propagated_outbound(self):
        mgr, bus = _make_manager()
        origin = {"channel": "tg", "chat_id": "123", "message_thread_id": 42}
        envelope = ResultEnvelope(status="ok", summary="Quick result.")

        await mgr._announce_result("t6", "lbl", "task", envelope, origin)

        out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
        assert out.metadata.get("message_thread_id") == 42

    @pytest.mark.asyncio
    async def test_message_thread_id_propagated_inbound(self):
        mgr, bus = _make_manager()
        origin = {"channel": "tg", "chat_id": "123", "message_thread_id": 99}
        envelope = ResultEnvelope(status="error", summary="Boom.")

        await mgr._announce_result("t7", "lbl", "task", envelope, origin)

        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert msg.metadata.get("message_thread_id") == 99


# ---------------------------------------------------------------------------
# _format_envelope_for_agent
# ---------------------------------------------------------------------------


class TestFormatEnvelopeForAgent:
    def test_includes_artifacts(self):
        mgr, _ = _make_manager()
        envelope = ResultEnvelope(
            status="ok",
            summary="Done",
            artifacts=[
                Artifact(path="/a.py", description="new file", kind="file"),
                Artifact(path="/b.py", description="edited", kind="diff"),
            ],
            error="minor issue",
            details="full trace",
        )
        text = mgr._format_envelope_for_agent("lbl", "do x", envelope)
        assert "Artifacts:" in text
        assert "[file] /a.py" in text
        assert "[diff] /b.py" in text
        assert "Error: minor issue" in text
        assert "Full details:" in text
        assert "full trace" in text
        assert "Summarize this naturally" in text

    def test_omits_details_when_same_as_summary(self):
        mgr, _ = _make_manager()
        envelope = ResultEnvelope(status="ok", summary="Done", details="Done")
        text = mgr._format_envelope_for_agent("lbl", "do x", envelope)
        assert "Full details:" not in text

    def test_status_text_partial(self):
        mgr, _ = _make_manager()
        envelope = ResultEnvelope(status="partial", summary="Half")
        text = mgr._format_envelope_for_agent("lbl", "task", envelope)
        assert "partial results" in text

    def test_status_text_error(self):
        mgr, _ = _make_manager()
        envelope = ResultEnvelope(status="error", summary="Fail")
        text = mgr._format_envelope_for_agent("lbl", "task", envelope)
        assert "failed" in text


# ---------------------------------------------------------------------------
# _build_envelope
# ---------------------------------------------------------------------------


class TestBuildEnvelope:
    def test_basic_envelope(self):
        from nanobot.agent.subagent import SubagentManager

        env = SubagentManager._build_envelope("All good.", "ok")
        assert env.status == "ok"
        assert env.summary == "All good."
        assert env.details == "All good."
        assert env.artifacts == []

    def test_envelope_with_tool_events(self):
        from nanobot.agent.subagent import SubagentManager

        events = [
            {"name": "write_file", "status": "ok", "arguments": {"path": "/tmp/x.py"}},
        ]
        env = SubagentManager._build_envelope("Done.", "ok", tool_events=events)
        assert len(env.artifacts) == 1
        assert env.artifacts[0].path == "/tmp/x.py"

    def test_error_envelope(self):
        from nanobot.agent.subagent import SubagentManager

        env = SubagentManager._build_envelope(
            "Boom.", "error", stop_reason="tool_error", error="div by zero",
        )
        assert env.error == "div by zero"
        assert env.stop_reason == "tool_error"
