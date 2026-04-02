"""Tests for nanobot.agent.verify — post-subagent artifact verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.result_envelope import Artifact, ResultEnvelope
from nanobot.agent.verify import verify_artifact_exists, verify_envelope


# ---------------------------------------------------------------------------
# verify_artifact_exists
# ---------------------------------------------------------------------------

class TestVerifyArtifactExists:
    def test_file_exists_verified_true(self, tmp_path: Path) -> None:
        """Existing file gets verified=True."""
        f = tmp_path / "hello.py"
        f.write_text("print('hello')")
        a = Artifact(path=str(f), description="hello", kind="file")
        result = verify_artifact_exists(a)
        assert result.verified is True

    def test_file_missing_verified_false(self, tmp_path: Path) -> None:
        """Nonexistent file gets verified=False."""
        a = Artifact(path=str(tmp_path / "nope.py"), description="missing", kind="file")
        result = verify_artifact_exists(a)
        assert result.verified is False

    def test_diff_exists_verified_true(self, tmp_path: Path) -> None:
        """Diff artifact for existing file gets verified=True."""
        f = tmp_path / "changed.py"
        f.write_text("x = 1")
        a = Artifact(path=str(f), description="edited", kind="diff")
        result = verify_artifact_exists(a)
        assert result.verified is True

    def test_url_artifact_skipped(self) -> None:
        """URL artifacts are not verified (verified stays False)."""
        a = Artifact(path="https://example.com", description="ref", kind="url")
        result = verify_artifact_exists(a)
        assert result.verified is False

    def test_metric_artifact_skipped(self) -> None:
        """Metric artifacts are not verified (verified stays False)."""
        a = Artifact(path="coverage:85%", description="cov", kind="metric")
        result = verify_artifact_exists(a)
        assert result.verified is False


# ---------------------------------------------------------------------------
# verify_envelope
# ---------------------------------------------------------------------------

class TestVerifyEnvelope:
    def test_all_verified_envelope_ok(self, tmp_path: Path) -> None:
        """All artifacts verified → status stays 'ok'."""
        f = tmp_path / "a.py"
        f.write_text("# ok")
        env = ResultEnvelope(
            status="ok",
            summary="done",
            artifacts=[Artifact(path=str(f), description="a", kind="file")],
        )
        result = verify_envelope(env)
        assert result.status == "ok"
        assert result.artifacts[0].verified is True

    def test_missing_files_downgrades_to_partial(self, tmp_path: Path) -> None:
        """Some missing files → status='partial', summary has warning."""
        exists = tmp_path / "yes.py"
        exists.write_text("# yes")
        missing = tmp_path / "no.py"
        env = ResultEnvelope(
            status="ok",
            summary="created two files",
            artifacts=[
                Artifact(path=str(exists), description="yes", kind="file"),
                Artifact(path=str(missing), description="no", kind="file"),
            ],
        )
        result = verify_envelope(env)
        assert result.status == "partial"
        assert "1 file(s) not found on disk" in result.summary

    def test_error_envelope_unchanged(self, tmp_path: Path) -> None:
        """Envelope with status='error' is returned as-is (no verification)."""
        missing = tmp_path / "ghost.py"
        env = ResultEnvelope(
            status="error",
            summary="boom",
            artifacts=[Artifact(path=str(missing), description="ghost", kind="file")],
            error="something broke",
        )
        result = verify_envelope(env)
        assert result is env  # returned unchanged
        assert result.artifacts[0].verified is False  # never checked

    def test_empty_artifacts_unchanged(self) -> None:
        """No artifacts → envelope returned as-is."""
        env = ResultEnvelope(status="ok", summary="nothing to verify")
        result = verify_envelope(env)
        assert result is env
        assert result.status == "ok"

    def test_envelope_preserves_other_fields(self, tmp_path: Path) -> None:
        """details, stop_reason, error are preserved through verification."""
        f = tmp_path / "x.py"
        f.write_text("pass")
        env = ResultEnvelope(
            status="ok",
            summary="wrote x",
            artifacts=[Artifact(path=str(f), description="x", kind="file")],
            details="full details here",
            stop_reason="max_turns",
            error=None,
        )
        result = verify_envelope(env)
        assert result.details == "full details here"
        assert result.stop_reason == "max_turns"
        assert result.error is None
        assert result.status == "ok"
        assert result.artifacts[0].verified is True

    def test_non_file_artifacts_not_counted_as_missing(self, tmp_path: Path) -> None:
        """URL/metric artifacts with verified=False don't trigger downgrade."""
        env = ResultEnvelope(
            status="ok",
            summary="mixed results",
            artifacts=[
                Artifact(path="https://example.com", description="ref", kind="url"),
                Artifact(path="cov:90%", description="cov", kind="metric"),
            ],
        )
        result = verify_envelope(env)
        # No file/diff is missing → status stays ok
        assert result.status == "ok"

    def test_partial_envelope_stays_partial(self, tmp_path: Path) -> None:
        """Already partial envelope with missing file stays partial."""
        missing = tmp_path / "ghost.py"
        env = ResultEnvelope(
            status="partial",
            summary="incomplete",
            artifacts=[Artifact(path=str(missing), description="ghost", kind="file")],
        )
        result = verify_envelope(env)
        assert result.status == "partial"
        # Summary should NOT be re-annotated (only ok → partial triggers annotation)
        assert "not found" not in result.summary
