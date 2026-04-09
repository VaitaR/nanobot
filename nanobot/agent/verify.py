"""Post-subagent artifact verification.

Validates that claimed file/diff artifacts actually exist on disk,
detecting the "illusion of execution" where a subagent reports files
it never wrote.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from nanobot.agent.result_envelope import Artifact, ResultEnvelope

log = structlog.get_logger(__name__)


def verify_artifact_exists(
    artifact: Artifact,
    *,
    correlation_id: str | None = None,
) -> Artifact:
    """Check if a file artifact exists on disk.

    Non-file artifacts (url, metric) are returned with ``verified=False``
    but are *not* considered missing — they simply can't be checked this way.
    """
    if artifact.kind not in ("file", "diff"):
        return artifact  # URLs/metrics can't be checked this way
    exists = Path(artifact.path).exists()
    return Artifact(
        path=artifact.path,
        description=artifact.description,
        kind=artifact.kind,
        verified=exists,
    )


def verify_envelope(
    envelope: ResultEnvelope,
    *,
    correlation_id: str | None = None,
) -> ResultEnvelope:
    """Verify all file/diff artifacts in the envelope.

    Downgrades status from ``"ok"`` to ``"partial"`` if any claimed file
    is missing on disk.  Returns a *new* ``ResultEnvelope`` (dataclass
    with slots=True, so we build fresh instances).
    """
    if not envelope.artifacts or envelope.status == "error":
        return envelope

    verified_artifacts = [
        verify_artifact_exists(a, correlation_id=correlation_id) for a in envelope.artifacts
    ]
    missing = [a for a in verified_artifacts if not a.verified and a.kind in ("file", "diff")]

    if missing and envelope.status == "ok":
        paths = ", ".join(a.path for a in missing)
        log.bind(correlation_id=correlation_id).warning(
            "Subagent claimed files that don't exist",
            missing_paths=paths,
        )
        return ResultEnvelope(
            status="partial",
            summary=(f"{envelope.summary} (warning: {len(missing)} file(s) not found on disk)"),
            artifacts=verified_artifacts,
            details=envelope.details,
            stop_reason=envelope.stop_reason,
            error=envelope.error,
        )

    # All verified (or already partial) — return with verified artifact list
    return ResultEnvelope(
        status=envelope.status,
        summary=envelope.summary,
        artifacts=verified_artifacts,
        details=envelope.details,
        stop_reason=envelope.stop_reason,
        error=envelope.error,
    )


# TODO: optional ruff check for .py artifacts (timeout 10s)
