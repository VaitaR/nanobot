"""Post-subagent artifact verification.

Validates that claimed file/diff artifacts actually exist on disk,
are non-empty, and (for Python files) pass basic lint checks,
detecting the "illusion of execution" where a subagent reports files
it never wrote or wrote incorrectly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger

from nanobot.agent.result_envelope import Artifact, ResultEnvelope


def verify_artifact_exists(artifact: Artifact) -> Artifact:
    """Check if a file artifact exists, is non-empty, and (for .py) passes ruff.

    Non-file artifacts (url, metric) are returned with ``verified=False``
    but are *not* considered missing — they simply can't be checked this way.
    """
    if artifact.kind not in ("file", "diff"):
        return artifact  # URLs/metrics can't be checked this way

    path = Path(artifact.path)
    exists = path.exists()

    if not exists:
        return Artifact(
            path=artifact.path,
            description=artifact.description,
            kind=artifact.kind,
            verified=False,
        )

    # Check non-empty
    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    if size == 0:
        logger.warning("Artifact is empty (0 bytes): {}", artifact.path)
        return Artifact(
            path=artifact.path,
            description=f"{artifact.description} (WARNING: empty file)",
            kind=artifact.kind,
            verified=False,
        )

    # For Python files, run ruff check (timeout 10s)
    if path.suffix == ".py":
        try:
            proc = subprocess.run(
                ["ruff", "check", "--quiet", str(path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                lint_errors = proc.stdout.strip()[:200] if proc.stdout else "lint errors"
                logger.warning("Artifact has lint errors: {} — {}", artifact.path, lint_errors)
                return Artifact(
                    path=artifact.path,
                    description=f"{artifact.description} (WARNING: lint errors)",
                    kind=artifact.kind,
                    verified=False,
                )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # ruff not available or timed out — skip lint check
            pass

    return Artifact(
        path=artifact.path,
        description=artifact.description,
        kind=artifact.kind,
        verified=True,
    )


def verify_envelope(envelope: ResultEnvelope) -> ResultEnvelope:
    """Verify all file/diff artifacts in the envelope.

    Downgrades status from ``"ok"`` to ``"partial"`` if any claimed file
    is missing, empty, or has lint errors.  Returns a *new*
    ``ResultEnvelope`` (dataclass with slots=True, so we build fresh
    instances).
    """
    if not envelope.artifacts or envelope.status == "error":
        return envelope

    verified_artifacts = [verify_artifact_exists(a) for a in envelope.artifacts]
    failed = [
        a for a in verified_artifacts
        if not a.verified and a.kind in ("file", "diff")
    ]

    if failed and envelope.status == "ok":
        paths = ", ".join(a.path for a in failed)
        logger.warning("Subagent artifacts failed verification: {}", paths)
        return ResultEnvelope(
            status="partial",
            summary=(
                f"{envelope.summary} "
                f"(warning: {len(failed)} artifact(s) failed verification)"
            ),
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
