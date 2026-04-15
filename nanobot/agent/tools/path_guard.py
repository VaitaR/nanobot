"""Protected path enforcement for agent tools.

Centralizes the protected-paths policy from
``product/architecture/protected-paths.md`` as runtime code so that
shell, file, and future tools share a single guard.
"""

import fnmatch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Patterns from product/architecture/protected-paths.md
# These are glob patterns matched against workspace-relative paths.
PROTECTED_PATTERNS: list[str] = [
    # Critical configuration
    ".env",
    ".env.*",
    "pyproject.toml",
    "poetry.lock",
    "package-lock.json",
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
    # Infrastructure
    "terraform/*",
    ".github/workflows/*",
    "deploy/*",
    "k8s/*",
    "kubernetes/*",
    # Security
    "*.pem",
    "*.key",
    "*.crt",
    ".ssh/*",
    "secrets/*",
    "vault/*",
    ".gitignore",
    # Agent configuration
    "nanobot.toml",
    "agents/*",
    "prompts/*",
    # Database
    "migrations/*",
    "alembic/*",
    "*.db",
    "*.sqlite",
]


class PathGuard:
    """Checks paths against the protected-paths policy.

    Args:
        extra_patterns: Additional glob patterns to protect.
        workspace_root: Root directory for resolving relative paths.
    """

    def __init__(
        self,
        extra_patterns: list[str] | None = None,
        workspace_root: str = ".",
    ) -> None:
        self.patterns = list(PROTECTED_PATTERNS) + (extra_patterns or [])
        self.workspace_root = Path(workspace_root).resolve()

    def is_protected(self, path: str) -> str | None:
        """Check whether *path* matches a protected pattern.

        Returns the matching pattern string if protected, ``None`` otherwise.
        """
        try:
            rel = str(Path(path).resolve().relative_to(self.workspace_root))
        except ValueError:
            rel = path

        name = Path(rel).name
        for pattern in self.patterns:
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                return pattern
        return None

    def check_write(self, path: str) -> str | None:
        """Return an error message if writing to *path* should be blocked."""
        match = self.is_protected(path)
        if match:
            logger.warning("Blocked write to protected path %s (matched %s)", path, match)
            return f"Path is protected ({match}). Requires human approval."
        return None
