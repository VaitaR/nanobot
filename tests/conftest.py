"""Root conftest: make nanobot_workspace importable across all test suites."""
from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE_SRC = Path("/root/.nanobot/workspace/src")
if _WORKSPACE_SRC.exists() and str(_WORKSPACE_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_SRC))
