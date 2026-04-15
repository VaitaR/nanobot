"""Exec tier cap gate for shell command execution.

Classifies shell commands into coarse risk tiers (0-3) and checks whether the
current execution context is allowed to run that tier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TierCapResult:
    """Decision returned by :func:`check_exec_tier`."""

    allowed: bool
    command_tier: int
    max_tier: int
    reason: str = ""


def _split_compound(cmd: str) -> list[str]:
    parts = re.split(r"\s*(?:&&|\|\||;|(?<!\|)\|(?!\|))\s*", cmd)
    return [p.strip() for p in parts if p.strip()]


def _classify_single(cmd: str) -> int:
    # Tier 3: destructive / privileged
    tier3 = [
        r"\brm\s+(-[rfRF]+\b|--force\b|--recursive\b)",
        r"\bgit\s+push\s+.*(?:--force\b|-f\b)",
        r"\bgit\s+reset\s+--hard\b",
        r"\b(sudo|doas|run0)\b",
        r"\b(shutdown|reboot|poweroff|halt)\b",
        r"\b(mkfs|fdisk|parted|diskpart)\b",
        r"\bdd\s+",
        r"\b(chmod\s+777|launchctl|systemctl|pkill|killall|kill\s+-9|crontab)\b",
    ]
    if any(re.search(p, cmd, re.IGNORECASE) for p in tier3):
        return 3

    # Tier 2: package managers, container builds, dependency mutation
    tier2 = [
        r"\b(npm|yarn|pnpm|bun)\b",
        r"\b(pip|pip3|pipenv|poetry|uv)\b",
        r"\b(docker\s+(build|run|compose|pull|image)|podman)\b",
        r"\b(apt(-get)?\s+(install|update|upgrade)|brew\b)\b",
        r"\b(make|cmake|ninja|cargo|go\s+(build|run|test|install|get|mod))\b",
    ]
    if any(re.search(p, cmd, re.IGNORECASE) for p in tier2):
        return 2

    # Tier 0: read-only inspection
    tier0 = [
        r"\bgit\s+(status|log|diff|show|branch|remote\s+-v|tag)\b",
        r"\b(ls|cat|head|tail|wc|find|which|type|file|stat|du|df|grep|pwd)\b",
        r"\b(date|env|printenv|uname|whoami|id|hostname|echo)\b",
    ]
    if any(re.search(p, cmd, re.IGNORECASE) for p in tier0):
        return 0

    # Tier 1 default for unknown/write-but-not-obviously-dangerous commands
    return 1


def classify_command(cmd: str) -> int:
    """Classify a shell command into risk tier 0-3."""
    if not cmd or not cmd.strip():
        return 0

    # Whole-command tier-3 patterns that can span separators.
    if any(
        re.search(pattern, cmd, re.IGNORECASE)
        for pattern in (
            r"\b(curl|wget)\b.*\|\s*(sh|bash)\b",
            r":\(\)\s*\{.*\};\s*:",
            r">\s*/dev/",
            r"\bDROP\s+(TABLE|DATABASE)\b",
            r"\bTRUNCATE\b",
        )
    ):
        return 3

    return max(_classify_single(part) for part in _split_compound(cmd))


def check_exec_tier(
    cmd: str,
    max_tier: int = 3,
    *,
    context: str = "interactive",
) -> TierCapResult:
    """Check whether *cmd* is allowed under the given tier cap."""
    if max_tier < 0:
        max_tier = 0
    if max_tier > 3:
        max_tier = 3

    tier = classify_command(cmd)
    if tier > max_tier:
        return TierCapResult(
            allowed=False,
            command_tier=tier,
            max_tier=max_tier,
            reason=f"Command tier {tier} exceeds {context} max tier {max_tier}",
        )

    return TierCapResult(allowed=True, command_tier=tier, max_tier=max_tier)
