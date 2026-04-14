from __future__ import annotations

import json
from pathlib import Path

from nanobot.heartbeat.drain import collect_pending


def test_collect_pending_normalizes_legacy_correlation_fields(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(
        json.dumps(
            {
                "version": 1,
                "notifications": [
                    {
                        "id": "n1",
                        "channel": "telegram",
                        "content": "hello",
                        "status": "pending",
                        "metadata": {
                            "source": "boredom",
                            "task_id": "TASK-9",
                            "correlation_id": "boredom_123",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    pending = collect_pending(path=queue_path)

    assert len(pending) == 1
    assert pending[0]["source"] == "boredom"
    assert pending[0]["task_id"] == "TASK-9"
    assert pending[0]["correlation_id"] == "boredom_123"
