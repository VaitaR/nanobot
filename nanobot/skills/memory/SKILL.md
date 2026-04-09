---
name: memory
description: Two-layer memory system with grep-based recall.
always: true
---

# Memory

## Structure

- `memory/MEMORY.md` — Long-term facts (preferences, project context, relationships). Always loaded into your context.
- `memory/HISTORY.md` — Append-only event log. NOT loaded into context. Search it with grep-style tools or in-memory filters. Each entry starts with [YYYY-MM-DD HH:MM].

## Search Past Events

Choose the search method based on file size:

- Small `memory/HISTORY.md`: use `read_file`, then search in-memory
- Large or long-lived `memory/HISTORY.md`: use the `exec` tool for targeted search

Examples:
- **Linux/macOS:** `grep -i "keyword" memory/HISTORY.md`
- **Windows:** `findstr /i "keyword" memory\HISTORY.md`
- **Cross-platform Python:** `python -c "from pathlib import Path; text = Path('memory/HISTORY.md').read_text(encoding='utf-8'); print('\n'.join([l for l in text.splitlines() if 'keyword' in l.lower()][-20:]))"`

Prefer targeted command-line search for large history files.

## When to Update MEMORY.md

Write important facts immediately using `edit_file` or `write_file`:
- User preferences ("I prefer dark mode")
- Project context ("The API uses OAuth2")
- Relationships ("Alice is the project lead")

## Provenance Tracking

Every new MEMORY.md entry should include a source comment so stale facts can be traced to their origin. Append an HTML comment on the line after the section heading:

```
## Session Facts (2026-04-09)
<!-- source: session_rotation 2026-04-09; task 20260409T102219 -->
- User prefers concise error messages
```

### Source types
| Source | When to use |
|--------|-------------|
| `user` | Fact stated directly by the user in conversation |
| `heartbeat` | Fact discovered during heartbeat cycle |
| `task` | Fact produced while executing a specific task |
| `session_rotation` | Fact extracted from session summarization |
| `consolidation` | Fact added during memory consolidation/compaction |

### Comment format
```
<!-- source: <source> <YYYY-MM-DD>[; task <task_id>] -->
```

- `source`: one of the types above
- `date`: the date the fact was written (not the date the event occurred)
- `task` (optional): task ID that triggered the write

### HISTORY.md traceability

HISTORY.md entries already include timestamps. When a session ID or correlation ID is available, append it to the header:

```
[2026-04-09 10:22 | session: abc123; corr: xyz789] Session summary
```

This lets you trace history entries back to specific sessions or request chains.

## Auto-consolidation

Old conversations are automatically summarized and appended to HISTORY.md when the session grows large. Long-term facts are extracted to MEMORY.md. You don't need to manage this.
