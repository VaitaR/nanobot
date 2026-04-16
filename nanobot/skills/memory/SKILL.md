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
- **Cross-platform Python:** `python -c "from pathlib import Path; text = Path('memory/HISTORY.md').read_text(encoding='utf-8'); print('\n'.join([l for l in text.splitlines() if 'keyword' in l.lower()][-20:]))"`

Prefer targeted command-line search for large history files.

## When to Update MEMORY.md

Write important facts immediately using `edit_file` or `write_file`.

**Only write facts that satisfy ALL four criteria:**

1. **Non-discoverable** — cannot be found by running a command or reading a file
2. **Non-inferable** — cannot be inferred from existing context (AGENTS.md, USER.md, HEARTBEAT.md, etc.)
3. **Stable** — won't change or expire within 30 days
4. **Actionable** — knowing this changes agent behavior

### What TO write
- User preferences ("Andrew prefers Russian, concise style")
- Stable architectural decisions not recorded elsewhere
- Gotchas that prevent recurring mistakes ("`.restart-pending` ≠ `.pending-reload`")
- Cross-session state ("Boredom engine is DISABLED — don't re-enable")

### What NOT to write
- ❌ Events or statuses ("Codex token expired 14.04") → HISTORY.md
- ❌ Discoverable facts ("Heartbeat runs every 30 min") → already in HEARTBEAT.md
- ❌ Task completions ("Task 20260415T102219 done") → tasks.py
- ❌ Code details (line numbers, function names, commit hashes) → source code
- ❌ Debugging narratives ("LLM hallucinated spawn status") → HISTORY.md
- ❌ Lists and counts ("3 open tasks") → query on demand

## Provenance Tracking

Every MEMORY.md section should include a source comment so stale facts can be traced:

```
## User Preferences
<!-- source: user 2026-04-09 -->
- Prefers concise error messages
```

### Source types
| Source | When to use |
|--------|-------------|
| `user` | Fact stated directly by the user in conversation |
| `heartbeat` | Fact discovered during heartbeat cycle |
| `task` | Fact produced while executing a specific task |
| `compaction` | Fact added during memory compaction |

### Comment format
```
<!-- source: <source> <YYYY-MM-DD>[; task <task_id>] -->
```

### HISTORY.md traceability

When a session ID or correlation ID is available, append it to HISTORY.md entries:

```
[2026-04-09 10:22 | session: abc123; corr: xyz789] Session summary
```

## Memory Compaction (Nightly)

A cron job triggers daily MEMORY.md review. Subagent classifies each entry:

| Action | Criteria | Destination |
|--------|----------|-------------|
| **Keep** | Passes all 4 criteria above | Stays in MEMORY.md |
| **Promote** | Stable behavioral rule or guard rail | AGENT_RULES.md / TOOLS.md / AGENTS.md |
| **Archive** | Stale, expired, discoverable, or one-time event | HISTORY.md with `[compacted from MEMORY]` |

**Rules for compaction subagent:**
- Never delete without tracing — always archive to HISTORY.md first
- Never edit AGENT_RULES.md or TOOLS.md directly — propose diff in task output for main agent to apply
- Max MEMORY.md budget: 50 lines
- After compaction, report: X kept, Y promoted, Z archived

## Auto-consolidation

Old conversations are automatically summarized and appended to HISTORY.md when the session grows large. Long-term facts are extracted to MEMORY.md following the criteria above. You don't need to manage this manually.
