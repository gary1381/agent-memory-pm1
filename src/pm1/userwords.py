"""Whose words is a record? Claude Code, Cowork and Codex capture some text written by other agents or by the harness
as role='user': context summaries, task notifications, another agent's message, Codex approval-review prompts (they carry
the agent's whole history). None of it is the user speaking. One place decides, for every consumer (rules, brief, miner,
conflicts, failsig). Stdlib only: the Claude Code hooks import it."""
from __future__ import annotations

import re

AGENT_SOURCE = 'mcp:memory_capture:'      # records an agent wrote through memory_capture (pm1.store.AGENT_SOURCE)
NOT_USER = ('<', 'Base directory for this skill', 'Another Claude session', '[SYSTEM NOTIFICATION',
            'This session is being continued', 'Caveat:', '[Request interrupted',
            'The following is the Codex agent history', 'The following is the Codex agent', '[Subagent')


def is_user_words(text: str | None, source: str | None = None) -> bool:
    """True when this role='user' record is plausibly something the user typed or said."""
    if (source or '').startswith(AGENT_SOURCE):
        return False
    t = (text or '').lstrip()
    if t.startswith('# ') and len(t) > 3000:
        return False              # a skill / command body the harness expanded into the turn (older captures kept it)
    return not t.startswith(NOT_USER)


PASTED = re.compile(r'<pasted_content\b[^>]*>.*?(?:</pasted_content\b[^>]*>|\Z)', re.S)
# the user's "quoted text::my reply" (also ：：); not C++ / Rust paths like std::vector
QUOTE_SEP = re.compile(r'(?<![A-Za-z0-9_])::|::(?![A-Za-z0-9_])|：：')


def own_words(text: str | None) -> str:
    """The part of a user turn the user wrote: without pasted blocks, and on a line "quoted::reply" only the reply
    (after the last separator). What they pasted or quoted is somebody else's text, not a rule or lesson of theirs."""
    t = PASTED.sub('\n', text or '')
    return '\n'.join(QUOTE_SEP.split(line)[-1] for line in t.split('\n')).strip()
