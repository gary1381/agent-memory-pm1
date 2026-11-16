"""Secret redaction: on every write (Store.add_episode, document blocks) and on every MCP result.

Captured transcripts and tool outputs hold API keys, tokens and passwords; pm1 returns records verbatim to agents
whose hosts send them to the cloud, so secrets are cut before they are stored and again before anything leaves.
``scrub`` cleans a database written before redaction existed. Patterns are deliberately simple: a false positive
costs a few characters of a record, a miss leaks a key.
"""
from __future__ import annotations

import re
from typing import Any

MARK = '[redacted]'
_B = r'(?<![A-Za-z0-9])'       # 'disk-usage-…' must not look like an 'sk-' key
# key=value: only a standalone literal with a digit in it -- not code ('token = self.tokenizer(...)',
# 'password=os.environ.get(...)', '${API_KEY:-}') or a word ('token: Optional'); on the live DB the looser
# '8+ non-space characters' matched ~1,900 places, nearly all code
_V = r'[A-Za-z0-9_\-+/=~!@#%^&]'
# a value may end the sentence ('token=abc123xyz.'), sit in markdown backticks or run on into '|' / ':'
_VAL = rf'(?={_V}*\d)[A-Za-z0-9_\-+/=~!@#%^&]{_V}{{7,}}?(?=\.?(?:$|[\s"\',;}})\]>\\`|:]))'
# a value with dots inside ('TOKEN=abc1234.def5678'): every dot-separated part must hold a digit, so 'token = item2.first' stays code
_W = r'[A-Za-z0-9_\-+/=~!@#%^&]'
_VALD = rf'(?!\d+\.)(?=\S{{12,}})(?:{_W}*\d{_W}*\.)+{_W}*\d{_W}*(?=\.?(?:$|[\s"\',;}})\]>\\`|:]))'
_VAL_ANY = f'(?:{_VAL}|{_VALD})'
SECRET = re.compile(
    rf'{_B}(?:sk-[A-Za-z0-9_-]{{16,}}|gh[pousr]_[A-Za-z0-9]{{20,}}|github_pat_[A-Za-z0-9_]{{20,}}|AKIA[0-9A-Z]{{16}}|'
    rf'xox[abpr]-[A-Za-z0-9-]{{10,}}|AIza[0-9A-Za-z_-]{{30,}}|hf_[A-Za-z0-9]{{30,}}|'
    rf'(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{{16,}}|gsk_[A-Za-z0-9]{{30,}}|glpat-[A-Za-z0-9_-]{{16,}}|'
    rf'xai-[A-Za-z0-9]{{30,}}|npm_[A-Za-z0-9]{{30,}}|ya29\.[A-Za-z0-9_-]{{20,}}|'
    rf'hooks\.slack\.com/services/[A-Za-z0-9/]{{20,}}|SG\.[A-Za-z0-9_-]{{16,}}\.[A-Za-z0-9_-]{{16,}}|GOCSPX-[A-Za-z0-9_-]{{20,}}|'
    rf'\d{{8,10}}:[A-Za-z0-9_-]{{35}}|'
    rf'eyJ[A-Za-z0-9_-]{{20,}}\.[A-Za-z0-9_-]{{10,}}\.[A-Za-z0-9_-]{{10,}})|'
    r'(?P<bearer>(?i:bearer)\s+)[A-Za-z0-9._~+/=-]{16,}|'
    r'(?P<auth>(?i:authorization:?["\']?\s*(?:token|basic)\s+))[A-Za-z0-9+/=._-]{16,}|'
    r'(?P<cu>\s-u\s+[A-Za-z0-9_.@-]{1,40}:)(?=[^\s"\']*\d)[^\s"\']{6,}|'
    r'(?P<my>\bmysql\b[^\n]{0,80}?\s-p)(?=[^\s]*\d)[^\s-][^\s]{6,}|'
    r'(?P<url>[A-Za-z][A-Za-z0-9+.-]*://[^\s:/@]+:)(?=[^\s@/]*\d)[^\s@/$]{4,}(?=@)|'
    r'(?P<flag>--(?i:api[_-]?key|token|secret|password|passwd)\s+)' + _VAL_ANY + '|'
    r'(?P<kaggle>["\']key["\']\s*:\s*["\'])[0-9a-f]{32}(?=["\'])|'
    r'(?P<key>(?i:api[_-]?key|token|secret(?:[_-]?(?:access[_-]?)?key)?|password|passwd|access[_-]?key)'
    r'["\']?\s*[=:]\s*["\']?)' + _VAL_ANY + '|'
    # a labelled short code ('**首次设置码**：`AbCd123xYz`', 'setup code: …', '验证码: 482913'): a FreeLLMAPI setup code was
    # stored and later quoted to an outside labeller (2026-10-27) -- the key=value rule wants an English label and a digit
    r'(?P<label>(?:(?i:(?:first[- ]?(?:time[- ]?)?|initial[- ]?|one[- ]?time[- ]?)?(?:setup|access|invite|verification|pairing|'
    r'recovery|activation|admin|login)[ _-]?(?:code|key|pin|password|token)|otp|pin[ _-]?code)'
    r'|(?:首次|初始|一次性)?(?:设置码|验证码|邀请码|访问码|配对码|恢复码|激活码|密码|口令|密钥|令牌))'
    r'\**\s*[:：=]\s*\**\s*[`"\'“]?)[A-Za-z0-9][A-Za-z0-9_\-+/=]{5,63}(?![A-Za-z0-9_\-+/=])|'
    r'-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)')


def _sub(m: re.Match) -> str:
    return next((m.group(g) for g in ('bearer', 'auth', 'url', 'cu', 'my', 'flag', 'kaggle', 'key', 'label') if m.group(g)),
                '') + MARK


def redact(text: str) -> str:
    return SECRET.sub(_sub, text) if text else (text or '')


def redact_obj(obj: Any) -> Any:
    """Redact every string inside a JSON-like result (dicts, lists, strings)."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj


def scrub(store, *, dry_run: bool = False) -> dict:
    """Redact records stored before redaction existed: episodes (re-split, re-indexed, vectors re-queued), document
    block bodies, logged MCP results and commit messages. Idempotent."""
    con = store.con
    out = dict.fromkeys(('episodes', 'doc_blocks', 'mcp_results', 'code_commits', 'mcp_calls', 'mcp_feedback', 'rules',
                         'rule_conflicts', 'judge_log'), 0)
    hits = [(r[0], r[1]) for r in con.execute('SELECT id, text FROM episodes') if SECRET.search(r[1])]
    for eid, text in hits:
        new = redact(text)
        if new != text:
            out['episodes'] += 1
            if not dry_run:
                store.rewrite_episode(eid, new)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table, key, col in (('doc_blocks', 'evidence_id', 'body'), ('mcp_results', 'call_id', 'result'),
                            ('code_commits', 'rowid', 'message'), ('mcp_calls', 'id', 'args'),
                            ('mcp_feedback', 'id', 'note'), ('rules', 'id', 'text'), ('rules', 'id', 'quote'),
                            ('rules', 'id', 'why'), ('rules', 'id', 'action'), ('rules', 'id', 'note'), ('rule_conflicts', 'id', 'excerpt'), ('judge_log', 'id', 'text')):
        if table not in tables:
            continue
        rows = [(r[0], r[1]) for r in con.execute(f'SELECT {key}, {col} FROM {table}') if r[1] and SECRET.search(r[1])]
        for k, v in rows:
            new = redact(v)
            if new != v:
                out[table] += 1
                if not dry_run:
                    with store.lock:
                        con.execute(f'UPDATE {table} SET {col}=? WHERE {key}=?', (new, k))
    return out


def main() -> None:
    import argparse
    import json
    import os

    from .paths import default_db
    from .store import Store
    ap = argparse.ArgumentParser(description='Redact secrets already stored in a pm1 database (back it up first).')
    ap.add_argument('--db', default=default_db())
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    store = Store(a.db)
    store.con.execute('PRAGMA busy_timeout=30000')
    print(json.dumps(scrub(store, dry_run=a.dry_run)))


if __name__ == '__main__':
    main()
