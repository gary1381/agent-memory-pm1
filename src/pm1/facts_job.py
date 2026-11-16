"""Background fact extraction (layer F, pm1.facts): once a session has been quiet for a while, one LLM call per chunk of
it writes short self-contained facts, each with the turns it came from; the facts are embedded for recall.

Off unless an extractor is configured. Real sessions must go to a LOCAL model (PM1_FACTS_URL = a llama-server on this
machine, e.g. Qwen3.5-4B); a hosted model only for synthetic or public benchmark data.

  python -m pm1.facts_job --db PATH --url http://127.0.0.1:18330 --model qwen35-4b [--idle-min 30] [--once]
Env: PM1_FACTS_URL / PM1_FACTS_MODEL / PM1_FACTS_KEY (sent as a bearer token; empty for a local server).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

PROMPT = """Below is part of a conversation session{who}, held on {date}. Turns are numbered.

Write down every fact a reader might later need about the people and things in it: events (what happened, when),
possessions, plans, decisions, preferences, opinions, relationships, places, activities, numbers, names, versions,
items shown in shared images.
Rules:
- one fact per line, self-contained: full names (never "he"/"she"/"I"/"we"), absolute dates when the text implies them
  ("yesterday" said on {date} -> the actual date);
- add the general category when it helps finding the fact later (e.g. "England (a European country)",
  "a Gamecube (a video game console)", "Harry Potter (fantasy books)");
- "turns": the numbers of the turns the fact comes from.
Respond in json: {{"facts": [{{"fact": "...", "turns": [3, 4]}}, ...]}}

SESSION
{turns}"""
MAX_CHARS = 12000                                    # one chunk of a session per call (large hosted model)
LOCAL_CHARS = 4000                                   # a 4B on an 8K slot: prompt + facts must fit
SCHEMA = {'type': 'object', 'properties': {'facts': {'type': 'array', 'items': {
    'type': 'object', 'properties': {'fact': {'type': 'string'}, 'turns': {'type': 'array', 'items': {'type': 'integer'}}},
    'required': ['fact', 'turns']}}}, 'required': ['facts']}


def chunks(turns: list[dict], max_chars: int = MAX_CHARS) -> list[list[dict]]:
    out, cur, n = [], [], 0
    for t in turns:
        if cur and n + len(t['text']) > max_chars:
            out.append(cur)
            cur, n = [], 0
        cur.append(t)
        n += min(len(t['text']), max_chars)
    return out + ([cur] if cur else [])


def parse_json(txt: str) -> dict:
    """The model's JSON; when the output was cut off (max_tokens), the facts that were complete before the cut."""
    txt = re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', (txt or '{}').strip())
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r'"facts"\s*:\s*\[', txt)
        if not m:
            raise
        dec, i, facts = json.JSONDecoder(), m.end(), []
        while True:
            while i < len(txt) and txt[i] in ' \t\r\n,':
                i += 1
            try:
                obj, i = dec.raw_decode(txt, i)
            except json.JSONDecodeError:
                return {'facts': facts, 'truncated': True}
            facts.append(obj)


def extract(call, turns: list[dict], date: str, who: str = '', max_chars: int = MAX_CHARS) -> list[dict]:
    """call(prompt) -> parsed JSON dict. turns: [{'episode_id', 'role', 'text'}]. Returns [{'text', 'episodes'}]."""
    facts = []
    for part in chunks(turns, max_chars):
        lines = '\n'.join(f"[{k}] {t['role']}: {t['text'][:max_chars]}" for k, t in enumerate(part))
        res = call(PROMPT.format(who=who, date=date, turns=lines)) or {}
        for f in res.get('facts') or []:
            if not isinstance(f, dict) or not str(f.get('fact') or '').strip():
                continue
            eps = [part[k]['episode_id'] for k in f.get('turns') or [] if isinstance(k, int) and 0 <= k < len(part)]
            if eps:
                facts.append({'text': str(f['fact']).strip(), 'episodes': eps})
    return facts


def http_call(url: str, model: str, key: str = ''):
    import httpx

    def call(prompt: str) -> dict:
        body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0.0,
                'max_tokens': 3000, 'chat_template_kwargs': {'enable_thinking': False},
                'response_format': {'type': 'json_schema', 'json_schema': {'name': 'facts', 'schema': SCHEMA}}}
        h = {'Authorization': f'Bearer {key}'} if key else {}
        r = httpx.post(url.rstrip('/') + '/v1/chat/completions', json=body, headers=h, timeout=900)
        r.raise_for_status()
        return parse_json(r.json()['choices'][0]['message']['content'])
    return call


def due_sessions(con, idle_min: int) -> list[tuple]:
    """(bank, scope, session, last_seq) for sessions quiet for idle_min minutes and new or grown since extraction."""
    cut = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(time.time() - idle_min * 60))
    return con.execute(
        'SELECT e.bank, e.scope, e.session, max(e.seq) FROM episodes e LEFT JOIN fact_sessions f ON f.bank = e.bank '
        'AND f.scope = e.scope AND f.session = e.session GROUP BY e.bank, e.scope, e.session '
        'HAVING max(e.occurred_at) < ? AND (max(f.last_seq) IS NULL OR max(e.seq) > max(f.last_seq))', (cut,)).fetchall()


def run_once(store, fx, call, model: str, idle_min: int = 30, limit: int = 50, max_chars: int = LOCAL_CHARS) -> int:
    n = 0
    for bank, scope, session, last in due_sessions(store.con, idle_min)[:limit]:
        rows = store.con.execute('SELECT id, role, text, occurred_at FROM episodes WHERE bank=? AND scope=? AND '
                                 'session=? ORDER BY seq', (bank, scope, session)).fetchall()
        turns = [{'episode_id': r[0], 'role': r[1], 'text': r[2]} for r in rows]
        try:
            facts = extract(call, turns, str(rows[0][3])[:10] if rows else '', max_chars=max_chars)
            fx.replace_session(bank, scope, session, facts, model=model, source='facts_job', last_seq=last)
        except Exception as exc:                    # keep going; the session is retried when it grows
            fx.replace_session(bank, scope, session, [], model=model, source='facts_job', last_seq=last,
                               error=f'{type(exc).__name__}: {exc}'[:300])
        n += 1
    while fx.embed_pending():
        pass
    return n


def main() -> None:
    from .facts import FactIndex
    from .store import Store
    from .worker import build_embedders
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--url', default=os.environ.get('PM1_FACTS_URL', ''))
    ap.add_argument('--model', default=os.environ.get('PM1_FACTS_MODEL', 'qwen35-4b'))
    ap.add_argument('--idle-min', type=int, default=30)
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    if not a.url:
        raise SystemExit('no extractor configured (--url / PM1_FACTS_URL): nothing to do')
    dense, _ = build_embedders(os.environ.get('PM1_DENSE', 'split'), 'none')
    store = Store(a.db)
    fx = FactIndex(store, dense)
    call = http_call(a.url, a.model, os.environ.get('PM1_FACTS_KEY', ''))
    while True:
        n = run_once(store, fx, call, a.model, a.idle_min)
        if a.once:
            print(f'{n} sessions extracted')
            return
        time.sleep(300 if n == 0 else 5)


if __name__ == '__main__':
    main()
