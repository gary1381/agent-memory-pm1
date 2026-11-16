"""Passive capture of coding-agent sessions (Codex rollouts, Claude Code transcripts) into pm1.

Nothing is asked of the agent: the host's own session files are tailed incrementally (byte offsets), so
capture costs no model calls and no agent tokens. What is stored:

  episodes  user / assistant text verbatim; each tool call + its output as one 'tool' record (command and
            output head/tail; the full output stays in the host file and `source` points at it:
            '<file>#<line ordinal>'), compaction events. scope = 'proj:<cwd>', session = '<host>:<id>'.
  usage     one row per model response: input / cached input / cache write / output / reasoning tokens
            (the ledger behind the memory ROI meter).
  tool_io   one row per tool call: name, input and output size (how much of the context is tool output).

  pm1-capture scan            ingest everything new
  pm1-capture watch [-i 60]   keep ingesting
  pm1-capture report [--days 7]  token ledger by host / day / project
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

from .paths import default_db
from .store import Store

CAP_SCHEMA = """
CREATE TABLE IF NOT EXISTS cap_files (path TEXT PRIMARY KEY, host TEXT NOT NULL, offset INTEGER NOT NULL DEFAULT 0,
  lines INTEGER NOT NULL DEFAULT 0, session TEXT, cwd TEXT, model TEXT, updated_at REAL);
CREATE TABLE IF NOT EXISTS cap_pending (call_id TEXT PRIMARY KEY, path TEXT, name TEXT, input TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS usage (id INTEGER PRIMARY KEY, host TEXT, session TEXT, cwd TEXT, model TEXT, ts TEXT,
  input INTEGER, cached INTEGER, cache_write INTEGER, output INTEGER, reasoning INTEGER, source TEXT UNIQUE);
CREATE INDEX IF NOT EXISTS ix_usage_ts ON usage(ts);
CREATE TABLE IF NOT EXISTS tool_io (id INTEGER PRIMARY KEY, host TEXT, session TEXT, cwd TEXT, ts TEXT, name TEXT,
  in_chars INTEGER, out_chars INTEGER, source TEXT UNIQUE);
"""
CAP_SCHEMA += """
CREATE TABLE IF NOT EXISTS mem_events (id INTEGER PRIMARY KEY, host TEXT, session TEXT, ts TEXT, system TEXT,
  kind TEXT, detail TEXT, out_chars INTEGER, source TEXT UNIQUE);
CREATE INDEX IF NOT EXISTS ix_mem_sess ON mem_events(host, session);
CREATE TABLE IF NOT EXISTS cloud_dropped (source TEXT PRIMARY KEY);
"""
# External memory systems in use, so token comparisons can separate vanilla agents from agents with a memory
# add-on. kind: 'query' = the agent consulted it; 'build' = maintaining it (costs tokens, not recall);
# 'available' = it was in the agent's instructions / tool list for that session (consulted or not).
import re as _re
MEM_SYSTEMS = {
    'graphify': {
        'query': _re.compile(r'graphify[-_.](?:mcp[-_.])?(?:query_graph|get_node|get_neighbors|get_community|god_nodes|'
                             r'graph_stats|shortest_path|get_pr_impact|list_prs|triage_prs)|mcp__graphify__|'
                             r'\bgraphify\s+(?:query|path|explain)\b|graphify-out/(?:GRAPH_REPORT\.md|graph\.json)|'
                             r'GRAPH_REPORT\.md', _re.I),
        'build': _re.compile(r'\bgraphify\s+(?:install|uninstall|update|build|build-all|watch|hook|label|clone|'
                             r'merge-graphs|diagnose)\b|\.graphify[-_]build|graphify-rebuild', _re.I),
        'available': _re.compile(r'graphify', _re.I),
    },
    'pm1': {'query': _re.compile(r'\b(?:memory_recall|search_notes|save_note|search_docs|search_sessions|code_history|open_record|mcp__pm1|'
                                 r'pm1[-_.]mcp)\b'), 'build': None,
            'available': _re.compile(r'memory_recall|search_notes|search_docs|search_sessions|code_history')},
    'hindsight': {'query': _re.compile(r'hindsight[-_.](?:recall|reflect)|/memories/recall', _re.I), 'build': None,
                  'available': _re.compile(r'hindsight', _re.I)},
    # OpenViking (Codex plugin 9/14-9/24, ov CLI, MCP). Work ON the OpenViking repo is development, not memory use,
    # so only its CLI / MCP / viking:// reads count as queries.
    'openviking': {'query': _re.compile(r'\bov(?:cli)?\s+(?:find|search|grep|ls|read|abstract|overview|recall)\b|'
                                        r'openviking[-_.](?:search|find|recall|read)|mcp__openviking__|viking://', _re.I),
                   'build': _re.compile(r'\bov(?:cli)?\s+(?:add|add-resource|add-memory|import|index|sync|rm)\b|'
                                        r'openviking[-_ ](?:server|serve|init)\b', _re.I),
                   'available': _re.compile(r'openviking|viking://', _re.I)},
}
# Calls to pm1 itself (and the older precision-memory server): counted in tool_io and the memory tags, but their
# output is not stored as a record -- it is a copy of records pm1 already holds, and stored it came back in later
# searches as if it were new evidence (25 such records by 2026-10-18)
SELF_TOOL = _re.compile(r'(?:^|__|-)(?:pm1|precision[-_]memory)(?:__|-|\.)', _re.I)
BANK = os.environ.get('PM1_BANK', 'personal')
HEAD, TAIL, CMD = 3000, 1200, 1500
SKIP_USER = ('<environment_context', '<user_instructions', '<permissions', '<model_switch', '# AGENTS.md',
             '<system-reminder', '<local-command', '<command-name', '<collaboration_mode')


def dt_iso(ms: int) -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ms / 1000))


def _clip(s: str, head: int = HEAD, tail: int = TAIL) -> str:
    if len(s) <= head + tail + 40:
        return s
    return f'{s[:head]}\n… [{len(s) - head - tail} chars omitted; full text at source] …\n{s[-tail:]}'


def _texts(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for c in content or []:
        if isinstance(c, dict):
            t = c.get('text') or c.get('content')
            if isinstance(t, list):
                t = _texts(t)
            if isinstance(t, str):
                out.append(t)
    return '\n'.join(out)


class Capture:
    def __init__(self, store: Store):
        self.store, self.con = store, store.con
        self.con.executescript(CAP_SCHEMA)

    # ---------- generic plumbing ----------
    tag_only = False          # retag(): re-read host files only to fill mem_events

    def _ep(self, f: dict, role: str, text: str, ts: str | None, src: str) -> None:
        if self.tag_only:
            return
        text = (text or '').strip()
        if not text or not f.get('session'):
            return
        self.store.add_episode(bank=BANK, scope=f"proj:{f.get('cwd') or 'unknown'}", text=text, role=role,
                               session=f"{f['host']}:{f['session']}", occurred_at=ts, source=src)

    def _mem(self, f: dict, ts, system, kind, detail, out_chars, src) -> None:
        self.con.execute('INSERT OR IGNORE INTO mem_events(host,session,ts,system,kind,detail,out_chars,source) '
                         'VALUES (?,?,?,?,?,?,?,?)', (f['host'], f.get('session'), ts, system, kind, detail[:200],
                                                      out_chars, src))

    def _tag_tool(self, f, name, inp, out, ts, src) -> None:
        probe = f'{name} {inp or ""}'
        for sysname, pat in MEM_SYSTEMS.items():
            for kind in ('query', 'build'):
                if pat[kind] is not None and pat[kind].search(probe):
                    self._mem(f, ts, sysname, kind, probe, len(out or ''), src)
                    break

    def _tag_context(self, f, text, ts, src) -> None:
        """Instructions / tool lists the host put in front of the model: the memory add-on was available."""
        for sysname, pat in MEM_SYSTEMS.items():
            m = pat['available'].search(text or '')
            if m:
                self._mem(f, ts, sysname, 'available', (text or '')[max(0, m.start() - 60):m.end() + 60], 0, src)

    def _tool(self, f: dict, call_id: str, name: str, inp: str, out: str, ts: str, src: str, ok=True) -> None:
        self._tag_tool(f, name, inp, out, ts, src)
        if 'mcp' in (name or '').lower() and 'tool' in (name or '').lower():
            self._tag_context(f, out, ts, src)
        if self.tag_only:
            return
        self.con.execute('INSERT OR IGNORE INTO tool_io(host,session,cwd,ts,name,in_chars,out_chars,source) '
                         'VALUES (?,?,?,?,?,?,?,?)', (f['host'], f['session'], f.get('cwd'), ts, name, len(inp or ''),
                                                      len(out or ''), src))
        if SELF_TOOL.search(name or ''):
            return
        self._ep(f, 'tool', f"[{name}{'' if ok else ' FAILED'}] {_clip(inp or '', CMD, 300)}\n→ {_clip(out or '')}",
                 ts, src)

    def _usage(self, f: dict, ts: str, u: dict, src: str) -> None:
        if self.tag_only:
            return
        self.con.execute('INSERT OR IGNORE INTO usage(host,session,cwd,model,ts,input,cached,cache_write,output,'
                         'reasoning,source) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                         (f['host'], f['session'], f.get('cwd'), f.get('model'), ts, *u, src))

    def _pend(self, call_id, path, name, inp, ts):
        if self.tag_only:
            self._tagpend[call_id] = (name, inp)
            return
        self.con.execute('INSERT OR REPLACE INTO cap_pending VALUES (?,?,?,?,?)', (call_id, path, name, inp, ts))

    def _take(self, call_id):
        if self.tag_only:
            return self._tagpend.pop(call_id, ('tool', ''))
        r = self.con.execute('SELECT name, input FROM cap_pending WHERE call_id=?', (call_id,)).fetchone()
        if r:
            self.con.execute('DELETE FROM cap_pending WHERE call_id=?', (call_id,))
            return r['name'], r['input']
        return 'tool', ''

    def ingest_file(self, path: str, host: str) -> int:
        row = self.con.execute('SELECT * FROM cap_files WHERE path=?', (path,)).fetchone()
        f = dict(row) if row else {'path': path, 'host': host, 'offset': 0, 'lines': 0, 'session': None, 'cwd': None,
                                   'model': None}
        size = os.path.getsize(path)
        if size < f['offset']:          # rewritten file: start over (dedupe by source keeps tables clean)
            f['offset'], f['lines'] = 0, 0
        if size == f['offset']:
            return 0
        n = 0
        with open(path, 'rb') as fh:
            fh.seek(f['offset'])
            for raw in fh:
                if not raw.endswith(b'\n'):
                    break                # partial line still being written
                f['offset'] += len(raw)
                f['lines'] += 1
                try:
                    d = json.loads(raw)
                except ValueError:
                    continue
                src = f"{path}#{f['lines']}"
                (self._codex if host == 'codex' else self._claude)(f, d, src)
                n += 1
        self.con.execute('INSERT OR REPLACE INTO cap_files(path,host,offset,lines,session,cwd,model,updated_at) '
                         'VALUES (?,?,?,?,?,?,?,?)', (path, host, f['offset'], f['lines'], f['session'], f['cwd'],
                                                      f['model'], time.time()))
        return n

    # ---------- Codex rollout ----------
    def _codex(self, f: dict, d: dict, src: str) -> None:
        t, p, ts = d.get('type'), d.get('payload') or {}, d.get('timestamp')
        if t == 'session_meta':
            f['session'], f['cwd'] = p.get('id') or p.get('session_id'), p.get('cwd')
        elif t == 'turn_context':
            f['cwd'] = p.get('cwd') or f['cwd']
            f['model'] = p.get('model') or f['model']
        elif t == 'world_state':
            f['model'] = ((p.get('state') or {}).get('model')) or f['model']
            self._tag_context(f, json.dumps(p, ensure_ascii=False), ts, src)
        elif t == 'token_usage_record':
            u = p.get('usage') or {}
            self._usage(f, ts, (u.get('input_tokens', 0), u.get('cached_input_tokens', 0),
                                u.get('cache_write_input_tokens', 0), u.get('output_tokens', 0),
                                u.get('reasoning_output_tokens', 0)), src)
        elif t == 'compacted':
            self._ep(f, 'event', '[context compacted] ' + _clip(str(p.get('message') or ''), 1500, 300), ts, src)
        elif t == 'response_item':
            k = p.get('type')
            if k == 'message' and p.get('role') == 'developer':
                self._tag_context(f, _texts(p.get('content')), ts, src)
            elif k == 'message' and p.get('role') in ('user', 'assistant'):
                text = _texts(p.get('content'))
                if p['role'] == 'user' and text.lstrip().startswith(SKIP_USER):
                    return
                self._ep(f, p['role'], text, ts, src)
            elif k in ('custom_tool_call', 'function_call'):
                inp = p.get('input') if k == 'custom_tool_call' else p.get('arguments')
                self._pend(p.get('call_id'), f['path'], p.get('name') or 'tool',
                           inp if isinstance(inp, str) else json.dumps(inp, ensure_ascii=False), ts)
            elif k in ('custom_tool_call_output', 'function_call_output'):
                out = p.get('output')
                out = _texts(out) if isinstance(out, list) else (out if isinstance(out, str) else json.dumps(out))
                name, inp = self._take(p.get('call_id'))
                self._tool(f, p.get('call_id'), name, inp, out, ts, src)

    # ---------- Claude Code transcript ----------
    def _claude(self, f: dict, d: dict, src: str) -> None:
        t, ts, m = d.get('type'), d.get('timestamp'), d.get('message') or {}
        if d.get('sessionId'):
            f['session'] = d['sessionId']
        if d.get('cwd'):
            f['cwd'] = d['cwd']
        if t in ('attachment', 'system'):
            self._tag_context(f, json.dumps(d.get('attachment') or d.get('content') or '', ensure_ascii=False), ts, src)
        if t == 'assistant':
            f['model'] = m.get('model') or f['model']
            u = m.get('usage') or {}
            if u:
                self._usage(f, ts, (u.get('input_tokens', 0) + u.get('cache_read_input_tokens', 0)
                                    + u.get('cache_creation_input_tokens', 0), u.get('cache_read_input_tokens', 0),
                                    u.get('cache_creation_input_tokens', 0), u.get('output_tokens', 0), 0), src)
            for c in m.get('content') or []:
                if c.get('type') == 'text':
                    self._ep(f, 'assistant', c.get('text'), ts, src)
                elif c.get('type') == 'tool_use':
                    self._pend(c.get('id'), f['path'], c.get('name'), json.dumps(c.get('input'), ensure_ascii=False), ts)
        elif t == 'user':
            content = m.get('content')
            if d.get('isMeta'):
                # harness-expanded text (a skill or command body, caveats): not the user's words, and boilerplate
                content = [c for c in content if c.get('type') == 'tool_result'] if isinstance(content, list) else []
            if isinstance(content, str):
                if not content.lstrip().startswith(SKIP_USER):
                    self._ep(f, 'user', content, ts, src)
                return
            for c in content or []:
                if c.get('type') == 'tool_result':
                    name, inp = self._take(c.get('tool_use_id'))
                    self._tool(f, c.get('tool_use_id'), name, inp, _texts(c.get('content')), ts, src,
                               ok=not c.get('is_error'))
                elif c.get('type') == 'text' and not (c.get('text') or '').lstrip().startswith(SKIP_USER):
                    self._ep(f, 'user', c.get('text'), ts, src)

    # ---------- cloud sessions (Cowork) exported by the session itself ----------
    def import_conversation(self, events, host: str, session: str, label: str = '') -> dict:
        """Normalized events from a cloud session's own export (see the cowork-conversation-export skill).

        Each event: {"k": stable key, "ts", "kind": user|assistant|turn|tool|usage, ...}. 'turn' events are
        older turns recovered from the server-side history (text only, no tool outputs). Keys are stable across
        re-exports, so importing the same or a larger export again only adds what is new."""
        self.con.execute('CREATE INDEX IF NOT EXISTS ix_episodes_source ON episodes(source)')
        f = {'host': host, 'session': session, 'cwd': f'{host}:{label}' if label else host, 'model': None}
        added = {'episodes': 0, 'tools': 0, 'usage': 0}
        for ev in events:
            src, kind, ts = f"{host}:{session}#{ev['k']}", ev.get('kind'), ev.get('ts')
            if kind == 'usage':
                f['model'] = ev.get('model') or f['model']
                u = [int(x or 0) for x in ev['u'][:4]]
                if self.con.execute('SELECT 1 FROM usage WHERE host=? AND session=? AND ts=? AND input=? AND output=?',
                                    (host, session, ts, u[0], u[3])).fetchone():
                    continue      # same response already imported by import-usage (keyed by transcript line)
                before = self.con.total_changes
                self._usage(f, ts, (*u, 0), src)
                added['usage'] += self.con.total_changes - before
                continue
            if self.con.execute('SELECT 1 FROM episodes WHERE source=? UNION ALL SELECT 1 FROM cloud_dropped '
                                'WHERE source=? LIMIT 1', (src, src)).fetchone():
                continue
            if kind == 'tool':
                self._tool(f, ev['k'], ev.get('name') or 'tool', ev.get('input') or '', ev.get('output') or '', ts,
                           src, ok=ev.get('ok', True))
                added['tools'] += 1
            elif kind in ('user', 'assistant', 'turn', 'summary', 'event'):
                role = ev.get('role') or ('event' if kind in ('summary', 'event') else kind)
                text = (ev.get('text') or '').strip()
                if not text or (role == 'user' and text.startswith(SKIP_USER)):
                    continue
                self._ep(f, role, text, ts, src)
                added['episodes'] += 1
        self.con.commit()
        tot = self.con.execute("SELECT count(*) FROM episodes WHERE session=?", (f'{host}:{session}',)).fetchone()[0]
        return {**added, 'session_episodes': tot}

    def dedupe_cloud(self, host: str = 'cowork', dry: bool = False) -> dict:
        """Remove duplicates that repeated exports of the same cloud session can leave:
        usage rows for the same response under two keys (transcript line vs message id); the same message text
        imported twice; and server-side backfill turns ('#turn:') whose text is already covered by the local,
        fully detailed records of that session. Local records are always the ones kept."""
        import re
        out = {'usage': 0, 'episodes_same_text': 0, 'turns_covered': 0, 'sessions': 0}
        dup_usage = [r[0] for r in self.con.execute(
            'SELECT u.id FROM usage u WHERE u.host=? AND EXISTS (SELECT 1 FROM usage v WHERE v.host=u.host AND '
            'v.session=u.session AND v.ts=u.ts AND v.input=u.input AND v.cached=u.cached AND v.output=u.output '
            'AND v.id<u.id)', (host,))]
        out['usage'] = len(dup_usage)
        if dup_usage and not dry:
            self.con.executemany('DELETE FROM usage WHERE id=?', [(i,) for i in dup_usage])
            self.con.commit()

        def norm(t):
            t = re.sub(r'\n?\[tools: [^\]]*\]\s*$', '', t or '')
            return re.sub(r'\s+', ' ', t).strip()

        drop = []
        sessions = [r[0] for r in self.con.execute('SELECT DISTINCT session FROM episodes WHERE session LIKE ?',
                                                   (f'{host}:%',))]
        out['sessions'] = len(sessions)
        for sess in sessions:
            eps = [dict(r) for r in self.con.execute(
                "SELECT id, scope, role, text, source, occurred_at FROM episodes WHERE bank=? AND session=? AND "
                "role IN ('user','assistant') ORDER BY id", (BANK, sess))]
            for e in eps:
                e['n'], e['turn'] = norm(e['text']), '#turn:' in (e['source'] or '')
            # a repeated message ("继续" said twice) is not a duplicate: local records only collide when the time
            # matches too; a backfill turn collides with a local record of the same text (>= 20 chars)
            kept_local, local_text = {}, set()
            for e in eps:
                if e['turn']:
                    continue
                key = (e['role'], e['n'], e['occurred_at'])
                if key in kept_local:
                    drop.append(e); out['episodes_same_text'] += 1
                else:
                    kept_local[key] = e
                    local_text.add((e['role'], e['n']))
            for e in eps:
                if e['turn'] and len(e['n']) >= 20 and (e['role'], e['n']) in local_text:
                    drop.append(e); out['episodes_same_text'] += 1
            gone = {d['id'] for d in drop}
            local = {}
            for e in eps:
                if not e['turn'] and e['id'] not in gone and len(e['n']) >= 20:
                    local.setdefault(e['role'], []).append(e['n'])
            for e in eps:
                if not e['turn'] or e['id'] in gone or not e['n']:
                    continue
                rest = e['n']
                for lt in local.get(e['role'], []):
                    if lt in rest:
                        rest = rest.replace(lt, ' ', 1)
                if len(rest.strip()) <= 0.1 * len(e['n']):
                    drop.append(e); out['turns_covered'] += 1
        if not dry:
            for e in drop:      # remember dropped keys so re-importing the same export does not bring them back
                self.con.execute('INSERT OR IGNORE INTO cloud_dropped(source) VALUES (?)', (e['source'],))
                self.store.forget_episode(BANK, e['scope'], e['id'])
            self.con.commit()
        return out

    # ---------- sources ----------
    @staticmethod
    def sources() -> list[tuple[str, str]]:
        home = Path.home()
        out = [(p, 'codex') for p in glob.glob(str(home / '.codex/sessions/*/*/*/rollout-*.jsonl'))]
        out += [(p, 'claude') for p in glob.glob(str(home / '.claude/projects/*/*.jsonl'))]
        return sorted(out, key=lambda x: os.path.getmtime(x[0]))

    # ---------- Cursor (composer chats in the global state.vscdb, read-only) ----------
    CURSOR_DB = Path.home() / '.config/Cursor/User/globalStorage/state.vscdb'

    def _cursor_workspaces(self) -> dict:
        """composerId -> workspace folder: global composerHeaders.workspaceId -> workspaceStorage/<id>/workspace.json."""
        import sqlite3
        from urllib.parse import unquote
        folders = {}
        for wj in glob.glob(str(Path.home() / '.config/Cursor/User/workspaceStorage/*/workspace.json')):
            try:
                folders[Path(wj).parent.name] = unquote((json.load(open(wj)).get('folder') or '').replace('file://', ''))
            except (OSError, ValueError):
                continue
        c = sqlite3.connect(f'file:{self.CURSOR_DB}?mode=ro', uri=True, timeout=10)
        try:
            return {cid: folders.get(ws) or ('cursor-no-folder' if ws == 'empty-window' else f'cursor-ws:{ws}')
                    for cid, ws in c.execute('SELECT composerId, workspaceId FROM composerHeaders')}
        except sqlite3.Error:
            return {}
        finally:
            c.close()

    def ingest_cursor(self) -> int:
        """Cursor keeps chats as composerData:<id> (bubble headers) + bubbleId:<id>:<bubble> rows. Bubbles are
        append-only, so each composer resumes at the last bubble index seen. Cursor stores no per-request token
        counts locally (tokenCount is 0); usage gets one context-size snapshot per changed composer instead."""
        import sqlite3
        if not self.CURSOR_DB.exists():
            return 0
        self.con.execute('CREATE TABLE IF NOT EXISTS cap_cursor (composer TEXT PRIMARY KEY, last_updated INTEGER, '
                         'n INTEGER NOT NULL DEFAULT 0)')
        cur = sqlite3.connect(f'file:{self.CURSOR_DB}?mode=ro', uri=True, timeout=10)
        seen = {r['composer']: dict(r) for r in self.con.execute('SELECT * FROM cap_cursor')}
        changed = [(k[13:], int(u)) for k, u in cur.execute(
            "SELECT key, json_extract(value,'$.lastUpdatedAt') FROM cursorDiskKV WHERE key LIKE 'composerData:%'")
            if u and int(u) > (seen.get(k[13:]) or {}).get('last_updated', 0)]
        wsmap = self._cursor_workspaces() if changed else {}
        n = 0
        for cid, upd in changed:
            d = json.loads(cur.execute('SELECT value FROM cursorDiskKV WHERE key=?', ('composerData:' + cid,)).fetchone()[0])
            heads = d.get('fullConversationHeadersOnly') or []
            f = {'host': 'cursor', 'session': cid, 'cwd': wsmap.get(cid) or 'cursor-unknown',
                 'model': (d.get('modelConfig') or {}).get('modelName'), 'path': str(self.CURSOR_DB)}
            start = (seen.get(cid) or {}).get('n', 0)
            busy = set(d.get('generatingBubbleIds') or [])
            stop = next((i for i, h in enumerate(heads) if h['bubbleId'] in busy), len(heads))
            heads = heads[:stop]                  # a bubble still being generated is taken on a later scan
            for h in heads[start:]:
                row = cur.execute('SELECT value FROM cursorDiskKV WHERE key=?', (f"bubbleId:{cid}:{h['bubbleId']}",)).fetchone()
                if not row:
                    continue
                b = json.loads(row[0])
                ts, src = b.get('createdAt'), f"cursor:{cid}:{h['bubbleId']}"
                tf = b.get('toolFormerData')
                if tf:
                    res = tf.get('result')
                    self._tool(f, tf.get('toolCallId') or h['bubbleId'], tf.get('name') or 'tool',
                               tf.get('rawArgs') or json.dumps(tf.get('params'), ensure_ascii=False),
                               res if isinstance(res, str) else json.dumps(res, ensure_ascii=False), ts, src,
                               ok=tf.get('status') != 'error')
                elif b.get('text'):
                    self._ep(f, 'user' if b.get('type') == 1 else 'assistant', b['text'], ts, src)
                n += 1
            if d.get('contextTokensUsed'):
                self._usage(f, dt_iso(upd), (int(d['contextTokensUsed']), 0, 0, 0, 0), f'cursor:{cid}@{upd}')
            self.con.execute('INSERT OR REPLACE INTO cap_cursor VALUES (?,?,?)', (cid, upd, len(heads)))
        cur.close()
        return n

    # ---------- memory add-on configuration timeline ----------
    @staticmethod
    def agent_config() -> dict:
        """Names only (MCP servers, skills, hook events) per host: no commands' env, no secrets."""
        home = Path.home()
        out = {}

        def names(d):
            return sorted(d) if isinstance(d, dict) else []
        try:
            import tomllib
            with open(home / '.codex/config.toml', 'rb') as fh:
                out['codex.mcp'] = names(tomllib.load(fh).get('mcp_servers'))
        except (OSError, ValueError, ImportError):
            pass
        try:
            out['cursor.mcp'] = names(json.load(open(home / '.cursor/mcp.json')).get('mcpServers'))
        except (OSError, ValueError):
            pass
        try:
            cj = json.load(open(home / '.claude.json'))
            out['claude.mcp'] = sorted(set(names(cj.get('mcpServers'))) | {
                f'{n}@{p}' for p, v in (cj.get('projects') or {}).items() for n in names((v or {}).get('mcpServers'))})
        except (OSError, ValueError):
            pass
        try:
            out['claude.hooks'] = names(json.load(open(home / '.claude/settings.json')).get('hooks'))
        except (OSError, ValueError):
            pass
        for key, d in (('claude.skills', '.claude/skills'), ('codex.skills', '.codex/skills'),
                       ('cursor.skills', '.cursor/skills-cursor'), ('cursor.plugins', '.cursor/plugins')):
            p = home / d
            if p.is_dir():
                out[key] = sorted(x.name for x in p.iterdir() if not x.name.startswith('.'))
        return out

    def snapshot_config(self) -> list[str]:
        """Record each host config list when it changes -> exact install / uninstall timeline of memory add-ons."""
        import hashlib
        self.con.execute('CREATE TABLE IF NOT EXISTS config_snapshots (id INTEGER PRIMARY KEY, ts TEXT, key TEXT, '
                         'items TEXT, sha TEXT, added TEXT, removed TEXT)')
        changed = []
        ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        for key, items in self.agent_config().items():
            blob = json.dumps(items, ensure_ascii=False)
            sha = hashlib.sha256(blob.encode()).hexdigest()[:16]
            last = self.con.execute('SELECT sha, items FROM config_snapshots WHERE key=? ORDER BY id DESC LIMIT 1',
                                    (key,)).fetchone()
            if last and last['sha'] == sha:
                continue
            prev = set(json.loads(last['items'])) if last else set()
            self.con.execute('INSERT INTO config_snapshots(ts,key,items,sha,added,removed) VALUES (?,?,?,?,?,?)',
                             (ts, key, blob, sha, json.dumps(sorted(set(items) - prev)),
                              json.dumps(sorted(prev - set(items)))))
            changed.append(key)
        return changed

    def scan(self) -> dict:
        prev = self.con.execute('PRAGMA synchronous').fetchone()[0]
        self.con.execute('PRAGMA synchronous=NORMAL')   # rebuildable from the host files
        try:
            res = {'files': 0, 'records': 0}
            cfg = self.snapshot_config()
            if cfg:
                res['config_changed'] = cfg
            for path, host in self.sources():
                n = self.ingest_file(path, host)
                res['files'] += bool(n)
                res['records'] += n
            if os.environ.get('PM1_CAPTURE_CURSOR', '1') != '0':
                res['cursor'] = self.ingest_cursor()
                res['records'] += res['cursor']
            if os.environ.get('PM1_CODE', '1') != '0':   # code history: file touches now, graphify graphs ~10 min
                from .codemap import CodeMap
                cm = CodeMap(self.store, BANK)
                t = cm.extract_touches()
                if t['touches'] or t['commits']:
                    res['code'] = t
                from .failsig import index_new       # failed shell commands -> signatures (hook: past fixes)
                with self.store.lock:
                    nf = index_new(self.con, BANK)
                if nf:
                    res['failures'] = nf
                if time.time() - getattr(self, '_graphs_at', 0) > 600:
                    self._graphs_at = time.time()
                    g = cm.import_graphs()
                    if g['repos']:
                        res['graphs'] = g
            return res
        finally:
            self.con.execute(f'PRAGMA synchronous={int(prev)}')

    def retag(self) -> dict:
        """Fill mem_events for everything already captured (re-reads host files; writes nothing else)."""
        self.tag_only, self._tagpend = True, {}
        self.con.execute('DELETE FROM mem_events')
        try:
            for path, host in self.sources():
                f = {'path': path, 'host': host, 'offset': 0, 'lines': 0, 'session': None, 'cwd': None, 'model': None}
                with open(path, 'rb') as fh:
                    for raw in fh:
                        if not raw.endswith(b'\n'):
                            break
                        f['lines'] += 1
                        try:
                            d = json.loads(raw)
                        except ValueError:
                            continue
                        (self._codex if host == 'codex' else self._claude)(f, d, f"{path}#{f['lines']}")
            if self.CURSOR_DB.exists():
                self.con.execute('CREATE TEMP TABLE IF NOT EXISTS cap_cursor_bak AS SELECT * FROM cap_cursor')
                self.con.execute('DELETE FROM cap_cursor')
                try:
                    self.ingest_cursor()
                finally:
                    self.con.execute('DELETE FROM cap_cursor')
                    self.con.execute('INSERT INTO cap_cursor SELECT * FROM cap_cursor_bak')
                    self.con.execute('DROP TABLE cap_cursor_bak')
        finally:
            self.tag_only = False
        return {r[0] + ':' + r[1]: r[2] for r in self.con.execute(
            'SELECT system, kind, count(*) FROM mem_events GROUP BY 1, 2')}

    def memory_report(self) -> list[dict]:
        """Per host and memory condition: vanilla (no add-on available in the session), available-unused,
        used (>=1 query). Observational only: sessions differ in task, project and time; use replay A/B for causes."""
        rows = []
        for r in self.con.execute(
                "WITH s AS (SELECT host, session, count(*) calls, sum(input) input, sum(input-cached) fresh, "
                "sum(output) output, min(ts) t0 FROM usage GROUP BY 1,2), "
                "m AS (SELECT host, session, system, max(kind='available') av, sum(kind='query') q, "
                "sum(CASE WHEN kind='query' THEN out_chars END) qchars, sum(kind='build') b FROM mem_events "
                "GROUP BY 1,2,3) "
                "SELECT s.host, COALESCE(m.system,'-') system, CASE WHEN m.q > 0 THEN 'used' WHEN m.av > 0 OR m.b > 0 "
                "THEN 'available' ELSE 'vanilla' END cond, count(*) sessions, sum(s.calls) calls, "
                "sum(s.input) input, sum(s.fresh) fresh, sum(s.output) output, sum(m.q) queries, "
                "sum(m.qchars) query_chars, min(s.t0) first, max(s.t0) last "
                "FROM s LEFT JOIN m ON m.host=s.host AND m.session=s.session AND m.system='graphify' "
                "GROUP BY 1,2,3 ORDER BY 1,3"):
            d = dict(r)
            d['input_per_call'] = round((d['input'] or 0) / max(1, d['calls']))
            rows.append(d)
        return rows

    def report(self, days: int = 7) -> dict:
        q = lambda sql, *a: [dict(r) for r in self.con.execute(sql, a)]  # noqa: E731
        since = time.strftime('%Y-%m-%d', time.gmtime(time.time() - days * 86400))
        by = q("SELECT host, substr(ts,1,10) AS day, count(*) AS calls, sum(input) AS input, sum(cached) AS cached, "
               "sum(output) AS output, sum(reasoning) AS reasoning FROM usage WHERE ts >= ? GROUP BY 1,2 ORDER BY 2,1",
               since)
        for r in by:
            r['cache_share'] = round((r['cached'] or 0) / max(1, r['input'] or 0), 3)
            r['fresh_input'] = (r['input'] or 0) - (r['cached'] or 0)
        tools = q("SELECT host, name, count(*) AS calls, sum(out_chars) AS out_chars, max(out_chars) AS max_out "
                  "FROM tool_io WHERE ts >= ? GROUP BY 1,2 ORDER BY out_chars DESC LIMIT 12", since)
        projects = q("SELECT cwd, host, count(*) AS calls, sum(input) AS input, sum(input - cached) AS fresh_input, "
                     "sum(output) AS output FROM usage WHERE ts >= ? GROUP BY 1,2 ORDER BY input DESC LIMIT 12", since)
        return {'since': since, 'by_day': by, 'tool_output': tools, 'projects': projects}


def _cloud_meta(cap: Capture, a) -> None:
    if not a.meta:
        return
    cap.con.execute('CREATE TABLE IF NOT EXISTS cloud_sessions (host TEXT, session TEXT, label TEXT, meta TEXT, '
                    'updated TEXT, PRIMARY KEY(host, session))')
    cap.con.execute('INSERT OR REPLACE INTO cloud_sessions VALUES (?,?,?,?,?)', (a.host, a.session, a.label, a.meta,
                    time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog='pm1-capture')
    ap.add_argument('--db', default=default_db())
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('scan')
    w = sub.add_parser('watch')
    w.add_argument('-i', '--interval', type=float, default=60)
    r = sub.add_parser('report')
    r.add_argument('--days', type=int, default=7)
    sub.add_parser('retag')
    sub.add_parser('memory')
    sub.add_parser('config')
    im = sub.add_parser('import-usage', help='per-response usage exported from a cloud session (Cowork)')
    im.add_argument('file', help='TSV: ts, model, input, cached, cache_write, output, line [, .gz / .gz.b64]')
    im.add_argument('--host', default='cowork')
    im.add_argument('--session', required=True)
    im.add_argument('--label', default='', help='project / workspace label stored as cwd')
    im.add_argument('--meta', default='', help='JSON with session totals (cost, title) to keep alongside')
    ic = sub.add_parser('import-conversation', help='conversation exported from a cloud session (Cowork)')
    ic.add_argument('file', help='normalized events, JSON lines (.jsonl / .jsonl.gz)')
    ic.add_argument('--host', default='cowork')
    ic.add_argument('--session', required=True)
    ic.add_argument('--label', default='')
    ic.add_argument('--meta', default='', help='JSON with session totals (cost, title) to keep alongside')
    dd = sub.add_parser('dedupe', help='remove duplicates left by repeated cloud-session imports')
    dd.add_argument('--host', default='cowork')
    dd.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    if getattr(a, 'session', None) and a.session.startswith(a.host + ':'):
        a.session = a.session[len(a.host) + 1:]      # '--session cowork:<id>' is the same session as '<id>'
    from .docs_cli import index_models
    os.makedirs(os.path.dirname(a.db) or '.', exist_ok=True)
    cap = Capture(Store(a.db, index_models=index_models()))
    if a.cmd == 'scan':
        print(json.dumps(cap.scan()))
    elif a.cmd == 'watch':
        import sqlite3
        while True:
            try:
                res = cap.scan()
            except sqlite3.OperationalError as e:   # another writer held the lock past the timeout
                if 'locked' not in str(e):
                    raise
                print(time.strftime('%H:%M:%S'), 'database locked; retrying next round', flush=True)
                time.sleep(a.interval)
                continue
            if res['records'] or res.get('config_changed'):
                print(time.strftime('%H:%M:%S'), json.dumps(res), flush=True)
            time.sleep(a.interval)
    elif a.cmd == 'retag':
        print(json.dumps(cap.retag()))
    elif a.cmd == 'import-usage':
        import base64
        import gzip
        raw = open(a.file, 'rb').read()
        if a.file.endswith('.b64'):
            raw = base64.b64decode(raw)
        if raw[:2] == b'\x1f\x8b':
            raw = gzip.decompress(raw)
        f = {'host': a.host, 'session': a.session, 'cwd': f'{a.host}:{a.label}' if a.label else a.host, 'model': None}
        n = 0
        for line in raw.decode().splitlines():
            if not line.strip():
                continue
            ts, model, inp, cr, cw, out, ln = line.split('\t')[:7]
            f['model'] = model
            before = cap.con.total_changes
            cap._usage(f, ts, (int(inp), int(cr), int(cw), int(out), 0), f'{a.host}:{a.session}#{ln}')
            n += cap.con.total_changes - before
        _cloud_meta(cap, a)
        tot = cap.con.execute('SELECT count(*), sum(input), sum(cached), sum(cache_write), sum(output) FROM usage '
                              'WHERE host=? AND session=?', (a.host, a.session)).fetchone()
        print(json.dumps({'new_rows': n, 'session_rows': tot[0], 'input': tot[1], 'cached': tot[2],
                          'cache_write': tot[3], 'output': tot[4]}))
    elif a.cmd == 'import-conversation':
        import gzip
        raw = open(a.file, 'rb').read()
        if raw[:2] == b'\x1f\x8b':
            raw = gzip.decompress(raw)
        events = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
        res = cap.import_conversation(events, a.host, a.session, a.label)
        _cloud_meta(cap, a)
        res['deduped'] = cap.dedupe_cloud(a.host)       # backfill turns now covered by local detail, etc.
        print(json.dumps({'events': len(events), **res}))
    elif a.cmd == 'dedupe':
        print(json.dumps(cap.dedupe_cloud(a.host, dry=a.dry_run)))
    elif a.cmd == 'config':
        cap.snapshot_config()
        for r in cap.con.execute('SELECT ts, key, added, removed FROM config_snapshots ORDER BY id'):
            print(r['ts'], r['key'], 'added', r['added'], 'removed', r['removed'])
    elif a.cmd == 'memory':
        json.dump(cap.memory_report(), sys.stdout, ensure_ascii=False, indent=1)
    else:
        json.dump(cap.report(a.days), sys.stdout, ensure_ascii=False, indent=1)
        print()


if __name__ == '__main__':
    main()
