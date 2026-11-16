"""Usage and benefit meter for pm1: summary, daily audit of real calls, HTML dashboard.

  pm1-meter summary [--days 7]                    JSON summary of real agent use
  pm1-meter audit [--day YYYY-MM-DD] [--max 120] [--dry-run] [--calibrate]
                                                  label yesterday's real retrieval calls with the local Grok/Composer
                                                  labelers (GPT only on their disagreements), store in mcp_audit
  pm1-meter audit-ingest --task <task>            (re)read a finished labeling task into mcp_audit
  pm1-meter dashboard [--out FILE] [--days 30]    self-contained HTML page (default ~/Documents/pm1-dashboard.html)

"Real" calls are those from agent clients (codex, cursor, claude-code, claude-desktop, cowork); tests, smoke runs and evaluation
runs (client codex-eval) are excluded. The audit sends the query and the returned records (secrets redacted) to
the labelers; it labels each returned item as answers / related / irrelevant to the query. It never changes memory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import sqlite3
import statistics
import subprocess
from pathlib import Path

from .paths import default_db
from .redact import redact

DB = default_db()
REAL = ('codex', 'cursor', 'claude-code', 'claude-desktop', 'cowork', 'claude')
HOOK = 'claude-code-hook'
AUDITED_TOOLS = ('search_sessions', 'search_docs', 'search_notes', 'memory_recall')   # memory_recall: calls before 11-15
LABEL_DIR = Path(os.path.expanduser(os.environ.get('PM1_LABEL_DIR', '~/pm1-labeling')))
LABEL_DUAL = Path(os.path.expanduser(os.environ.get('PM1_LABEL_DUAL', '~/pm1-labeling/label_dual.py'))).parent.parent
BACKFILL_LOG = Path(os.path.expanduser('~/.local/state/pm1/backfill.jsonl'))
LABELS = ('answers', 'related', 'irrelevant')


def connect(db: str = DB) -> sqlite3.Connection:
    con = sqlite3.connect(db, timeout=30)
    con.row_factory = sqlite3.Row
    from .sessions import LOG_SCHEMA
    con.executescript(LOG_SCHEMA)
    return con


def _in(seq) -> str:
    return ','.join("'" + s + "'" for s in seq)


# ---------------------------------------------------------------- summary
def summary(con: sqlite3.Connection, days: int = 7, today: dt.date | None = None) -> dict:
    today = today or dt.datetime.now(dt.timezone.utc).date()
    since = (today - dt.timedelta(days=days - 1)).isoformat()
    rows = con.execute(f'SELECT * FROM mcp_calls WHERE ts >= ? AND client IN ({_in(REAL)})', (since,)).fetchall()
    by_client: dict[str, int] = {}
    by_tool: dict[str, dict] = {}
    active_days = set()
    for r in rows:
        by_client[r['client']] = by_client.get(r['client'], 0) + 1
        t = by_tool.setdefault(r['tool'], {'calls': 0, 'errors': 0, 'lat': [], 'tok': []})
        t['calls'] += 1
        t['errors'] += bool(r['error'])
        t['lat'].append(r['latency_ms'] or 0)
        t['tok'].append(r['out_tokens'] or 0)
        active_days.add(r['ts'][:10])
    tools = {k: {'calls': v['calls'], 'errors': v['errors'],
                 'median_ms': round(statistics.median(v['lat'])) if v['lat'] else None,
                 'avg_tokens': round(sum(v['tok']) / len(v['tok'])) if v['tok'] else None}
             for k, v in sorted(by_tool.items(), key=lambda kv: -kv[1]['calls'])}
    hooks = con.execute('SELECT tool, count(*) n, sum(out_tokens) tok FROM mcp_calls WHERE ts >= ? AND client=? '
                        'GROUP BY tool', (since, HOOK)).fetchall()
    fb = con.execute('SELECT verdict, count(*) n FROM mcp_feedback WHERE ts >= ? GROUP BY verdict',
                     (since,)).fetchall()
    au = audit_stats(con, since)
    rerank = rerank_stats(con, since)
    return {'since': since, 'days': days, 'real_calls': len(rows), 'active_days': len(active_days),
            'by_client': by_client, 'by_tool': tools,
            'hooks': {h['tool']: {'events': h['n'], 'tokens': h['tok']} for h in hooks},
            'feedback': {f['verdict']: f['n'] for f in fb}, 'audit': au, 'rerank': rerank,
            'vectors': vector_stats(con), 'backfill_runs': backfill_runs(3)}


def rerank_stats(con: sqlite3.Connection, since: str) -> dict:
    out = {'reranked': 0, 'fallback': 0}
    for r in con.execute(f"SELECT m.result FROM mcp_results m JOIN mcp_calls c ON c.id = m.call_id WHERE c.ts >= ? "
                         f"AND c.client IN ({_in(REAL)}) AND c.tool IN ({_in(AUDITED_TOOLS)})", (since,)):
        try:
            route = json.loads(r['result']).get('route') or {}
        except ValueError:
            continue                                   # clipped result
        out['reranked' if route.get('rerank_model') else 'fallback'] += 1
    return out


def audit_stats(con: sqlite3.Connection, since: str) -> dict:
    rows = con.execute('SELECT a.call_id, a.item, a.label, a.agreed FROM mcp_audit a JOIN mcp_calls c ON '
                       'c.id = a.call_id WHERE c.ts >= ? ORDER BY a.call_id, a.rowid', (since,)).fetchall()
    calls: dict[int, list] = {}
    for r in rows:
        calls.setdefault(r['call_id'], []).append(r)
    if not calls:
        return {'calls': 0}
    top1 = [c[0]['label'] for c in calls.values()]        # items are stored in result order
    items = [r['label'] for r in rows]
    return {'calls': len(calls), 'items': len(items),
            'top1_answers': round(top1.count('answers') / len(top1), 3),
            'any_answers': round(sum(any(r['label'] == 'answers' for r in c) for c in calls.values()) / len(calls), 3),
            'items_irrelevant': round(items.count('irrelevant') / len(items), 3),
            'agreement': round(sum(r['agreed'] for r in rows) / len(rows), 3)}


def vector_stats(con: sqlite3.Connection) -> dict:
    rows = con.execute("SELECT state, count(*) n FROM jobs WHERE kind='dense' GROUP BY state").fetchall()
    return {r['state']: r['n'] for r in rows}


def backfill_runs(n: int) -> list:
    if not BACKFILL_LOG.exists():
        return []
    return [json.loads(x) for x in BACKFILL_LOG.read_text().splitlines()[-n:] if x.strip()]


# ---------------------------------------------------------------- audit
INSTRUCTIONS = """# pm1 retrieval audit ({task})

Each line is one search an AI coding agent made in the user's memory tool (pm1), with the query and the records pm1
returned, in the order returned. Records are excerpts of past agent sessions or documents. Judge each record ONLY
against the query, using only the text given (no outside knowledge, do not guess what the agent wanted beyond the
query).

Labels for each record:
- "answers": the record itself states the fact / decision / number / file / command the query asks about (fully or
  the main part of it).
- "related": same topic or project, but it does not contain what the query asks for.
- "irrelevant": a different topic, or only shares words with the query.

Output one line per input line:
{{"id": "<id>", "response": {{"labels": {{"<item id>": "answers|related|irrelevant", ...}}, "confidence": 0.0-1.0}}}}

Rules: every item id of the input must appear in "labels" exactly once; keep ids exactly as given. Judge each line
independently. If a record is cut off, judge what is shown. "[redacted]" marks removed secrets; ignore it.
"""


def _items(result: dict) -> list[tuple[str, str]]:
    out = []
    from .sessions import flat_items
    for it in flat_items(result)[:6]:
        iid = it.get('episode_id') or it.get('evidence_id') or it.get('span_id')
        if iid is None:
            continue
        head = ' · '.join(str(it[k]) for k in ('when', 'project', 'role', 'path', 'where') if it.get(k))
        body = (it.get('text') or '')[:1500]
        out.append((str(iid), (head + '\n' if head else '') + body))
    return out


def audit_prepare(con: sqlite3.Connection, day: str, max_calls: int = 120) -> tuple[str, int]:
    rows = con.execute(
        f"SELECT c.id, c.client, c.tool, c.args, m.result FROM mcp_calls c JOIN mcp_results m ON m.call_id = c.id "
        f"WHERE substr(c.ts,1,10) = ? AND c.client IN ({_in(REAL)}) AND c.tool IN ({_in(AUDITED_TOOLS)}) "
        f"AND c.error IS NULL AND c.id NOT IN (SELECT call_id FROM mcp_audit) ORDER BY c.id LIMIT ?",
        (day, max_calls)).fetchall()
    task = f"pm1_audit_{day.replace('-', '')}"
    lines = []
    for r in rows:
        try:
            res = json.loads(r['result'])
            query = json.loads(r['args']).get('query') or ''
        except ValueError:
            continue
        items = _items(res)
        if not query or not items:
            continue
        prompt = f'Query ({r["tool"]}): {redact(query)}\n\n' + '\n\n'.join(
            f'[item {iid}]\n{redact(text)}' for iid, text in items)
        lines.append({'id': f'call{r["id"]}', 'prompt': prompt, 'item_ids': [i for i, _ in items]})
    if lines:
        (LABEL_DIR / 'inbox').mkdir(parents=True, exist_ok=True)
        with open(LABEL_DIR / 'inbox' / f'{task}.jsonl', 'w') as fh:
            for x in lines:
                fh.write(json.dumps(x, ensure_ascii=False) + '\n')
        (LABEL_DIR / 'inbox' / f'{task}.md').write_text(INSTRUCTIONS.format(task=task))
    return task, len(lines)


def audit_ingest(con: sqlite3.Connection, task: str) -> dict:
    """Combine every labeler's output for the task: majority label per item; a GPT review (run only on the Cursor
    labelers' disagreements) decides ties."""
    src = {}
    inbox = LABEL_DIR / 'inbox' / f'{task}.jsonl'
    for line in inbox.read_text().splitlines():
        x = json.loads(line)
        src[x['id']] = x['item_ids']
    votes: dict[tuple[str, str], dict[str, str]] = {}
    conf: dict[str, list] = {}
    for f in sorted((LABEL_DIR / 'outbox').glob(f'*/{task}.out.jsonl')):
        who = f.parent.name
        for line in f.read_text().splitlines():
            try:
                x = json.loads(line)
                resp = x['response'] if isinstance(x['response'], dict) else json.loads(x['response'])
            except (ValueError, KeyError, TypeError):
                continue
            for iid, lab in (resp.get('labels') or {}).items():
                if x['id'] in src and str(iid) in src[x['id']] and lab in LABELS:
                    votes.setdefault((x['id'], str(iid)), {})[who] = lab
            if isinstance(resp.get('confidence'), (int, float)):
                conf.setdefault(x['id'], []).append(float(resp['confidence']))
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    n = 0
    for cid, iids in src.items():
        for iid in iids:
            v = votes.get((cid, iid))
            if not v:
                continue
            gpt = [lab for who, lab in v.items() if who.startswith('gpt')]
            counts = {lab: list(v.values()).count(lab) for lab in set(v.values())}
            best = max(counts.values())
            top = [lab for lab, c in counts.items() if c == best]
            label = top[0] if len(top) == 1 else (gpt[0] if gpt else sorted(top, key=LABELS.index)[-1])
            c = conf.get(cid) or [0.0]
            con.execute('INSERT OR REPLACE INTO mcp_audit(call_id, item, label, agreed, confidence, task, ts) '
                        'VALUES (?,?,?,?,?,?,?)', (int(cid[4:]), iid, label, int(len(counts) == 1),
                                                   round(min(c), 2), task, ts))
            n += 1
    con.commit()
    return {'task': task, 'calls': len(src), 'items_labelled': n}


def audit(con: sqlite3.Connection, day: str, max_calls: int, dry: bool, calibrate: bool) -> dict:
    from . import judge
    if not judge.enabled() or not (LABEL_DUAL / 'scripts/label_dual.py').exists():
        return {'task': None, 'calls': 0, 'ran': False, 'skipped': 'judges off (PM1_JUDGE=off) or label_dual not installed: '
                'nothing is sent to a cloud labeler'}
    task, n = audit_prepare(con, day, max_calls)
    if not n or dry:
        return {'task': task, 'calls': n, 'ran': False}
    cmd = [str(LABEL_DUAL / '.venv/bin/python'), 'scripts/label_dual.py', task, '--dir', str(LABEL_DIR)]
    first = not any((LABEL_DIR / 'outbox').glob('pm1_audit_*.compare.json'))
    if calibrate or first:          # a new task type is calibrated once (4 labelers) before the default pair
        cmd.append('--calibrate')
    p = subprocess.run(cmd, cwd=LABEL_DUAL, capture_output=True, text=True, timeout=7200)
    out = audit_ingest(con, task)
    return {**out, 'label_dual_exit': p.returncode, 'stderr_tail': p.stderr[-300:] if p.returncode else ''}


# ---------------------------------------------------------------- dashboard
def daily(con: sqlite3.Connection, days: int) -> tuple[list[str], dict[str, list[int]]]:
    today = dt.datetime.now(dt.timezone.utc).date()
    dates = [(today - dt.timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    series = {c: [0] * days for c in REAL}
    for r in con.execute(f'SELECT substr(ts,1,10) d, client, count(*) n FROM mcp_calls WHERE ts >= ? AND client IN '
                         f'({_in(REAL)}) GROUP BY 1, 2', (dates[0],)):
        if r['d'] in dates:
            series[r['client']][dates.index(r['d'])] = r['n']
    return dates, {k: v for k, v in series.items() if any(v)}


COLORS = {'codex': '#2a78c6', 'cursor': '#7a52c7', 'claude-code': '#d0743c', 'cowork': '#2f9e6e', 'claude': '#b5853a'}


def _bars(dates: list[str], series: dict[str, list[int]]) -> str:
    w, h, pad = 720, 180, 28
    tot = [sum(v[i] for v in series.values()) for i in range(len(dates))]
    mx = max(tot + [1])
    bw = (w - pad) / len(dates)
    parts = [f'<svg viewBox="0 0 {w} {h + 24}" role="img" aria-label="calls per day">',
             f'<text x="0" y="12" class="ax">{mx}</text><line x1="{pad}" y1="{h}" x2="{w}" y2="{h}" class="base"/>']
    for i, d in enumerate(dates):
        y = h
        for c, v in series.items():
            if not v[i]:
                continue
            bh = (h - 16) * v[i] / mx
            y -= bh
            parts.append(f'<rect x="{pad + i * bw + 1:.1f}" y="{y:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" '
                         f'fill="{COLORS.get(c, "#888")}"><title>{d} {c}: {v[i]}</title></rect>')
        if i % max(1, len(dates) // 6) == 0 or i == len(dates) - 1:
            parts.append(f'<text x="{pad + i * bw:.1f}" y="{h + 16}" class="ax">{d[5:]}</text>')
    parts.append('</svg>')
    legend = ' '.join(f'<span class="lg"><i style="background:{COLORS.get(c, "#888")}"></i>{c}</span>' for c in series)
    return ''.join(parts) + f'<div class="legend">{legend}</div>'


def dashboard(con: sqlite3.Connection, days: int = 30) -> str:
    s7 = summary(con, 7)
    s30 = summary(con, days)
    dates, series = daily(con, days)
    au, fb, rr, vec = s7['audit'], s7['feedback'], s7['rerank'], s7['vectors']
    e = html.escape

    def kpi(label, value, sub=''):
        return f'<div class="kpi"><div class="v">{e(str(value))}</div><div class="l">{e(label)}</div>' \
               f'<div class="s">{e(sub)}</div></div>'

    adopted = fb.get('helpful', 0) + fb.get('wrong', 0)
    err = f"{fb.get('wrong', 0)}/{adopted}" if adopted else '—'
    pend = vec.get('pending', 0)
    kpis = ''.join([
        kpi('real agent calls, 7 days', s7['real_calls'], f"{s7['active_days']} active days"),
        kpi('clients', len(s7['by_client']), ', '.join(f'{k} {v}' for k, v in s7['by_client'].items()) or '—'),
        kpi('audited: top result answers', f"{au['top1_answers']:.0%}" if au.get('calls') else '—',
            f"{au.get('calls', 0)} calls audited" if au.get('calls') else 'audit not run yet'),
        kpi('reported wrong / adopted', err, f"{sum(fb.values())} feedback reports"),
        kpi('reranked searches', f"{rr['reranked']}/{rr['reranked'] + rr['fallback']}" if any(rr.values()) else '—',
            'rest fell back to first-stage order'),
        kpi('vectors pending', f'{pend:,}', f"{vec.get('done', 0):,} done"),
    ])
    gate_use = s7['active_days'] >= 3 and s7['real_calls'] >= 10
    gate_err = (fb.get('wrong', 0) / adopted < 0.01) if adopted >= 20 else None
    gate = [('Weekly real use (≥3 active days, ≥10 calls in 7 days)', gate_use),
            ('<1 error per 100 adopted recalls (needs ≥20 adopted)', gate_err),
            ('Beats agent searching alone on real questions', None),
            ('Handoff brief saves ≥30% tokens', None)]
    mark = {True: '<b class="ok">met</b>', False: '<b class="no">not met</b>', None: '<b class="na">open</b>'}
    tools = ''.join(f"<tr><td>{e(k)}</td><td>{v['calls']}</td><td>{v['errors']}</td><td>{v['median_ms']}</td>"
                    f"<td>{v['avg_tokens']}</td></tr>" for k, v in s30['by_tool'].items())
    hooks = ''.join(f"<tr><td>{e(k)}</td><td>{v['events']}</td><td>{v['tokens'] or 0:,}</td></tr>"
                    for k, v in s30['hooks'].items()) or '<tr><td colspan="3">none</td></tr>'
    runs = ''.join(f"<tr><td>{e(r['ended'][:16])}</td><td>{e(r['reason'])}</td><td>{r['dense']:,}</td>"
                   f"<td>{r['minutes']}</td><td>{r['pending_after']:,}</td></tr>" for r in s7['backfill_runs']) \
        or '<tr><td colspan="5">no nightly run yet (first one tonight at 01:00)</td></tr>'
    recent = con.execute(f"SELECT c.id, c.ts, c.client, c.tool, c.args, c.n_items, c.latency_ms, "
                         f"(SELECT verdict FROM mcp_feedback f WHERE f.call_id=c.id LIMIT 1) fb, "
                         f"(SELECT label FROM mcp_audit a WHERE a.call_id=c.id ORDER BY rowid LIMIT 1) top "
                         f"FROM mcp_calls c WHERE c.client IN ({_in(REAL)}) ORDER BY c.id DESC LIMIT 25").fetchall()
    rec = ''.join(f"<tr><td>{e(r['ts'][5:16].replace('T', ' '))}</td><td>{e(r['client'])}</td><td>{e(r['tool'])}</td>"
                  f"<td class='q'>{e((json.loads(r['args'] or '{}').get('query') or json.loads(r['args'] or '{}').get('target') or '')[:90])}</td>"
                  f"<td>{r['n_items']}</td><td>{round(r['latency_ms'] or 0)}</td><td>{e(r['top'] or '')}</td>"
                  f"<td>{e(r['fb'] or '')}</td></tr>" for r in recent)
    # every client's last call, the Claude Code hook included: agent calls can be hours apart while Claude Code is
    # busy (the hook still fires on each session start), so "latest calls" alone can look as if a client stopped
    last = con.execute(f"SELECT client, max(ts) t, sum(ts >= ?) n FROM mcp_calls WHERE client IN ({_in(REAL + (HOOK,))}) "
                       f"GROUP BY client ORDER BY t DESC", (s7['since'],)).fetchall()
    lastrows = ''.join(f"<tr><td>{e(r['client'])}{' (automatic, not an agent call)' if r['client'] == HOOK else ''}</td>"
                       f"<td>{e(r['t'][5:16].replace('T', ' '))}</td><td>{r['n']}</td></tr>" for r in last)
    now = dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>pm1 Usage</title><style>
:root{{--bg:#fbfaf7;--fg:#1f2328;--mut:#6a6f76;--card:#fff;--line:#e4e1da;--ok:#2f9e6e;--no:#c4473a;--na:#8a8f96}}
@media (prefers-color-scheme: dark){{:root{{--bg:#16181b;--fg:#e8e6e1;--mut:#9aa0a6;--card:#1f2226;--line:#33373c}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:20px 16px 48px}} h1{{font-size:22px;margin:0 0 2px}}
h2{{font-size:15px;margin:28px 0 8px}} .sub{{color:var(--mut)}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:16px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}}
.kpi .v{{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}} .kpi .l{{font-size:12px}}
.kpi .s{{font-size:11px;color:var(--mut)}} svg{{width:100%;height:auto}} .ax{{font-size:10px;fill:var(--mut)}}
.base{{stroke:var(--line)}} .legend{{font-size:12px;color:var(--mut)}} .lg{{margin-right:12px}}
.lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}}
table{{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}}
td,th{{text-align:left;padding:5px 6px;border-bottom:1px solid var(--line)}} th{{color:var(--mut);font-weight:500}}
.wrap{{overflow-x:auto}} .q{{max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.ok{{color:var(--ok)}} .no{{color:var(--no)}} .na{{color:var(--na)}}
</style></head><body><main>
<h1>pm1 usage and benefit</h1><div class="sub">Generated {e(now)} from the local pm1 database. Real agent calls
only (tests and evaluation runs excluded).</div>
<div class="kpis">{kpis}</div>
<h2>Last activity by client</h2><div class="wrap"><table><tr><th>client</th><th>last call (UTC)</th>
<th>calls, 7 days</th></tr>{lastrows}</table></div>
<h2>Calls per day, last {days} days</h2>{_bars(dates, series)}
<h2>11/1 gate</h2><table>{''.join(f'<tr><td>{e(t)}</td><td>{mark[v]}</td></tr>' for t, v in gate)}</table>
<h2>Tools, last {days} days</h2><div class="wrap"><table><tr><th>tool</th><th>calls</th><th>errors</th>
<th>median ms</th><th>avg tokens</th></tr>{tools}</table></div>
<h2>Audit, last 7 days</h2><table><tr><td>calls audited</td><td>{au.get('calls', 0)}</td></tr>
<tr><td>top result answers the query</td><td>{au.get('top1_answers', '—')}</td></tr>
<tr><td>some result answers the query</td><td>{au.get('any_answers', '—')}</td></tr>
<tr><td>returned items irrelevant</td><td>{au.get('items_irrelevant', '—')}</td></tr>
<tr><td>labeler agreement</td><td>{au.get('agreement', '—')}</td></tr></table>
<h2>Agent feedback, last 7 days</h2><table>{''.join(f'<tr><td>{e(k)}</td><td>{fb.get(k, 0)}</td></tr>'
                                                    for k in ('helpful', 'not_helpful', 'wrong', 'missing'))}</table>
<h2>Claude Code hooks, last {days} days</h2><table><tr><th>event</th><th>count</th><th>tokens</th></tr>{hooks}</table>
<h2>Nightly vector backfill</h2><div class="wrap"><table><tr><th>ended</th><th>reason</th><th>vectors</th>
<th>minutes</th><th>pending after</th></tr>{runs}</table></div>
<h2>Latest calls</h2><div class="wrap"><table><tr><th>time (UTC)</th><th>client</th><th>tool</th><th>query</th>
<th>items</th><th>ms</th><th>audit top</th><th>feedback</th></tr>{rec}</table></div>
</main></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(prog='pm1-meter')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('summary')
    s.add_argument('--days', type=int, default=7)
    a_ = sub.add_parser('audit')
    a_.add_argument('--day', default=(dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).isoformat())
    a_.add_argument('--max', type=int, default=120)
    a_.add_argument('--dry-run', action='store_true')
    a_.add_argument('--calibrate', action='store_true')
    i = sub.add_parser('audit-ingest')
    i.add_argument('--task', required=True)
    d = sub.add_parser('dashboard')
    d.add_argument('--out', default=os.path.expanduser('~/Documents/pm1-dashboard.html'))
    d.add_argument('--days', type=int, default=30)
    a = ap.parse_args()
    con = connect()
    if a.cmd == 'summary':
        print(json.dumps(summary(con, a.days), ensure_ascii=False, indent=1))
    elif a.cmd == 'audit':
        print(json.dumps(audit(con, a.day, a.max, a.dry_run, a.calibrate), ensure_ascii=False))
    elif a.cmd == 'audit-ingest':
        print(json.dumps(audit_ingest(con, a.task), ensure_ascii=False))
    elif a.cmd == 'dashboard':
        # the desktop app's shared server logs as 'cowork' until it restarts on code that names its connection
        # 'claude-desktop' (mcp.handle); every call through it is Code tab or Cowork alike, so fold the old label in
        con.execute("UPDATE mcp_calls SET client='claude-desktop' WHERE client='cowork'")
        con.execute("UPDATE mcp_feedback SET client='claude-desktop' WHERE client='cowork'")
        con.commit()
        Path(a.out).write_text(dashboard(con, a.days))
        print(a.out)


if __name__ == '__main__':
    main()
