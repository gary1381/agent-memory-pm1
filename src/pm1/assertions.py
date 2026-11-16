"""Append-only assertion ledger (structure candidate C2, proposed by grok-4.7 in the A1 design consult, 2026-10-15).

The model's ONLY writes: resolve_entity (exact-name rule) and append_assertion (quote-locked, closed lists).
Everything else is a VIEW rebuilt by code from the log and never edited: current rows per (entity, facet, scope)
with history and conflict sets (incl. later denials), loops (proposal -> decision by topic_key), the standing
instruction register, and a per-date session index. Deleting the views and rebuilding must reproduce them.
Stdlib only (runs inside the isolated Hermes container).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

MODALITY = ('user_claim', 'user_decision', 'user_proposal', 'user_preference', 'user_instruction', 'third_party',
            'assistant_advice', 'assistant_example')
POLARITY = ('assert', 'deny')
FACETS = ('identity', 'age', 'location', 'job', 'pay', 'offer', 'goal', 'relationship', 'course', 'certification',
          'portfolio', 'project', 'tool', 'subscription', 'budget', 'spend', 'count', 'event', 'schedule', 'deadline',
          'preference', 'health', 'trait', 'plan', 'other')
TYPES = ('person', 'org', 'place', 'event', 'work', 'product', 'goal', 'project', 'topic')
SCHEMA = """
CREATE TABLE IF NOT EXISTS al_entity (id INTEGER PRIMARY KEY, name TEXT NOT NULL, norm TEXT NOT NULL UNIQUE, type TEXT,
  aliases TEXT DEFAULT '[]', first_turn TEXT);
CREATE TABLE IF NOT EXISTS al_assertion (id INTEGER PRIMARY KEY, turn_id TEXT NOT NULL, said_at TEXT, role TEXT,
  about_time TEXT, entity_id INTEGER NOT NULL, facet TEXT NOT NULL, value_text TEXT, number REAL, unit TEXT,
  scope TEXT DEFAULT '', modality TEXT NOT NULL, polarity TEXT NOT NULL, quote TEXT NOT NULL, topic_key TEXT,
  mentioned TEXT DEFAULT '[]', seq INTEGER);
CREATE TABLE IF NOT EXISTS al_log (id INTEGER PRIMARY KEY, step TEXT, op TEXT, args TEXT, why TEXT, ok INTEGER,
  result TEXT, actor TEXT);
"""
_W = re.compile(r'\s+')


def norm(s):
    return _W.sub(' ', re.sub(r'[^\w\s$%.,:/-]', ' ', (s or '').lower())).strip()


class Ledger:
    def __init__(self, con, raw, max_turn=None):
        """raw(turn_id) -> (role, text, day) or None.
        max_turn: when set, turn_ids after the current step are rejected. Absent / None = legacy."""
        self.con, self.raw = con, raw
        self.max_turn = int(max_turn) if max_turn is not None else None
        con.executescript(SCHEMA)

    def _check_turn_bound(self, turn_id):
        if self.max_turn is None:
            return
        m = re.fullmatch(r't?(\d+)', str(turn_id))
        if m and int(m.group(1)) > self.max_turn:
            raise ValueError(
                f'turn t{m.group(1)} is after the current step; only turns ≤ t{self.max_turn} are available')

    # ---------- writes ----------
    def apply(self, op, args, step=None, actor=None, why=None):
        try:
            fn = {'resolve_entity': self.resolve_entity, 'append_assertion': self.append_assertion}.get(op)
            if fn is None:
                raise ValueError(f'unknown op {op}')
            res, ok = fn(**{k: v for k, v in args.items() if k != 'why'}), True
        except (ValueError, TypeError, KeyError) as e:
            res, ok = {'error': f'{type(e).__name__}: {e}'}, False
        self.con.execute('INSERT INTO al_log(step, op, args, why, ok, result, actor) VALUES (?,?,?,?,?,?,?)',
                         (step, op, json.dumps(args, ensure_ascii=False), why or args.get('why'), int(ok),
                          json.dumps(res, ensure_ascii=False), actor))
        self.con.commit()
        return {'ok': ok, **res}

    def resolve_entity(self, surface_name, type='person', turn_id=None, aliases=None):
        if type not in TYPES:
            raise ValueError(f'type must be one of {TYPES}')
        if turn_id is not None:
            self._check_turn_bound(turn_id)
        n = norm(surface_name)
        if not n:
            raise ValueError('empty name')
        r = self.con.execute('SELECT id, name FROM al_entity WHERE norm=?', (n,)).fetchone()
        if r:
            return {'entity_id': r[0], 'name': r[1], 'existed': True}
        for i, name, al in self.con.execute('SELECT id, name, aliases FROM al_entity'):     # explicit aliases only
            if n in [norm(a) for a in json.loads(al or '[]')]:
                return {'entity_id': i, 'name': name, 'existed': True, 'via_alias': True}
        cur = self.con.execute('INSERT INTO al_entity(name, norm, type, aliases, first_turn) VALUES (?,?,?,?,?)',
                               (surface_name.strip(), n, type, json.dumps(aliases or []), turn_id))
        return {'entity_id': cur.lastrowid, 'name': surface_name.strip(), 'existed': False}

    def append_assertion(self, turn_id, entity_id, facet, quote, modality, polarity='assert', value_text=None,
                         number=None, unit=None, scope='', about_time=None, topic_key=None, mentioned=None):
        if modality not in MODALITY:
            raise ValueError(f'modality must be one of {MODALITY}')
        if polarity not in POLARITY:
            raise ValueError('polarity must be assert or deny')
        if facet not in FACETS:
            raise ValueError(f'facet must be one of {FACETS} (use other if none fits)')
        self._check_turn_bound(turn_id)
        r = self.raw(turn_id)
        if r is None:
            raise ValueError(f'turn {turn_id} not found')
        role, text, day = r
        if modality.startswith('user_') and role != 'user':
            raise ValueError(f'{modality} needs a user turn; {turn_id} is {role}')
        if modality.startswith('assistant_') and role != 'assistant':
            raise ValueError(f'{modality} needs an assistant turn; {turn_id} is {role}')
        if not quote or norm(quote) not in norm(text):
            raise ValueError('quote must be an exact substring of the turn (copy it, do not paraphrase)')
        for n in re.findall(r'\d[\d,.]*', f'{value_text or ""} {"" if number is None else number}'):
            n = n.rstrip('.,')
            cand = {n, n.replace(',', ''), re.sub(r'\.0+$', '', n)}
            if not any(c and (c in quote or c in quote.replace(',', '')) for c in cand):
                raise ValueError(f'number {n} does not appear in the quote')
        if about_time and not re.fullmatch(r'\d{4}(-\d\d(-\d\d)?)?', str(about_time)):
            raise ValueError('about_time must be YYYY, YYYY-MM or YYYY-MM-DD')
        eid = int(str(entity_id).lstrip('e:'))
        if not self.con.execute('SELECT 1 FROM al_entity WHERE id=?', (eid,)).fetchone():
            raise ValueError(f'unknown entity_id {entity_id}; resolve_entity first')
        seq = int(re.sub(r'\D', '', str(turn_id)) or 0)
        cur = self.con.execute(
            'INSERT INTO al_assertion(turn_id, said_at, role, about_time, entity_id, facet, value_text, number, unit, '
            'scope, modality, polarity, quote, topic_key, mentioned, seq) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (str(turn_id), day, role, about_time, eid, facet, value_text, number, unit, (scope or '').strip().lower(),
             modality, polarity, quote, (topic_key or '').strip().lower() or None, json.dumps(mentioned or []), seq))
        return {'assertion_id': cur.lastrowid}

    # ---------- views (code only) ----------
    def _rows(self):
        names = dict(self.con.execute('SELECT id, name FROM al_entity'))
        cols = ('id', 'turn_id', 'said_at', 'about_time', 'entity_id', 'facet', 'value_text', 'number', 'unit',
                'scope', 'modality', 'polarity', 'quote', 'topic_key', 'seq')
        rows = [dict(zip(cols, r)) for r in self.con.execute(f"SELECT {', '.join(cols)} FROM al_assertion ORDER BY seq, id")]
        for r in rows:
            r['entity'] = names.get(r['entity_id'], '?')
        return rows

    def views(self):
        rows = self._rows()
        val = lambda r: r['value_text'] or (f"{r['number']:g} {r['unit'] or ''}".strip() if r['number'] is not None else r['quote'])
        chains = defaultdict(list)
        for r in rows:
            if r['modality'] in ('user_claim', 'user_decision', 'user_preference', 'third_party') and r['facet'] != 'other':
                chains[(r['entity'], r['facet'], r['scope'])].append(r)
        current = []
        for (ent, facet, scope), ch in chains.items():
            asserts = [r for r in ch if r['polarity'] == 'assert' and r['modality'] != 'third_party']
            denies = [r for r in ch if r['polarity'] == 'deny']
            latest = asserts[-1] if asserts else None
            distinct = {norm(val(r)) for r in asserts}
            conflict = bool(denies) or len(distinct) > 1
            current.append({'entity': ent, 'facet': facet, 'scope': scope, 'value': val(latest) if latest else None,
                            'turn': latest['turn_id'] if latest else None, 'conflict': conflict,
                            'history': [(r['said_at'], r['turn_id'], ('DENY ' if r['polarity'] == 'deny' else '') +
                                         (f"[{r['modality']}] " if r['modality'] != 'user_claim' else '') + val(r))
                                        for r in ch]})
        loops = defaultdict(list)
        for r in rows:
            if r['topic_key']:
                loops[r['topic_key']].append(r)
        loop_view = []
        for k, ch in loops.items():
            mods = [r['modality'] for r in ch]
            status = ('committed' if 'user_decision' in mods else 'open_proposal' if 'user_proposal' in mods else
                      'advised' if 'assistant_advice' in mods else 'noted')
            loop_view.append({'topic_key': k, 'status': status, 'items': [
                (r['said_at'], r['turn_id'], r['modality'], ('DENY ' if r['polarity'] == 'deny' else '') + val(r)) for r in ch]})
        instructions = [{'turn': r['turn_id'], 'said_at': r['said_at'], 'text': r['quote']}
                        for r in rows if r['modality'] == 'user_instruction' and r['polarity'] == 'assert']
        sessions = defaultdict(list)
        for r in rows:
            sessions[r['said_at']].append(r['id'])
        advice = [r for r in rows if r['modality'] in ('assistant_advice', 'assistant_example')]
        return {'current': current, 'loops': loop_view, 'instructions': instructions,
                'sessions': {d: ids for d, ids in sorted(sessions.items())}, 'advice': advice, 'assertions': rows}

    def items(self):
        """Retrievable spans for the hybrid index: one per current row (with history + conflict flag), loop,
        instruction, assistant advice/example; assertions with facet=other individually."""
        v, out = self.views(), []
        for c in v['current']:
            h = ' → '.join(f'{v_} ({d}, {t})' for d, t, v_ in c['history'])
            out.append({'layer': 'current' + (':conflict' if c['conflict'] else ''), 'id': f"c:{c['entity']}:{c['facet']}:{c['scope']}",
                        'text': f"{c['entity']} — {c['facet']}{' [' + c['scope'] + ']' if c['scope'] else ''}: "
                                f"current {c['value']}{' (CONFLICT)' if c['conflict'] else ''}; history: {h}",
                        'evidence': [t for _d, t, _v in c['history']], 'day': c['history'][-1][0]})
        for l in v['loops']:
            out.append({'layer': f"loop:{l['status']}", 'id': f"l:{l['topic_key']}",
                        'text': f"{l['topic_key']} ({l['status']}): " + ' | '.join(f'{d} {m}: {x}' for d, _t, m, x in l['items']),
                        'evidence': [t for _d, t, _m, _x in l['items']], 'day': l['items'][-1][0]})
        for i in v['instructions']:
            out.append({'layer': 'instruction', 'id': f"i:{i['turn']}", 'text': f"Standing instruction ({i['said_at']}): {i['text']}",
                        'evidence': [i['turn']], 'day': i['said_at']})
        for r in v['advice']:
            out.append({'layer': r['modality'], 'id': f"a:{r['id']}", 'text': f"{r['said_at']} assistant "
                        f"{'example' if r['modality'] == 'assistant_example' else 'advice'} on {r['entity']}/{r['facet']}: {r['quote']}",
                        'evidence': [r['turn_id']], 'day': r['said_at']})
        for r in v['assertions']:
            if r['facet'] == 'other' and r['modality'].startswith('user_'):
                out.append({'layer': 'assertion', 'id': f"s:{r['id']}", 'text': f"{r['said_at']} {r['entity']}: {r['quote']}",
                            'evidence': [r['turn_id']], 'day': r['said_at']})
        return out
