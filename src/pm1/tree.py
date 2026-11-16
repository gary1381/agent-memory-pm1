"""Memory tree v2 (design: project doc claude/design-memory-tree-v2-2026-10-15.md).

Derived layers over the raw records (raw stays authoritative):
  tree_entity   entities with aliases
  tree_version  entity -> attribute -> versions (bi-temporal: valid_from / valid_to, stated_at), asserted_by
                user|assistant, status stated|proposed|accepted|retracted|contradicted, evidence = raw turn ids
  tree_event    dated events linked to entities and to the versions they change
  tree_digest   1-3 line digests of turns, tagged by source (user / assistant / tool)
  tree_rollup   memory-agent summaries per entity / topic / period (derived, never evidence on their own)
  tree_oplog    every operation applied, with its reason and check result (training data + undo trail)

The ONLY way to change the tree is Tree.apply(op). Each op is checked automatically (evidence exists, dates parse,
assistant text never becomes a user-stated value, ...); a failed check rejects the op and is logged.
"""
from __future__ import annotations

import datetime as _dt
import json
import re

SCHEMA = """
CREATE TABLE IF NOT EXISTS tree_entity (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, name TEXT NOT NULL, type TEXT,
  aliases TEXT DEFAULT '[]', merged_into INTEGER);
CREATE TABLE IF NOT EXISTS tree_version (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, entity_id INTEGER NOT NULL, attribute TEXT NOT NULL,
  akey TEXT NOT NULL, value TEXT NOT NULL, valid_from TEXT, valid_to TEXT, stated_at TEXT, asserted_by TEXT NOT NULL,
  status TEXT NOT NULL, caused_by_event INTEGER, evidence TEXT NOT NULL, closed_by INTEGER,
  qualifier TEXT DEFAULT '', value_kind TEXT DEFAULT 'fact', unit TEXT, norm_value REAL, raw_text TEXT,
  date_precision TEXT, synthetic INTEGER DEFAULT 0, sensitive INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_tv ON tree_version(bank, scope, entity_id, akey);
CREATE TABLE IF NOT EXISTS tree_relation (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, src INTEGER NOT NULL, rel TEXT NOT NULL,
  dst INTEGER NOT NULL, status TEXT DEFAULT 'stated', evidence TEXT NOT NULL, valid_to TEXT);
CREATE TABLE IF NOT EXISTS tree_loop (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, text TEXT NOT NULL, kind TEXT DEFAULT 'task',
  owner TEXT, due TEXT, entities TEXT DEFAULT '[]', status TEXT DEFAULT 'open', resolution TEXT, evidence TEXT NOT NULL,
  opened TEXT, closed TEXT);
CREATE TABLE IF NOT EXISTS tree_event (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, at TEXT, when_text TEXT, what TEXT NOT NULL,
  entities TEXT DEFAULT '[]', changes TEXT DEFAULT '[]', evidence TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tree_digest (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, turns TEXT NOT NULL, source TEXT NOT NULL,
  text TEXT NOT NULL, day TEXT);
CREATE TABLE IF NOT EXISTS tree_rollup (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, target TEXT NOT NULL, period TEXT, text TEXT NOT NULL,
  sources TEXT DEFAULT '[]', model TEXT);
CREATE TABLE IF NOT EXISTS tree_oplog (
  id INTEGER PRIMARY KEY, bank TEXT NOT NULL, scope TEXT NOT NULL, step TEXT, op TEXT NOT NULL, args TEXT NOT NULL,
  why TEXT, ok INTEGER NOT NULL, result TEXT, actor TEXT);
"""

STATUSES = {'stated', 'proposed', 'accepted', 'rejected', 'deferred', 'retracted', 'contradicted'}
# teacher consultation 2026-10-15: targets, measurements and progress must not supersede each other
VALUE_KINDS = {'fact', 'preference', 'target', 'measurement', 'estimate', 'claim', 'progress', 'plan'}
DATE_PRECISION = {None, '', 'day', 'week', 'month', 'year', 'range', 'unresolved'}
_STOP = frozenset('the a an of for to in on my our user users current currently its their his her'.split())


def akey(attribute: str) -> str:
    """normalised attribute key: lowercase content words, plural 's' stripped, sorted."""
    ws = [w for w in re.findall(r'[a-z0-9]+', (attribute or '').lower()) if w not in _STOP]
    ws = [w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w for w in ws]
    return ' '.join(sorted(set(ws)))


def _date_ok(s) -> bool:
    if s in (None, ''):
        return True
    try:
        _dt.date.fromisoformat(str(s)[:10])
        return True
    except ValueError:
        return False


class CheckError(ValueError):
    pass


class Tree:
    def __init__(self, store, bank: str, scope: str, raw=None, max_turn=None):
        """raw: callable turn_id -> (role, text, day) or None; used by the evidence checks.
        max_turn: when set, evidence / turns citing a later turn id are rejected (step-bounded builds).
        Absent / None keeps legacy behaviour."""
        self.con, self.bank, self.scope, self.raw = store.con, bank, scope, raw
        self.max_turn = int(max_turn) if max_turn is not None else None
        self.con.executescript(SCHEMA)

    # ---------- helpers ----------
    def _q(self, sql, *a):
        return self.con.execute(sql, a)

    def _entity(self, ref) -> int:
        """entity id from an id or a name / alias (follows merges)."""
        if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
            r = self._q('SELECT id, merged_into FROM tree_entity WHERE id=? AND bank=? AND scope=?', int(ref),
                        self.bank, self.scope).fetchone()
        else:
            r = None
            low = (ref or '').strip().lower()
            for i, name, al, mi in self._q('SELECT id, name, aliases, merged_into FROM tree_entity WHERE bank=? AND scope=?',
                                           self.bank, self.scope):
                if name.lower() == low or low in [a.lower() for a in json.loads(al or '[]')]:
                    r = (i, mi)
                    if mi is None:
                        break
        if r is None:
            raise CheckError(f'unknown entity {ref!r}')
        while r[1] is not None:
            r = self._q('SELECT id, merged_into FROM tree_entity WHERE id=?', r[1]).fetchone()
        return r[0]

    def _evidence(self, ev, need_role=None, value=None):
        ev = [str(e) for e in (ev or [])]
        if not ev:
            raise CheckError('evidence is required')
        if self.max_turn is not None:
            for e in ev:
                m = re.fullmatch(r't?(\d+)', e)
                if m and int(m.group(1)) > self.max_turn:
                    raise CheckError(
                        f'turn t{m.group(1)} is after the current step; only turns ≤ t{self.max_turn} are available')
        if self.raw is None:
            return ev
        found = [self.raw(e) for e in ev]
        if any(f is None for f in found):
            raise CheckError(f'evidence turn not found: {[e for e, f in zip(ev, found) if f is None]}')
        if need_role and not any(f[0] == need_role for f in found):
            raise CheckError(f'needs a {need_role} turn as evidence (assistant text cannot be a user-stated value)')
        if value:
            nums = re.findall(r'\d[\d,.]*', value)
            text = ' '.join(f[1] for f in found).lower()
            if nums and not any(n.rstrip('.,') in text for n in nums):
                raise CheckError(f'value numbers {nums} not found in the evidence')
        return ev

    # ---------- the operation API ----------
    def apply(self, op: dict, step: str | None = None, actor: str | None = None) -> dict:
        name = op.get('op')
        args = {k: v for k, v in op.items() if k not in ('op', 'why')}
        try:
            fn = getattr(self, f'op_{name}', None)
            if fn is None:
                raise CheckError(f'unknown op {name!r}')
            res = fn(**args)
            ok = True
        except (CheckError, TypeError, ValueError, KeyError) as e:   # bad args / unresolved $ref -> rejected, logged
            res, ok = {'error': f'{type(e).__name__}: {e}' if not isinstance(e, CheckError) else str(e)}, False
        self._q('INSERT INTO tree_oplog(bank, scope, step, op, args, why, ok, result, actor) VALUES (?,?,?,?,?,?,?,?,?)',
                self.bank, self.scope, step, str(name), json.dumps(args, ensure_ascii=False), op.get('why'), int(ok),
                json.dumps(res, ensure_ascii=False), actor)
        return {'ok': ok, **res}

    def op_noop(self, **_):
        return {}

    def op_create_entity(self, name, type=None, aliases=None):
        try:
            return {'entity': self._entity(name), 'existed': True}
        except CheckError:
            pass
        cur = self._q('INSERT INTO tree_entity(bank, scope, name, type, aliases) VALUES (?,?,?,?,?)', self.bank,
                      self.scope, name, type, json.dumps(aliases or []))
        return {'entity': cur.lastrowid}

    def op_merge_entities(self, keep, drop, why=None):
        a, b = self._entity(keep), self._entity(drop)
        if a == b:
            raise CheckError('same entity')
        name, al = self._q('SELECT name, aliases FROM tree_entity WHERE id=?', b).fetchone()
        ka = json.loads(self._q('SELECT aliases FROM tree_entity WHERE id=?', a).fetchone()[0] or '[]')
        self._q('UPDATE tree_entity SET aliases=? WHERE id=?', json.dumps(sorted(set(ka + [name] + json.loads(al or '[]')))), a)
        self._q('UPDATE tree_entity SET merged_into=? WHERE id=?', a, b)
        self._q('UPDATE tree_version SET entity_id=? WHERE entity_id=?', a, b)   # undo trail: tree_oplog
        return {'entity': a, 'merged': b}

    def _open(self, eid, key, qualifier='', kind='fact'):
        return self._q("SELECT id FROM tree_version WHERE entity_id=? AND akey=? AND qualifier=? AND value_kind=? "
                       "AND valid_to IS NULL AND closed_by IS NULL AND status IN ('stated','accepted') "
                       "ORDER BY id DESC LIMIT 1", eid, key, qualifier or '', kind or 'fact').fetchone()

    def op_add_version(self, entity, attribute, value, valid_from=None, asserted_by='user', status=None,
                       evidence=None, stated_at=None, caused_by_event=None, qualifier='', value_kind='fact',
                       unit=None, norm_value=None, raw_text=None, date_precision=None, synthetic=False,
                       sensitive=False):
        status = status or ('stated' if asserted_by == 'user' else 'proposed')
        if status not in STATUSES:
            raise CheckError(f'bad status {status!r}')
        if value_kind not in VALUE_KINDS:
            raise CheckError(f'value_kind must be one of {sorted(VALUE_KINDS)}')
        if date_precision not in DATE_PRECISION:
            raise CheckError('bad date_precision')
        if synthetic and status in ('stated', 'accepted'):
            raise CheckError('example / synthetic data never becomes a stated or accepted value')
        if asserted_by not in ('user', 'assistant'):
            raise CheckError('asserted_by must be user or assistant')
        if asserted_by == 'assistant' and status in ('stated', 'accepted'):
            raise CheckError('an assistant value can only be proposed; use promote_proposal when the user accepts it')
        if not (_date_ok(valid_from) and _date_ok(stated_at)):
            raise CheckError('dates must be ISO (YYYY-MM-DD)')
        ev = self._evidence(evidence, need_role='user' if status == 'stated' else None, value=str(value))
        if stated_at is None and self.raw is not None:
            stated_at = max((self.raw(e)[2] or '') for e in ev) or None
        eid, key = self._entity(entity), akey(attribute)
        prev = self._open(eid, key, qualifier, value_kind) if status == 'stated' else None
        cur = self._q('INSERT INTO tree_version(bank, scope, entity_id, attribute, akey, value, valid_from, stated_at, '
                      'asserted_by, status, caused_by_event, evidence, qualifier, value_kind, unit, norm_value, raw_text, '
                      'date_precision, synthetic, sensitive) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', self.bank,
                      self.scope, eid, attribute, key, str(value), valid_from or stated_at, stated_at, asserted_by,
                      status, caused_by_event, json.dumps(ev), qualifier or '', value_kind, unit, norm_value, raw_text,
                      date_precision, int(bool(synthetic)), int(bool(sensitive)))
        vid = cur.lastrowid
        if prev:
            self._q('UPDATE tree_version SET valid_to=?, closed_by=? WHERE id=?', valid_from or stated_at, vid, prev[0])
        return {'version': vid, 'closed': prev[0] if prev else None}

    def op_close_version(self, version, valid_to=None, why=None):
        if not _date_ok(valid_to):
            raise CheckError('dates must be ISO')
        n = self._q('UPDATE tree_version SET valid_to=COALESCE(?, valid_to, stated_at) WHERE id=? AND bank=? AND scope=?',
                    valid_to, int(version), self.bank, self.scope).rowcount
        if not n:
            raise CheckError(f'no version {version}')
        return {'version': int(version)}

    def op_promote_proposal(self, version, evidence=None, valid_from=None):
        r = self._q('SELECT entity_id, akey, status, qualifier, value_kind, synthetic FROM tree_version WHERE id=? AND '
                    'bank=? AND scope=?', int(version), self.bank, self.scope).fetchone()
        if r is None or r[2] not in ('proposed', 'deferred'):
            raise CheckError('promote_proposal needs a proposed (or deferred) version')
        if r[5]:
            raise CheckError('example / synthetic data cannot be promoted')
        ev = self._evidence(evidence, need_role='user')
        prev = self._open(r[0], r[1], r[3], r[4])
        self._q('UPDATE tree_version SET status=?, evidence=?, valid_from=COALESCE(?, valid_from) WHERE id=?', 'accepted',
                json.dumps(sorted(set(json.loads(self._q('SELECT evidence FROM tree_version WHERE id=?', int(version))
                                                 .fetchone()[0]) + ev))), valid_from, int(version))
        if prev:
            self._q('UPDATE tree_version SET valid_to=COALESCE(?, valid_to), closed_by=? WHERE id=?', valid_from,
                    int(version), prev[0])
        return {'version': int(version), 'closed': prev[0] if prev else None}

    def op_set_status(self, version, status, evidence=None):
        """reject / defer / retract a version (e.g. the user turns a proposal down)."""
        if status not in ('rejected', 'deferred', 'retracted'):
            raise CheckError('set_status is for rejected, deferred or retracted')
        self._evidence(evidence, need_role='user')
        n = self._q('UPDATE tree_version SET status=? WHERE id=? AND bank=? AND scope=?', status, int(version),
                    self.bank, self.scope).rowcount
        if not n:
            raise CheckError(f'no version {version}')
        return {'version': int(version)}

    def op_add_relation(self, src, rel, dst, evidence=None, status='stated'):
        if status not in STATUSES:
            raise CheckError(f'bad status {status!r}')
        a, b = self._entity(src), self._entity(dst)
        ev = self._evidence(evidence)
        cur = self._q('INSERT INTO tree_relation(bank, scope, src, rel, dst, status, evidence) VALUES (?,?,?,?,?,?,?)',
                      self.bank, self.scope, a, rel.strip().lower().replace(' ', '_'), b, status, json.dumps(ev))
        return {'relation': cur.lastrowid}

    def op_open_loop(self, text, evidence=None, kind='task', owner=None, due=None, entities=None):
        if kind not in ('task', 'question', 'decision', 'followup', 'issue'):
            raise CheckError('kind must be task, question, decision, followup or issue')
        if not _date_ok(due):
            raise CheckError('dates must be ISO')
        ev = self._evidence(evidence)
        opened = max((self.raw(e)[2] or '') for e in ev) if self.raw else None
        cur = self._q('INSERT INTO tree_loop(bank, scope, text, kind, owner, due, entities, evidence, opened) '
                      'VALUES (?,?,?,?,?,?,?,?,?)', self.bank, self.scope, text, kind, owner, due,
                      json.dumps([self._entity(e) for e in (entities or [])]), json.dumps(ev), opened)
        return {'loop': cur.lastrowid}

    def op_close_loop(self, loop, resolution, evidence=None, status='done'):
        if status not in ('done', 'dropped', 'superseded'):
            raise CheckError('status must be done, dropped or superseded')
        ev = self._evidence(evidence)
        closed = max((self.raw(e)[2] or '') for e in ev) if self.raw else None
        n = self._q('UPDATE tree_loop SET status=?, resolution=?, closed=? WHERE id=? AND bank=? AND scope=?', status,
                    resolution, closed, int(loop), self.bank, self.scope).rowcount
        if not n:
            raise CheckError(f'no loop {loop}')
        return {'loop': int(loop)}

    def op_search(self, query, layer=None, k=8):
        """read-only; keyword overlap stand-in until tree items go through the hybrid index (recall.py)."""
        qw = set(re.findall(r'[a-z0-9]+', query.lower())) - _STOP
        scored = []
        for it in self.items():
            if layer and not it['layer'].startswith(layer):
                continue
            s = len(qw & set(re.findall(r'[a-z0-9]+', it['text'].lower())))
            if s:
                scored.append((s, it))
        scored.sort(key=lambda x: -x[0])
        return {'results': [{'id': it['id'], 'layer': it['layer'], 'text': it['text'][:300]} for _s, it in scored[:k]]}

    def op_mark_contradiction(self, v1, v2, why=None):
        for v in (v1, v2):
            if not self._q('SELECT 1 FROM tree_version WHERE id=?', int(v)).fetchone():
                raise CheckError(f'no version {v}')
        self._q("UPDATE tree_version SET status='contradicted' WHERE id IN (?,?) AND status='stated'", int(v1), int(v2))
        return {'versions': [int(v1), int(v2)]}

    def op_add_event(self, what, evidence=None, at=None, when_text=None, entities=None, changes=None):
        if not _date_ok(at):
            raise CheckError('dates must be ISO')
        ev = self._evidence(evidence)
        eids = [self._entity(e) for e in (entities or [])]
        cur = self._q('INSERT INTO tree_event(bank, scope, at, when_text, what, entities, changes, evidence) '
                      'VALUES (?,?,?,?,?,?,?,?)', self.bank, self.scope, at, when_text, what, json.dumps(eids),
                      json.dumps([int(c) for c in (changes or [])]), json.dumps(ev))
        for c in changes or []:
            self._q('UPDATE tree_version SET caused_by_event=? WHERE id=?', cur.lastrowid, int(c))
        return {'event': cur.lastrowid}

    def op_add_digest(self, turns, source, text):
        if source not in ('user', 'assistant', 'tool'):
            raise CheckError('source must be user, assistant or tool')
        if len(text) > 400:
            raise CheckError('digest longer than 400 chars')
        ev = self._evidence(turns, need_role=source if source != 'tool' else None)
        day = max((self.raw(e)[2] or '') for e in ev) if self.raw else None
        cur = self._q('INSERT INTO tree_digest(bank, scope, turns, source, text, day) VALUES (?,?,?,?,?,?)', self.bank,
                      self.scope, json.dumps(ev), source, text, day)
        return {'digest': cur.lastrowid}

    def op_rollup(self, target, text, period=None, sources=None, model=None):
        cur = self._q('INSERT INTO tree_rollup(bank, scope, target, period, text, sources, model) VALUES (?,?,?,?,?,?,?)',
                      self.bank, self.scope, target, period, text, json.dumps(sources or []), model)
        return {'rollup': cur.lastrowid}

    # ---------- read ----------
    def history(self, entity, attribute) -> list[dict]:
        eid, key = self._entity(entity), akey(attribute)
        rows = self._q('SELECT id, value, valid_from, valid_to, asserted_by, status, evidence FROM tree_version '
                       'WHERE entity_id=? AND akey=? ORDER BY COALESCE(valid_from, stated_at), id', eid, key).fetchall()
        return [dict(zip(('id', 'value', 'valid_from', 'valid_to', 'asserted_by', 'status', 'evidence'), r)) for r in rows]

    def current(self, entity, attribute):
        h = [v for v in self.history(entity, attribute) if v['status'] in ('stated', 'accepted') and v['valid_to'] is None]
        return h[-1] if h else None

    def items(self) -> list[dict]:
        """Every derived item as a retrievable span (for the hybrid index): one per attribute chain, event, digest,
        rollup; text carries the history so keyword and embedding search both see it."""
        out = []
        names = {i: n for i, n in self._q('SELECT id, name FROM tree_entity WHERE bank=? AND scope=?', self.bank, self.scope)}
        chains = {}
        for r in self._q('SELECT id, entity_id, attribute, akey, value, COALESCE(valid_from, stated_at), asserted_by, '
                         'status, evidence, qualifier, value_kind, synthetic FROM tree_version WHERE bank=? AND scope=? '
                         'ORDER BY 6, id', self.bank, self.scope):
            chains.setdefault((r[1], r[3], r[9] or '', r[10] or 'fact'), []).append(r)
        for (eid, _k, qual, kind), ch in chains.items():
            steps = ' → '.join(f"{r[4]} ({r[5]}{', ' + r[6] + ' ' + r[7] if r[7] != 'stated' else ''}"
                               f"{', example' if r[11] else ''})" for r in ch)
            label = ch[-1][2] + (f' [{qual}]' if qual else '') + (f' ({kind})' if kind != 'fact' else '')
            out.append({'layer': 'version', 'id': f'v{ch[-1][0]}', 'text': f'{names.get(eid, "?")} — {label}: {steps}',
                        'evidence': sorted({e for r in ch for e in json.loads(r[8])}), 'day': ch[-1][5]})
        for i, s, rel, d, st, ev in self._q('SELECT id, src, rel, dst, status, evidence FROM tree_relation WHERE bank=? '
                                           'AND scope=?', self.bank, self.scope):
            out.append({'layer': 'relation', 'id': f'l{i}', 'text': f'{names.get(s, "?")} {rel} {names.get(d, "?")}'
                        + (f' ({st})' if st != 'stated' else ''), 'evidence': json.loads(ev), 'day': None})
        for i, text, kind, owner, due, st, res, ev, op in self._q(
                'SELECT id, text, kind, owner, due, status, resolution, evidence, opened FROM tree_loop WHERE bank=? '
                'AND scope=?', self.bank, self.scope):
            out.append({'layer': 'loop', 'id': f'o{i}', 'text': f'[{kind}, {st}] {text}' + (f' (due {due})' if due else '')
                        + (f' (owner {owner})' if owner else '') + (f' → {res}' if res else ''),
                        'evidence': json.loads(ev), 'day': op})
        for i, at, wt, what, ev in self._q('SELECT id, at, when_text, what, evidence FROM tree_event WHERE bank=? AND scope=?',
                                           self.bank, self.scope):
            out.append({'layer': 'event', 'id': f'e{i}', 'text': f"{at or wt or ''}: {what}".strip(': '),
                        'evidence': json.loads(ev), 'day': at})
        for i, turns, src, text, day in self._q('SELECT id, turns, source, text, day FROM tree_digest WHERE bank=? AND scope=?',
                                                self.bank, self.scope):
            out.append({'layer': f'digest:{src}', 'id': f'd{i}', 'text': text, 'evidence': json.loads(turns), 'day': day})
        for i, tgt, per, text in self._q('SELECT id, target, period, text FROM tree_rollup WHERE bank=? AND scope=?',
                                         self.bank, self.scope):
            out.append({'layer': 'rollup', 'id': f'r{i}', 'text': f'{tgt} ({per or "all"}): {text}', 'evidence': [], 'day': per})
        return out
