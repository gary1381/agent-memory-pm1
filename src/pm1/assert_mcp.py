"""stdio MCP server for the append-only assertion ledger (structure candidate C2) — stdlib only, for the isolated
Hermes teacher. Writes: resolve_entity, append_assertion. Reads: ledger_search (BM25 over code-built views),
ledger_entity (all views for one entity), ledger_entities, raw_turn.
Active conversation: $PM1_AL_DIR/CURRENT.json {"conv": .., "step": .., "max_turn": ..}; store $PM1_AL_DIR/conv<conv>.sqlite;
raw turns $PM1_RAW_DIR/conv<conv>.sqlite (read-only).
"""
import importlib.util, json, math, os, re, sqlite3, sys
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location('pm1_assertions', Path(__file__).resolve().parent / 'assertions.py')
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
Ledger, MODALITY, POLARITY, FACETS, TYPES = _m.Ledger, _m.MODALITY, _m.POLARITY, _m.FACETS, _m.TYPES

ROOT = Path(os.environ.get('PM1_AL_DIR', '/work/ledger'))
RAWS = Path(os.environ.get('PM1_RAW_DIR', '/raw'))
S = {'type': 'string'}


def _p(props, req=()):
    return {'type': 'object', 'properties': props, 'required': list(req), 'additionalProperties': True}


TOOLS = {
    'ledger_entities': ('List entities (id, name, type).', _p({})),
    'ledger_search': ('Search the code-built views (current rows with history/conflicts, loops, instructions, assistant '
                      'advice). Returns ids + text.', _p({'query': S, 'k': {'type': 'integer'}}, ['query'])),
    'ledger_entity': ('All current rows, loops and assertions for one entity id.', _p({'entity_id': {'type': ['integer', 'string']}}, ['entity_id'])),
    'raw_turn': ('Read one raw conversation turn by id (e.g. "t123").', _p({'turn': S}, ['turn'])),
    'resolve_entity': (f'Get or create an entity by EXACT normalized name (no fuzzy merge). type: {"|".join(TYPES)}.',
                       _p({'surface_name': S, 'type': S, 'turn_id': S, 'aliases': {'type': 'array', 'items': S}, 'why': S},
                          ['surface_name', 'turn_id'])),
    'append_assertion': ('The ONLY way to record memory. quote = exact substring of the turn; every number must appear in '
                         f'the quote. modality: {"|".join(MODALITY)}; polarity: assert|deny ("I never..." = deny, never '
                         f'delete); facet: {"|".join(FACETS)}. Optional: value_text, number, unit, scope, about_time '
                         '(YYYY[-MM[-DD]]), topic_key (shared by advice/proposal/decision/denial about one subject).',
                         _p({'turn_id': S, 'entity_id': {'type': ['integer', 'string']}, 'facet': S, 'quote': S, 'modality': S,
                             'polarity': S, 'value_text': S, 'number': {'type': ['number', 'string', 'null']}, 'unit': S,
                             'scope': S, 'about_time': S, 'topic_key': S, 'why': S},
                            ['turn_id', 'entity_id', 'facet', 'quote', 'modality'])),
}


def _ledger():
    cur = json.loads((ROOT / 'CURRENT.json').read_text())
    raw_db = sqlite3.connect(f'file:{RAWS / ("conv" + str(cur["conv"]) + ".sqlite")}?mode=ro&immutable=1', uri=True)   # read-only mount
    max_turn = cur.get('max_turn')

    def raw(tid):
        m = re.fullmatch(r't?(\d+)', str(tid))
        r = raw_db.execute('SELECT role, text, substr(occurred_at,1,10) FROM episodes WHERE id=?',
                           (int(m.group(1)),)).fetchone() if m else None
        return tuple(r) if r else None
    return Ledger(sqlite3.connect(ROOT / f"conv{cur['conv']}.sqlite"), raw, max_turn=max_turn), cur, raw, max_turn


def _bound_err(turn, max_turn):
    if max_turn is None:
        return None
    m = re.fullmatch(r't?(\d+)', str(turn))
    if m and int(m.group(1)) > int(max_turn):
        return (f'turn t{m.group(1)} is after the current step; '
                f'only turns ≤ t{max_turn} are available')
    return None


def _bm25(items, q, k):
    tok = lambda s: re.findall(r'[a-z0-9$]+(?:[.,][0-9]+)*', s.lower())
    docs = [tok(i['layer'] + ' ' + i['text']) for i in items]
    df = Counter(w for d in docs for w in set(d)); n = len(docs); avg = sum(map(len, docs)) / max(1, n)
    out = []
    for it, d in zip(items, docs):
        tf = Counter(d)
        s = sum(math.log(1 + (n - df[w] + .5) / (df[w] + .5)) * tf[w] * 2.2 / (tf[w] + 1.2 * (.25 + .75 * len(d) / avg))
                for w in tok(q) if w in tf)
        if s > 0:
            out.append((s, it))
    out.sort(key=lambda x: -x[0])
    return [{'id': it['id'], 'layer': it['layer'], 'text': it['text'][:500]} for _s, it in out[:k]]


def call(name, a):
    led, cur, raw, max_turn = _ledger()
    if name == 'ledger_entities':
        return {'entities': [{'id': i, 'name': n, 'type': t} for i, n, t in led.con.execute('SELECT id, name, type FROM al_entity')]}
    if name == 'ledger_search':
        return {'results': _bm25(led.items(), a['query'], int(a.get('k', 8)))}
    if name == 'ledger_entity':
        eid = int(str(a['entity_id']).lstrip('e:'))
        name_ = (led.con.execute('SELECT name FROM al_entity WHERE id=?', (eid,)).fetchone() or ['?'])[0]
        v = led.views()
        return {'entity': name_, 'current': [c for c in v['current'] if c['entity'] == name_],
                'assertions': [{k: r[k] for k in ('id', 'turn_id', 'said_at', 'facet', 'scope', 'modality', 'polarity',
                                                  'value_text', 'quote', 'topic_key')} for r in v['assertions']
                               if r['entity_id'] == eid][-40:]}
    if name == 'raw_turn':
        err = _bound_err(a['turn'], max_turn)
        if err:
            return {'error': err}
        r = raw(a['turn'])
        return {'turn': a['turn'], 'role': r[0], 'day': r[2], 'text': r[1][:4000]} if r else {'error': 'not found'}
    return led.apply(name, a, step=cur.get('step'), actor='hermes', why=a.get('why'))


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        mid, method = msg.get('id'), msg.get('method')
        if mid is None:
            continue
        if method == 'initialize':
            result = {'protocolVersion': msg.get('params', {}).get('protocolVersion', '2024-11-05'),
                      'capabilities': {'tools': {}}, 'serverInfo': {'name': 'pm1ledger', 'version': '0.1'}}
        elif method == 'tools/list':
            result = {'tools': [{'name': n, 'description': d, 'inputSchema': s} for n, (d, s) in TOOLS.items()]}
        elif method == 'tools/call':
            p = msg.get('params', {})
            try:
                out = call(p.get('name'), p.get('arguments') or {})
                err = isinstance(out, dict) and out.get('ok') is False
            except Exception as e:  # noqa: BLE001
                out, err = {'error': f'{type(e).__name__}: {e}'}, True
            result = {'content': [{'type': 'text', 'text': json.dumps(out, ensure_ascii=False, default=str)}], 'isError': err}
        elif method == 'ping':
            result = {}
        else:
            sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32601, 'message': method}}) + '\n')
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': result}) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
