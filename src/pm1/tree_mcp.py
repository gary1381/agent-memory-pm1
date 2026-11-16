"""Minimal stdio MCP server exposing ONLY the memory-tree operations (pm1.tree) — for the Hermes teacher.

No terminal, files or network: every tool maps to Tree.apply (checked + logged) or a read. Standard library only, so
it runs inside the isolated Hermes container with the pm1 source mounted read-only.

Which tree is active is set by the driver in $PM1_TREE_DIR/CURRENT.json: {"conv": "2", "step": "r3:...", "max_turn": 10}. max_turn (optional) bounds raw_turn and evidence.
Trees: $PM1_TREE_DIR/conv<conv>.sqlite (bank $PM1_TREE_BANK); raw turns: $PM1_RAW_DIR/conv<conv>.sqlite (read-only).
"""
import json, os, re, sqlite3, sys
from pathlib import Path

import importlib.util  # noqa: E402  (load tree.py alone: pm1/__init__ pulls numpy via recall)
_spec = importlib.util.spec_from_file_location('pm1_tree', Path(__file__).resolve().parent / 'tree.py')
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Tree = _mod.Tree

TREES = Path(os.environ.get('PM1_TREE_DIR', '/work/trees'))
RAWS = Path(os.environ.get('PM1_RAW_DIR', '/raw'))
BANK = os.environ.get('PM1_TREE_BANK', 'beam10m')
OBJ = {'type': 'object', 'additionalProperties': True}


def _p(props, req=()):
    return {'type': 'object', 'properties': props, 'required': list(req), 'additionalProperties': True}


S, A, I = {'type': 'string'}, {'type': 'array', 'items': {'type': 'string'}}, {'type': ['integer', 'string']}
TOOLS = {
    'tree_search': ('Search your memory tree (version chains, events, loops, relations, digests).',
                    _p({'query': S, 'layer': S, 'k': {'type': 'integer'}}, ['query'])),
    'tree_entities': ('List the entities in your memory tree.', _p({})),
    'raw_turn': ('Read one raw conversation turn by id (e.g. "t123") to check evidence.', _p({'turn': S}, ['turn'])),
    'create_entity': ('Create an entity (returns the existing one if the name or alias exists).',
                      _p({'name': S, 'type': S, 'aliases': A}, ['name'])),
    'merge_entities': ('Merge two entities that are the same real-world object.', _p({'keep': S, 'drop': S}, ['keep', 'drop'])),
    'add_version': ('Add a value for entity+attribute (+qualifier, value_kind). Fields: value_kind fact|preference|'
                    'target|measurement|estimate|claim|progress|plan; asserted_by user|assistant; status stated|proposed;'
                    ' valid_from YYYY-MM-DD; date_precision; unit; raw_text; synthetic; sensitive; evidence [turn ids].',
                    _p({'entity': S, 'attribute': S, 'value': S, 'evidence': A}, ['entity', 'attribute', 'value', 'evidence'])),
    'promote_proposal': ('The user explicitly adopted a proposed version.', _p({'version': I, 'evidence': A}, ['version', 'evidence'])),
    'set_status': ('Mark a version rejected | deferred | retracted (user evidence).',
                   _p({'version': I, 'status': S, 'evidence': A}, ['version', 'status', 'evidence'])),
    'close_version': ('Close a version (valid_to YYYY-MM-DD).', _p({'version': I, 'valid_to': S}, ['version'])),
    'mark_contradiction': ('Two versions contradict each other.', _p({'v1': I, 'v2': I}, ['v1', 'v2'])),
    'add_event': ('Add a dated event; changes = version ids it caused.',
                  _p({'what': S, 'at': S, 'when_text': S, 'entities': A, 'changes': {'type': 'array'}, 'evidence': A},
                     ['what', 'evidence'])),
    'add_relation': ('Relate two entities (uses, part_of, candidate_for, replaces, ...).',
                     _p({'src': S, 'rel': S, 'dst': S, 'evidence': A}, ['src', 'rel', 'dst', 'evidence'])),
    'open_loop': ('Open a task / question / decision / followup / issue.',
                  _p({'text': S, 'kind': S, 'owner': S, 'due': S, 'entities': A, 'evidence': A}, ['text', 'evidence'])),
    'close_loop': ('Close an open loop (done | dropped | superseded).',
                   _p({'loop': I, 'resolution': S, 'status': S, 'evidence': A}, ['loop', 'resolution', 'evidence'])),
    'add_digest': ('1-3 line digest (<=400 chars) of turns; source user|assistant.',
                   _p({'turns': A, 'source': S, 'text': S}, ['turns', 'source', 'text'])),
}


class _Store:
    def __init__(self, path):
        self.con = sqlite3.connect(path)


def _current():
    return json.loads((TREES / 'CURRENT.json').read_text())


def _raw(conv):
    db = sqlite3.connect(f"file:{RAWS / f'conv{conv}.sqlite'}?mode=ro&immutable=1", uri=True)

    def f(tid):
        m = re.fullmatch(r't?(\d+)', str(tid))
        r = db.execute('SELECT role, text, substr(occurred_at,1,10) FROM episodes WHERE id=?',
                       (int(m.group(1)),)).fetchone() if m else None
        return tuple(r) if r else None
    return f


def _bound_err(turn, max_turn):
    if max_turn is None:
        return None
    m = re.fullmatch(r't?(\d+)', str(turn))
    if m and int(m.group(1)) > int(max_turn):
        return (f'turn t{m.group(1)} is after the current step; '
                f'only turns ≤ t{max_turn} are available')
    return None


def call(name, args):
    cur = _current()
    raw = _raw(cur['conv'])
    max_turn = cur.get('max_turn')
    t = Tree(_Store(TREES / f"conv{cur['conv']}.sqlite"), BANK, str(cur['conv']), raw=raw, max_turn=max_turn)
    if name == 'tree_entities':
        return {'entities': [n for (n,) in t.con.execute(
            'SELECT name FROM tree_entity WHERE merged_into IS NULL AND bank=? AND scope=?', (t.bank, t.scope))]}
    if name == 'raw_turn':
        err = _bound_err(args.get('turn'), max_turn)
        if err:
            return {'error': err}
        r = raw(args.get('turn'))
        return {'turn': args.get('turn'), 'role': r[0], 'day': r[2], 'text': r[1][:4000]} if r else {'error': 'not found'}
    op = {'search': 'search'}.get(name, name.replace('tree_', ''))
    args = {k: (int(v[1:]) if isinstance(v, str) and re.fullmatch(r'[vloe]\d+', v) else v) for k, v in args.items()}
    res = t.apply({'op': op, **args}, step=cur.get('step'), actor='hermes')
    t.con.commit()
    return res


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        mid, method = msg.get('id'), msg.get('method')
        if mid is None:
            continue                                   # notifications
        if method == 'initialize':
            result = {'protocolVersion': msg.get('params', {}).get('protocolVersion', '2024-11-05'),
                      'capabilities': {'tools': {}}, 'serverInfo': {'name': 'pm1tree', 'version': '0.1'}}
        elif method == 'tools/list':
            result = {'tools': [{'name': n, 'description': d, 'inputSchema': s} for n, (d, s) in TOOLS.items()]}
        elif method == 'tools/call':
            p = msg.get('params', {})
            try:
                out = call(p.get('name'), p.get('arguments') or {})
                result = {'content': [{'type': 'text', 'text': json.dumps(out, ensure_ascii=False)}],
                          'isError': bool(isinstance(out, dict) and out.get('ok') is False)}
            except Exception as e:                     # noqa: BLE001 — report, never crash the server
                result = {'content': [{'type': 'text', 'text': f'error: {e}'}], 'isError': True}
        elif method == 'ping':
            result = {}
        else:
            sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': mid,
                                         'error': {'code': -32601, 'message': f'unknown method {method}'}}) + '\n')
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': result}) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
