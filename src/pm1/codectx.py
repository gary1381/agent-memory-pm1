"""Code context from a graphify graph: what an agent needs before changing a file or symbol, in one call.

graphify (AST, no LLM) gives nodes with file and line (functions 'f()', methods '.m()', classes 'C', files 'x.py')
and edges with the call / import site: calls, indirect_call, method, contains, imports, imports_from, references,
uses, inherits. This module reads that graph plus the source files of the same tree; no database, no model calls.

  code_context(target)      a symbol: definition and signature, docstring, its class, callers and callees with
                            call sites, tests that use it, other symbols in its file; a file: outline with
                            signatures, who imports it, what it imports, its tests. Several matches: the best one
                            in full, the others listed.
  repo_outline(path, depth) directories and code files with their top-level symbols.

  python -m pm1.codectx serve --root DIR [--graph DIR/graphify-out/graph.json]    stdio MCP server (read-only tools)
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

CALLS = ('calls', 'indirect_call')
MAX_CHARS = 7000
TEST_RE = re.compile(r'(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$')
CODE_EXT = ('.py', '.pyi', '.ts', '.tsx', '.js', '.jsx', '.mjs', '.go', '.rs', '.java', '.kt', '.c', '.cc', '.cpp', '.h',
            '.hpp', '.rb', '.php', '.cs', '.swift', '.scala', '.lua')
SKIP_DIRS = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', 'graphify-out', '.mypy_cache', '.pytest_cache',
             'dist', 'build', '.tox'}


def _line(n: dict) -> int:
    m = re.match(r'L(\d+)', str(n.get('source_location') or ''))
    return int(m.group(1)) if m else 0


def _name(label: str) -> str:
    """'.method()' / 'func()' / 'Class' -> bare name"""
    return label.strip().lstrip('.').removesuffix('()')


def is_test(path: str) -> bool:
    return bool(TEST_RE.search(path or ''))


class Graph:
    def __init__(self, root: str, graph_path: str | None = None):
        self.root = os.path.abspath(root)
        path = graph_path or os.path.join(self.root, 'graphify-out', 'graph.json')
        with open(path) as f:
            g = json.load(f)
        self.nodes = {n['id']: n for n in g.get('nodes', []) if n.get('file_type', 'code') == 'code'}
        self.out, self.inc = defaultdict(list), defaultdict(list)
        for e in g.get('links') or g.get('edges') or []:
            s, t = e.get('source'), e.get('target')
            if s in self.nodes or t in self.nodes:
                self.out[s].append(e)
                self.inc[t].append(e)
        self.by_name = defaultdict(list)
        self.by_file = defaultdict(list)
        for n in self.nodes.values():
            self.by_name[_name(n.get('label', '')).lower()].append(n)
            if n.get('source_file'):
                self.by_file[n['source_file']].append(n)
        self._src: dict[str, list[str]] = {}

    # ---------- helpers ----------
    def lines(self, rel: str) -> list[str]:
        if rel not in self._src:
            try:
                with open(os.path.join(self.root, rel), errors='replace') as f:
                    self._src[rel] = f.read().splitlines()
            except OSError:
                self._src[rel] = []
        return self._src[rel]

    def signature(self, n: dict) -> str:
        """the def / class line(s) up to the colon that ends them (at most 8 lines)"""
        src, i = self.lines(n.get('source_file', '')), _line(n) - 1
        if not (0 <= i < len(src)):
            return ''
        out = []
        for j in range(i, min(i + 8, len(src))):
            out.append(src[j].strip())
            if src[j].rstrip().endswith(':'):
                break
        return ' '.join(out)[:300]

    def docstring(self, n: dict) -> str:
        src, i = self.lines(n.get('source_file', '')), _line(n) - 1
        for j in range(i + 1, min(i + 12, len(src))):
            s = src[j].strip()
            if s.startswith(('"""', "'''", 'r"""')):
                body = s.lstrip('r')[3:]
                return (body.split('"""')[0].split("'''")[0] or (src[j + 1].strip() if j + 1 < len(src) else ''))[:200]
            if s and not s.startswith(('#', '@')) and not src[j].rstrip().endswith((',', '(')) and j > i + 1:
                break
        return ''

    def where(self, n: dict) -> str:
        return f"{n.get('source_file', '?')}:{_line(n)}"

    def file_node(self, rel: str) -> dict | None:
        base = os.path.basename(rel)
        return next((n for n in self.by_file.get(rel, []) if n.get('label') == base), None)

    def resolve_file(self, target: str) -> str | None:
        t = target.strip().lstrip('./')
        if t.startswith(self.root):
            t = os.path.relpath(t, self.root)
        if t in self.by_file:
            return t
        hits = [f for f in self.by_file if f.endswith('/' + t) or f == t]
        if not hits and '/' not in t and t.endswith('.py'):
            hits = [f for f in self.by_file if os.path.basename(f) == t]
        hits.sort(key=lambda f: (is_test(f), len(f)))
        return hits[0] if hits else None

    def resolve_symbol(self, target: str) -> list[dict]:
        t = target.strip()
        cls = None
        if '.' in t and not t.endswith('.py'):          # Class.method or module.func
            cls, t = t.rsplit('.', 1)
        cands = list(self.by_name.get(_name(t).lower(), []))
        if cls:
            owners = {e['target'] for c in self.by_name.get(_name(cls).lower(), []) for e in self.out[c['id']]
                      if e.get('relation') == 'method'}
            narrowed = [c for c in cands if c['id'] in owners or
                        os.path.splitext(os.path.basename(c.get('source_file', '')))[0] == cls.split('.')[-1]]
            cands = narrowed or cands
        exact = [c for c in cands if _name(c.get('label', '')) == _name(t)]
        cands = exact or cands
        cands.sort(key=lambda c: (is_test(c.get('source_file', '')), -len(self.inc[c['id']]), c.get('source_file', '')))
        return cands

    def _edges(self, nid: str, rels, incoming: bool, cap: int) -> list[str]:
        es = self.inc[nid] if incoming else self.out[nid]
        out, seen = [], set()
        for e in es:
            if e.get('relation') not in rels:
                continue
            other = self.nodes.get(e['source'] if incoming else e['target'])
            if other is None:
                continue
            site = f"{e.get('source_file') or other.get('source_file', '?')}:{str(e.get('source_location') or '').lstrip('L')}"
            key = (other['id'], site)
            if key in seen:
                continue
            seen.add(key)
            out.append(f"{other.get('label')} ({self.where(other)})" + ('' if incoming is False else f" at {site}"))
        out.sort(key=lambda s: ('/test' in s or 'test_' in s, s))
        return out[:cap] + ([f'... {len(out) - cap} more'] if len(out) > cap else [])

    def owner(self, nid: str) -> dict | None:
        for e in self.inc[nid]:
            if e.get('relation') == 'method' and e['source'] in self.nodes:
                return self.nodes[e['source']]
        return None

    def tests_mentioning(self, name: str, cap: int = 8) -> list[str]:
        if len(name) < 3:
            return []
        pat = re.compile(r'\b' + re.escape(name) + r'\b')
        hits = []
        for f in sorted(x for x in self.by_file if is_test(x)):
            for i, line in enumerate(self.lines(f), 1):
                if pat.search(line):
                    hits.append(f'{f}:{i}')
                    break
            if len(hits) >= cap:
                break
        return hits

    def text_calls(self, name: str, defined: dict, cap: int = 12) -> list[str]:
        """call sites found by name in non-test code (graphify cannot resolve obj.method() calls): unresolved, so
        another symbol of the same name can match too"""
        if len(name) < 3:
            return []
        pat = re.compile(r'(?<![\w])' + re.escape(name) + r'\s*\(')
        skip = (defined.get('source_file'), _line(defined))
        hits = []
        for f in sorted(x for x in self.by_file if x.endswith(CODE_EXT) and not is_test(x)):
            for i, line in enumerate(self.lines(f), 1):
                st = line.lstrip()
                if (f, i) == skip or st.startswith(('def ', 'async def ', 'class ', '#')):
                    continue
                if pat.search(line):
                    hits.append(f'{f}:{i}: {st[:110]}')
                    if len(hits) > cap:
                        return hits[:cap] + ['... more']
        return hits

    def outline(self, rel: str, with_sig: bool = True, cap: int = 60) -> list[str]:
        syms = [n for n in self.by_file.get(rel, []) if n.get('label') != os.path.basename(rel)]
        methods = {e['target'] for n in syms for e in self.out[n['id']] if e.get('relation') == 'method'}
        nested = {e['target'] for n in syms for e in self.out[n['id']] if e.get('relation') == 'contains'}
        out = []
        for n in sorted(syms, key=_line):
            if n['id'] in nested and n['id'] not in methods:
                continue                                  # inner functions: left out of the outline
            ind = '    ' if n['id'] in methods else ''
            sig = self.signature(n) if with_sig else ''
            out.append(f"{ind}L{_line(n)} {sig or n.get('label')}")
        return out[:cap] + ([f'... {len(out) - cap} more'] if len(out) > cap else [])

    # ---------- tools ----------
    def code_context(self, target: str) -> dict:
        if not target or not target.strip():
            raise ValueError('target is required: a file path or a symbol name')
        rel = self.resolve_file(target) if (target.endswith('.py') or '/' in target) else None
        if rel:
            return self._file_context(rel)
        cands = self.resolve_symbol(target)
        if not cands:
            rel = self.resolve_file(target)
            if rel:
                return self._file_context(rel)
            near = sorted({_name(n.get('label', '')) for k, ns in self.by_name.items()
                           if _name(target).lower() in k for n in ns})[:15]
            return {'target': target, 'found': False, 'similar_names': near,
                    'hint': 'no symbol or file with that name in the code graph; try a similar name, a file path, '
                            'or repo_outline'}
        n = cands[0]
        own = self.owner(n['id'])
        name = _name(n.get('label', ''))
        out = {'target': target, 'kind': 'class' if n.get('_callable_class') else
               ('method' if own else 'function'), 'name': (f"{own.get('label')}." if own else '') + name,
               'defined_at': self.where(n), 'signature': self.signature(n), 'doc': self.docstring(n),
               'called_by': self._edges(n['id'], CALLS, True, 15),
               'calls': self._edges(n['id'], CALLS, False, 15),
               'used_by': self._edges(n['id'], ('references', 'uses', 'inherits'), True, 8),
               'call_sites_by_name': [],
               'tests_mentioning_it': self.tests_mentioning(name)}
        if own or not out['called_by']:
            out['call_sites_by_name'] = self.text_calls(name, n)     # unresolved: matched by name only
        if not out['call_sites_by_name']:
            del out['call_sites_by_name']
        if n.get('_callable_class'):
            out['methods'] = [f"L{_line(m)} {self.signature(m)}" for m in
                              sorted((self.nodes[e['target']] for e in self.out[n['id']]
                                      if e.get('relation') == 'method' and e['target'] in self.nodes), key=_line)][:30]
            out['subclasses'] = self._edges(n['id'], ('inherits',), True, 10)
        out['same_file'] = [s for s in self.outline(n.get('source_file', ''), with_sig=False, cap=25)]
        if len(cands) > 1:
            out['other_matches'] = [f"{c.get('label')} ({self.where(c)})" for c in cands[1:10]]
        return _fit(out)

    def _file_context(self, rel: str) -> dict:
        fn = self.file_node(rel)
        imported_by, imports = [], []
        if fn:
            imported_by = sorted({self.nodes[e['source']].get('source_file', '?') for e in self.inc[fn['id']]
                                  if e.get('relation') in ('imports', 'imports_from') and e['source'] in self.nodes})
            imports = sorted({(self.nodes[e['target']].get('source_file') or self.nodes[e['target']].get('label'))
                              for e in self.out[fn['id']]
                              if e.get('relation') in ('imports', 'imports_from') and e['target'] in self.nodes})
        # files whose symbols call into this file count as users too (graphify links imports to symbols)
        users = set(imported_by)
        for n in self.by_file.get(rel, []):
            for e in self.inc[n['id']]:
                src = self.nodes.get(e['source'])
                if src and src.get('source_file') and src['source_file'] != rel and e.get('relation') in \
                        CALLS + ('imports', 'imports_from', 'uses', 'inherits'):
                    users.add(src['source_file'])
        mod = os.path.splitext(os.path.basename(rel))[0]
        out = {'target': rel, 'kind': 'file', 'lines': len(self.lines(rel)), 'outline': self.outline(rel),
               'used_by_files': sorted(u for u in users if not is_test(u))[:25],
               'tests': sorted({u for u in users if is_test(u)} | {t.split(':')[0] for t in self.tests_mentioning(mod)})[:12],
               'imports': [i for i in imports if i != rel][:25]}
        return _fit(out)

    def repo_outline(self, path: str = '', depth: int = 2, symbols: int = 6) -> dict:
        path = path.strip().strip('/')
        files = sorted(f for f in self.by_file if f.endswith(CODE_EXT) and
                       (not path or f == path or f.startswith(path + '/')))
        if not files:
            raise ValueError(f'no code files under {path!r}')
        base = path.count('/') + 1 if path else 0
        while True:
            entries: dict[str, str] = {}
            dirs: dict[str, int] = defaultdict(int)
            for f in files:
                parts = f.split('/')
                if len(parts) - base > depth:
                    dirs['/'.join(parts[:base + depth])] += 1
                    continue
                fn = self.file_node(f)
                inner = {m['id'] for m in self.by_file[f] if fn is None or m['id'] != fn['id']}
                # top level: not the file node, not a method, not inside another function or class of this file
                tops = [n for n in self.by_file[f] if n['id'] in inner and not self.owner(n['id'])
                        and not any(e.get('relation') == 'contains' and e['source'] in inner for e in self.inc[n['id']])]
                tops.sort(key=_line)
                names = ', '.join(_name(n.get('label', '')) for n in tops[:symbols])
                more = f' +{len(tops) - symbols}' if len(tops) > symbols else ''
                entries[f] = f"{f} ({len(self.lines(f))} lines): {names}{more}"
            for d, k in dirs.items():
                entries[d + '/'] = f"{d}/ ... {k} code files (repo_outline path='{d}')"
            tree = [entries[k] for k in sorted(entries)]
            out = {'root': path or '.', 'files': len(files), 'depth': depth, 'tree': tree}
            if depth <= 1 or len(json.dumps(out, ensure_ascii=False)) <= MAX_CHARS:
                return _fit(out)
            depth -= 1                                   # a big tree: fewer levels, more directory summaries


def _fit(out: dict) -> dict:
    """keep the answer under MAX_CHARS by trimming the longest lists"""
    while len(json.dumps(out, ensure_ascii=False)) > MAX_CHARS:
        k = max((k for k, v in out.items() if isinstance(v, list) and len(v) > 3), default=None,
                key=lambda k: len(json.dumps(out[k], ensure_ascii=False)))
        if k is None:
            break
        out[k] = out[k][:max(3, len(out[k]) * 2 // 3)] + ['... (trimmed)']
    return out


# ---------------------------------------------------------------- MCP server
TOOLS = [
    {'name': 'code_context',
     'description': 'Before changing code: one call returns what you need about a function, class, method or file. '
                    'Symbol: definition (file:line), signature, docstring, its class, callers and callees with call '
                    'sites, tests that use it, other symbols in its file. File: outline with signatures, files that '
                    'use it, its tests, its imports. Accepts a name (parse_args, Store.add_episode) or a path.',
     'inputSchema': {'type': 'object', 'properties': {
         'target': {'type': 'string', 'description': 'symbol name, Class.method, or file path'}},
         'required': ['target']}},
    {'name': 'repo_outline',
     'description': 'Map of the code: directories and code files with their line counts and top-level functions '
                    'and classes. Use it first in an unfamiliar repo, or to find where something lives.',
     'inputSchema': {'type': 'object', 'properties': {
         'path': {'type': 'string', 'description': 'sub-directory to show (default: whole repo)'},
         'depth': {'type': 'integer', 'minimum': 1, 'maximum': 6, 'default': 2}}}},
]
RO = {'readOnlyHint': True, 'openWorldHint': False}


def serve(g: Graph, stdin=sys.stdin, stdout=sys.stdout) -> None:
    for line in stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        mid, method = msg.get('id'), msg.get('method')
        if mid is None:
            continue
        if method == 'initialize':
            res = {'protocolVersion': (msg.get('params') or {}).get('protocolVersion') or '2025-06-18',
                   'capabilities': {'tools': {}}, 'serverInfo': {'name': 'pm1-code', 'version': '0.1'}}
        elif method == 'ping':
            res = {}
        elif method == 'tools/list':
            res = {'tools': [{**t, 'annotations': RO} for t in TOOLS]}
        elif method == 'tools/call':
            p = msg.get('params') or {}
            a = p.get('arguments') or {}
            try:
                if p.get('name') == 'code_context':
                    payload = g.code_context(str(a.get('target') or ''))
                elif p.get('name') == 'repo_outline':
                    payload = g.repo_outline(str(a.get('path') or ''), int(a.get('depth') or 2))
                else:
                    raise ValueError(f"unknown tool {p.get('name')}")
                res = {'content': [{'type': 'text', 'text': json.dumps(payload, ensure_ascii=False)}], 'isError': False}
            except (ValueError, KeyError) as exc:
                res = {'content': [{'type': 'text', 'text': f'error: {exc}'}], 'isError': True}
        else:
            stdout.write(json.dumps({'jsonrpc': '2.0', 'id': mid,
                                     'error': {'code': -32601, 'message': f'method not found: {method}'}}) + '\n')
            stdout.flush()
            continue
        stdout.write(json.dumps({'jsonrpc': '2.0', 'id': mid, 'result': res}, ensure_ascii=False) + '\n')
        stdout.flush()


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(prog='pm1-code')
    ap.add_argument('cmd', choices=('serve', 'context', 'outline'))
    ap.add_argument('target', nargs='?', default='')
    ap.add_argument('--root', default='.')
    ap.add_argument('--graph')
    a = ap.parse_args(argv)
    g = Graph(a.root, a.graph)
    if a.cmd == 'serve':
        serve(g)
    elif a.cmd == 'context':
        print(json.dumps(g.code_context(a.target), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(g.repo_outline(a.target), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
