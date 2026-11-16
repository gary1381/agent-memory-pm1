"""pm1-docs: operator CLI for the document layer (same database as the pm1 MCP server).

  pm1-docs add ~/notes [--scope docs]      register a folder (files stay in place)
  pm1-docs scan [--root ID]                index new / changed files, retire deleted ones
  pm1-docs watch [--interval 120]          scan periodically (cheap: unchanged files are skipped by size+mtime)
  pm1-docs status | roots | disable ID
  pm1-docs search "query" [-k 6]           same search as the MCP tool search_docs
  pm1-docs cite ev_...                     resolve an evidence id

Env: PM1_DB, PM1_BANK (personal), PM1_DOC_SCOPE (docs), PM1_DENSE / PM1_TOKEN (which vector jobs to queue;
the pm1 worker computes them), PM1_RERANK_* for search.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .paths import default_db
from .docs import DocIndex
from .store import Store

_DENSE = {'local': 'e5-small', 'http': 'e5-small', 'split': 'e5-small-q8', 'none': None}
_TOKEN = {'pplx': 'pplx-0.6b', 'none': None}


def index_models() -> tuple[str, ...]:
    if os.environ.get('PM1_INDEX_MODELS') is not None:
        return tuple(m for m in os.environ['PM1_INDEX_MODELS'].split(',') if m)
    from .worker import default_dense
    d = _DENSE.get(os.environ.get('PM1_DENSE') or default_dense())
    t = _TOKEN.get(os.environ.get('PM1_TOKEN', 'none'))
    return tuple(x for x in (d and f'dense:{d}', t and f'tok:{t}') if x)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog='pm1-docs')
    ap.add_argument('--db', default=default_db())
    ap.add_argument('--bank', default=os.environ.get('PM1_BANK', 'personal'))
    ap.add_argument('--scope', default=os.environ.get('PM1_DOC_SCOPE', 'docs'))
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('add').add_argument('path')
    sub.add_parser('scan').add_argument('--root', type=int)
    w = sub.add_parser('watch')
    w.add_argument('--interval', type=float, default=120)
    sub.add_parser('status')
    sub.add_parser('roots')
    sub.add_parser('disable').add_argument('root_id', type=int)
    s = sub.add_parser('search')
    s.add_argument('query')
    s.add_argument('-k', type=int, default=6)
    s.add_argument('--budget', type=int, default=2000)
    c = sub.add_parser('cite')
    c.add_argument('evidence_id')
    c.add_argument('--context', type=int, default=1)
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(a.db) or '.', exist_ok=True)
    store = Store(a.db, index_models=index_models())
    docs = DocIndex(store)
    out = None
    if a.cmd == 'add':
        out = docs.add_root(a.bank, a.scope, a.path)
    elif a.cmd == 'scan':
        out = docs.scan(a.root)
    elif a.cmd == 'watch':
        while True:
            t0 = time.time()
            res = docs.scan()
            if any(res[k] for k in ('new', 'changed', 'deleted', 'errors')):
                print(time.strftime('%H:%M:%S'), json.dumps(res), f'{time.time() - t0:.1f}s', flush=True)
            time.sleep(a.interval)
    elif a.cmd == 'status':
        out = docs.status() | {'index': store.stats(a.bank, a.scope)}
    elif a.cmd == 'roots':
        out = docs.roots(False)
    elif a.cmd == 'disable':
        out = docs.disable_root(a.root_id)
    elif a.cmd == 'search':
        from .recall import Recaller
        from .rerank import router_from_env
        from .worker import build_embedders, default_dense
        dense, token = build_embedders(os.environ.get('PM1_DENSE') or default_dense(), os.environ.get('PM1_TOKEN', 'none'))
        rec = Recaller(store, dense=dense, token=token, router=router_from_env(),
                       profile=os.environ.get('PM1_RERANK_PROFILE', 'balanced'))
        out = docs.search(rec, a.bank, a.scope, a.query, k=a.k, budget_tokens=a.budget)
    elif a.cmd == 'cite':
        out = docs.cite(a.evidence_id, context=a.context)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
    print()


if __name__ == '__main__':
    main()
