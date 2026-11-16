"""Recall-only pass over BEAM-10M (no LLM): writes a recall_meta-style json for beam_coverage.py.

usage: PM1_RERANK_QWEN=... PM1_RERANK_GTE=... .venv-gpu/bin/python eval/beam_recall_eval.py <out.json> [convs...]
env  : PM1_PLANNING=0 disables query planning (baseline), PM1_DIRECTIVES=<n> (default 3)
"""
import json, os, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from beam_qa import BUDGET, CATS, STORES, questions  # noqa: E402
from beam_cfg import conv_ids  # noqa: E402
from pm1 import Recaller, Store  # noqa: E402
from pm1.embed import FastE5  # noqa: E402
from pm1.rerank import router_from_env  # noqa: E402

out = Path(sys.argv[1])
convs = sys.argv[2:] or conv_ids()
meta = json.loads(out.read_text()) if out.exists() else {}
dense = FastE5(gpu=os.environ.get('PM1_GPU', '1') != '0')
router = router_from_env()
for conv in convs:
    store = Store(STORES / f'conv{conv}.sqlite')
    rec = Recaller(store, dense=dense, router=router, k_first=100,
                   profile=os.environ.get('PM1_RERANK_PROFILE', 'balanced'),
                   planning=os.environ.get('PM1_PLANNING', '1') != '0',
                   n_directives=int(os.environ.get('PM1_DIRECTIVES', 3)),
                   timeline_chars=int(os.environ.get('PM1_TIMELINE_CHARS', 700)),
                   multi_cap=int(os.environ.get('PM1_MULTI_CAP', 99)),
                   multi_user_bias=float(os.environ.get('PM1_MULTI_UB', 0.5)),
                   multi_chars=int(os.environ.get('PM1_MULTI_CHARS', 700)),
                   multi_weight=float(os.environ.get('PM1_MULTI_W', 1.5)),
                   directive_min_sim=float(os.environ['PM1_DIR_MIN']) if os.environ.get('PM1_DIR_MIN') else None)
    qs = questions(conv)
    for cat in CATS:
        for i, q in enumerate(qs[cat]):
            t0 = time.perf_counter()
            res = rec.recall('beam', conv, q['question'], budget_tokens=BUDGET, max_items=80)
            d = res.diagnostics
            meta[f'{conv}|{cat}|{i}'] = {
                'recall_ms': round((time.perf_counter() - t0) * 1e3, 1), 'tokens': d.get('tokens'),
                'intent': d.get('intent'), 'mode': d.get('mode'), 'threads': d.get('threads'),
                'directives': d.get('directives'), 'items': len(res.items),
                'source_chat_ids': [int(it['source'].split(':')[1]) for it in res.items if it.get('source')]}
    store.close()
    out.write_text(json.dumps(meta, indent=1))
    ms = [v['recall_ms'] for k, v in meta.items() if k.startswith(f'{conv}|')]
    print(f'conv {conv}: p50 {statistics.median(ms):.0f} ms, max {max(ms):.0f} ms', flush=True)
