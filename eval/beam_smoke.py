"""One BEAM-10M question end to end (recall -> answer -> official judge), to validate the pipeline."""
import sys
import time
from pathlib import Path

import beam_qa
from beam_official.prompts import answer_generation_for_rag

from pm1 import Recaller, Store
from pm1.embed import HttpE5
from pm1.rerank import router_from_env

conv, cat, idx = (sys.argv[1:] + ['1', 'information_extraction', '0'])[:3]
q = beam_qa.questions(conv)[cat][int(idx)]
store = Store(Path.home() / f'.cache/pm1/beam10m/conv{conv}.sqlite')
rec = Recaller(store, dense=HttpE5('http://127.0.0.1:18170/v1/embeddings', name='e5-small-q8'),
               router=router_from_env(), k_first=100)
t = time.time()
res = rec.recall('beam', conv, q['question'], budget_tokens=12000, max_items=80)
d = res.diagnostics
print('recall s', round(time.time() - t, 2), d.get('route'), d.get('rerank_model'), d.get('tokens'), len(res.items), flush=True)
gold = set(q.get('source_chat_ids') or [])
print('gold ids', sorted(gold)[:10], 'retrieved:', bool(gold & {int(i['source'].split(':')[1]) for i in res.items}), flush=True)
resp, model = beam_qa.ask(answer_generation_for_rag.replace('<context>', res.context).replace('<question>', q['question']),
                          beam_qa.ANSWER_MODELS)
print('Q:', q['question'])
print('A:', model, '|', resp[:300])
print('gold answer:', q.get('answer') or q.get('ideal_response'))
from beam_official import compute_metrics as cm  # noqa: E402
cm.initialize_models = lambda: None
llm = beam_qa.FreeLLM(beam_qa.JUDGE_MODELS)
fn = getattr(cm, f'evaluate_{cat}')
r = fn(rubric=q['rubric'], llm_response=resp, probing_question=q['question'], model=llm)
print('judge score', r['llm_judge_score'], llm.used)
