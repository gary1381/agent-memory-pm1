"""Build a Chinese reranking test set from LongMemEval-S dev.

Step 1 (export): for each scorable dev question, the shipped RRF top-K candidate spans with evidence labels.
Step 2 (translate): question + every candidate span -> Simplified Chinese via freellmapi (cached, resumable).
Labels carry over unchanged: a span is evidence in Chinese iff it was evidence in English.

  .venv/bin/python eval/zh_build.py export      # -> eval/data/lme_dev_pool_en.json
  .venv/bin/python eval/zh_build.py translate   # -> eval/data/lme_dev_pool_zh.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
DATADIR = Path(__file__).parent / 'data'
POOL_EN = DATADIR / 'lme_dev_pool_en.json'
POOL_ZH = DATADIR / 'lme_dev_pool_zh.json'
CACHE = Path.home() / '.cache/pm1/zh_translate.sqlite'
MODELS = ['command-a-translate', 'gpt-oss-120b', 'qwen3.8-27b', 'gemini-3.6-flash', 'gemini-3.5-flash', 'command-a', 'aya-expanse-32b', 'gemini-3.5-flash-lite', 'command-r-2']
_cool: dict[str, float] = {}   # model -> time until which it is rate-limited
_rr = [0]
PROMPT = ('Translate each value of this JSON object into natural Simplified Chinese, as a native speaker would say it '
          'in a chat with an AI assistant. Keep every fact, number, date and name; transliterate or keep names as is '
          'when that is customary. Keep code, URLs and markup unchanged; translate only natural-language text. '
          'Return ONLY a JSON object with exactly the same keys.\n')


def export(k=32):
    from lme import DATA, VecCache, fill, ingest, span_texts, split_of
    from pm1 import Recaller
    from pm1.embed import SplitE5
    dense, cache = SplitE5(), VecCache()
    out = []
    for q in json.loads(DATA.read_text()):
        if split_of(q['question_id']) != 'dev' or q['question_id'].endswith('_abs'):
            continue
        store, meta = ingest(q)
        fill(store, dense, 'dense', cache)
        order, _, _ = Recaller(store, dense=dense).rank('lme', q['question_id'], q['question'], 'rrf')
        head = order[:k]
        texts = span_texts(store, set(head))
        info = {sid: (role, when) for sid, role, when in store.con.execute(
            'SELECT s.id, e.role, e.occurred_at FROM spans s JOIN episodes e ON e.id=s.episode_id')}
        out.append({'qid': q['question_id'], 'type': q['question_type'], 'question': q['question'],
                    'question_date': q['question_date'], 'answer': str(q['answer']),
                    'answer_sessions': q['answer_session_ids'],
                    'cands': [{'sid': s, 'text': texts[s], 'evidence': meta[s]['evidence'],
                               'session': meta[s]['session'], 'role': info[s][0], 'date': info[s][1]} for s in head]})
        store.close()
    DATADIR.mkdir(exist_ok=True)
    POOL_EN.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    n_ev = sum(any(c['evidence'] for c in x['cands']) for x in out)
    print(f'{len(out)} questions, {sum(len(x["cands"]) for x in out)} candidates, evidence in pool for {n_ev}')


_lock = threading.Lock()


def _db():
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(CACHE), check_same_thread=False)
    con.execute('CREATE TABLE IF NOT EXISTS t (h TEXT PRIMARY KEY, src TEXT, zh TEXT, model TEXT)')
    return con


def _cjk_ratio(s):
    letters = [c for c in s if c.isalpha()]
    return sum('一' <= c <= '鿿' for c in letters) / max(1, len(letters))


def _parse(txt):
    txt = re.sub(r'^```(?:json)?|```$', '', txt.strip(), flags=re.M).strip()
    i, j = txt.find('{'), txt.rfind('}')
    return json.loads(txt[i:j + 1])


def _translate_batch(con, items: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Rotate over free models; a model that errors (429/413/bad JSON) cools down for a minute."""
    import time as _t
    from llm import chat
    for _round in range(12):
        with _lock:
            start = _rr[0] = (_rr[0] + 1) % len(MODELS)
        order = MODELS[start:] + MODELS[:start]
        live = [m for m in order if _cool.get(m, 0) < _t.time()]
        for model in live:
            try:
                txt, used = chat([{'role': 'user', 'content': PROMPT + json.dumps(items, ensure_ascii=False)}],
                                 model=model, max_tokens=3000, timeout=120, retries=1)
                got = _parse(txt)
                ok = {k: (str(got[k]), used) for k in items if k in got and str(got[k]).strip()}
                if len(ok) == len(items):
                    return ok
            except Exception:
                pass
            _cool[model] = _t.time() + 60
        _t.sleep(15)
    return {}


NQ, NK = int(os.environ.get('PM1_ZH_NQ', 60)), int(os.environ.get('PM1_ZH_NK', 16))   # default scope agreed 2026-10-12
if (NQ, NK) != (60, 16):
    POOL_ZH = DATADIR / f'lme_dev_pool_zh_{NQ}x{NK}.json'


def _scope(pool):
    return [dict(x, cands=x['cands'][:NK]) for x in pool[:NQ]]


def snapshot(con=None):
    """Write POOL_ZH from whatever is cached; a question is `complete` when it and all its candidates are translated."""
    pool = _scope(json.loads(POOL_EN.read_text()))
    con = con or _db()
    zh = {h: z for h, z in con.execute('SELECT h, zh FROM t')}
    complete = 0
    for x in pool:
        x['question_en'] = x['question']
        x['question'] = zh.get(hashlib.sha1(x['question_en'].encode()).hexdigest())
        for c in x['cands']:
            c['text_en'] = c['text']
            c['text'] = zh.get(hashlib.sha1(c['text_en'].encode()).hexdigest())
            if c['text'] is not None and _cjk_ratio(c['text']) < 0.3:
                c['low_cjk'] = True
        x['complete'] = x['question'] is not None and all(c['text'] is not None for c in x['cands'])
        complete += x['complete']
    POOL_ZH.write_text(json.dumps(pool, ensure_ascii=False, indent=1))
    print(f'snapshot {POOL_ZH}: {complete}/{len(pool)} questions complete', flush=True)
    return complete, len(pool)


def translate(batch_chars=1800, workers=4):
    import time as _t
    pool = _scope(json.loads(POOL_EN.read_text()))
    con = _db()
    texts = {}
    for x in pool:                       # question order: early questions complete first
        texts[hashlib.sha1(x['question'].encode()).hexdigest()] = x['question']
        for c in x['cands']:
            texts[hashlib.sha1(c['text'].encode()).hexdigest()] = c['text']
    have = {h for (h,) in con.execute('SELECT h FROM t')}
    todo = [(h, s) for h, s in texts.items() if h not in have]
    print(f'{len(texts)} texts in scope ({NQ} q x top-{NK}), {len(todo)} to translate', flush=True)
    batches, cur, size = [], {}, 0
    for h, s in todo:
        if cur and size + len(s) > batch_chars:
            batches.append(cur); cur, size = {}, 0
        cur[h] = s; size += len(s)
    if cur:
        batches.append(cur)
    done = [0]

    def store(pairs):
        with _lock:
            for h, s, z, m in pairs:
                con.execute('INSERT OR REPLACE INTO t VALUES (?,?,?,?)', (h, s, z, m))
            con.commit()

    def work(b):
        while True:                      # never give up: quotas reset daily, the job just waits
            res = _translate_batch(con, {str(i): s for i, s in enumerate(b.values())})
            if res:
                store([(h, s, *res[str(i)]) for i, (h, s) in enumerate(b.items())])
                break
            if len(b) > 1:
                for h, s in b.items():
                    work({h: s})
                break
            _t.sleep(300)
        with _lock:
            done[0] += 1
            if done[0] % 10 == 0:
                print(f'{done[0]}/{len(batches)} batches', flush=True)
                snapshot(con)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, batches))
    c, n = snapshot(con)
    print(f'wrote {POOL_ZH}: {c}/{n} complete', flush=True)


def export_todo(scope='scope'):
    """Untranslated texts as JSONL {"id","en"} for an external batch translator."""
    pool = json.loads(POOL_EN.read_text())
    if scope == 'scope':
        pool = _scope(pool)
    con = _db()
    have = {h for (h,) in con.execute('SELECT h FROM t')}
    seen, rows = set(), []
    for x in pool:
        for s in [x['question']] + [c['text'] for c in x['cands']]:
            h = hashlib.sha1(s.encode()).hexdigest()
            if h not in have and h not in seen:
                seen.add(h)
                rows.append({'id': h, 'en': s})
    out = DATADIR / f'zh_todo_{scope}.jsonl'
    out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print(f'{out}: {len(rows)} texts, {sum(len(r["en"]) for r in rows)} chars')


def import_file(path, model='external'):
    """Accepts JSONL/JSON of {"id","zh"} (or {"id","text"}); keeps rows whose id is known and text is non-empty."""
    raw = Path(path).read_text(encoding='utf-8').strip()
    rows = json.loads(raw) if raw.startswith('[') else [json.loads(l) for l in raw.split('\n') if l.strip()]  # not splitlines(): U+2028 inside text
    pool = json.loads(POOL_EN.read_text())
    src = {}
    for x in pool:
        for s in [x['question']] + [c['text'] for c in x['cands']]:
            src[hashlib.sha1(s.encode()).hexdigest()] = s
    con = _db()
    ok = bad = 0
    for r in rows:
        zh = r.get('zh') or r.get('text') or r.get('translation')
        if r.get('id') in src and zh and zh.strip():
            con.execute('INSERT OR REPLACE INTO t VALUES (?,?,?,?)', (r['id'], src[r['id']], zh.strip(), model))
            ok += 1
        else:
            bad += 1
    con.commit()
    print(f'imported {ok}, skipped {bad}')
    snapshot(con)


if __name__ == '__main__':
    cmd, args = sys.argv[1], sys.argv[2:]
    {'export': export, 'translate': translate, 'snapshot': snapshot,
     'export-todo': export_todo, 'import': import_file}[cmd](*args)
