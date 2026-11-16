"""All-local answer tier for BEAM: send the exported answer prompts to a local llama.cpp server.

usage: beam_local_answer.py <prompts.jsonl> <out.jsonl> [base_url] [model_label]
Resumable (skips ids already in out). Output rows {"id", "answer"} feed `beam_qa.py import-answers`.
"""
import json, re, sys, time
from pathlib import Path
import httpx

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
url = (sys.argv[3] if len(sys.argv) > 3 else 'http://127.0.0.1:18190').rstrip('/') + '/v1/chat/completions'
label = sys.argv[4] if len(sys.argv) > 4 else 'local'
done = {json.loads(l)['id'] for l in dst.read_text().splitlines() if l.strip()} if dst.exists() else set()
rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
THINK = re.compile(r'<think>.*?</think>', re.S)
with dst.open('a', encoding='utf-8') as f, httpx.Client(timeout=600) as c:
    for r in rows:
        if r['id'] in done:
            continue
        t0 = time.perf_counter()
        body = {'messages': [{'role': 'user', 'content': r['prompt']}], 'temperature': 0, 'max_tokens': 1024,
                'chat_template_kwargs': {'enable_thinking': False}}
        resp = c.post(url, json=body)
        resp.raise_for_status()
        j = resp.json()
        ans = THINK.sub('', j['choices'][0]['message'].get('content') or '').strip()
        f.write(json.dumps({'id': r['id'], 'answer': ans, 'model': label,
                            'prompt_tokens': j.get('usage', {}).get('prompt_tokens'),
                            'sec': round(time.perf_counter() - t0, 1)}, ensure_ascii=False) + '\n')
        f.flush()
        print(r['id'], f'{time.perf_counter() - t0:.1f}s', flush=True)
