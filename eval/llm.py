"""Minimal client for the local freellmapi (OpenAI-compatible). The key is read from disk and never printed."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = 'http://127.0.0.1:3001/v1/chat/completions'
KEY_FILE = Path.home() / '.config/precision-memory/freellmapi.key'


def _key() -> str:
    m = re.search(r'freellmapi-[A-Za-z0-9_-]{8,}', KEY_FILE.read_text())
    if not m:
        raise RuntimeError(f'no freellmapi key in {KEY_FILE}')
    return m.group(0)


def chat(messages, model='qwen3.8-27b', temperature=0.0, max_tokens=4096, timeout=180, retries=4,
         response_format=None) -> tuple[str, str]:
    """Returns (text, model actually used)."""
    body = {'model': model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens}
    if response_format:
        body['response_format'] = response_format
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), method='POST',
                                         headers={'Content-Type': 'application/json',
                                                  'Authorization': f'Bearer {_key()}'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            return d['choices'][0]['message']['content'] or '', d.get('model', model)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'freellmapi failed after {retries} attempts: {last}')
