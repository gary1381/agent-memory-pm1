"""Query translation for keyword search across languages (2026-10-29).

The author's sessions are mostly Chinese, but agents often search them in English: in a replay of real agent queries 7 of the 9 runs that
missed their evidence sent only English queries for Chinese records (keyword search cannot match those, and the e5
vectors did not bridge them); 1 was the reverse. search_sessions can add the query's translation to its keyword stage:
a query with Chinese in it gets an English translation, an English one a Chinese translation.

One small service holds the models (quickmt en-zh / zh-en, Marian, CC-BY-4.0, CTranslate2 int8, ~0.3 s a query on
2 cores) so the MCP processes stay light:

    python -m pm1.xlate serve [--port 18192]      # POST /translate {"text": ..., "to": "zh"|"en"} -> {"text": ...}

Client: Xlate(url, dirs=...).other(query) returns the translation, or None (direction off, service down, timeout).

Glossary (xlate_glossary.tsv + ~/.config/pm1/xlate_glossary.tsv): fixed terms the small models get wrong (重排 ->
"rearrangements"); a Chinese term is replaced by its English one before translating, and the glossary terms found
in the query are added to the translation in either direction.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = {'zh': os.path.expanduser(os.environ.get('PM1_XLATE_MODEL_ZH', '~/models/quickmt-en-zh')),
          'en': os.path.expanduser(os.environ.get('PM1_XLATE_MODEL_EN', '~/models/quickmt-zh-en'))}
GLOSSARIES = (os.path.join(os.path.dirname(__file__), 'xlate_glossary.tsv'),
              os.path.expanduser('~/.config/pm1/xlate_glossary.tsv'))
_CJK = re.compile(r'[㐀-鿿]')
_ALPHA = re.compile(r'[A-Za-z㐀-鿿]')


def mostly_english(text: str) -> bool:
    letters = _ALPHA.findall(text)
    return bool(letters) and sum(bool(_CJK.match(c)) for c in letters) / len(letters) < 0.1


class Xlate:
    def __init__(self, url: str, timeout: float = 1.5, dirs: str = 'en-zh,zh-en'):
        self.url, self.timeout = url.rstrip('/') + '/translate', timeout
        self.dirs = {d.strip() for d in dirs.split(',')}

    @functools.lru_cache(maxsize=512)
    def other(self, text: str) -> str | None:
        """the query in the other language, when that direction is on"""
        to = 'en' if _CJK.search(text) else 'zh' if mostly_english(text) else None
        if to is None or ('zh-en' if to == 'en' else 'en-zh') not in self.dirs:
            return None
        try:
            req = urllib.request.Request(self.url, json.dumps({'text': text, 'to': to}).encode(),
                                         {'Content-Type': 'application/json'})
            out = json.load(urllib.request.urlopen(req, timeout=self.timeout)).get('text') or ''
        except Exception:
            return None
        return out.strip() or None


class Glossary:
    """zh <-> en terms, read again when a glossary file changes"""

    def __init__(self, paths=GLOSSARIES):
        self.paths, self._seen, self.zh, self.en = paths, None, [], []

    def _load(self) -> None:
        seen = tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in self.paths)
        if seen == self._seen:
            return
        pairs = []
        for p in self.paths:
            if os.path.exists(p):
                for line in open(p, encoding='utf-8'):
                    if line.strip() and not line.startswith('#') and '\t' in line:
                        zh, en = line.rstrip('\n').split('\t', 1)
                        pairs.append((zh.strip(), [e.strip() for e in en.split('|') if e.strip()]))
        self.zh = sorted(pairs, key=lambda x: -len(x[0]))                     # longest first: 重排服务 before 重排
        self.en = sorted(((e, zh) for zh, ens in pairs for e in ens), key=lambda x: -len(x[0]))
        self._seen = seen

    def to_en(self, text: str) -> tuple[str, list[str]]:
        """(text with glossary Chinese replaced by English, every English term found)"""
        self._load()
        found = []
        for zh, ens in self.zh:
            if zh in text:
                text = text.replace(zh, f' {ens[0]} ')
                found += ens
        return text, list(dict.fromkeys(found))

    def to_zh(self, text: str) -> list[str]:
        """Chinese terms for the glossary English words in the text"""
        self._load()
        low, found = text.lower(), []
        for en, zh in self.en:
            if re.search(r'(?<![\w-])' + re.escape(en.lower()) + r'(?![\w-])', low):
                found.append(zh)
        return list(dict.fromkeys(found))


def translate(mt: dict, gl: Glossary, text: str, to: str) -> str:
    text = text[:1000]
    if to == 'en':
        text, extra = gl.to_en(text)
    else:
        extra = gl.to_zh(text)
    tr, src, tgt = mt[to]
    hyp = tr.translate_batch([src.encode(text, out_type=str)], beam_size=2, max_decoding_length=128)[0].hypotheses[0]
    out = tgt.decode(hyp)
    return ' '.join([out] + [e for e in extra if e.lower() not in out.lower()])


def serve(port: int, models: dict[str, str], threads: int) -> None:
    import ctranslate2
    import sentencepiece as spm
    mt = {to: (ctranslate2.Translator(d, device='cpu', compute_type='int8', inter_threads=1, intra_threads=threads),
               spm.SentencePieceProcessor(model_file=os.path.join(d, 'src.spm.model')),
               spm.SentencePieceProcessor(model_file=os.path.join(d, 'tgt.spm.model')))
          for to, d in models.items() if os.path.isdir(d)}
    gl = Glossary()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)))
                text = translate(mt, gl, req['text'], req.get('to') or 'zh')
                body, code = json.dumps({'text': text}, ensure_ascii=False).encode(), 200
            except Exception as exc:
                body, code = json.dumps({'error': str(exc)[:200]}).encode(), 400
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):              # health
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')

        def log_message(self, *a):
            pass

    print(f'pm1-xlate on 127.0.0.1:{port} (to {sorted(mt)})', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog='pm1-xlate')
    ap.add_argument('cmd', choices=['serve'])
    ap.add_argument('--port', type=int, default=18192)
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args(argv)
    serve(a.port, MODELS, a.threads)


if __name__ == '__main__':
    main()
