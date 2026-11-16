"""Multilingual tokenisation and span splitting (no domain vocabulary)."""
from __future__ import annotations

import re

# Generic English function words only; nothing topic- or dataset-specific.
STOP = frozenset('a an the and or of to in on at for with by from is are was were be been i you he she it we they '
                 'me my your our their this that these those do does did have has had what which who how when '
                 'where why can could would should will just so as if not no yes'.split())
_LATIN = re.compile(r'[a-z0-9]+(?:[\'.-][a-z0-9]+)*')
_CJK = re.compile(r'[㐀-鿿]+')
_SENT_END = re.compile(r'(?<=[.!?。！？\n])\s*')
MAX_SPAN_CHARS = 1200


def tokenize(text: str) -> list[str]:
    """Lower-cased latin words + jieba words and character bigrams for CJK runs."""
    text = text.lower()
    toks = [t for t in _LATIN.findall(text) if t not in STOP]
    runs = _CJK.findall(text)
    if runs:
        import jieba
        for run in runs:
            toks += [w for w in jieba.lcut(run) if w.strip()]
            toks += [run[i:i + 2] for i in range(len(run) - 1)]
    return toks


def split_spans(text: str, max_chars: int = MAX_SPAN_CHARS) -> list[tuple[int, int]]:
    """Character ranges covering ``text``; short texts stay whole, long ones break at sentence ends."""
    if len(text) <= max_chars:
        return [(0, len(text))]
    spans, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max((m.end() for m in _SENT_END.finditer(window) if m.end() > max_chars // 2), default=0)
            if cut:
                end = start + cut
        spans.append((start, end))
        start = end
    return spans
