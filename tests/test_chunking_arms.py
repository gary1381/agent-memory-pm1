import sys
from pathlib import Path

from pm1 import Recaller, Store

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
from chunk_eval import cut_spans  # noqa: E402


def covers(text, spans):
    pos = 0
    for a, b in spans:
        assert a <= pos and b > pos
        pos = b
    assert pos == len(text)


def test_cut_keeps_short_records_whole_and_covers_long_ones():
    assert cut_spans('short.', 1200, 120) == [(0, 6)]
    text = ''.join(f'Sentence number {i} goes here.\n\n' for i in range(300))
    spans = cut_spans(text, 1200, 120)
    covers(text, spans)
    assert all(b - a <= 1200 for a, b in spans)
    assert all(text[b - 1] == '\n' for _, b in spans[:-1])          # broke at a blank line
    assert all(spans[i][1] - spans[i + 1][0] > 0 for i in range(len(spans) - 1))


def test_cut_prefers_a_code_fence_and_survives_one_long_word():
    text = 'word ' * 150 + '\n```\ncode\n```\n' + 'word ' * 150
    assert text[cut_spans(text, 1200, 0)[0][1]:].startswith('```')
    covers('x' * 5000, cut_spans('x' * 5000, 1200, 120))


def store_with(tmp_path, text):
    s = Store(str(tmp_path / 'm.sqlite'))
    s.add_episode(bank='personal', scope='research:llm.c', session='issue-#1', seq=0, role='user', text=text,
                  source='a', occurred_at='2026-01-02T03:04:05')
    return s


def test_header_and_widen(tmp_path):
    long_text = 'alpha beta gamma. ' * 200
    s = store_with(tmp_path, long_text)
    sid = s.span_ids(1)
    r = s.spans('personal', 'research:llm.c', sid)[sid[1]]
    assert s.span_header(r) == 'llm.c · ' + ' '.join(long_text.split())[:80] + ' · 2026-01-02 · user'
    wide = s.widen(r, 2000)
    assert 1900 < len(wide) <= 2004 and r['text'].strip() in wide
    assert s.widen(r, 100) == r['text']


def test_parts_lets_a_second_span_of_a_record_in(tmp_path, monkeypatch):
    s = store_with(tmp_path, 'needle alpha beta. ' * 200)
    assert len(s.span_ids(1)) >= 2
    one = Recaller(s).recall('personal', 'research:llm.c', 'needle', budget_tokens=4000)
    monkeypatch.setenv('PM1_RECALL_PARTS', '2')
    two = Recaller(s).recall('personal', 'research:llm.c', 'needle', budget_tokens=4000)
    assert len(one.items) == 1 and len(two.items) == 2
