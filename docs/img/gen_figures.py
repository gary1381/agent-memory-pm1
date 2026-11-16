"""Draw the figures in docs/img/ (standard library only): python3 docs/img/gen_figures.py

architecture.svg  the pipeline from capture to MCP results
locomo_4k.svg     LoCoMo accuracy at a 4K-token budget and the LLM calls each system needs to ingest

Numbers are copied from docs/RESULTS.md; change them there first.
"""
import html
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = 'Helvetica, Arial, sans-serif'
C = {  # fill, stroke, text
    'live': ('#E1F5EE', '#0F6E56', '#04342C'),
    'off': ('#F1EFE8', '#888780', '#444441'),
    'info': ('#E6F1FB', '#185FA5', '#042C53'),
}


def e(s):
    return html.escape(s, quote=False)


class Svg:
    def __init__(self, w):
        self.w, self.out = w, []

    def band(self, y, h, title):
        self.out.append(f'<rect x="20" y="{y}" width="{self.w - 40}" height="{h}" rx="12" fill="none" '
                        'stroke="#B4B2A9" stroke-width="0.8" stroke-dasharray="4 3"/>')
        self.out.append(f'<text x="34" y="{y + 20}" font-size="13" font-weight="600" fill="#5F5E5A">{e(title)}</text>')

    def box(self, x, y, w, h, lines, kind='live', size=12):
        f, s, t = C[kind]
        self.out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{h}" rx="8" fill="{f}" stroke="{s}" '
                        'stroke-width="1"/>')
        n, lh = len(lines), size + 4
        y0 = y + h / 2 - (n - 1) * lh / 2 + size / 2 - 1
        for i, line in enumerate(lines):
            wgt = ' font-weight="600"' if i == 0 and n > 1 else ''
            self.out.append(f'<text x="{x + w / 2:.1f}" y="{y0 + i * lh:.1f}" text-anchor="middle" font-size="{size}"'
                            f'{wgt} fill="{t}">{e(line)}</text>')

    def row(self, y, h, items, gap=10, x0=34):
        total = self.w - 68
        w = (total - gap * (len(items) - 1)) / len(items)
        for i, it in enumerate(items):
            self.box(x0 + i * (w + gap), y, w, h, list(it[1:]), it[0])

    def text(self, x, y, s, size=12, fill='#5F5E5A', anchor='start', weight=''):
        wgt = f' font-weight="{weight}"' if weight else ''
        self.out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{wgt}>'
                        f'{e(s)}</text>')

    def write(self, name, h, title):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{h}" viewBox="0 0 {self.w} {h}" '
               f'font-family="{FONT}">\n<title>{e(title)}</title>\n'
               f'<rect width="{self.w}" height="{h}" fill="#FFFFFF"/>\n' + '\n'.join(self.out) + '\n</svg>\n')
        with open(os.path.join(HERE, name), 'w') as f:
            f.write(svg)


def architecture():
    s = Svg(1000)
    s.text(30, 34, 'pm1 (Precision Memory): raw records are the evidence; no LLM at write time, no generative model at read time', 17,
           '#2C2C2A', weight='700')
    y = 56
    s.band(y, 78, '1  Sources (captured read-only)')
    s.row(y + 30, 38, [
        ('live', 'Claude Code', 'transcripts + hooks'),
        ('live', 'Codex', 'session files'),
        ('live', 'Cursor', 'chat database'),
        ('live', 'Cowork', 'exported sessions'),
        ('live', 'Document folders', 'MD, PDF, DOCX, XLSX'),
    ])
    y = 146
    s.band(y, 112, '2  Ingestion (no LLM)')
    s.row(y + 30, 38, [
        ('live', 'Redact secrets', 'before storage'),
        ('live', 'Split into spans', 'at most 1,200 chars'),
        ('live', 'SQLite store', 'raw records only'),
        ('live', 'FTS5 BM25 index', 'Porter stemming, jieba'),
        ('live', 'e5-small vectors', 'CPU ONNX or GGUF service'),
    ])
    s.row(y + 76, 28, [
        ('info', 'records grouped by session and project'),
        ('info', 'vectors filled in by a background worker'),
        ('off', 'LLM fact index: eval only, off'),
    ])
    y = 270
    s.band(y, 118, '3  Retrieval: search_sessions (production)')
    s.row(y + 30, 40, [
        ('live', 'zh-en translation', 'optional keyword lane'),
        ('live', 'BM25', 'common words dropped'),
        ('live', 'e5 dense', 'exact cosine scan'),
        ('live', 'RRF fusion', 'pool of 100'),
        ('live', 'Qwen3-Reranker-0.6B', 'all 100, waits up to 8 s'),
        ('live', 'Pack to budget', '4K tokens, by session'),
    ])
    s.text(34, y + 92, 'Without a reranker the fused order is kept: CPU-only search over about 745 spans takes 6 ms '
           'p50 on two threads.')
    s.text(34, y + 108, 'With the reranker: production p50 3.7 s per call, almost all of it reranking.')
    y = 400
    s.band(y, 96, '4  MCP server (stdio) and governance')
    s.row(y + 30, 28, [
        ('live', 'search_sessions, open_record, session_brief, code_history'),
        ('live', 'search_docs, cite, save_note, search_notes'),
    ])
    s.row(y + 62, 28, [
        ('live', 'results wrapped as evidence, not instructions'),
        ('live', 'rules: agents propose, the user approves'),
        ('live', 'one local SQLite file'),
    ])
    y = 508
    s.band(y, 100, '5  Tested and not adopted (pre-registered rule not met, or cost too high)')
    s.row(y + 30, 28, [
        ('off', '1024-d / 2560-d vectors'),
        ('off', 'Qwen3-Reranker-4B'),
        ('off', 'graph two-hop recall'),
        ('off', 'second-round recall'),
    ])
    s.row(y + 62, 28, [
        ('off', 'planning before search'),
        ('off', 'budgets above 4K'),
        ('off', 'grep-style literal channel'),
        ('off', 'LLM fact index (eval only)'),
    ])
    s.text(34, y + 112, 'Green: in production. Blue: notes. Grey: off by default or not adopted. Details: '
           'docs/DESIGN.md, docs/EVALUATION.md.', 11, '#888780')
    s.write('architecture.svg', y + 124, 'pm1 architecture')


def locomo():
    rows = [  # label, 4K accuracy (%), LLM calls to ingest, is pm1
        ('pm1 + LLM fact index (eval variant)', 92.2, '275', True),
        ('Mem0 2.2.1', 90.6, '6,453', False),
        ('pm1 production, LLM-free', 89.4, '0', True),
        ('Hindsight 1.0.0', 85.7, '742', False),
        ('Graphiti 0.31.2', 71.6, '18,835', False),
        ('graphify 0.9.74', 56.0, '302', False),
        ('OpenViking 0.5.2', 47.4, '1,002', False),
    ]
    W, x0, x1, top, bh, gap = 860, 270, 700, 84, 26, 12
    s = Svg(W)
    s.text(24, 30, 'LoCoMo, 1,540 questions, 4K-token budget: answer accuracy', 16, '#2C2C2A', weight='700')
    s.text(24, 50, 'Same answer model (GPT-6-Luna), same Mem0 answer prompt, blind judge (composer-2.5)', 12)
    s.text(x1 + 70, top - 26, 'LLM calls to ingest', 12, '#5F5E5A', 'middle', '600')

    def X(v):
        return x0 + (x1 - x0) * v / 100

    h = top + len(rows) * (bh + gap)
    for t in range(0, 101, 20):
        s.out.append(f'<line x1="{X(t):.1f}" y1="{top - 2}" x2="{X(t):.1f}" y2="{h - gap + 2}" stroke="#E4E2DA" '
                     'stroke-width="1"/>')
        s.text(X(t), h + 8, f'{t}%', 11, '#888780', 'middle')
    for i, (label, v, calls, ours) in enumerate(rows):
        y = top + i * (bh + gap)
        fill = '#0F6E56' if ours else '#A9A79E'
        s.text(x0 - 10, y + bh / 2 + 4, label, 12, '#2C2C2A' if ours else '#444441', 'end', '600' if ours else '')
        s.out.append(f'<rect x="{x0}" y="{y}" width="{X(v) - x0:.1f}" height="{bh}" rx="3" fill="{fill}"/>')
        s.text(X(v) - 6, y + bh / 2 + 4, f'{v:.1f}%', 12, '#FFFFFF' if ours else '#2C2C2A', 'end', '600')
        s.text(x1 + 70, y + bh / 2 + 4, calls, 12, '#2C2C2A', 'middle', '600' if calls == '0' else '')
    xb = X(94.9)
    s.out.append(f'<line x1="{xb:.1f}" y1="{top - 6}" x2="{xb:.1f}" y2="{h - gap + 6}" stroke="#185FA5" '
                 'stroke-width="1.2" stroke-dasharray="4 3"/>')
    s.text(xb, top - 12, 'gold evidence only: 94.9%', 11, '#185FA5', 'middle')
    s.text(24, h + 30, 'Production pm1 and Mem0 come from separate reports (about 1 point apart, no paired test). '
           'The fact-index variant calls', 11, '#5F5E5A')
    s.text(24, h + 45, 'an LLM once per session and is not in production. Source: docs/RESULTS.md.', 11, '#5F5E5A')
    s.write('locomo_4k.svg', h + 58, 'LoCoMo accuracy at 4K tokens')


if __name__ == '__main__':
    architecture()
    locomo()
