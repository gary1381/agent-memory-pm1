"""Document parsing with source locators (ported from knowledge_base/knowledge_hub/files.py, parser
'source-locators-v1' / 'source-locators-html-v2').

Each block carries a locator (line range / page / block / sheet+cell) so a citation can point back into
the original file. Only permissively licensed parsers: pypdf (BSD), python-docx (MIT), openpyxl (MIT),
all imported lazily (optional extra 'docs').
"""
from __future__ import annotations

import io
import logging
import re
from html.parser import HTMLParser

logging.getLogger('pypdf').setLevel(logging.ERROR)   # malformed-but-readable PDFs log a warning per object

EXTENSIONS = {'.txt', '.md', '.html', '.htm', '.pdf', '.docx', '.xlsx'}
BLOCK_CHARS = 1800


def parser_name(suffix: str) -> str:
    return 'source-locators-html-v2' if suffix in ('.html', '.htm') else 'source-locators-v1'


class _VisibleHTML(HTMLParser):
    """Reviewable semantic blocks without browser chrome or scripts."""
    BLOCKS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'pre', 'li', 'dt', 'dd', 'caption', 'th', 'td'}
    SKIP = {'script', 'style', 'svg', 'noscript', 'template'}
    VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source',
            'track', 'wbr'}

    def __init__(self, main_only=False):
        super().__init__(convert_charrefs=True)
        self.main_only, self.tree_depth, self.main_level, self.main_tag = main_only, 0, None, None
        self.skip, self.capture, self.depth, self.parts, self.blocks, self.section = 0, None, 0, [], [], ''

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if self.main_only and self.main_level is None and dict(attrs).get('role', '').lower() == 'main':
            self.main_level, self.main_tag = self.tree_depth, tag
        active = not self.main_only or self.main_level is not None
        if tag not in self.VOID:
            self.tree_depth += 1
        if not active:
            return
        if tag in self.SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        if self.capture is None and tag in self.BLOCKS:
            self.capture, self.depth, self.parts = tag, 1, []
        elif self.capture is not None:
            if tag == 'br':
                self.parts.append('\n')
            if tag not in self.VOID:
                self.depth += 1

    def handle_endtag(self, tag):
        tag = tag.lower()
        active = not self.main_only or self.main_level is not None
        closing_main = (self.main_level is not None and self.tree_depth - 1 == self.main_level
                        and tag == self.main_tag)
        self.tree_depth = max(0, self.tree_depth - 1)
        if not active:
            return
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
            if closing_main:
                self.main_level = self.main_tag = None
            return
        if self.skip or self.capture is None:
            if closing_main:
                self.main_level = self.main_tag = None
            return
        self.depth -= 1
        if self.depth:
            return
        text = '\n'.join(' '.join(line.split()) for line in ''.join(self.parts).splitlines() if line.strip())
        if text:
            if self.capture.startswith('h'):
                self.section = text
            self.blocks.append((self.capture, self.section, text))
        self.capture, self.parts = None, []
        if closing_main:
            self.main_level = self.main_tag = None

    def handle_data(self, data):
        if not self.skip and self.capture is not None:
            self.parts.append(data)


def parse(data: bytes, suffix: str) -> list[dict]:
    """-> [{'body': str, 'locator': dict}] in document order. Raises ValueError for unsupported/empty files."""
    blocks: list[dict] = []

    def add(text, locator):
        if not text or not text.strip():
            return
        for start in range(0, len(text), BLOCK_CHARS):
            body = text[start:start + BLOCK_CHARS]
            blocks.append({'body': body, 'locator': {**locator, 'char_start': start, 'char_end': start + len(body)}})

    if suffix in ('.txt', '.md'):
        text = data.decode('utf-8-sig', errors='replace')
        lines, section, start = [], '', 1
        for i, line in enumerate(text.splitlines(keepends=True), 1):
            if line.startswith('#') and suffix == '.md':
                if lines:
                    add(''.join(lines), {'line_start': start, 'line_end': i - 1, 'section': section})
                lines, start, section = [], i, line.strip('# \r\n')
            lines.append(line)
            if not line.strip():
                add(''.join(lines), {'line_start': start, 'line_end': i, 'section': section})
                lines, start = [], i + 1
        add(''.join(lines), {'line_start': start, 'line_end': len(text.splitlines()), 'section': section})
    elif suffix in ('.html', '.htm'):
        text = data.decode('utf-8-sig', errors='replace')
        has_main = bool(re.search(r"\brole\s*=\s*(['\"])main\1", text, flags=re.IGNORECASE))
        p = _VisibleHTML(main_only=has_main)
        p.feed(text)
        for i, (element, section, body) in enumerate(p.blocks, 1):
            add(body, {'block': i, 'element': element, 'section': section, 'extraction': 'html-visible-text'})
    elif suffix == '.pdf':
        from pypdf import PdfReader
        for i, page in enumerate(PdfReader(io.BytesIO(data)).pages, 1):
            add(page.extract_text() or '', {'page': i, 'extraction': 'pypdf-text', 'ocr': False})
    elif suffix == '.docx':
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        doc, section = Document(io.BytesIO(data)), ''
        for i, block in enumerate(doc.iter_inner_content(), 1):
            if isinstance(block, Paragraph):
                if block.style and block.style.name.startswith('Heading'):
                    section = block.text
                add(block.text, {'block': i, 'section': section})
            elif isinstance(block, Table):
                for j, row in enumerate(block.rows, 1):
                    add('\t'.join(c.text for c in row.cells), {'table_block': i, 'row': j, 'section': section})
    elif suffix == '.xlsx':
        from openpyxl import load_workbook
        formula = load_workbook(io.BytesIO(data), read_only=True, data_only=False)
        values = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        try:
            for sheet in formula:
                if (sheet.max_row or 0) * (sheet.max_column or 0) > 2_000_000:
                    raise ValueError('sheet_too_large')
                cached_rows = values[sheet.title].iter_rows()
                for row in sheet.iter_rows():
                    cached = next(cached_rows)
                    for cell, value in zip(row, cached):
                        if cell.value is None:
                            continue
                        number = value.value if cell.data_type == 'f' else cell.value
                        add(f'{sheet.title}!{cell.coordinate}: {cell.value}'
                            + (f' [cached value: {number}]' if cell.data_type == 'f' else ''),
                            {'sheet': sheet.title, 'cell': cell.coordinate,
                             'formula': cell.value if cell.data_type == 'f' else None})
        finally:
            formula.close()
            values.close()
    else:
        raise ValueError('unsupported_file_type')
    if not blocks:
        raise ValueError('no_extractable_text_requires_ocr_or_review')
    return blocks


def merge_small(blocks: list[dict], target: int = 900) -> list[dict]:
    """Join consecutive small blocks (one-line paragraphs, single cells) up to ~target chars, keeping the
    first locator plus the span of the rest; an evidence unit should be a readable passage, not a line."""
    out: list[dict] = []
    for b in blocks:
        if out and len(out[-1]['body']) + len(b['body']) + 1 <= target and \
                out[-1]['locator'].get('section') == b['locator'].get('section') and \
                out[-1]['locator'].get('sheet') == b['locator'].get('sheet') and \
                out[-1]['locator'].get('page') == b['locator'].get('page'):
            last = out[-1]
            sep = '' if last['body'].endswith('\n') else '\n'
            last['body'] += sep + b['body']
            loc = last['locator']
            if 'line_end' in b['locator']:
                loc['line_end'] = b['locator']['line_end']
            if 'block' in b['locator']:
                loc['block_end'] = b['locator']['block']
            if 'cell' in b['locator']:
                loc['cell_end'] = b['locator']['cell']
            if 'row' in b['locator']:
                loc['row_end'] = b['locator']['row']
            loc.pop('char_start', None), loc.pop('char_end', None)
        else:
            out.append({'body': b['body'], 'locator': dict(b['locator'])})
    return out
