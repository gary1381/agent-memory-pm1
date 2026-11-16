"""Write absolute dates next to relative times, from the record's own date (2026-10-31; no model).

"I went yesterday" said on 8 May 2023 -> "I went yesterday [7 May 2023]". An answering model then copies a date instead
of doing the arithmetic, the commonest temporal error on LoCoMo. English and Chinese; vague phrases ("a few days ago")
say so ("[shortly before 8 May 2023]"); phrases that need the speaker's intent ("this weekend", "on Friday") are left.
"""
from __future__ import annotations

import datetime as dt
import re

DAY = dt.timedelta(days=1)
WEEKDAYS = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday')
ZH_WEEKDAYS = '一二三四五六日'
ZH_NUM = {'一': 1, '两': 2, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
          '十一': 11, '十二': 12}
EN_NUM = {'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
          'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'a couple of': 2, 'a couple': 2, 'couple of': 2}
NUM = r'(\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a couple(?: of)?|couple of)'

EN = re.compile(
    r'\b(?:(?P<before>the day before yesterday)|(?P<after>the day after tomorrow)|(?P<yest>yesterday|last night)'
    r'|(?P<tom>tomorrow)|(?P<lastwd>(?:last|this past) (?:' + '|'.join(WEEKDAYS) + r'))'
    r'|(?P<nextwd>next (?:' + '|'.join(WEEKDAYS) + r'))|(?P<lw>last week)|(?P<nw>next week)'
    r'|(?P<lwe>last weekend)|(?P<nwe>next weekend)|(?P<lm>last month)|(?P<nm>next month)|(?P<ly>last year)'
    r'|(?P<ny>next year)|(?P<season>last (?:spring|summer|fall|autumn))'
    r'|(?P<few>(?:a few|several) (?:days|weeks) ago)|(?P<n>' + NUM + r') (?P<unit>day|week|month|year)s? ago)\b',
    re.I)
ZH = re.compile(
    r'(?P<d3>大前天)|(?P<d2>前天)|(?P<yest>昨天|昨晚|昨日|昨夜)|(?P<tom>明天)|(?P<dat>后天)'
    r'|(?P<wd>[上下])(?:周|星期|礼拜)(?P<day>[一二三四五六日天])|(?P<w>[上下])(?:周|星期|礼拜|个星期)(?![一二三四五六日天])'
    r'|(?P<m>[上下])个?月|(?P<ly>去年)|(?P<lly>前年)|(?P<ny>明年)'
    r'|(?P<n>\d+|十[一二]?|[一两二三四五六七八九])(?:个)?(?P<unit>天|周|星期|月|年)前')


def fmt(d: dt.date) -> str:
    return f'{d.day} {d:%B %Y}'


def _month(d: dt.date, n: int) -> str:
    y, m = divmod(d.year * 12 + d.month - 1 + n, 12)
    return dt.date(y, m + 1, 1).strftime('%B %Y')


def _week(d: dt.date, n: int) -> dt.date:
    return d - d.weekday() * DAY + n * 7 * DAY


def _ago(d: dt.date, n: int, unit: str) -> str:
    if unit in ('day', '天'):
        return fmt(d - n * DAY)
    if unit in ('week', '周', '星期'):
        return f'around {fmt(d - n * 7 * DAY)}'
    return f'around {_month(d, -n)}' if unit in ('month', '月') else f'around {d.year - n}'


def _en(m: re.Match, d: dt.date) -> str | None:
    g = m.lastgroup
    w = m.group(0).lower()
    if g == 'before':
        return fmt(d - 2 * DAY)
    if g == 'after':
        return fmt(d + 2 * DAY)
    if g in ('yest', 'tom'):
        return fmt(d + (DAY if g == 'tom' else -DAY))
    if g == 'lastwd':
        return f'{fmt(d - ((d.weekday() - WEEKDAYS.index(w.split()[-1]) - 1) % 7 + 1) * DAY)}'
    if g == 'nextwd':
        return f'{fmt(d + ((WEEKDAYS.index(w.split()[-1]) - d.weekday() - 1) % 7 + 1) * DAY)}'
    if g in ('lw', 'nw'):
        return f'the week of {fmt(_week(d, 1 if g == "nw" else -1))}'
    if g in ('lwe', 'nwe'):
        sat = d + (((5 - d.weekday() - 1) % 7 + 1) * DAY if g == 'nwe' else -((d.weekday() - 5 - 1) % 7 + 1) * DAY)
        return f'the weekend of {fmt(sat)}'
    if g in ('lm', 'nm'):
        return _month(d, 1 if g == 'nm' else -1)
    if g in ('ly', 'ny'):
        return str(d.year + (1 if g == 'ny' else -1))
    if g == 'season':
        s = w.split()[-1]
        end = {'spring': 5, 'summer': 8}.get(s, 11)
        return f'{s} {d.year if d.month > end else d.year - 1}'
    if g == 'few':
        return f'shortly before {fmt(d)}'
    n, unit = m.group('n').lower(), m.group('unit').lower()
    return _ago(d, int(n) if n.isdigit() else EN_NUM[n], unit)


def _zh(m: re.Match, d: dt.date) -> str | None:
    g = m.groupdict()
    if g['d3'] or g['d2']:
        return fmt(d - (3 if g['d3'] else 2) * DAY)
    if g['yest'] or g['tom'] or g['dat']:
        return fmt(d + (-1 if g['yest'] else 1 if g['tom'] else 2) * DAY)
    if g['wd']:
        i = 6 if g['day'] in '日天' else ZH_WEEKDAYS.index(g['day'])
        return fmt(_week(d, -1 if g['wd'] == '上' else 1) + i * DAY)
    if g['w']:
        return f'the week of {fmt(_week(d, -1 if g["w"] == "上" else 1))}'
    if g['m']:
        return _month(d, -1 if g['m'] == '上' else 1)
    if g['ly'] or g['lly'] or g['ny']:
        return str(d.year + (-1 if g['ly'] else -2 if g['lly'] else 1))
    n = g['n']
    return _ago(d, int(n) if n.isdigit() else ZH_NUM[n], g['unit'])


def annotate(text: str, when: dt.date | dt.datetime | str | None) -> str:
    """``text`` with "[absolute date]" after each relative time, counted from ``when``"""
    if not when or not text:
        return text
    if isinstance(when, str):
        when = dt.datetime.fromisoformat(when.replace('Z', '+00:00'))
    d = when.date() if isinstance(when, dt.datetime) else when
    text = EN.sub(lambda m: f'{m.group(0)} [{_en(m, d)}]', text)
    return ZH.sub(lambda m: f'{m.group(0)} [{_zh(m, d)}]', text)
