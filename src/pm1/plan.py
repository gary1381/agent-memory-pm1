"""Query planning for recall: decide how to search before searching. Pure rules, no model call.

intent
  fact      one thing to find (default)              -> ranked top-k, as before
  timeline  progression / order / summary over time  -> time-window + one representative per day-thread
  multi     several facts to combine or compare      -> one search per sub-question, merged round-robin
window   (start, end) ISO dates if the query names a range
want     number of items the query asks for ("twenty items")
subs     sub-questions for 'multi'
openers  also bring the opening turn of matched day-threads (phase starts: "when I started X")
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field

_MONTHS = {m: i + 1 for i, m in enumerate(
    ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october',
     'november', 'december'])}
_MONTHS |= {k[:3]: v for k, v in list(_MONTHS.items())} | {'sept': 9}
_NUM = {w: i for i, w in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen '
    'seventeen eighteen nineteen twenty'.split())} | {'thirty': 30, 'forty': 40, 'fifty': 50}
_CN_NUM = {c: i for i, c in enumerate('零一二三四五六七八九十')}

_ISO = re.compile(r'\b(20\d\d)[-/.](\d{1,2})[-/.](\d{1,2})\b')
_CN_DATE = re.compile(r'(20\d\d)\s*年\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*[日号])?')
_MDY = re.compile(r'\b(' + '|'.join(sorted(_MONTHS, key=len, reverse=True)) +
                  r')\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?\b', re.I)
_MY = re.compile(r'\b(' + '|'.join(sorted(_MONTHS, key=len, reverse=True)) + r')\.?\s+(20\d\d)\b', re.I)

_TIMELINE = re.compile(
    r'\b(progress(ed|ion)?|evolv(e|ed|ing)|evolution|in (chronological )?order|order(ed)? of|'
    r'sequence (in which|of)|the order (in which|that)|in what order|which order|'
    r'chronolog\w*|timeline|summari[sz]e|(a|an|the) (\w+ ){0,2}summary of|summary of how|(an|the) overview of|'
    r'over time|journey|develop(ed|ment)s? (over|from|between)|'
    r'how (did|has|have) .{0,80}\b(change|chang(ed|ing)|develop(ed)?|grow|grew|progress(ed)?)|'
    r'trajectory|walk me through)\b', re.I)
_TIMELINE_ZH = re.compile(r'进展|进度|演变|变化过程|顺序|先后|时间线|按时间|总结|概括|回顾|历程|经过|阶段')
_MULTI = re.compile(
    r'\b(in total|combined|combining|altogether|sum of|add(ed)? up|across (all |my |the )?'
    r'(sessions|projects|conversations)|both|each of|compare|comparison|difference|versus|vs\.?|'
    r'how many days|how many weeks|how many months|how long (after|before|between)|between when|'
    r'days (between|after|before)|(after|before) I (started|began|start|begin))\b', re.I)
_MULTI_ZH = re.compile(r'总共|一共|合计|加起来|分别|比较|对比|相差|之间.{0,6}(多少|几)|多少天|多久')
_TEMPORAL = re.compile(r'\b(when|days?|weeks?|months?|how long|start(ed)?|began|begin|launch(ed)?|kick(ed)? off)\b'
                       r'|什么时候|哪天|开始', re.I)
_WANT = re.compile(r'\b(?:only (?:and only )?)?(\d{1,2}|' + '|'.join(_NUM) + r')\s+(?:items|events|points|steps|'
                   r'phases|stages|things|bullets?)\b', re.I)
_LEAD = re.compile(r'^\s*(how many|how much|what is the|what\'s the|what are the|what was the|can you|could you|'
                   r'please|tell me|do you know)\b[^,]*?\b(between|of|for|across)?\s*', re.I)


@dataclass
class Plan:
    intent: str = 'fact'
    window: tuple[str, str] | None = None
    want: int | None = None
    subs: list[str] = field(default_factory=list)
    openers: bool = False
    aggregate: bool = False      # counts / totals / lists across sessions -> wide context budget
    anchor: tuple[str, str] | None = None   # date window a relative phrase points at ("a week ago"), given `now`


def _iso(y: int, m: int, d: int) -> str | None:
    if 1 <= m <= 12 and 1 <= d <= 31:
        return f'{y:04d}-{m:02d}-{d:02d}'
    return None


def _month_end(y: int, m: int) -> int:
    return [31, 29 if y % 4 == 0 and (y % 100 or y % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]


def dates_in(text: str) -> list[tuple[int, str, str]]:
    """(position, start, end) for each date mention; a bare month spans the whole month."""
    out: list[tuple[int, str, str]] = []
    taken: list[tuple[int, int]] = []

    def free(a, b):
        return all(b <= x or a >= y for x, y in taken)

    for m in _ISO.finditer(text):
        d = _iso(int(m[1]), int(m[2]), int(m[3]))
        if d:
            out.append((m.start(), d, d)); taken.append(m.span())
    for m in _CN_DATE.finditer(text):
        if not free(*m.span()):
            continue
        y, mo = int(m[1]), int(m[2])
        if 1 <= mo <= 12:
            a = _iso(y, mo, int(m[3]) if m[3] else 1)
            b = _iso(y, mo, int(m[3]) if m[3] else _month_end(y, mo))
            out.append((m.start(), a, b)); taken.append(m.span())
    years = [int(y) for y in re.findall(r'\b(20\d\d)\b', text)]
    for m in _MDY.finditer(text):
        if not free(*m.span()):
            continue
        mo, day = _MONTHS[m[1].lower()], int(m[2])
        y = int(m[3]) if m[3] else (years[-1] if years else None)
        if y and day <= 31:
            d = _iso(y, mo, day)
            out.append((m.start(), d, d)); taken.append(m.span())
    for m in _MY.finditer(text):
        if not free(*m.span()):
            continue
        y, mo = int(m[2]), _MONTHS[m[1].lower()]
        out.append((m.start(), _iso(y, mo, 1), _iso(y, mo, _month_end(y, mo)))); taken.append(m.span())
    return sorted(out)


def _window(text: str) -> tuple[str, str] | None:
    ds = dates_in(text)
    if len(ds) >= 2:
        return min(d[1] for d in ds), max(d[2] for d in ds)
    if len(ds) == 1 and ds[0][1] != ds[0][2]:       # a whole month named alone
        return ds[0][1], ds[0][2]
    return None


def _want(text: str) -> int | None:
    m = _WANT.search(text)
    if not m:
        return None
    w = m[1].lower()
    return int(w) if w.isdigit() else _NUM.get(w)


def _subs(text: str) -> list[str]:
    """Split a compound question into the things to look up separately."""
    q = text.strip().rstrip('?？.。')
    m = re.search(r'\bbetween\s+(?:when\s+)?(.+?)\s+and\s+(?:when\s+)?(.+)$', q, re.I)
    if m:
        return [m[1].strip(), m[2].strip()]
    m = re.search(r'\b(?:after|before)\s+(?:when\s+)?I\s+(.+?)\s+(?:did|do|does)\s+I\s+(.+)$', q, re.I)
    if m:
        return [m[1].strip(), m[2].strip()]
    m = re.search(r'(.+?)(?:之间|和|与|跟)(.+?)(?:之间|相差|相隔|一共|总共|合计)', q)
    if m and re.search(r'[一-鿿]', q):
        return [m[1].strip(), m[2].strip()]
    body = _LEAD.sub('', q)
    parts = [x.strip() for x in re.split(r',\s*(?:and\s+)?|\s+and\s+|\s+as well as\s+|\s+plus\s+|、|以及|和', body)
             if x.strip()]
    if len(parts) < 2:
        return []
    # "<head> across A, B and C": the first part carries the shared head; give it to every item.
    m = re.match(r'(.*\b(?:across|on|for|of|with|combining|combine|between|in|about|from|to|at)\b)\s+(.+)$',
                 parts[0], re.I)
    head, first = (m[1], m[2]) if m else ('', parts[0])
    keep = ' '.join(w for w in re.findall(r"[\w$%.'-]+", head) if w.lower() not in _HEAD_STOP)
    items = [first] + parts[1:]
    tail = re.split(r'\b(?:based on|during|according to|in my|from my|across my|combined)\b', items[-1], 1)
    if len(tail) == 2:
        items[-1] = tail[0].strip()
        keep = (keep + ' ' + ' '.join(w for w in re.findall(r"[\w$%.'-]+", tail[1])
                                      if w.lower() not in _HEAD_STOP)).strip()
    subs = [f'{keep} {it}'.strip() for it in items if it]
    return subs if len(subs) >= 2 else []


_HEAD_STOP = frozenset('how many much what which total in am i have did do does my the a an when across combining combine '
                       'combined on for of with to and is are was were me our we all overall altogether'.split())


_ASPECTS = re.compile(r'\b(?:including|such as|covering|especially|particularly|like|focusing on|in terms of|'
                      r'with respect to|regarding)\s+(.+?)(?:,?\s+(?:(?:in order|and how|between|during|over)\b|from \d)|[?.]|$)',
                      re.I)


def _aspects(text: str) -> list[str]:
    """'... including venue options, guest capacities, and permit fees' -> the listed aspects."""
    m = _ASPECTS.search(text)
    if not m:
        return []
    parts = [x.strip(' ,') for x in re.split(r',\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+', m[1]) if x.strip(' ,')]
    parts = [re.sub(r'^(?:how|what|my|the|I)\s+', '', x, flags=re.I) for x in parts]
    return [x for x in parts if len(x) > 3][:8]


_AGG = re.compile(r'\bhow (?:many|much|often)\b|\b(?:total|number of|in all|altogether|combined|list (?:all|every)|all (?:the|of the) '
                  r'\w+ (?:I|we)\b|every time|each time)\b', re.I)
_AGG_ZH = re.compile(r'多少|几(?:个|次|天|周|件|本|家)|总共|一共|合计|所有')


def _parse_now(now) -> _dt.date | None:
    if now is None:
        return None
    if isinstance(now, _dt.datetime):
        return now.date()
    if isinstance(now, _dt.date):
        return now
    m = re.search(r'(20\d\d)\D{0,3}(\d{1,2})\D{0,3}(\d{1,2})', str(now))
    try:
        return _dt.date(int(m[1]), int(m[2]), int(m[3])) if m else None
    except ValueError:
        return None


def _nth_weekday(y: int, m: int, wd: int, n: int) -> _dt.date:
    """n-th weekday wd (Mon=0) of month m; n=-1 for the last one."""
    if n > 0:
        d = _dt.date(y, m, 1)
        return d + _dt.timedelta(days=(wd - d.weekday()) % 7 + 7 * (n - 1))
    d = _dt.date(y, m, _month_end(y, m))
    return d - _dt.timedelta(days=(d.weekday() - wd) % 7)


def _easter(y: int) -> _dt.date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 19 * l) // 433
    mo = (h + l - 7 * m + 90) // 25
    return _dt.date(y, mo, (h + l - 7 * m + 33 * mo + 19) % 32)


_HOLIDAYS = [
    (r"valentine'?s?(?: day)?|情人节", lambda y: _dt.date(y, 2, 14)),
    (r"christmas eve|平安夜", lambda y: _dt.date(y, 12, 24)),
    (r"christmas(?: day)?|xmas|圣诞节?", lambda y: _dt.date(y, 12, 25)),
    (r"new year'?s eve|跨年夜", lambda y: _dt.date(y, 12, 31)),
    (r"new year'?s(?: day)?|元旦", lambda y: _dt.date(y, 1, 1)),
    (r"halloween|万圣节", lambda y: _dt.date(y, 10, 31)),
    (r"(?:fourth|4th) of july|july (?:4th|fourth)|independence day", lambda y: _dt.date(y, 7, 4)),
    (r"thanksgiving|感恩节", lambda y: _nth_weekday(y, 11, 3, 4)),
    (r"easter(?: sunday)?|复活节", _easter),
    (r"mother'?s day|母亲节", lambda y: _nth_weekday(y, 5, 6, 2)),
    (r"father'?s day|父亲节", lambda y: _nth_weekday(y, 6, 6, 3)),
    (r"memorial day", lambda y: _nth_weekday(y, 5, 0, -1)),
    (r"labou?r day", lambda y: _nth_weekday(y, 9, 0, 1)),
    (r"st\.? patrick'?s day", lambda y: _dt.date(y, 3, 17)),
]
_HOLIDAY_RE = [(re.compile(r'\b(?:' + p + r')', re.I), f) for p, f in _HOLIDAYS]
_UNIT = {'day': 1, 'week': 7, 'month': 30, 'year': 365}
_NUMW = r'(\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a couple of|a few|several)'
_REL_AGO = re.compile(_NUMW + r'\s+(day|week|month|year)s?\s+ago\b', re.I)
_REL_PAST = re.compile(r'\b(?:in |over |during )?the (?:past|last)\s+' + _NUMW + r'?\s*(day|week|month)s?\b', re.I)
_CN_AGO = re.compile(r'(\d{1,3}|[一二两三四五六七八九十]+)\s*(天|周|星期|个月|年)(?:以)?前')


def _numw(w: str | None) -> int:
    if not w:
        return 1
    w = w.lower()
    if w.isdigit():
        return int(w)
    return {'a': 1, 'an': 1, 'a couple of': 2, 'a few': 3, 'several': 4}.get(w, _NUM.get(w, 1))


def _cn_num(s: str) -> int:
    if s.isdigit():
        return int(s)
    s = s.replace('两', '二')
    if s == '十':
        return 10
    if '十' in s:
        a, _, b = s.partition('十')
        return (_CN_NUM.get(a, 1) if a else 1) * 10 + (_CN_NUM.get(b, 0) if b else 0)
    return _CN_NUM.get(s, 1)


def anchor(text: str, now) -> tuple[str, str] | None:
    """Date window that a relative time phrase points at, resolved against `now` (the time of asking):
    "yesterday", "3 days ago", "a week ago", "last week(end)", "last month", "the past two weeks",
    "on Valentine's Day", "on March 3rd" (no year: the latest such day not after `now`), "in May".
    Approximate on purpose: the window widens with distance, since people round ("a week ago" ~ 4-10 days)."""
    d = _parse_now(now)
    if d is None:
        return None
    td = _dt.timedelta
    t = text.lower()

    def win(a: _dt.date, b: _dt.date) -> tuple[str, str]:
        return a.isoformat(), min(b, d).isoformat()

    if re.search(r'\byesterday\b|昨天', t):
        return win(d - td(1), d - td(1))
    if re.search(r'\bday before yesterday\b|前天', t):
        return win(d - td(2), d - td(2))
    m = _REL_AGO.search(t)
    if m:
        n, u = _numw(m[1]), _UNIT[m[2].lower()]
        back = n * u
        slack = 1 if back <= 3 else max(2, round(back * 0.3)) if u <= 7 else max(7, round(back * 0.25))
        return win(d - td(back + slack), d - td(max(0, back - slack)))
    m = _CN_AGO.search(text)
    if m:
        u = {'天': 1, '周': 7, '星期': 7, '个月': 30, '年': 365}[m[2]]
        back = _cn_num(m[1]) * u
        slack = 1 if back <= 3 else max(2, round(back * 0.3)) if u <= 7 else max(7, round(back * 0.25))
        return win(d - td(back + slack), d - td(max(0, back - slack)))
    m = _REL_PAST.search(t)
    if m:
        return win(d - td(_numw(m[1]) * _UNIT[m[2].lower()]), d)
    if re.search(r'\blast weekend\b|上个?周末', t):
        sat = d - td((d.weekday() - 5) % 7 or 7)        # the latest Saturday strictly before today
        return win(sat - td(1), sat + td(2))
    if re.search(r'\blast week\b|上个?(?:周|星期|礼拜)', t):
        mon = d - td(d.weekday() + 7)
        return win(mon - td(2), mon + td(8))
    if re.search(r'\bthis week\b|这个?(?:周|星期|礼拜)', t):
        return win(d - td(d.weekday()), d)
    if re.search(r'\blast month\b|上个月', t):
        y, mo = (d.year, d.month - 1) if d.month > 1 else (d.year - 1, 12)
        return win(_dt.date(y, mo, 1), _dt.date(y, mo, _month_end(y, mo)))
    if re.search(r'\bthis month\b|这个月|本月', t):
        return win(d.replace(day=1), d)
    if re.search(r'\blast year\b|去年', t):
        return win(_dt.date(d.year - 1, 1, 1), _dt.date(d.year - 1, 12, 31))
    for rx, f in _HOLIDAY_RE:
        if rx.search(text):
            h = f(d.year)
            if h > d:
                h = f(d.year - 1)
            return win(h - td(1), h + td(1))
    if re.search(r'\b20\d\d\b', text):
        return None                         # an explicit year: dates_in/_window handle it
    m = _MDY.search(text)
    if m and int(m[2]) <= 31:
        mo, day = _MONTHS[m[1].lower()], int(m[2])
        for y in (d.year, d.year - 1):
            try:
                h = _dt.date(y, mo, min(day, _month_end(y, mo)))
            except ValueError:
                continue
            if h <= d:
                return win(h - td(1), h + td(1))
    m = re.search(r'\b(?:in|during|back in|early|late|mid)[ -](' + '|'.join(
        k for k in _MONTHS if len(k) > 3 or k == 'may') + r')\b', t)
    if m:
        mo = _MONTHS[m[1]]
        y = d.year if mo <= d.month else d.year - 1
        return win(_dt.date(y, mo, 1), _dt.date(y, mo, _month_end(y, mo)))
    return None


def plan(query: str, now=None) -> Plan:
    p = Plan(window=_window(query), want=_want(query))
    p.anchor = anchor(query, now) if now is not None else None
    timeline = bool(_TIMELINE.search(query) or _TIMELINE_ZH.search(query))
    multi = bool(_MULTI.search(query) or _MULTI_ZH.search(query)
                 or re.search(r'^\s*how (?:many|much)\b.*\btotal\b', query, re.I))
    if timeline and (p.window or p.want or not multi):
        p.intent = 'timeline'
        p.subs = _aspects(query)
    elif multi:
        subs = _subs(query)
        if subs:
            p.intent, p.subs = 'multi', subs[:6]
    p.aggregate = p.intent != 'fact' or bool(_AGG.search(query) or _AGG_ZH.search(query))
    p.openers = p.intent != 'fact' and bool(_TEMPORAL.search(query))
    if p.intent == 'timeline':
        p.openers = True
    return p
