"""pm1.timeref: relative times get an absolute date from the record's date; everything else is left alone."""
import datetime as dt

import pytest

from pm1.timeref import annotate

MON = dt.date(2023, 5, 8)


@pytest.mark.parametrize('text,want', [
    ('I went yesterday', '[7 May 2023]'), ('Last night was fun', '[7 May 2023]'),
    ('the day before yesterday', '[6 May 2023]'), ('see you tomorrow', '[9 May 2023]'),
    ('last Saturday', '[6 May 2023]'), ('last Monday', '[1 May 2023]'), ('next Friday', '[12 May 2023]'),
    ('last week', '[the week of 1 May 2023]'), ('next week', '[the week of 15 May 2023]'),
    ('last weekend', '[the weekend of 6 May 2023]'), ('next weekend', '[the weekend of 13 May 2023]'),
    ('last month', '[April 2023]'), ('next month', '[June 2023]'), ('last year', '[2022]'),
    ('last summer', '[summer 2022]'), ('two days ago', '[6 May 2023]'), ('3 weeks ago', '[around 17 April 2023]'),
    ('a month ago', '[around April 2023]'), ('a couple of years ago', '[around 2021]'),
    ('a few days ago', '[shortly before 8 May 2023]'),
    ('昨天去了', '[7 May 2023]'), ('前天', '[6 May 2023]'), ('大前天', '[5 May 2023]'), ('明天', '[9 May 2023]'),
    ('上周五', '[5 May 2023]'), ('上星期天', '[7 May 2023]'), ('下周一', '[15 May 2023]'),
    ('上周去的', '[the week of 1 May 2023]'), ('上个月', '[April 2023]'), ('去年', '[2022]'), ('前年', '[2021]'),
    ('三天前', '[5 May 2023]'), ('两周前', '[around 24 April 2023]'), ('3个月前', '[around February 2023]'),
])
def test_annotates(text, want):
    assert want in annotate(text, MON)


@pytest.mark.parametrize('text', ['the last time we met', 'last one', 'this weekend', 'on Friday', 'today', '今天',
                                  'last-minute plans'])
def test_leaves_alone(text):
    assert annotate(text, MON) == text


def test_year_wrap_and_date_strings():
    assert '[December 2022]' in annotate('last month', '2023-01-15T10:00:00Z')
    assert 'yesterday [31 December 2022]' in annotate('yesterday', '2023-01-01')
    assert annotate('yesterday', None) == 'yesterday'
