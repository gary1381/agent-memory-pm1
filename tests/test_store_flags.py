from pm1.store import Store


def _store(texts):
    s = Store(':memory:')
    for i, (role, t) in enumerate(texts):
        s.add_episode(bank='b', scope='s', text=t, role=role, session='x', seq=i, occurred_at='2024-01-01')
    return s


def test_negations_user_only():
    s = _store([('user', "I've never set up diagnostic logs for shard errors."),
                ('assistant', 'You have never mentioned that before.'),
                ('user', 'I set up diagnostic logs yesterday.'),
                ('user', '我从来没有用过 Milvus。')])
    texts = s.spans('b', 's', s.negations('b', 's'))
    got = sorted(r['text'] for r in texts.values())
    assert got == ["I've never set up diagnostic logs for shard errors.", '我从来没有用过 Milvus。']


def test_directives():
    s = _store([('user', 'Always include frame rate data when I ask about rendering performance.'),
                ('user', 'I always forget my keys.'),
                ('user', 'Never use abbreviations when I ask for a summary.')])
    texts = sorted(r['text'] for r in s.spans('b', 's', s.directives('b', 's')).values())
    assert texts == ['Always include frame rate data when I ask about rendering performance.',
                     'Never use abbreviations when I ask for a summary.']


def test_span_meta_threads_and_slash_dates():
    s = Store(':memory:')
    s.add_episode(bank='b', scope='s', text='hi', role='assistant', session='a', seq=0, occurred_at='2023/05/20 (Sat) 02:21')
    s.add_episode(bank='b', scope='s', text='start', role='user', session='a', seq=1, occurred_at='2023/05/20 (Sat) 02:22')
    m = s.span_meta('b', 's')
    assert set(m['day']) == {'2023-05-20'} and len(m['opener']) == 1
    assert s.bm25('b', 's', 'start', window=('2023-05-20', '2023-05-20'))


def test_directive_topic_extraction():
    from pm1.recall import Recaller
    assert Recaller.directive_topic('Always include exact error messages when I ask about debugging issues. ->-> 2,33') \
        == 'debugging issues'
    assert Recaller.directive_topic('Never use emojis.') == 'Never use emojis'
