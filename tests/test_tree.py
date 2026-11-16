from pm1.store import Store
from pm1.tree import Tree, akey

RAW = {'t1': ('user', 'I set my holiday gift budget to $400 total.', '2024-12-05'),
       't2': ('assistant', 'Maybe raise it to $450 so the family exchange fits?', '2024-12-05'),
       't3': ('user', "OK, let's go with $450.", '2024-12-06'),
       't4': ('user', 'I adopted a dog named Rex.', '2024-12-07')}


def tree(tmp_path):
    return Tree(Store(tmp_path / 't.sqlite'), 'b', 's', raw=RAW.get)


def test_akey_plural():
    assert akey('holiday gifts budget') == akey('Holiday gift budget')


def test_versions_proposal_promotion(tmp_path):
    t = tree(tmp_path)
    assert t.apply({'op': 'create_entity', 'name': 'User', 'type': 'person'})['ok']
    v1 = t.apply({'op': 'add_version', 'entity': 'User', 'attribute': 'holiday gift budget', 'value': '$400',
                  'evidence': ['t1']})
    assert v1['ok'] and t.current('User', 'holiday gifts budget')['value'] == '$400'
    # assistant cannot state a user value
    bad = t.apply({'op': 'add_version', 'entity': 'User', 'attribute': 'holiday gift budget', 'value': '$450',
                   'asserted_by': 'assistant', 'status': 'stated', 'evidence': ['t2']})
    assert not bad['ok']
    p = t.apply({'op': 'add_version', 'entity': 'User', 'attribute': 'holiday gift budget', 'value': '$450',
                 'asserted_by': 'assistant', 'evidence': ['t2']})
    assert p['ok'] and t.current('User', 'holiday gift budget')['value'] == '$400'     # proposal is not current
    pr = t.apply({'op': 'promote_proposal', 'version': p['version'], 'evidence': ['t3'], 'valid_from': '2024-12-06'})
    assert pr['ok'] and pr['closed'] == v1['version']
    assert t.current('User', 'holiday gift budget')['value'] == '$450'
    h = t.history('User', 'holiday gift budget')
    assert [v['value'] for v in h] == ['$400', '$450'] and h[0]['valid_to'] == '2024-12-06'


def test_checks(tmp_path):
    t = tree(tmp_path)
    t.apply({'op': 'create_entity', 'name': 'User'})
    assert not t.apply({'op': 'add_version', 'entity': 'User', 'attribute': 'x', 'value': '1', 'evidence': ['t9']})['ok']
    assert not t.apply({'op': 'add_version', 'entity': 'User', 'attribute': 'budget', 'value': '$999',
                        'evidence': ['t1']})['ok']                       # number not in evidence
    assert not t.apply({'op': 'add_event', 'what': 'x', 'at': 'next week', 'evidence': ['t1']})['ok']
    assert not t.apply({'op': 'add_digest', 'turns': ['t2'], 'source': 'user', 'text': 'x'})['ok']
    assert t.apply({'op': 'add_digest', 'turns': ['t2'], 'source': 'assistant',
                    'text': 'Suggested raising the gift budget to $450.'})['ok']
    log = t.con.execute('SELECT count(*), sum(ok) FROM tree_oplog').fetchone()
    assert tuple(log) == (6, 2)


def test_merge_and_items(tmp_path):
    t = tree(tmp_path)
    t.apply({'op': 'create_entity', 'name': 'Rex', 'type': 'pet'})
    t.apply({'op': 'create_entity', 'name': 'my dog', 'type': 'pet'})
    t.apply({'op': 'add_version', 'entity': 'my dog', 'attribute': 'name', 'value': 'Rex', 'evidence': ['t4']})
    assert t.apply({'op': 'merge_entities', 'keep': 'Rex', 'drop': 'my dog'})['ok']
    assert t.current('my dog', 'name')['value'] == 'Rex'                  # alias resolves to the kept entity
    e = t.apply({'op': 'add_event', 'what': 'User adopted Rex', 'at': '2024-12-07', 'entities': ['Rex'],
                 'evidence': ['t4']})
    assert e['ok']
    layers = {i['layer'] for i in t.items()}
    assert {'version', 'event'} <= layers


def test_kinds_do_not_supersede_and_loops(tmp_path):
    raw = {'u1': ('user', 'My target is 15 interviews.', '2024-07-01'),
           'u2': ('user', 'I have done 12 so far.', '2024-07-03'),
           'a1': ('assistant', 'Example data: Normal(200, 50).', '2024-07-03'),
           'u3': ('user', 'I need to fix the Redis connection by Friday.', '2024-07-03')}
    t = Tree(Store(tmp_path / 'k.sqlite'), 'b', 's', raw=raw.get)
    t.apply({'op': 'create_entity', 'name': 'Interviews'})
    t.apply({'op': 'add_version', 'entity': 'Interviews', 'attribute': 'count', 'value': '15', 'value_kind': 'target',
             'evidence': ['u1']})
    p = t.apply({'op': 'add_version', 'entity': 'Interviews', 'attribute': 'count', 'value': '12',
                 'value_kind': 'progress', 'evidence': ['u2']})
    assert p['ok'] and p['closed'] is None                    # progress does not close the target
    assert not t.apply({'op': 'add_version', 'entity': 'Interviews', 'attribute': 'latency', 'value': '200',
                        'asserted_by': 'assistant', 'status': 'accepted', 'synthetic': True, 'evidence': ['a1']})['ok']
    lp = t.apply({'op': 'open_loop', 'text': 'Fix the Redis connection', 'due': '2024-07-05', 'evidence': ['u3']})
    assert lp['ok']
    texts = ' | '.join(i['text'] for i in t.items())
    assert '(target)' in texts and '(progress)' in texts and '[task, open]' in texts
    assert t.apply({'op': 'search', 'query': 'redis connection'})['results'][0]['layer'] == 'loop'
