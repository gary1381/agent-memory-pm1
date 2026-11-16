import sqlite3

from pm1.assertions import Ledger

RAW = {'t1': ('user', "I'm 60 and earning $85,000 annually, aiming for a raise.", '2024-03-15'),
       't2': ('assistant', 'You could ask for $10,000 more. Example: "led 15 successful series".', '2024-03-15'),
       't3': ('user', "Good news: I'm now earning $110,000 after the promotion.", '2024-05-02'),
       't4': ('user', "I've never asked for a raise before.", '2024-06-01'),
       't5': ('user', 'Always include salary ranges when I ask about job offers.', '2024-06-01')}


def led():
    return Ledger(sqlite3.connect(':memory:'), RAW.get)


def test_quote_lock_modality_numbers():
    l = led()
    e = l.apply('resolve_entity', {'surface_name': 'Crystal', 'type': 'person', 'turn_id': 't1'})['entity_id']
    ok = l.apply('append_assertion', {'turn_id': 't1', 'entity_id': e, 'facet': 'pay', 'quote': 'earning $85,000 annually',
                                      'modality': 'user_claim', 'number': 85000, 'unit': 'USD/year'})
    assert ok['ok']
    assert not l.apply('append_assertion', {'turn_id': 't1', 'entity_id': e, 'facet': 'pay', 'quote': 'earns 85k a year',
                                            'modality': 'user_claim'})['ok']              # paraphrase rejected
    assert not l.apply('append_assertion', {'turn_id': 't2', 'entity_id': e, 'facet': 'pay', 'quote': 'ask for $10,000 more',
                                            'modality': 'user_claim'})['ok']              # assistant turn as user claim
    assert not l.apply('append_assertion', {'turn_id': 't1', 'entity_id': e, 'facet': 'pay', 'quote': 'earning $85,000',
                                            'modality': 'user_claim', 'number': 95000})['ok']   # number not in quote
    assert l.apply('append_assertion', {'turn_id': 't2', 'entity_id': e, 'facet': 'resume_choice' if False else 'other',
                                        'quote': 'led 15 successful series', 'modality': 'assistant_example'})['ok']


def test_views_history_conflict_instruction():
    l = led()
    e = l.apply('resolve_entity', {'surface_name': 'Crystal', 'turn_id': 't1'})['entity_id']
    assert l.apply('resolve_entity', {'surface_name': ' crystal ', 'turn_id': 't3'})['existed']
    l.apply('append_assertion', {'turn_id': 't1', 'entity_id': e, 'facet': 'pay', 'quote': 'earning $85,000 annually',
                                 'modality': 'user_claim', 'number': 85000})
    l.apply('append_assertion', {'turn_id': 't3', 'entity_id': e, 'facet': 'pay', 'quote': "I'm now earning $110,000",
                                 'modality': 'user_claim', 'number': 110000, 'topic_key': 'raise'})
    l.apply('append_assertion', {'turn_id': 't4', 'entity_id': e, 'facet': 'pay', 'quote': "I've never asked for a raise",
                                 'modality': 'user_claim', 'polarity': 'deny', 'topic_key': 'raise'})
    l.apply('append_assertion', {'turn_id': 't5', 'entity_id': e, 'facet': 'preference', 'modality': 'user_instruction',
                                 'quote': 'Always include salary ranges when I ask about job offers.'})
    v = l.views()
    pay = next(c for c in v['current'] if c['facet'] == 'pay')
    assert pay['value'].startswith('110000') and pay['conflict'] and len(pay['history']) == 3     # nothing deleted
    assert v['instructions'] and v['instructions'][0]['turn'] == 't5'
    assert any(i['layer'] == 'current:conflict' for i in l.items())
