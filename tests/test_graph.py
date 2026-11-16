from pm1 import Store
from pm1.graph import Graph
from pm1.ledger import Ledger


def _ledger(tmp_path):
    st = Store(tmp_path / 'g.sqlite')
    led = Ledger(st)
    led.add_extraction('b', 's', 'sess1|2024-01-10|0', '2024-01-10', {
        'entities': [{'name': 'Dr. Selim Kaya', 'type': 'person', 'aliases': ['Dr. Kaya']}],
        'events': [{'date': '2024-01-10', 'what': 'The user saw Dr. Selim Kaya for a checkup.', 'kind': 'done',
                    'entities': ['Dr. Selim Kaya']}],
        'state': [{'subject': 'Dr. Selim Kaya', 'attribute': 'role', 'value': 'primary care physician'}]})
    led.add_extraction('b', 's', 'sess2|2024-03-01|0', '2024-03-01', {
        'entities': [{'name': 'Dr. Kaya', 'type': 'person', 'aliases': []}],
        'events': [{'date': '2024-03-01', 'what': 'Dr. Kaya prescribed aspirin 81 mg.', 'kind': 'done',
                    'entities': ['Dr. Kaya']}],
        'state': [{'subject': 'User', 'attribute': 'medication', 'value': 'aspirin 81 mg daily'},
                  {'subject': 'Dr. Kaya', 'attribute': 'role', 'value': 'cardiology referral coordinator'}]})
    led.resolve('b', 's')
    return st


def test_graph_merges_aliases_and_chains(tmp_path):
    st = _ledger(tmp_path)
    g = Graph(st, 'b', 's')
    kaya = [n for n in g.nodes if 'kaya' in n['name'].lower()]
    assert len(kaya) == 1                                   # "Dr. Selim Kaya" and "Dr. Kaya" are one node
    n = kaya[0]
    assert len(n['events']) == 2
    text = g.expand(n['id'])
    assert 'cardiology referral coordinator (since 2024-03-01)' in text and 'earlier: primary care physician' in text
    assert '[src sess1|2024-01-10|0]' in text
    m = g.map('What did Dr. Kaya prescribe?')
    assert m and m[0].startswith(f"[node {n['id']}]") and '1 earlier values' in m[0]
    tl = g.timeline(start='2024-02-01')
    assert 'aspirin' in tl and 'checkup' not in tl
    user = next(x for x in g.nodes if x['name'] == 'User')
    assert 'aspirin 81 mg daily' in g.expand(user['id'])


def test_update_notes(tmp_path):
    st = _ledger(tmp_path)
    notes = Ledger(st).update_notes('b', 's')
    assert notes[('sess1', '2024-01-10')] == [
        'Dr. Selim Kaya — role: "primary care physician" was later updated to "cardiology referral coordinator" (2024-03-01)']
