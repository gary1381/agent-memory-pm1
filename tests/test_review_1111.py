"""Regressions reproduced during the 2026-11-11 audit; synthetic data only."""
import pytest

from pm1 import Recaller, Store
from pm1.mcp import Service


def test_source_deduplication_preserves_namespaces():
    s = Store()
    for bank, scope, session in [('a', 'one', 's1'), ('b', 'one', 's1'),
                                 ('a', 'two', 's1'), ('a', 'one', 's2')]:
        out = s.add_episode(bank=bank, scope=scope, session=session, text='shared source text',
                            source='transcript.jsonl#1')
        assert not out['duplicate']
        assert s.bm25(bank, scope, 'shared source text')
        # A replay within the same namespace still deduplicates after a changed seq.
        again = s.add_episode(bank=bank, scope=scope, session=session, seq=100,
                              text='shared source text', source='transcript.jsonl#1')
        assert again['duplicate'] and again['episode_id'] == out['episode_id']
    s.close()


@pytest.mark.parametrize('value', ['"oops"', '[1, 2]', 'true', '42'])
def test_non_table_config_is_ignored(tmp_path, monkeypatch, value):
    from pm1.paths import apply_config
    cfg = tmp_path / 'config.toml'
    cfg.write_text(f'pm1 = {value}\n')
    monkeypatch.setenv('PM1_CONFIG', str(cfg))
    assert apply_config() == []


def test_config_uses_tomli_when_tomllib_is_unavailable(tmp_path, monkeypatch):
    import builtins
    import sys
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    from pm1.paths import apply_config
    cfg = tmp_path / 'config.toml'
    cfg.write_text('[pm1]\nthreads = 3\n')
    monkeypatch.setenv('PM1_CONFIG', str(cfg))
    monkeypatch.setenv('PM1_THREADS', 'placeholder')  # restore even when apply_config adds an initially absent key
    monkeypatch.delenv('PM1_THREADS', raising=False)
    monkeypatch.setitem(sys.modules, 'tomli', tomllib)
    original = builtins.__import__

    def without_tomllib(name, *args, **kwargs):
        if name == 'tomllib':
            raise ModuleNotFoundError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_tomllib)
    assert apply_config() == ['PM1_THREADS']


def test_capture_same_note_in_two_scopes_is_recallable_in_both():
    s = Store()
    service = Service(s, Recaller(s), bank='personal', default_scope='one', client='test')
    a = service.call('memory_capture', {'text': 'zebra queue policy', 'scope': 'one'})
    b = service.call('memory_capture', {'text': 'zebra queue policy', 'scope': 'two'})
    assert a['episode_id'] != b['episode_id']
    assert service.call('memory_recall', {'query': 'zebra queue policy', 'scope': 'two'})['items']
    s.close()
