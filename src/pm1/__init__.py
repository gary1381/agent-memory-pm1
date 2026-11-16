"""Precision Memory v1 (pm1).

Raw episodes are the only source of truth; every index is derived and
rebuildable; recall needs no generative model and returns evidence.
"""
__all__ = ['Store', 'Recaller', 'RecallResult']
__version__ = '0.1.0.dev0'


def __getattr__(name):
    # lazy: hooks import light modules (pm1.failsig) on every tool call and must not pay for numpy
    if name == 'Store':
        from .store import Store
        return Store
    if name in ('Recaller', 'RecallResult'):
        from . import recall
        return getattr(recall, name)
    raise AttributeError(name)
