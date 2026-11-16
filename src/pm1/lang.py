"""Cheap, deterministic language tags for routing (no model, no dependency).

Tags: 'en', 'zh', 'ja', 'ko', 'other' (another script or a non-English Latin language), 'und' (no letters).
Routing only needs one question answered reliably: may an English-only model see this text? So the rules lean
towards 'other' whenever a text is not clearly English.
"""
from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z']+")
# Frequent English function words; non-English Latin text rarely hits these.
_EN = set('the a an and or but of to in on at for with from by about as is are was were be been being have has had '
          'do does did i you he she it we they me my your his her our their this that these those what which who '
          'when where why how not no can could will would should just so if then than there here all any some '
          'very also into out up down over again more most such only own same too'.split())

# Function words of the Latin-script languages most likely to appear unaccented.
_OTHER = {
    'es': set('el la los las de del que y en un una por con para es son está como pero muy más'.split()),
    'fr': set('le la les des du de et est une un pour dans pas que qui sur avec ce cette sont'.split()),
    'de': set('der die das und ist nicht ein eine ich du wir sie mit auf für von zu den dem auch'.split()),
    'it': set('il lo gli le di che e non per una un sono con della del questo anche'.split()),
    'pt': set('o os as de do da que e não um uma para com por são está mais'.split()),
    'nl': set('de het een en van is niet dat ik je met voor op zijn ook'.split()),
}


def detect(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 'und'
    n = len(letters)
    han = sum('一' <= c <= '鿿' or '㐀' <= c <= '䶿' for c in letters)
    kana = sum('぀' <= c <= 'ヿ' for c in letters)
    hangul = sum('가' <= c <= '힯' or 'ᄀ' <= c <= 'ᇿ' for c in letters)
    latin = sum(c.isascii() for c in letters)
    if kana / n > 0.1:
        return 'ja'
    if hangul / n > 0.2:
        return 'ko'
    if han / n > 0.2:
        return 'zh'
    if latin / n < 0.9:                      # Cyrillic, Arabic, accented Latin languages, ...
        return 'other'
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < 6:                       # too short to judge by function words: ASCII-only -> English
        return 'en'
    en = sum(w in _EN for w in words) / len(words)
    other = max(sum(w in v for w in words) for v in _OTHER.values()) / len(words)
    # Code, lists and terse notes have few function words of any language: still English-compatible.
    return 'other' if other >= 0.08 and other > en else 'en'


def english_compatible(tag: str) -> bool:
    """May an English-only model score this text?"""
    return tag in ('en', 'und')
