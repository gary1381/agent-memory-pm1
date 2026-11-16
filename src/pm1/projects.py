"""Logical projects: one name for the scopes that hold the same piece of work.

Session scopes are one per folder (``proj:<cwd>``) or Cowork label (``proj:cowork:<label>``), so one project is often
spread over its Cowork copy, sub-folders, worktrees and renamed folders (one project can have a dozen scopes). Automatic
rule: the first folder under the work roots (``PM1_WORK_ROOTS``, default ``work,code,office``; a numeric prefix such as
``1.work`` is ignored, so ``~/work/<X>``, ``~/1.work/<X>`` and ``~/work/office/<X>`` all name ``<X>``), or the Cowork label; lower-cased, ``_`` -> ``-``, without a ``N.`` prefix, a trailing number or ``-gpu`` / ``-next``. Then the alias file
(``PM1_PROJECTS``, default ``~/.config/pm1/projects.json``: ``{"canonical": ["other name", ...]}``) merges names the
rule cannot (e.g. an old name and a new name). ``isolated`` (a list in the same file) marks projects whose records only show
up when that project is the current one or is asked for by name.
"""
from __future__ import annotations

import json
import os
import re

ROOTS = tuple(x.strip() for x in os.environ.get('PM1_WORK_ROOTS', 'work,code,office').split(',') if x.strip())
HOME = os.path.expanduser('~')


def _norm(name: str) -> str:
    n = name.strip().lower().replace('_', '-').replace(' ', '-')
    n = re.sub(r'^\d+\.', '', n) or n           # '1.work' -> 'work', '0.notes' -> 'notes'
    n = re.sub(r'(-?\d+)$', '', n) or n
    n = re.sub(r'-(gpu|next)$', '', n)
    return n


def auto_name(scope_or_path: str) -> str:
    s = scope_or_path[5:] if scope_or_path.startswith('proj:') else scope_or_path
    if s.startswith('cowork:'):
        return _norm(s[7:])
    p = os.path.normpath(s)
    rel = os.path.relpath(p, HOME) if p.startswith(HOME) else p.lstrip('/')
    parts = [x for x in rel.split(os.sep) if x and x != '.']
    if not parts:
        return 'home'
    if _norm(parts[0]) not in ROOTS:
        return _norm(parts[0])
    while len(parts) > 1 and _norm(parts[0]) in ROOTS:      # ~/1.work/office/pdf -> pdf
        parts = parts[1:]
    return _norm(parts[0])


class Projects:
    def __init__(self, path: str | None = None):
        self.path = path or os.environ.get('PM1_PROJECTS') or os.path.join(HOME, '.config/pm1/projects.json')
        self.alias: dict[str, str] = {}
        self.isolated: set[str] = set()
        try:
            cfg = json.load(open(self.path))
        except (OSError, ValueError):
            cfg = {}
        for canon, others in cfg.items():
            if canon == 'isolated':
                continue
            for o in [canon] + list(others or []):
                self.alias[_norm(o)] = _norm(canon)
        self.isolated = {self.name_of(x) for x in cfg.get('isolated', [])}

    def name_of(self, scope_or_name: str) -> str:
        """logical project of a scope, a folder path or a bare name"""
        if scope_or_name.startswith(('proj:', '/', 'cowork:')):
            n = auto_name(scope_or_name)
        else:
            n = _norm(scope_or_name)
        return self.alias.get(n, n)

    def current(self) -> str | None:
        """the project of the folder this server was started in (Claude Code, Codex and Cursor start MCP servers in
        the project folder; Cowork starts them in $HOME: no current project)"""
        env = os.environ.get('PM1_PROJECT')
        if env:
            return self.name_of(env)
        cwd = os.getcwd()
        if os.path.normpath(cwd) == HOME or os.path.basename(cwd).startswith('scratch-'):
            return None
        n = self.name_of(cwd)
        return None if n in ROOTS or n == 'home' else n

    @staticmethod
    def current_scope() -> str:
        """the session scope of the client's own folder (PM1_PROJECT when it is a folder or proj: scope)"""
        env = os.environ.get('PM1_PROJECT', '')
        if env.startswith('proj:'):
            return env
        return 'proj:' + (env if env.startswith('/') else os.getcwd())
