"""Where pm1 keeps its data by default. One place, so the MCP server, the hooks, the worker and pm1-rule never
open different databases when nothing is configured (they used to: ~/.local/state/pm1 vs pm1-merge)."""
from __future__ import annotations

import os


def config_file() -> str:
    return os.path.expanduser(os.environ.get('PM1_CONFIG') or os.path.join(
        os.environ.get('XDG_CONFIG_HOME') or os.path.expanduser('~/.config'), 'pm1', 'config.toml'))


# keys the config file may set. Never a program to run (label_dual) or a place the judge writes; URLs only to this machine.
SAFE_KEYS = {'db', 'bank', 'scope', 'doc_scope', 'dense', 'token', 'threads', 'judge', 'rerank_profile', 'quality_floor',
             'tool_tier', 'hook_toolout_limit', 'tooloutdir', 'toolout_dir', 'index_models', 'capture_cursor', 'docs', 'code',
             'beam_dense', 'gpu'}
URL_KEYS = {'e5_url', 'e5_doc_url', 'e5_query_url', 'pplx_url', 'rerank_qwen', 'rerank_qwen_vllm', 'rerank_gte', 'rerank_bge',
            'rerank_qwen_cpu', 'rerank_gte_cpu'}


def _loopback(url: str) -> bool:
    import urllib.parse
    return (urllib.parse.urlparse(str(url)).hostname or '') in ('127.0.0.1', 'localhost', '::1')


def env_int(name: str, default: int) -> int:
    """An integer setting that never takes a process down: a typo falls back to the default."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def apply_config() -> list[str]:
    """Read ~/.config/pm1/config.toml (or $PM1_CONFIG) and set the environment variables it names, unless the
    environment already has them: an environment variable always wins over the file. Flat table:

        [pm1]
        db = "~/pm1/memory.sqlite"      # -> PM1_DB
        dense = "none"                  # -> PM1_DENSE
        judge = "off"                   # -> PM1_JUDGE

    Only the keys in SAFE_KEYS (and loopback URLs for URL_KEYS) are taken. A missing or unreadable file changes nothing. Returns the variables it set."""
    try:
        fh = open(config_file(), 'rb')          # no file, no import: the hooks must stay fast
    except OSError:
        return []
    try:
        with fh:
            try:
                import tomllib
            except ModuleNotFoundError:       # Python 3.10 (the minimum supported version)
                import tomli as tomllib
            table = tomllib.load(fh).get('pm1') or {}
    except Exception:
        return []
    if not isinstance(table, dict):
        return []
    done = []
    for k, v in table.items():
        key = str(k).lower().replace('-', '_')
        name = 'PM1_' + key.upper()
        if key not in SAFE_KEYS and not (key in URL_KEYS and _loopback(v)):
            continue                           # unknown keys, programs and remote endpoints are not taken from the file
        if isinstance(v, (str, int, float, bool)) and name not in os.environ:
            os.environ[name] = ('1' if v else '0') if isinstance(v, bool) else str(v)
            done.append(name)
    return done


apply_config()          # every entry point imports this module before it reads any PM1_* setting


def state_dir() -> str:
    return os.path.join(os.environ.get('XDG_STATE_HOME') or os.path.expanduser('~/.local/state'), 'pm1')


def default_db() -> str:
    """PM1_DB, else $XDG_STATE_HOME/pm1/memory.sqlite (~/.local/state/pm1/memory.sqlite)."""
    return os.path.expanduser(os.environ.get('PM1_DB') or os.path.join(state_dir(), 'memory.sqlite'))
