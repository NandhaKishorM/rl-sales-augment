"""Tiny zero-dependency .env loader, so API keys never have to live in code.

Providers call `load_env()` automatically before reading credentials: it looks for a `.env`
file in the current working directory (walking up a couple of parents), parses KEY=VALUE
lines, and fills os.environ WITHOUT overriding variables that are already set. Real
environment variables therefore always win over the file. Never commit `.env` to git.
"""
from __future__ import annotations
import os

_loaded: set = set()


def load_env(path: str = None) -> dict:
    """Load a .env file into os.environ (missing keys only). Returns the vars it added.

    path=None searches ./.env, then up to two parent directories. Safe to call repeatedly;
    each file is only read once per process. Also exposed as `rl_sales_augment.load_env`.
    """
    if path is None:
        d = os.getcwd()
        for _ in range(3):
            cand = os.path.join(d, ".env")
            if os.path.isfile(cand):
                path = cand
                break
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        if path is None:
            return {}
    path = os.path.abspath(path)
    if path in _loaded:
        return {}
    _loaded.add(path)
    added = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
                    added[k] = v
    except OSError:
        pass
    return added
