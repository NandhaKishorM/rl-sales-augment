"""Model bundle resolver: the trained bundles live on GitHub Releases, not in the wheel.

Resolution order for each variant (first hit wins):
  1. $RSA_MODEL_DIR/<file>            explicit local dir (offline / air-gapped deployments)
  2. <package>/data/<file>            a file dropped next to the code (dev installs)
  3. ~/.cache/rl-sales-augment/<file> the download cache
  4. download from GitHub Releases    sha256-verified, atomic rename into the cache

Zero dependencies (urllib). Set RSA_MODEL_DIR to skip networking entirely.
"""
from __future__ import annotations
import hashlib
import os
import urllib.request

_BASE = "https://github.com/NandhaKishorM/rl-sales-augment/releases/download/models-v1/"

REGISTRY = {
    # variant: (filename, sha256)
    "e4b": ("rl_sales_agent_v3.pt", "dcf85e22a1cb7e7eb3fe5cc3fc1443ec38a0482cb5898c6dd687fcdc6755ed95"),
    "e2b": ("rl_sales_agent_v2.pt", "c3cecd141df1b4a715ac5f85d6c2dcded4318003b9a6ff0f4ed62d580471c35c"),
}


def _cache_dir():
    return os.environ.get("RSA_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "rl-sales-augment")


def ensure_model(variant: str = "e4b") -> str:
    """Return a local path to the bundle for `variant` ("e4b" default, "e2b"), downloading
    from the public GitHub release on first use."""
    if variant not in REGISTRY:
        raise ValueError(f"unknown model variant {variant!r}; choose from {sorted(REGISTRY)}")
    fname, sha = REGISTRY[variant]

    env_dir = os.environ.get("RSA_MODEL_DIR")
    if env_dir and os.path.exists(os.path.join(env_dir, fname)):
        return os.path.join(env_dir, fname)
    pkg_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", fname)
    if os.path.exists(pkg_local):
        return pkg_local
    cached = os.path.join(_cache_dir(), fname)
    if os.path.exists(cached):
        return cached

    url = _BASE + fname
    os.makedirs(_cache_dir(), exist_ok=True)
    tmp = cached + ".part"
    print(f"rl-sales-augment: downloading {fname} (one-time, cached in {_cache_dir()}) ...")
    urllib.request.urlretrieve(url, tmp)
    digest = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
    if digest != sha:
        os.remove(tmp)
        raise RuntimeError(f"sha256 mismatch for {fname}: got {digest[:16]}..., "
                           f"expected {sha[:16]}...; download corrupted, please retry")
    os.replace(tmp, cached)
    print(f"rl-sales-augment: {fname} ready ({os.path.getsize(cached)/1e6:.0f} MB)")
    return cached
