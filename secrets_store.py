"""
NOVA — SecretStore: the single gateway to credentials (project guide hardening).

Wraps the `keyring` package (Secret Service backend on this desktop). No
NOVA component reads os.environ or a plaintext file for a credential
directly — everything asks this module.

Migration: the environment is a one-time SEED, not a source of truth. If a
secret is missing from the keyring but present in the environment, it is
written into the keyring and returned; once seeded, the env var can be
removed from the shell profile. The same applies to the legacy plaintext
MSAL token cache file (see outlook.py, which migrates it through here).

Known secrets:
  GEMINI_API_KEY / GROQ_API_KEY  — cloud free-tier keys (llm.py, preflight)
  MS_TOKEN_CACHE                 — serialized MSAL token cache (outlook.py)
"""

from __future__ import annotations

import logging
import os

_SERVICE = "nova"

log = logging.getLogger("nova")

_keyring = None  # lazy: keyring import touches D-Bus; tests inject a fake


def configure(backend) -> None:
    """Tests inject an in-memory fake (get_password/set_password/
    delete_password). Production never calls this."""
    global _keyring
    _keyring = backend


def _kr():
    global _keyring
    if _keyring is None:
        import keyring
        _keyring = keyring
    return _keyring


def get_secret(name: str) -> str | None:
    """The keyring wins; the environment seeds it exactly once."""
    value = _kr().get_password(_SERVICE, name)
    if value is not None:
        return value
    env_value = os.environ.get(name)
    if env_value:
        set_secret(name, env_value)
        log.info("secret %s migrated from environment into the keyring — "
                 "the env var can now be removed", name)
        return env_value
    return None


def set_secret(name: str, value: str) -> None:
    _kr().set_password(_SERVICE, name, value)


def delete_secret(name: str) -> None:
    try:
        _kr().delete_password(_SERVICE, name)
    except Exception:  # noqa: BLE001 — deleting a missing secret is not an error
        pass
