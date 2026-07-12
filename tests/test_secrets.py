"""SecretStore: the ONLY gateway to credentials. The keyring is the source
of truth; the environment seeds it exactly once (migration path); the MSAL
token cache lives in the keyring, never as a plaintext file."""

import outlook
import secrets_store


# --- core store ---------------------------------------------------------------

def test_set_get_roundtrip(temp_secrets):
    secrets_store.set_secret("GEMINI_API_KEY", "k-123")
    assert secrets_store.get_secret("GEMINI_API_KEY") == "k-123"
    assert temp_secrets.store[("nova", "GEMINI_API_KEY")] == "k-123"


def test_env_var_seeds_the_keyring_exactly_once(temp_secrets, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert secrets_store.get_secret("GEMINI_API_KEY") == "from-env"
    assert temp_secrets.store[("nova", "GEMINI_API_KEY")] == "from-env"  # migrated

    monkeypatch.delenv("GEMINI_API_KEY")               # env var removed by owner
    assert secrets_store.get_secret("GEMINI_API_KEY") == "from-env"  # keyring holds it


def test_keyring_wins_over_a_stale_env_var(temp_secrets, monkeypatch):
    secrets_store.set_secret("GROQ_API_KEY", "rotated-key")
    monkeypatch.setenv("GROQ_API_KEY", "old-key-still-in-profile")
    assert secrets_store.get_secret("GROQ_API_KEY") == "rotated-key"


def test_missing_everywhere_is_none(temp_secrets):
    assert secrets_store.get_secret("GEMINI_API_KEY") is None


def test_delete_is_idempotent(temp_secrets):
    secrets_store.set_secret("GROQ_API_KEY", "x")
    secrets_store.delete_secret("GROQ_API_KEY")
    secrets_store.delete_secret("GROQ_API_KEY")       # second delete: no raise
    assert secrets_store.get_secret("GROQ_API_KEY") is None


# --- MSAL token cache (outlook.py) ---------------------------------------------

def test_token_cache_roundtrips_through_the_keyring(temp_secrets):
    cache = outlook._load_token_cache()
    assert cache.serialize() is not None              # empty but valid
    cache.deserialize('{"AccessToken": {}}')
    cache.has_state_changed = True
    outlook._save_token_cache(cache)
    assert secrets_store.get_secret(outlook.TOKEN_CACHE_SECRET) == cache.serialize()

    again = outlook._load_token_cache()
    assert again.serialize() == cache.serialize()


def test_save_is_a_noop_when_nothing_changed(temp_secrets):
    cache = outlook._load_token_cache()
    cache.has_state_changed = False
    outlook._save_token_cache(cache)
    assert secrets_store.get_secret(outlook.TOKEN_CACHE_SECRET) is None


def test_legacy_plaintext_cache_migrates_in_and_the_file_is_removed(
        temp_secrets, temp_backlog, tmp_path, monkeypatch):
    legacy = tmp_path / ".ms_token_cache.json"
    legacy.write_text('{"RefreshToken": {"legacy": true}}')
    monkeypatch.setattr(outlook, "LEGACY_TOKEN_CACHE", legacy)

    cache = outlook._load_token_cache()

    assert not legacy.exists()                        # plaintext copy is gone
    assert secrets_store.get_secret(outlook.TOKEN_CACHE_SECRET) is not None
    assert "RefreshToken" in cache.serialize()
    assert any(e["action_type"] == "secret_migrated"
               for e in temp_backlog.log_entries())   # audit-logged (§4)


def test_legacy_file_survives_if_the_keyring_write_fails(temp_secrets, tmp_path,
                                                         monkeypatch):
    """Never delete the only copy of a credential: if the keyring write is
    not verifiable, the plaintext file must stay."""
    legacy = tmp_path / ".ms_token_cache.json"
    legacy.write_text('{"RefreshToken": {}}')
    monkeypatch.setattr(outlook, "LEGACY_TOKEN_CACHE", legacy)

    class WriteDropsEverything:
        def get_password(self, service, name):
            return None

        def set_password(self, service, name, value):
            pass  # silently loses the write (locked keyring, backend error)

        def delete_password(self, service, name):
            pass

    secrets_store.configure(WriteDropsEverything())
    outlook._load_token_cache()
    assert legacy.exists()                            # not deleted
