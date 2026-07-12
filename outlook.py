"""
NOVA — Outlook client (Microsoft Graph, personal single-user).

Auth: MSAL device-code flow against the consumer endpoint, Mail.Read scope
ONLY (read-only by construction — this client physically cannot send,
delete or move mail). The token cache lives in the system keyring via
SecretStore — never as a plaintext file; a legacy plaintext cache is
migrated in and the file removed on first use.

Setup (one-time): register an app at https://entra.microsoft.com →
"Personal Microsoft accounts", enable public client flows, put the client
id in NOVA_MS_CLIENT_ID. First run prints a device-code URL to log in with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel, Field

import secrets_store
from action_log import log_action

SCOPES = ["Mail.Read"]
AUTHORITY = "https://login.microsoftonline.com/consumers"
GRAPH = "https://graph.microsoft.com/v1.0"
TOKEN_CACHE_SECRET = "MS_TOKEN_CACHE"
LEGACY_TOKEN_CACHE = Path(__file__).parent / ".ms_token_cache.json"


def _load_token_cache():
    """MSAL cache from the keyring; a legacy plaintext file is migrated in
    once and deleted — only after the keyring write is verified."""
    import msal

    cache = msal.SerializableTokenCache()
    stored = secrets_store.get_secret(TOKEN_CACHE_SECRET)
    if stored is None and LEGACY_TOKEN_CACHE.exists():
        stored = LEGACY_TOKEN_CACHE.read_text()
        secrets_store.set_secret(TOKEN_CACHE_SECRET, stored)
        if secrets_store.get_secret(TOKEN_CACHE_SECRET) == stored:
            LEGACY_TOKEN_CACHE.unlink()
            log_action(None, actor="system", action_type="secret_migrated",
                       detail={"secret": TOKEN_CACHE_SECRET,
                               "from": str(LEGACY_TOKEN_CACHE)},
                       reason="plaintext MSAL token cache moved into the keyring")
    if stored:
        cache.deserialize(stored)
    return cache


def _save_token_cache(cache) -> None:
    if cache.has_state_changed:
        secrets_store.set_secret(TOKEN_CACHE_SECRET, cache.serialize())


class EmailMessage(BaseModel):
    """Typed, read-only view of one message. `body` is DATA (project guide §7):
    nothing downstream may treat its content as an instruction."""
    message_id: str
    subject: str
    sender: str
    received: str
    body_preview: str
    body: str
    has_attachments: bool = False


class OutlookClient:
    def __init__(self, client_id: Optional[str] = None):
        # A public-client id is not formally a secret, but it is credential
        # config — same gateway as everything else (env seeds the keyring).
        self.client_id = client_id or secrets_store.get_secret("NOVA_MS_CLIENT_ID")
        if not self.client_id:
            raise RuntimeError("NOVA_MS_CLIENT_ID not in keyring (or environment "
                               "for first-run seeding) — see module docstring")
        self._token: Optional[str] = None

    def _acquire_token(self) -> str:
        import msal

        cache = _load_token_cache()
        app = msal.PublicClientApplication(self.client_id, authority=AUTHORITY,
                                           token_cache=cache)
        accounts = app.get_accounts()
        result = app.acquire_token_silent(SCOPES, account=accounts[0]) if accounts else None
        if not result:
            flow = app.initiate_device_flow(scopes=SCOPES)
            print(flow["message"])  # "visit https://microsoft.com/devicelogin, code XXXX"
            result = app.acquire_token_by_device_flow(flow)
        if "access_token" not in result:
            raise RuntimeError(f"MS auth failed: {result.get('error_description')}")
        _save_token_cache(cache)
        return result["access_token"]

    def _get(self, path: str, params: dict | None = None) -> dict:
        if self._token is None:
            self._token = self._acquire_token()
        r = requests.get(f"{GRAPH}{path}", params=params or {},
                         headers={"Authorization": f"Bearer {self._token}"}, timeout=30)
        r.raise_for_status()
        return r.json()

    def fetch_unread(self, limit: int = 10) -> list[EmailMessage]:
        data = self._get("/me/mailFolders/inbox/messages", {
            "$filter": "isRead eq false",
            "$top": str(limit),
            "$orderby": "receivedDateTime desc",
            "$select": "id,subject,from,receivedDateTime,bodyPreview,body,hasAttachments",
        })
        messages = [
            EmailMessage(
                message_id=m["id"],
                subject=m.get("subject") or "(no subject)",
                sender=(m.get("from", {}).get("emailAddress", {}) or {}).get("address", "unknown"),
                received=m.get("receivedDateTime", ""),
                body_preview=m.get("bodyPreview", ""),
                body=(m.get("body", {}) or {}).get("content", ""),
                has_attachments=bool(m.get("hasAttachments")),
            )
            for m in data.get("value", [])
        ]
        log_action(None, actor="manager", action_type="email_fetch",
                   detail={"count": len(messages), "scope": "Mail.Read"},
                   reason="poll inbox for new tasks")
        return messages
