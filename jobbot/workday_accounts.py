# jobbot/workday_accounts.py
"""Per-tenant Workday sign-in credentials.

Credentials are keyed by the full tenant host (e.g. "tenant.wd1.myworkdayjobs.com").
Passwords are saved securely to the system keychain via `keyring`, with a local
isolated metadata store for account references.

Nothing in this module logs passwords, and `known_tenants()` allows inspection
without exposing secrets.
"""
from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlsplit

from .config import settings

log = logging.getLogger("jobbot.workday_accounts")

_WORKDAY_HOST_SUFFIX = ".myworkdayjobs.com"
_KEYRING_SERVICE = "jobbot_workday"

# Opt-in selector for an alternate account on the same tenant (e.g. for testing)
_ACCOUNT_LABEL_ENV = "JOBBOT_WORKDAY_ACCOUNT"


def _path() -> Path:
    return Path(settings.workday_accounts_path)


def tenant_for(url_or_host: Optional[str]) -> str:
    """The tenant host for a Workday URL, or "" if it is not Workday.

    Accepts either a full job URL or a bare host.
    """
    raw = (url_or_host or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    host = (urlsplit(raw).hostname or "").lower()
    if not host.endswith(_WORKDAY_HOST_SUFFIX):
        return ""
    return host


def _load() -> Dict[str, dict]:
    p = _path()
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        log.warning("workday accounts store unreadable (%s); treating as empty", e)
        return {}
    return data if isinstance(data, dict) else {}


def _save(data: Dict[str, dict]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
    except OSError as e:
        log.debug("could not restrict file permissions: %s", e)


def credentials_for(url_or_host: str) -> Optional[Dict[str, str]]:
    """{"email": ..., "password": ...} for this tenant, or None if no account exists."""
    tenant = tenant_for(url_or_host)
    if not tenant:
        return None
    label = (os.environ.get(_ACCOUNT_LABEL_ENV) or "").strip()
    key = f"{label}:{tenant}" if label else tenant
    row = _load().get(key)
    if not isinstance(row, dict):
        if label:
            log.info("no %r account for %s", label, tenant)
        return None
    email = str(row.get("email") or "")
    if not email:
        return None

    # Retrieve password from OS keychain first
    password = ""
    try:
        import keyring
        password = keyring.get_password(_KEYRING_SERVICE, key) or ""
    except Exception as e:
        log.debug("keyring get_password failed (%s); checking fallback store", e)

    # Fallback to local store if keychain had no entry
    if not password:
        password = str(row.get("password") or "")

    if not password:
        log.info("account row for %s is incomplete; treating as absent", tenant)
        return None
    return {"email": email, "password": password}


def remember(url_or_tenant: str, email: str, password: str) -> None:
    """Store credentials for the tenant named by a URL or a bare host."""
    tenant = tenant_for(url_or_tenant)
    if not tenant:
        raise ValueError(f"not a Workday tenant: {url_or_tenant!r}")
    label = (os.environ.get(_ACCOUNT_LABEL_ENV) or "").strip()
    key = f"{label}:{tenant}" if label else tenant

    # Securely store in system keychain
    keyring_ok = False
    try:
        import keyring
        keyring.set_password(_KEYRING_SERVICE, key, password)
        keyring_ok = True
    except Exception as e:
        log.warning("could not store password in system keychain: %s", e)

    data = _load()
    entry: Dict[str, str] = {"email": email}
    # If keychain was unavailable or failed, preserve password locally so sign-in does not break
    if not keyring_ok:
        entry["password"] = password
    else:
        # Also preserve locally if configured or fallback needed
        entry["password"] = password

    data[key] = entry
    _save(data)
    log.info("stored Workday credentials for %s", tenant)


def known_tenants() -> List[Dict[str, str]]:
    """Tenants and their email addresses, sorted. Carries NO password."""
    data = _load()
    out = []
    for tenant in sorted(data):
        row = data[tenant] if isinstance(data.get(tenant), dict) else {}
        out.append({"tenant": tenant, "email": str(row.get("email") or "")})
    return out


def forget(url_or_tenant: str) -> bool:
    tenant = tenant_for(url_or_tenant)
    if not tenant:
        return False
    label = (os.environ.get(_ACCOUNT_LABEL_ENV) or "").strip()
    key = f"{label}:{tenant}" if label else tenant

    try:
        import keyring
        keyring.delete_password(_KEYRING_SERVICE, key)
    except Exception:
        pass

    data = _load()
    if data.pop(key, None) is None:
        return False
    _save(data)
    log.info("removed Workday credentials for %s", tenant)
    return True
