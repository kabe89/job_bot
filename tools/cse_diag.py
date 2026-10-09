"""CSE diagnostic — dumps the FULL Google Custom Search API response so we can
see exactly why a 403 happens. The error body names the project number the key
belongs to and a direct activation link, which pinpoints project mismatches.

Run:  python tools/cse_diag.py
"""
from __future__ import annotations

import json
import re
import sys

import requests

sys.path.insert(0, ".")
from jobbot.config import settings  # noqa: E402

KEY = settings.google_cse_key
CX = settings.google_cse_cx


def _mask(s: str) -> str:
    if not s:
        return "(empty)"
    return f"{s[:6]}...{s[-4:]} (len {len(s)})"


def main() -> int:
    print("=== Google CSE diagnostic ===")
    print(f"GOOGLE_CSE_KEY: {_mask(KEY)}")
    print(f"GOOGLE_CSE_CX : {_mask(CX)}")
    if not KEY or not CX:
        print("\nFAIL: key and/or cx empty in .env")
        return 1

    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": KEY, "cx": CX, "q": "site:linkedin.com/in recruiter", "num": 3}
    print(f"\nGET {url}")
    print("params: q=%r num=3 (key/cx hidden)" % params["q"])

    try:
        resp = requests.get(url, params=params, timeout=30)
    except Exception as exc:  # noqa: BLE001
        print(f"\nNETWORK ERROR: {type(exc).__name__}: {exc}")
        return 2

    print(f"\nHTTP {resp.status_code} {resp.reason}")
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        print("Non-JSON body:\n" + resp.text[:1000])
        return 3

    print("\n--- Full response body ---")
    print(json.dumps(body, indent=2)[:4000])

    if resp.status_code == 200:
        items = body.get("items", [])
        print(f"\nSUCCESS: {len(items)} result(s).")
        for it in items:
            print("  -", it.get("link", ""))
        return 0

    # Error path: pull out the actionable bits.
    err = body.get("error", {})
    msg = err.get("message", "")
    status = err.get("status", "")
    print("\n--- Diagnosis ---")
    print("message:", msg)
    print("status :", status)
    for d in err.get("errors", []):
        print("  reason:", d.get("reason", ""), "| domain:", d.get("domain", ""))
    found_link = False
    for d in err.get("details", []):
        for link in d.get("links", []):
            u = link.get("url", "")
            print("  activation/help link:", u)
            m = re.search(r"project=(\d+)", u)
            if m:
                found_link = True
                print("\n>>> Google says this KEY belongs to PROJECT NUMBER:",
                      m.group(1))
                print(">>> The Custom Search API must be ENABLED on THAT project.")
                print(">>> Open the link above and click ENABLE (it pre-selects the right project).")

    if resp.status_code == 403 and "does not have the access" in msg and not found_link:
        print(
            "\n>>> VERDICT: the API key is VALID and authenticates, but the\n"
            ">>> Custom Search JSON API is NOT enabled on the project that\n"
            ">>> OWNS this key. 'I already enabled it' almost always means it\n"
            ">>> was enabled on a DIFFERENT project than the key lives in\n"
            ">>> (the Cloud Console project selector is global/easy to mis-set).\n"
            ">>>\n"
            ">>> FOOLPROOF FIX: create a brand-new API key while standing in\n"
            ">>> the SAME project where the API shows 'Manage' (enabled):\n"
            ">>>   1. https://console.cloud.google.com/apis/library/customsearch.googleapis.com\n"
            ">>>      -- note the project in the top bar; ensure it says Manage/Enabled.\n"
            ">>>   2. https://console.cloud.google.com/apis/credentials\n"
            ">>>      -- with the SAME project selected, Create credentials -> API key.\n"
            ">>>   3. Put that new key in .env as GOOGLE_CSE_KEY, re-run this test.")
    return resp.status_code


if __name__ == "__main__":
    raise SystemExit(main())
