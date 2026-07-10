"""One-time Google Drive OAuth bootstrap (ARCHITECTURE.md section 11).

Run locally ONCE to mint a long-lived refresh token, then paste the three
GDRIVE_* values into the Coolify env and set STORAGE_BACKEND=gdrive. Nothing here
runs in production.

Usage:
    uv run python scripts/gdrive_auth.py --client-secrets client_secret.json

Create the OAuth client (type: Desktop app) in Google Cloud Console and download
its client_secret.json first. The 'drive.file' scope limits access to files this
app creates.
"""

from __future__ import annotations

import argparse
import json
import sys

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Mint a Google Drive refresh token")
    parser.add_argument("--client-secrets", required=True, help="Path to client_secret.json")
    parser.add_argument("--port", type=int, default=0, help="Local redirect port (0 = auto)")
    args = parser.parse_args()

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("google-auth-oauthlib is required: uv sync", file=sys.stderr)
        return 2

    flow = InstalledAppFlow.from_client_secrets_file(args.client_secrets, SCOPES)
    creds = flow.run_local_server(port=args.port, prompt="consent")

    if not creds.refresh_token:
        print("No refresh token returned. Revoke prior grants and retry with prompt=consent.", file=sys.stderr)
        return 1

    out = {
        "GDRIVE_CLIENT_ID": creds.client_id,
        "GDRIVE_CLIENT_SECRET": creds.client_secret,
        "GDRIVE_REFRESH_TOKEN": creds.refresh_token,
    }
    print("\nSet these in Coolify (mark as secrets), then STORAGE_BACKEND=gdrive:\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
