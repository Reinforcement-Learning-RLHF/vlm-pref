"""
Generate a valid .streamlit/secrets.toml from a Google service account JSON key file.
The private_key field must be on a single line with \\n escape sequences — this script
handles that automatically.

Usage:
    python scripts/gen_secrets_toml.py path/to/service_account.json YOUR_SPREADSHEET_ID
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ".streamlit" / "secrets.toml"


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python scripts/gen_secrets_toml.py <service_account.json> <spreadsheet_id>")
        sys.exit(1)

    json_path = Path(sys.argv[1])
    spreadsheet_id = sys.argv[2]

    with open(json_path) as f:
        sa = json.load(f)

    # private_key must be a single TOML line — replace actual newlines with \n
    private_key = sa["private_key"].replace("\n", "\\n")

    toml = f"""[gcp_service_account]
type                        = "{sa['type']}"
project_id                  = "{sa['project_id']}"
private_key_id              = "{sa['private_key_id']}"
private_key                 = "{private_key}"
client_email                = "{sa['client_email']}"
client_id                   = "{sa['client_id']}"
auth_uri                    = "{sa['auth_uri']}"
token_uri                   = "{sa['token_uri']}"
auth_provider_x509_cert_url = "{sa['auth_provider_x509_cert_url']}"
client_x509_cert_url        = "{sa['client_x509_cert_url']}"

[sheets]
spreadsheet_id = "{spreadsheet_id}"

[huggingface]
video_repo_id = "nafisatibrahim/vlm-pref-videos"
"""

    OUT.write_text(toml, encoding="utf-8")
    print(f"Written to {OUT}")
    print("Never commit this file — it is already in .gitignore.")


if __name__ == "__main__":
    main()
