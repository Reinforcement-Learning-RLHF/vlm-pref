"""
Google Sheets read/write for preference labels.

Credentials are loaded in this order:
1. GOOGLE_APPLICATION_CREDENTIALS env var  -- local dev, path to a service account JSON file
2. st.secrets["gcp_service_account"]       -- Streamlit Cloud deployment

The target spreadsheet is identified by:
1. SHEETS_SPREADSHEET_ID env var
2. st.secrets["sheets"]["spreadsheet_id"]

If neither is set, spreadsheet_id() returns None and callers fall back to local CSV.
"""

from __future__ import annotations
import os

import gspread
import streamlit as st


@st.cache_resource
def _client() -> gspread.Client:
    """Cached gspread client — created once per Streamlit session, reused across calls."""
    try:
        info = dict(st.secrets["gcp_service_account"])
        return gspread.service_account_from_dict(info)
    except Exception:
        pass

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        raise RuntimeError(
            "No GCP credentials found.\n"
            "  Local: set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json\n"
            "  Cloud: add [gcp_service_account] to .streamlit/secrets.toml"
        )
    return gspread.service_account(filename=creds_path)


def spreadsheet_id() -> str | None:
    """Return the configured spreadsheet ID, or None (triggers CSV fallback)."""
    sid = os.environ.get("SHEETS_SPREADSHEET_ID")
    if sid:
        return sid
    try:
        return st.secrets["sheets"]["spreadsheet_id"]
    except Exception as e:
        print(f"[sheets] could not read spreadsheet_id: {e}")
        return None


def read_rows(sid: str, sheet_name: str) -> list[dict]:
    """Return all data rows as a list of dicts, matching csv.DictReader output."""
    ws = _client().open_by_key(sid).worksheet(sheet_name)
    return ws.get_all_records()


def append_row(sid: str, sheet_name: str, row: dict, fieldnames: list[str]) -> None:
    """
    Append one row to the sheet.
    If the sheet is empty, writes header + data in a single API call to avoid race conditions.
    """
    ws = _client().open_by_key(sid).worksheet(sheet_name)
    data_row = [str(row.get(f, "")) for f in fieldnames]

    if not ws.row_values(1):
        # Write header and data together in one call
        ws.append_rows([fieldnames, data_row], value_input_option="RAW")
    else:
        ws.append_rows([data_row], value_input_option="RAW")
