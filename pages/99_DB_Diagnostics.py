# pages/99_DB_Diagnostics.py
from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from reporting_shared import DB_PATH, get_db_status

st.set_page_config(page_title="DB Diagnostics", layout="wide")
st.title("Database Diagnostics")

status = get_db_status(DB_PATH)

st.subheader("Database Path")
st.code(status["DB_PATH"], language="text")

c1, c2, c3 = st.columns(3)
c1.metric("DB Exists", "Yes" if status["exists"] else "No")
c2.metric("DB Size", f"{status['size_bytes']:,} bytes")
c3.metric("Git LFS Pointer", "Yes" if status["is_lfs_pointer"] else "No")

if status["error"]:
    st.error(status["error"])

if not status["exists"]:
    st.warning("maintenance_master.db was not found in the repo root.")
    st.stop()

if status["is_lfs_pointer"]:
    st.error("The DB file appears to be a Git LFS pointer, not the real SQLite database.")
    st.stop()

tables = status["tables"]
st.subheader("Tables Found")
st.write(f"{len(tables):,} tables found.")

required = [
    "Locations_Master",
    "Assets_Master",
    "Asset_History_Merged",
    "Purchase_Orders",
    "Mobile_Service_Report_History",
    "Mobile_Service_Report",
    "Vendors_Master",
    "mx_vendor_audit_current",
    "mx_vendor_audit_manual",
]

with sqlite3.connect(DB_PATH) as conn:
    rows = []
    for t in required:
        exists = t in tables
        count = None
        if exists:
            try:
                count = pd.read_sql_query(f'SELECT COUNT(*) AS n FROM "{t}"', conn)["n"].iloc[0]
            except Exception:
                count = "error"
        rows.append({"Expected Table": t, "Exists": exists, "Rows": count})
    req_df = pd.DataFrame(rows)

st.subheader("Expected Reporting Tables")
st.dataframe(req_df, use_container_width=True, hide_index=True)

st.subheader("All Tables")
st.dataframe(pd.DataFrame({"Table": tables}), use_container_width=True, hide_index=True)
