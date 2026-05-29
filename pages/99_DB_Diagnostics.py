from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from auth_helper import require_login
from reporting_shared import DB_PATH, get_db_status

st.set_page_config(page_title="DB Diagnostics", layout="wide")
require_login()

st.title("Database Diagnostics")
status = get_db_status(DB_PATH)

st.write("Database path:")
st.code(status.get("db_path", ""), language="text")

c1, c2, c3 = st.columns(3)
c1.metric("DB Exists", "Yes" if status.get("exists") else "No")
c2.metric("DB Size MB", status.get("size_mb", 0))
c3.metric("Table Count", status.get("table_count", 0))

if status.get("error"):
    st.error(status["error"])

tables = status.get("tables", []) or []
if not tables:
    st.warning("No SQLite tables were found.")
    st.stop()

st.subheader("Tables")
st.dataframe(pd.DataFrame({"Table": tables}), width="stretch", hide_index=True)

st.subheader("Row Counts")
rows = []
try:
    with sqlite3.connect(DB_PATH) as conn:
        for table in tables:
            try:
                count = pd.read_sql_query(f'SELECT COUNT(*) AS row_count FROM "{table}"', conn)["row_count"].iloc[0]
            except Exception as exc:
                count = f"Error: {exc}"
            rows.append({"Table": table, "Rows": count})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
except Exception as exc:
    st.error(f"Could not read row counts: {exc}")
