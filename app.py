# Reporting_App.py
# Main starter app for the Maintenance Reporting dashboard.

from __future__ import annotations

import sqlite3
from pathlib import Path

import streamlit as st

from reporting_shared import DB_PATH, load_locations, load_assets, get_valid_locations


st.set_page_config(page_title="Maintenance Reporting", layout="wide")

st.title("Maintenance Reporting")
st.caption("Reporting dashboard using maintenance_master.db from this GitHub repo")

db_path = Path(DB_PATH)

st.subheader("Database Status")
st.code(str(db_path), language="text")

if not db_path.exists():
    st.error("maintenance_master.db was not found in the repo root.")
    st.info("Upload maintenance_master.db beside app.py, not inside the pages folder.")
    st.stop()

try:
    with sqlite3.connect(str(db_path)) as conn:
        integrity = conn.execute("PRAGMA integrity_check;").fetchone()[0]
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;").fetchall()
    if str(integrity).lower() != "ok":
        st.error(f"SQLite integrity check failed: {integrity}")
        st.stop()
except Exception as exc:
    st.error("The database file exists, but SQLite could not open it.")
    st.exception(exc)
    st.info("This often means GitHub uploaded a Git LFS pointer file instead of the real .db file, or the file was corrupted during upload.")
    st.stop()

locations_df = load_locations(DB_PATH)
assets_df = load_assets(DB_PATH)

st.session_state["reporting_locations_df"] = locations_df
st.session_state["reporting_assets_df"] = assets_df
st.session_state["reporting_valid_locations"] = get_valid_locations(locations_df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Database Found", "Yes")
c2.metric("SQLite Integrity", "OK")
c3.metric("Tables", f"{len(tables):,}")
c4.metric("DB Size", f"{db_path.stat().st_size / (1024*1024):,.1f} MB")

c5, c6, c7 = st.columns(3)
c5.metric("Valid Locations", f"{len(st.session_state['reporting_valid_locations']):,}")
c6.metric("Assets Loaded", f"{len(assets_df):,}")
c7.metric("Location Rows Loaded", f"{len(locations_df):,}")

with st.expander("Detected SQLite Tables", expanded=False):
    st.write([t[0] for t in tables])

st.subheader("Pages")
st.markdown(
    """
    - **Asset History Report**
    - **Purchase Order Report**
    - **Mobile Service Report**
    - **Vendor Report**
    - **Inventory Analysis Report**
    - **Inventory Re-Stock Report**
    """
)

st.info("Use the page menu in the sidebar to open each report page.")
