# Reporting_App.py
# Main starter app for the Maintenance Reporting dashboard.

from __future__ import annotations

import streamlit as st

from reporting_shared import DB_PATH, load_locations, load_assets, get_valid_locations
from auth_helper import require_login


st.set_page_config(page_title="Maintenance Reporting", layout="wide")
require_login()

st.title("Maintenance Reporting")
st.caption("Starter reporting dashboard using maintenance_master.db")

locations_df = load_locations(DB_PATH)
assets_df = load_assets(DB_PATH)

st.session_state["reporting_locations_df"] = locations_df
st.session_state["reporting_assets_df"] = assets_df
st.session_state["reporting_valid_locations"] = get_valid_locations(locations_df)

c1, c2, c3 = st.columns(3)
c1.metric("Valid Locations", f"{len(st.session_state['reporting_valid_locations']):,}")
c2.metric("Assets Loaded", f"{len(assets_df):,}")
c3.metric("Location Rows Loaded", f"{len(locations_df):,}")

st.subheader("Data Source")
st.code(DB_PATH, language="text")

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
