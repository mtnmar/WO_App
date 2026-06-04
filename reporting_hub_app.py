# Reporting_App.py
# Main app for the Maintenance Reporting dashboard.

from __future__ import annotations

import streamlit as st

from reporting_shared import DB_PATH, load_locations, load_assets, get_valid_locations
from auth_helper import require_login


st.set_page_config(page_title="Maintenance Reporting", layout="wide")
require_login()

st.title("Maintenance Reporting")
st.caption("Maintenance reporting dashboard using maintenance_master.db")

locations_df = load_locations(DB_PATH)
assets_df = load_assets(DB_PATH)

st.session_state["reporting_locations_df"] = locations_df
st.session_state["reporting_assets_df"] = assets_df
st.session_state["reporting_valid_locations"] = get_valid_locations(locations_df)

c1, c2, c3 = st.columns(3)
c1.metric("Valid Locations", f"{len(st.session_state['reporting_valid_locations']):,}")
c2.metric("Assets Loaded", f"{len(assets_df):,}")
c3.metric("Location Rows Loaded", f"{len(locations_df):,}")

st.subheader("CMMS Program Overview")
st.info(
    "Placeholder for the CMMS program overview/status section. "
    "A future app/table in maintenance_master.db can feed current status, warnings, trends, and program messages here."
)

with st.container(border=True):
    st.markdown("**Current Status / Message Placeholder**")
    st.write(
        "Use this section later for written overview notes, system warnings, data quality alerts, "
        "reporting trends, active rollout items, and CMMS program updates."
    )

overview_cols = st.columns(4)
overview_cols[0].metric("CMMS Status", "Placeholder")
overview_cols[1].metric("Active Warnings", "Pending")
overview_cols[2].metric("Trend Alerts", "Pending")
overview_cols[3].metric("Program Messages", "Pending")

with st.expander("Future overview database hook", expanded=False):
    st.markdown(
        """
        Planned future table options:
        - `cmms_program_overview`
        - `cmms_program_messages`
        - `cmms_status_warnings`
        - `cmms_trending_alerts`

        Suggested fields:
        - `message_date`
        - `message_type`
        - `status_level`
        - `title`
        - `message`
        - `location`
        - `owner`
        - `active`
        - `sort_order`
        """
    )

st.subheader("Data Source")
st.code(DB_PATH, language="text")

st.subheader("Pages")

pages = [
    ("01", "Asset History Report", "Asset history, cost, work order, PO, and issue review reporting."),
    ("02", "Purchase Order Report", "PO reporting, CMMS monitored cost, non-CMMS cost, and financial exports."),
    ("03", "Mobile Service Report", "Mobile service history, due/overdue service status, and meter/service reporting."),
    ("04", "Vendor Report", "Vendor contact, lead time, and vendor audit reporting."),
    ("05", "Inventory Analysis Report", "Inventory movement, stock value, usage, and transaction analysis."),
    ("06", "Inventory Re-Stock Report", "Restock recommendations, RFQ/cart support, and purchasing prep."),
    ("07", "Straight Transactions Report", "Finance-ready straight inventory transactions and bag usage reporting."),
    ("08", "MX vs NetSuite PO Cross-Check", "MaintainX vs NetSuite PO comparison, KPI, bypass, and review reporting."),
    ("09", "Asset Risk & Replacement Planner", "Placeholder for future asset replacement risk and lifecycle planning."),
]

st.dataframe(
    [{"Page": p, "Report": name, "Purpose": purpose} for p, name, purpose in pages],
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    """
    **Available report pages:**

    - **01 — Asset History Report**
    - **02 — Purchase Order Report**
    - **03 — Mobile Service Report**
    - **04 — Vendor Report**
    - **05 — Inventory Analysis Report**
    - **06 — Inventory Re-Stock Report**
    - **07 — Straight Transactions Report**
    - **08 — MX vs NetSuite PO Cross-Check**
    - **09 — Asset Risk & Replacement Planner**
    """
)

st.info("Use the page menu in the sidebar to open each report page.")
