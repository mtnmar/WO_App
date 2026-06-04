# 09_Asset_Risk_Replacement_Planner.py
# Placeholder page for future Asset Risk & Replacement Planning system.
# This online-version placeholder does not connect to the production database yet.

import streamlit as st

try:
    from auth_helper import require_login
except Exception:
    def require_login():
        return None
import pandas as pd
from datetime import date

st.set_page_config(
    page_title="Asset Risk & Replacement Planner",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------------------------
# Page Header
# ------------------------------------------------------------
st.title("📊 Asset Risk & Replacement Planner")
st.caption("Placeholder dashboard for future continuous asset replacement, lifecycle, and risk planning.")

st.info(
    "This page is currently a placeholder for the future Asset Risk & Replacement Planning system. "
    "The full desktop version will eventually calculate replacement priority using CMMS, financial, "
    "downtime, criticality, maintenance cost, parts risk, and condition data."
)

# ------------------------------------------------------------
# Overview / Status Placeholder
# ------------------------------------------------------------
st.subheader("Program Overview")

overview_text = """
The Asset Risk & Replacement Planner will be used to continuously evaluate equipment health,
replacement risk, capital exposure, and lifecycle priority across the maintenance program.

Future functionality will include:
- Top at-risk assets
- Replacement priority scoring
- Remaining useful life estimates
- Maintenance cost vs. replacement value
- Downtime and production impact
- Parts availability and vendor lead-time risk
- Capital replacement forecasting
- Location and department-level risk summaries
"""

st.markdown(overview_text)

# ------------------------------------------------------------
# Placeholder KPI Row
# ------------------------------------------------------------
st.subheader("Future KPI Summary")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Assets Evaluated", "Pending")

with kpi2:
    st.metric("High Risk Assets", "Pending")

with kpi3:
    st.metric("Estimated 5-Year Capital Need", "Pending")

with kpi4:
    st.metric("Deferred Replacement Risk", "Pending")

# ------------------------------------------------------------
# Placeholder Filters
# ------------------------------------------------------------
st.subheader("Future Filters")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    st.selectbox(
        "Location",
        ["All Locations", "Placeholder"],
        disabled=True,
        help="Future filter will use All Parent Locations / NS Location data.",
    )

with filter_col2:
    st.selectbox(
        "Asset Type",
        ["All Asset Types", "Placeholder"],
        disabled=True,
        help="Future filter will use asset type/category data.",
    )

with filter_col3:
    st.selectbox(
        "Risk Level",
        ["All Risk Levels", "Highly Critical", "Critical", "Important", "Standard"],
        disabled=True,
        help="Future filter will use calculated replacement risk bands.",
    )

# ------------------------------------------------------------
# Placeholder Top Risk Table
# ------------------------------------------------------------
st.subheader("Top At-Risk Assets")

placeholder_assets = pd.DataFrame(
    [
        {
            "Rank": 1,
            "Asset": "Example Primary Crusher",
            "Location": "Example Location",
            "Asset Criticality": "Highly Critical",
            "Replacement Risk Score": 96,
            "Recommendation": "Replace / Capital Review",
            "Estimated Replacement Year": "2027",
        },
        {
            "Rank": 2,
            "Asset": "Example Cone Crusher",
            "Location": "Example Location",
            "Asset Criticality": "Critical",
            "Replacement Risk Score": 88,
            "Recommendation": "Rebuild / Evaluate",
            "Estimated Replacement Year": "2028",
        },
        {
            "Rank": 3,
            "Asset": "Example Loader",
            "Location": "Example Location",
            "Asset Criticality": "Critical",
            "Replacement Risk Score": 82,
            "Recommendation": "Monitor / Capital Forecast",
            "Estimated Replacement Year": "2029",
        },
    ]
)

st.dataframe(
    placeholder_assets,
    use_container_width=True,
    hide_index=True,
)

# ------------------------------------------------------------
# Future Scoring Model
# ------------------------------------------------------------
st.subheader("Future Replacement Priority Inputs")

input_data = pd.DataFrame(
    [
        {"Input": "Asset Criticality", "Purpose": "Weights risk based on production, safety, downtime, redundancy, and operational impact."},
        {"Input": "Asset Age / Expected Life", "Purpose": "Compares current age against expected useful life and remaining useful life."},
        {"Input": "Maintenance Cost", "Purpose": "Compares repair cost trends against current replacement value."},
        {"Input": "Downtime Cost", "Purpose": "Captures production loss and operational disruption."},
        {"Input": "Failure Frequency / MTBF", "Purpose": "Identifies declining reliability and repeat failure patterns."},
        {"Input": "MTTR / Repair Duration", "Purpose": "Measures maintainability and repair complexity."},
        {"Input": "Parts Availability Risk", "Purpose": "Raises priority when parts are obsolete, scarce, or long lead time."},
        {"Input": "Vendor Lead-Time Risk", "Purpose": "Uses vendor performance and lead-time scoring where available."},
        {"Input": "Condition Score", "Purpose": "Adds inspection-based condition assessment."},
        {"Input": "Replacement Cost", "Purpose": "Supports capital planning and replacement forecasting."},
    ]
)

st.dataframe(
    input_data,
    use_container_width=True,
    hide_index=True,
)

# ------------------------------------------------------------
# Future Placeholder Sections
# ------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Future Capital Forecast")
    st.warning("Capital forecast charts and tables will be added in the desktop version.")

with right:
    st.subheader("Future Risk Trend")
    st.warning("Risk trend charts will be added after the scoring database is created.")

# ------------------------------------------------------------
# Development Notes
# ------------------------------------------------------------
with st.expander("Development Notes", expanded=False):
    st.markdown(
        """
        Future database/table ideas:

        - `asset_replacement_scores`
        - `asset_replacement_runs`
        - `asset_condition_assessments`
        - `asset_replacement_overrides`
        - `asset_capital_forecast`

        Suggested calculated fields:

        - `replacement_priority_score`
        - `remaining_useful_life`
        - `maintenance_cost_ratio`
        - `downtime_cost_estimate`
        - `parts_risk_score`
        - `vendor_lead_time_score`
        - `condition_score`
        - `recommended_action`
        - `estimated_replacement_year`
        - `capital_budget_year`
        """
    )

st.divider()
st.caption(f"Placeholder page only — no live database connection. Last updated: {date.today().strftime('%Y-%m-%d')}")
