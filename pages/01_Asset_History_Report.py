# 01_Asset_History_Report.py
# Streamlit page for Asset_history_merged.db reporting

from __future__ import annotations

import io
import os
import sqlite3
from datetime import date, datetime

import pandas as pd
import streamlit as st

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


st.set_page_config(page_title="Location / Asset CMMS Report", layout="wide")

# =========================
# DATABASE CONFIG
# =========================
try:
    from reporting_shared import DB_PATH
except Exception:
    from pathlib import Path
    DB_PATH = str(Path(__file__).resolve().parents[1] / "maintenance_master.db")
TABLE_NAME = "Asset_History_Merged"
LOCATIONS_TABLE = "Locations_Master"
ASSETS_TABLE = "Assets_Master"
REQUIRED_COLS = {"Location", "ASSET"}
DATE_CANDIDATES = ["_completed_dt", "COMPLETED ON", "Completed On", "completed_on"]
COST_CANDIDATES = ["_Cost", "Total cost", "RECEIVED COST", "TOTAL ITEM COST", "Total Parts Cost", "Additional Costs"]

# Work-order audit tables created by the Work Order Audit app.
WO_AUDIT_TABLE = "mx_work_orders_audit"
WO_ISSUES_TABLE = "mx_work_orders_audit_issues"
WO_REVIEW_TABLE = "mx_work_orders_audit_reviewed"

REVIEW_REASON_CODES = [
    "Confirmed error",
    "False positive",
    "In-Proper attention to department allocation",
    "Improper asset allocation",
    "Bulk Work Order, asset not needed",
    "Failure to properly assign parts",
    "Data timing issue",
    "Already corrected",
    "Training issue noted",
    "Other",
]

REVIEW_REASON_DETAILS = {
    "Confirmed error": "Flag confirmed as valid. For parts-related review, this can also mean there was not enough information in the title, description, or notes to verify inventory, PO, or consumable usage.",
    "False positive": "Flag reviewed and determined not to be an actual error.",
    "In-Proper attention to department allocation": "Work order was assigned to the wrong department code, distorting department-level maintenance reporting and KPI rollups.",
    "Improper asset allocation": "Asset allocation/location assignment was not properly maintained in the CMMS, impacting asset history and location accountability.",
    "Bulk Work Order, asset not needed": "Used when the work order intentionally does not require a specific asset, such as a bulk or general work order.",
    "Failure to properly assign parts": "Parts were used, but no inventory transaction or PO was tied to the work order.",
    "Data timing issue": "The supporting data likely exists but was not yet present at the time of review or audit run.",
    "Already corrected": "Issue was already fixed in the CMMS by the time it was reviewed.",
    "Training issue noted": "The review identified a process or training gap that should be addressed to prevent repeats.",
    "Other": "Reviewed item does not fit the predefined reason categories.",
}

HELPER_KPI_TARGETS = {
    "Presently Overdue": 5.0,
    "Backlog": 5.0,
    "Completed Overdue": 5.0,
    "Completed No Start": 2.0,
    "Completed No Due": 2.0,
}

REVIEW_REASON_DEFAULT_TARGETS = {
    "Confirmed error": 3.0,
    "False positive": 1.0,
    "In-Proper attention to department allocation": 1.0,
    "Improper asset allocation": 0.5,
    "Bulk Work Order, asset not needed": 2.0,
    "Failure to properly assign parts": 1.0,
    "Data timing issue": 1.0,
    "Already corrected": 1.0,
    "Training issue noted": 1.0,
    "Other": 1.0,
}


# -----------------------------
# Helpers
# -----------------------------
def _norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"


@st.cache_data(show_spinner=False)
def list_sqlite_tables(db_path: str) -> list[str]:
    if not db_path or not os.path.exists(db_path):
        return []
    with sqlite3.connect(db_path) as conn:
        q = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        return pd.read_sql_query(q, conn)["name"].tolist()


@st.cache_data(show_spinner=False)
def load_data():
    conn = sqlite3.connect(DB_PATH)

    query = f"""
        SELECT *
        FROM {TABLE_NAME}
    """

    df = pd.read_sql_query(query, conn)

    conn.close()

    # Date handling
    if "_completed_dt" in df.columns:
        df["_completed_dt"] = pd.to_datetime(
            df["_completed_dt"],
            errors="coerce"
        )

    # Cost handling
    if "_Cost" in df.columns:
        df["_Cost"] = pd.to_numeric(
            df["_Cost"],
            errors="coerce"
        ).fillna(0)

    return df


@st.cache_data(show_spinner=False)
def load_reference_table(table_name: str) -> pd.DataFrame:
    """Load a shared reference table from the main maintenance database.

    These reference tables are used to control reporting inputs so legacy
    history values do not drive user-facing filter choices.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            q = "SELECT name FROM sqlite_master WHERE type='table' AND name = ?"
            exists = pd.read_sql_query(q, conn, params=[table_name])
            if exists.empty:
                return pd.DataFrame()
            return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_reference_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Central reference data loader for this reporting app.

    Keep this same pattern for later pages so all reporting pages use the
    same proper locations and asset reference source.
    """
    locations_df = load_reference_table(LOCATIONS_TABLE)
    assets_df = load_reference_table(ASSETS_TABLE)
    return locations_df, assets_df


def get_proper_location_options(locations_df: pd.DataFrame, history_df: pd.DataFrame) -> list[str]:
    """Return proper Location filter choices from Locations_Master[All Parents]."""
    if not locations_df.empty and "All Parents" in locations_df.columns:
        proper_locations = sorted({
            _norm_text(x)
            for x in locations_df["All Parents"].dropna().tolist()
            if _norm_text(x)
        })
    else:
        proper_locations = []

    if not proper_locations:
        # Fallback only if Locations_Master is unavailable.
        return sorted({
            _norm_text(x)
            for x in history_df.get("Location", pd.Series(dtype=str)).dropna().tolist()
            if _norm_text(x)
        })

    history_locations = {
        _norm_text(x)
        for x in history_df.get("Location", pd.Series(dtype=str)).dropna().tolist()
        if _norm_text(x)
    }

    # For this page, show proper locations that actually have asset-history rows.
    filtered_locations = [x for x in proper_locations if x in history_locations]
    return filtered_locations or proper_locations




def prepare_df(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, str | None]:
    if df.empty:
        return df, None, None

    df = df.copy()
    date_col = next((c for c in DATE_CANDIDATES if c in df.columns), None)
    cost_col = next((c for c in COST_CANDIDATES if c in df.columns), None)

    if date_col:
        df["Report Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    else:
        df["Report Date"] = pd.NaT

    if cost_col:
        df["Report Cost"] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0)
    else:
        df["Report Cost"] = 0.0

    for c in ["Location", "ASSET", "DB_TYPE", "STATUS", "TITLE", "Description", "WORKORDER", "PO", "P/N", "Vendors"]:
        if c in df.columns:
            df[c] = df[c].map(_norm_text)

    return df, date_col, cost_col


def date_window(df: pd.DataFrame, mode: str, selected_month: date | None, custom_start: date | None, custom_end: date | None):
    valid_dates = df["Report Date"].dropna()
    if valid_dates.empty:
        today = pd.Timestamp.today().normalize()
        return today, today

    max_dt = valid_dates.max().normalize()

    if mode == "YTD":
        return pd.Timestamp(year=max_dt.year, month=1, day=1), max_dt

    if mode == "Monthly":
        base = pd.Timestamp(selected_month or max_dt.date())
        start = pd.Timestamp(year=base.year, month=base.month, day=1)
        end = start + pd.offsets.MonthEnd(0)
        return start, end

    start = pd.Timestamp(custom_start or valid_dates.min().date())
    end = pd.Timestamp(custom_end or max_dt.date())
    return start, end


def filtered_data(df: pd.DataFrame, locations: list[str], assets: list[str], start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out = out[out["Report Date"].notna()]
    out = out[(out["Report Date"] >= start_dt) & (out["Report Date"] <= end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]

    if locations:
        out = out[out["Location"].isin(locations)]
    if assets:
        out = out[out["ASSET"].isin(assets)]

    return out.sort_values("Report Date", ascending=False)


def build_pdf(df: pd.DataFrame, filters: dict, summary: dict) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not installed in this environment.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        rightMargin=0.35 * inch,
        leftMargin=0.35 * inch,
        topMargin=0.35 * inch,
        bottomMargin=0.35 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Location / Asset CMMS Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now():%Y-%m-%d %I:%M %p}", styles["Normal"]))
    story.append(Spacer(1, 8))

    filter_lines = [f"<b>{k}:</b> {v}" for k, v in filters.items()]
    story.append(Paragraph("<br/>".join(filter_lines), styles["Normal"]))
    story.append(Spacer(1, 8))

    summary_data = [["Metric", "Value"]] + [[k, v] for k, v in summary.items()]
    summary_tbl = Table(summary_data, hAlign="LEFT", colWidths=[2.2 * inch, 2.2 * inch])
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_tbl)
    story.append(Spacer(1, 10))

    export_cols = [c for c in ["Report Date", "DB_TYPE", "WORKORDER", "PO", "ASSET", "Location", "TITLE", "P/N", "Vendors", "Report Cost"] if c in df.columns]
    preview = df[export_cols].head(80).copy()
    if "Report Date" in preview.columns:
        preview["Report Date"] = pd.to_datetime(preview["Report Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "Report Cost" in preview.columns:
        preview["Report Cost"] = preview["Report Cost"].map(_money)

    table_data = [export_cols] + preview.fillna("").astype(str).values.tolist()
    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.2, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(Paragraph("Raw Data Preview - first 80 filtered rows", styles["Heading2"]))
    story.append(tbl)

    doc.build(story)
    return buffer.getvalue()



# -----------------------------
# Work Order Audit KPI Helpers
# -----------------------------
def _table_exists(table_name: str) -> bool:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            return not pd.read_sql_query(q, conn, params=[table_name]).empty
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def load_optional_db_table(table_name: str) -> pd.DataFrame:
    if not _table_exists(table_name):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    except Exception:
        return pd.DataFrame()


def _text_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[col].fillna("").astype(str).str.strip()


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(0)


def _norm_id_series(series: pd.Series) -> pd.Series:
    s = series.copy().astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower().strip(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower().strip() in lower:
            return lower[c.lower().strip()]
    return None


def _wo_key_set_from_history(history_df: pd.DataFrame) -> set[str]:
    if history_df.empty:
        return set()
    wo_col = _first_existing_col(history_df, ["WORKORDER", "Work Order", "Work Order ID", "WO", "WO ID"])
    if not wo_col:
        return set()
    return set(_norm_id_series(history_df[wo_col]).dropna().astype(str).tolist())


def prepare_wo_audit_view(audit_df: pd.DataFrame, selected_locations: list[str], selected_assets: list[str], start_dt: pd.Timestamp, end_dt: pd.Timestamp, history_filtered: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty:
        return pd.DataFrame()

    view = audit_df.copy()
    created_col = _first_existing_col(view, ["Created on", "Created", "Created_dt"])
    completed_col = _first_existing_col(view, ["Completed on", "Completed", "Completed_dt"])
    location_col = _first_existing_col(view, ["Location", "NS Location", "Asset_All Parent Locations", "All Parent Locations"])
    asset_col = _first_existing_col(view, ["Asset", "Asset_Name", "Asset Name", "Name"])
    id_col = _first_existing_col(view, ["ID", "wo_id", "Work Order ID"])

    view["__Created_dt"] = pd.to_datetime(view[created_col], errors="coerce") if created_col else pd.NaT
    view["__Completed_dt"] = pd.to_datetime(view[completed_col], errors="coerce") if completed_col else pd.NaT
    view["__Location"] = _text_series(view, location_col) if location_col else ""
    view["__Asset"] = _text_series(view, asset_col) if asset_col else ""
    view["__WO_ID"] = _norm_id_series(view[id_col]) if id_col else pd.Series(pd.NA, index=view.index)

    # Do NOT restrict KPI rows to Work Orders that appear in Asset_History_Merged.
    # Asset history only contains WOs with history/cost activity, while the Work Order Audit
    # app counts directly from mx_work_orders_audit. Restricting by history WO IDs makes the
    # KPI totals lower than the audit app.

    if selected_locations:
        view = view[view["__Location"].isin(selected_locations)].copy()
    if selected_assets and "__Asset" in view.columns:
        view = view[view["__Asset"].isin(selected_assets)].copy()

    # Do not apply the report date range to the audit KPI population.
    # The Work Order Audit app itself already limits the audit table to Created on >= 2026-01-01.
    # Keeping the same population makes the KPI cards match the audit app. Location/Asset filters
    # still apply above when selected.

    return view


def filter_issues_reviews_to_audit(issues_df: pd.DataFrame, reviews_df: pd.DataFrame, audit_view: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter issue/review tables to the same WO population as the audit view.

    Important: this intentionally does NOT date-filter by first_seen_ts or reviewed_ts.
    The Work Order Audit app counts issue/review state directly from the audit tables.
    Date-filtering by review date caused the reporting KPI counts to differ from the audit app.
    """
    issues_view = issues_df.copy() if not issues_df.empty else pd.DataFrame()
    reviews_view = reviews_df.copy() if not reviews_df.empty else pd.DataFrame()

    valid_wo_ids = set(audit_view["__WO_ID"].dropna().astype(str).tolist()) if not audit_view.empty and "__WO_ID" in audit_view.columns else set()

    if valid_wo_ids:
        if not issues_view.empty and "wo_id" in issues_view.columns:
            issues_view["__wo_id"] = _norm_id_series(issues_view["wo_id"])
            issues_view = issues_view[issues_view["__wo_id"].astype(str).isin(valid_wo_ids)].copy()
        if not reviews_view.empty and "wo_id" in reviews_view.columns:
            reviews_view["__wo_id"] = _norm_id_series(reviews_view["wo_id"])
            reviews_view = reviews_view[reviews_view["__wo_id"].astype(str).isin(valid_wo_ids)].copy()

    return issues_view, reviews_view

def kpi_rate(flagged: int, total: int) -> str:
    pct = round((flagged / total) * 100, 1) if total else 0.0
    return f"{flagged:,}/{total:,}", f"{pct}%"


def render_cmms_kpi_tab(audit_view: pd.DataFrame, issues_view: pd.DataFrame, reviews_view: pd.DataFrame):
    st.subheader("Location / Asset CMMS KPI Review")
    st.caption("Read-only KPI review pulled from the existing Work Order Audit tables. Counts now mirror the Work Order Audit app: KPI totals are based on mx_work_orders_audit / issues / reviewed directly, not on Asset_History_Merged rows. Location and Asset filters still apply when selected.")

    if audit_view.empty:
        st.info("No work-order audit rows match the current filters, or the Work Order Audit tables have not been built yet.")
        return

    for c in ["Completed_WO_Flag", "Open_WO_Flag", "Open_Overdue_Flag", "Backlog_Flag", "Days_Overdue", "No_Start_Date_Flag", "No_Due_Date_Flag"]:
        if c not in audit_view.columns:
            audit_view[c] = 0

    total_all = len(audit_view)
    completed_view = audit_view[_num_series(audit_view, "Completed_WO_Flag") == 1].copy()
    open_view = audit_view[_num_series(audit_view, "Open_WO_Flag") == 1].copy()
    total_completed = len(completed_view)
    total_open = len(open_view)

    presently_overdue_cnt = int((_num_series(open_view, "Open_Overdue_Flag") == 1).sum()) if not open_view.empty else 0
    backlog_cnt = int((_num_series(open_view, "Backlog_Flag") == 1).sum()) if not open_view.empty else 0
    completed_overdue_cnt = int((_num_series(completed_view, "Days_Overdue") > 0).sum()) if not completed_view.empty else 0
    completed_no_start_cnt = int((_num_series(completed_view, "No_Start_Date_Flag") == 1).sum()) if not completed_view.empty else 0
    completed_no_due_cnt = int((_num_series(completed_view, "No_Due_Date_Flag") == 1).sum()) if not completed_view.empty else 0

    st.markdown("### Helper KPI Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    val, delta = kpi_rate(presently_overdue_cnt, total_open)
    c1.metric("Presently Overdue", val, f"{delta} of open WOs")
    val, delta = kpi_rate(backlog_cnt, total_open)
    c2.metric("Backlog", val, f"{delta} of open WOs")
    val, delta = kpi_rate(completed_overdue_cnt, total_all)
    c3.metric("Completed Overdue", val, f"{delta} of total WOs")
    val, delta = kpi_rate(completed_no_start_cnt, total_all)
    c4.metric("Completed No Start", val, f"{delta} of total WOs")
    val, delta = kpi_rate(completed_no_due_cnt, total_all)
    c5.metric("Completed No Due", val, f"{delta} of total WOs")

    helper_detail = pd.DataFrame([
        {"KPI": "Presently Overdue", "Flagged": presently_overdue_cnt, "Relevant Population": total_open, "Total Work Orders": total_all, "% of Total WOs": round((presently_overdue_cnt / total_all) * 100, 1) if total_all else 0.0, "Definition": "Open work orders with a due date before today. Due today is not overdue.", "Ramification": "Indicates current schedule risk and growing maintenance exposure."},
        {"KPI": "Backlog", "Flagged": backlog_cnt, "Relevant Population": total_open, "Total Work Orders": total_all, "% of Total WOs": round((backlog_cnt / total_all) * 100, 1) if total_all else 0.0, "Definition": "Open work orders with no planned start date and no due date, and at least 3 days since Created on.", "Ramification": "Represents unplanned work sitting in the system without clear scheduling direction."},
        {"KPI": "Completed Overdue", "Flagged": completed_overdue_cnt, "Relevant Population": total_completed, "Total Work Orders": total_all, "% of Total WOs": round((completed_overdue_cnt / total_all) * 100, 1) if total_all else 0.0, "Definition": "Completed work orders with Days Overdue > 0.", "Ramification": "Shows completion after the due date and highlights planning / execution slippage."},
        {"KPI": "Completed No Start", "Flagged": completed_no_start_cnt, "Relevant Population": total_completed, "Total Work Orders": total_all, "% of Total WOs": round((completed_no_start_cnt / total_all) * 100, 1) if total_all else 0.0, "Definition": "Completed work orders missing a planned start date, except work completed within 3 days.", "Ramification": "Shows weak planning discipline and reduces the usefulness of scheduling KPIs."},
        {"KPI": "Completed No Due", "Flagged": completed_no_due_cnt, "Relevant Population": total_completed, "Total Work Orders": total_all, "% of Total WOs": round((completed_no_due_cnt / total_all) * 100, 1) if total_all else 0.0, "Definition": "Completed work orders missing a due date, except work completed within 3 days.", "Ramification": "Reduces the value of overdue tracking, prioritization, and schedule compliance."},
    ])
    st.dataframe(helper_detail, use_container_width=True, hide_index=True)

    rollup = helper_detail.copy()
    rollup["Target %"] = rollup["KPI"].map(HELPER_KPI_TARGETS).fillna(1.0)
    rollup["Status"] = rollup.apply(lambda r: "On Target" if float(r["% of Total WOs"]) <= float(r["Target %"]) else "Above Target", axis=1)
    st.markdown("### KPI Standards Rollup")
    st.dataframe(rollup[["KPI", "Flagged", "Relevant Population", "Total Work Orders", "% of Total WOs", "Target %", "Status"]], use_container_width=True, hide_index=True)

    st.markdown("### Reviewed Flag KPI Summary")

    # Match the Work Order Audit app display labels and reviewed KPI layout.
    # Open issue counts are still shown for management context, but the reviewed
    # KPI summary mirrors the audit app labels: Reviewed Flags, Viewed No Error,
    # and Viewed Error.
    open_issue_view = issues_view.copy() if not issues_view.empty else pd.DataFrame()
    if not open_issue_view.empty:
        open_status = _text_series(open_issue_view, "status").str.upper()
        open_issue_view = open_issue_view[
            open_status.eq("OPEN") &
            (_num_series(open_issue_view, "is_bypassed") == 0)
        ].copy()

        if not reviews_view.empty and "issue_key" in reviews_view.columns and "issue_key" in open_issue_view.columns:
            reviewed_keys = reviews_view["issue_key"].dropna().astype(str).str.strip()
            open_issue_view = open_issue_view[
                ~open_issue_view["issue_key"].astype(str).str.strip().isin(reviewed_keys)
            ].copy()

    open_unbypassed_flags = len(open_issue_view)
    open_unbypassed_wos = (
        open_issue_view["wo_id"].dropna().astype(str).str.strip().nunique()
        if not open_issue_view.empty and "wo_id" in open_issue_view.columns else 0
    )

    total_flags = len(issues_view) if not issues_view.empty else 0
    reviewed_flags = len(reviews_view) if not reviews_view.empty else 0

    reviewed_status = (
        _text_series(reviews_view, "review_status").str.upper()
        if not reviews_view.empty else pd.Series(dtype=str)
    )
    viewed_no_error_cnt = int(reviewed_status.eq("VIEWED_NO_ERROR").sum()) if not reviews_view.empty else 0
    viewed_with_error_cnt = int(reviewed_status.eq("VIEWED_WITH_ERROR").sum()) if not reviews_view.empty else 0

    # Audit app matching display labels.
    r1, r2, r3 = st.columns(3)
    val, delta = kpi_rate(reviewed_flags, total_all)
    r1.metric("Reviewed Flags", val, f"{delta} of total WOs")
    val, delta = kpi_rate(viewed_no_error_cnt, total_all)
    r2.metric("Viewed No Error", val, f"{delta} of total WOs")
    val, delta = kpi_rate(viewed_with_error_cnt, total_all)
    r3.metric("Viewed Error", val, f"{delta} of total WOs")

    # Extra open-flag context retained from the reporting page, but kept separate
    # so the reviewed KPI display stays aligned with the audit app.
    with st.expander("Open Flag Context", expanded=False):
        o1, o2 = st.columns(2)
        o1.metric("Open WOs w/ Flags", f"{open_unbypassed_wos:,}")
        o2.metric("Open Flags", f"{open_unbypassed_flags:,}")
        flag_summary_df = pd.DataFrame([
            {"KPI": "Open WOs w/ Flags", "Count": open_unbypassed_wos, "Definition": "Unique work orders with open, unbypassed, unreviewed audit flags."},
            {"KPI": "Open Flags", "Count": open_unbypassed_flags, "Definition": "Open, unbypassed audit issue rows that have not been reviewed/closed."},
            {"KPI": "Reviewed Flags", "Count": reviewed_flags, "Definition": "Reviewed audit rows stored in mx_work_orders_audit_reviewed."},
            {"KPI": "Viewed No Error", "Count": viewed_no_error_cnt, "Definition": "Reviewed rows marked VIEWED_NO_ERROR."},
            {"KPI": "Viewed Error", "Count": viewed_with_error_cnt, "Definition": "Reviewed rows marked VIEWED_WITH_ERROR."},
        ])
        st.dataframe(flag_summary_df, use_container_width=True, hide_index=True)

    st.markdown("### Review Reason KPI Cards")

    if not reviews_view.empty and "reason_code" in reviews_view.columns:
        grouped = reviews_view.groupby("reason_code", dropna=False).size()
        reason_counts_map = {str(k): int(v) for k, v in grouped.items() if pd.notna(k)}
    else:
        reason_counts_map = {}

    reason_cards = st.columns(3)
    for i, reason in enumerate(REVIEW_REASON_CODES):
        count = int(reason_counts_map.get(reason, 0))
        val, delta = kpi_rate(count, total_all)
        # The metric label is intentionally the display label from the audit app.
        reason_cards[i % 3].metric(reason, val, f"{delta} of total WOs")

    reason_rows = []
    for reason in REVIEW_REASON_CODES:
        count = int(reason_counts_map.get(reason, 0))
        pct_total = round((count / total_all) * 100, 1) if total_all else 0.0
        target_pct = float(REVIEW_REASON_DEFAULT_TARGETS.get(reason, 1.0))
        reason_rows.append({
            "Review Reason": reason,
            "Flag Count": count,
            "Total Flags": total_flags,
            "Total Work Orders": total_all,
            "% of Total WOs": pct_total,
            "Target %": target_pct,
            "Status": "On Target" if pct_total <= target_pct else "Above Target",
            "Definition / Ramification": REVIEW_REASON_DETAILS.get(reason, "No description provided."),
        })

    reason_df = pd.DataFrame(reason_rows)
    st.markdown("### Review Reason KPI Detail")
    st.dataframe(reason_df, use_container_width=True, hide_index=True)

    if reviews_view.empty:
        st.info("No reviewed work-order flags match the current filters.")
    else:
        display_cols = [c for c in ["issue_key", "wo_id", "issue_type", "review_status", "reason_code", "review_note", "reviewed_by", "reviewed_ts"] if c in reviews_view.columns]
        st.markdown("### Reviewed Flag Detail")
        st.dataframe(reviews_view[display_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download CMMS KPI Review CSV",
            data=reviews_view[display_cols].to_csv(index=False).encode("utf-8-sig"),
            file_name=f"location_asset_cmms_kpi_review_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("### Monthly Helper KPI Trend")
    if not audit_view.empty and "__Created_dt" in audit_view.columns:
        month_df = audit_view.copy()
        month_df["Created Month"] = pd.to_datetime(month_df["__Created_dt"], errors="coerce").dt.to_period("M").astype(str)
        monthly = month_df.groupby("Created Month", dropna=False).agg(
            Total_WOs=("__WO_ID", "count"),
            Presently_Overdue=("Open_Overdue_Flag", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            Backlog=("Backlog_Flag", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            Completed_Overdue=("Days_Overdue", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
            Completed_No_Start=("No_Start_Date_Flag", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            Completed_No_Due=("No_Due_Date_Flag", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        ).reset_index().sort_values("Created Month")
        st.dataframe(monthly, use_container_width=True, hide_index=True)
        chart_cols = [c for c in ["Presently_Overdue", "Backlog", "Completed_Overdue", "Completed_No_Start", "Completed_No_Due"] if c in monthly.columns]
        if chart_cols:
            st.line_chart(monthly.set_index("Created Month")[chart_cols], use_container_width=True)

# -----------------------------
# Page
# -----------------------------
st.title("Location / Asset CMMS Report")
st.caption("Filtered location / asset CMMS reporting from Asset_History_Merged with work-order audit KPI review.")

with st.sidebar:
    st.header("Location / Asset CMMS Reporting")
    st.caption("Connected to maintenance_master.db")
    st.code(DB_PATH, language="text")
    st.code(TABLE_NAME, language="text")

raw_df = load_data()
locations_ref_df, assets_ref_df = load_reference_data()
source_label = f"{TABLE_NAME}"

df, source_date_col, source_cost_col = prepare_df(raw_df)

if df.empty:
    st.warning("No asset history data loaded. Check the database path/table or upload a CSV reference file.")
    st.stop()

missing = [c for c in ["Location", "ASSET"] if c not in df.columns]
if missing:
    st.error(f"Missing required column(s): {', '.join(missing)}")
    st.stop()

st.caption(f"Source: {source_label} | Date column: {source_date_col or 'not found'} | Cost column: {source_cost_col or 'not found'}")

proper_location_options = get_proper_location_options(locations_ref_df, df)
if locations_ref_df.empty or "All Parents" not in locations_ref_df.columns:
    st.warning("Locations_Master was not found or does not contain 'All Parents'. Location filter is using history fallback values.")
else:
    st.caption(f"Location filter controlled by {LOCATIONS_TABLE} → All Parents. Proper location choices available: {len(proper_location_options):,}")

# Filters
f1, f2, f3, f4 = st.columns([1.25, 1.25, 1, 1])

with f1:
    loc_options = proper_location_options
    selected_locations = st.multiselect("Location", loc_options)

asset_base = df.copy()
if selected_locations:
    asset_base = asset_base[asset_base["Location"].isin(selected_locations)]

with f2:
    asset_options = sorted([x for x in asset_base["ASSET"].dropna().astype(str).unique().tolist() if x])
    selected_assets = st.multiselect("Asset", asset_options)

with f3:
    period_mode = st.radio("Date Range", ["YTD", "Monthly", "Custom"], horizontal=False)

valid_dates = df["Report Date"].dropna()
def_min = valid_dates.min().date() if not valid_dates.empty else date.today()
def_max = valid_dates.max().date() if not valid_dates.empty else date.today()

with f4:
    if period_mode == "Monthly":
        selected_month = st.date_input("Month", value=date(def_max.year, def_max.month, 1))
        custom_start = custom_end = None
    elif period_mode == "Custom":
        custom_start = st.date_input("Start", value=date(def_max.year, 1, 1), min_value=def_min, max_value=def_max)
        custom_end = st.date_input("End", value=def_max, min_value=def_min, max_value=def_max)
        selected_month = None
    else:
        selected_month = custom_start = custom_end = None

start_dt, end_dt = date_window(df, period_mode, selected_month, custom_start, custom_end)
fdf = filtered_data(df, selected_locations, selected_assets, start_dt, end_dt)

report_tab, cmms_kpi_tab = st.tabs(["Report", "CMMS KPI Review"])

with report_tab:
    # KPIs
    workorders = fdf["WORKORDER"].replace("", pd.NA).dropna().nunique() if "WORKORDER" in fdf.columns else 0
    pos = fdf["PO"].replace("", pd.NA).dropna().nunique() if "PO" in fdf.columns else 0
    assets = fdf["ASSET"].replace("", pd.NA).dropna().nunique() if "ASSET" in fdf.columns else 0
    locations = fdf["Location"].replace("", pd.NA).dropna().nunique() if "Location" in fdf.columns else 0
    cost = float(fdf["Report Cost"].sum()) if "Report Cost" in fdf.columns else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Filtered Cost", _money(cost))
    k2.metric("Rows", f"{len(fdf):,}")
    k3.metric("Work Orders", f"{workorders:,}")
    k4.metric("Assets", f"{assets:,}")
    k5.metric("POs", f"{pos:,}")

    st.caption(f"Showing {start_dt:%Y-%m-%d} through {end_dt:%Y-%m-%d}")

    # Charts
    st.subheader("Cost Trend")
    chart_df = fdf.copy()
    if not chart_df.empty:
        # Daily for short ranges, monthly for longer ranges.
        days = max((end_dt - start_dt).days, 1)
        if days <= 45:
            chart_df["Period"] = chart_df["Report Date"].dt.date
        else:
            chart_df["Period"] = chart_df["Report Date"].dt.to_period("M").astype(str)

        trend = chart_df.groupby("Period", as_index=False)["Report Cost"].sum().sort_values("Period")
        st.bar_chart(trend, x="Period", y="Report Cost", use_container_width=True)
    else:
        st.info("No rows match the selected filters.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cost by DB Type")
        if "DB_TYPE" in fdf.columns and not fdf.empty:
            db_chart = fdf.groupby("DB_TYPE", as_index=False)["Report Cost"].sum().sort_values("Report Cost", ascending=False)
            st.bar_chart(db_chart, x="DB_TYPE", y="Report Cost", use_container_width=True)
        else:
            st.info("No DB_TYPE data available.")

    with c2:
        st.subheader("Top Assets by Cost")
        if not fdf.empty:
            top_assets = fdf.groupby("ASSET", as_index=False)["Report Cost"].sum().sort_values("Report Cost", ascending=False).head(15)
            st.bar_chart(top_assets, x="ASSET", y="Report Cost", use_container_width=True)
        else:
            st.info("No asset data available.")

    # Raw table + exports
    st.subheader("Raw Filtered Data")
    show_cols_default = [c for c in [
        "Report Date", "DB_TYPE", "WORKORDER", "TITLE", "STATUS", "PO", "P/N", "QUANTITY RECEIVED",
        "ITEM COST", "TOTAL ITEM COST", "RECEIVED COST", "Total Parts Cost", "Additional Costs",
        "Total cost", "Report Cost", "Vendors", "ASSET", "Location", "TX_TYPE", "TX_DIRECTION", "TX_REASON"
    ] if c in fdf.columns]

    with st.expander("Column Display", expanded=False):
        display_cols = st.multiselect("Columns", list(fdf.columns), default=show_cols_default)

    view = fdf[display_cols].copy() if display_cols else fdf.copy()
    st.dataframe(view, use_container_width=True, hide_index=True)

    csv_bytes = view.to_csv(index=False).encode("utf-8-sig")

    filters_for_pdf = {
        "Location": ", ".join(selected_locations) if selected_locations else "All",
        "Asset": ", ".join(selected_assets) if selected_assets else "All",
        "Date Range": f"{start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}",
        "Source": source_label,
    }
    summary_for_pdf = {
        "Filtered Cost": _money(cost),
        "Rows": f"{len(fdf):,}",
        "Work Orders": f"{workorders:,}",
        "Assets": f"{assets:,}",
        "Locations": f"{locations:,}",
        "POs": f"{pos:,}",
    }

    b1, b2 = st.columns([1, 1])
    with b1:
        st.download_button(
            "Download Filtered CSV",
            data=csv_bytes,
            file_name=f"location_asset_cmms_filtered_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with b2:
        if REPORTLAB_AVAILABLE:
            pdf_bytes = build_pdf(view, filters_for_pdf, summary_for_pdf)
            st.download_button(
                "Download Filtered PDF",
                data=pdf_bytes,
                file_name=f"location_asset_cmms_report_{datetime.now():%Y%m%d_%H%M}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning("PDF export requires ReportLab. Install with: pip install reportlab")

with cmms_kpi_tab:
    audit_current = load_optional_db_table(WO_AUDIT_TABLE)
    audit_issues = load_optional_db_table(WO_ISSUES_TABLE)
    audit_reviews = load_optional_db_table(WO_REVIEW_TABLE)
    audit_view = prepare_wo_audit_view(audit_current, selected_locations, selected_assets, start_dt, end_dt, fdf)
    issues_view, reviews_view = filter_issues_reviews_to_audit(audit_issues, audit_reviews, audit_view, start_dt, end_dt)
    render_cmms_kpi_tab(audit_view, issues_view, reviews_view)
