# 03_Mobile_Service_Report.py
# Streamlit reporting page for Mobile Service Report History
# Report-only version: no MaintainX API, no WO upload, no build/refresh, no meter input page.

from __future__ import annotations

import io
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from auth_helper import require_login

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

try:
    from reporting_shared import (
        DB_PATH,
        load_table,
        load_locations,
        get_valid_locations,
        norm_text,
        money,
    )
except Exception:
    DB_PATH = str(Path(__file__).resolve().parents[1] / "maintenance_master.db")

    def norm_text(x) -> str:
        if pd.isna(x):
            return ""
        return str(x).strip()

    def money(x) -> str:
        try:
            return f"${float(x):,.2f}"
        except Exception:
            return "$0.00"

    @st.cache_data(show_spinner=False)
    def load_table(db_path: str, table_name: str) -> pd.DataFrame:
        with sqlite3.connect(db_path) as conn:
            chk = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=[table_name],
            )
            if chk.empty:
                return pd.DataFrame()
            return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)

    @st.cache_data(show_spinner=False)
    def load_locations(db_path: str = DB_PATH) -> pd.DataFrame:
        try:
            return load_table(db_path, "Locations_Master")
        except Exception:
            return pd.DataFrame()

    def get_valid_locations(locations_df: pd.DataFrame) -> list[str]:
        col = next((c for c in ["All Parents", "All Parent Locations", "Location", "Name"] if c in locations_df.columns), None)
        if not col:
            return []
        vals = locations_df[col].dropna().astype(str).map(str.strip)
        return sorted(vals[vals.ne("")].unique().tolist())


st.set_page_config(page_title="Mobile Service Report", layout="wide")
require_login()

REPORT_HISTORY_TABLE = "Mobile_Service_Report_History"
REPORT_TABLE = "Mobile_Service_Report"
LOCATIONS_TABLE = "Locations_Master"

KPI_STATUS_ORDER = ["Over Due", "Coming Due", "Missing Meter", "Needs New Reading", "Current"]

DATE_ONLY_COLUMNS = {
    "Current Date",
    "Date of Last Service",
    "Next Service Date",
    "Predicted Service Date",
    "Report Run Date",
}

DATETIME_COLUMNS = {
    "Report Run Timestamp",
}


def normalize_date_series(s: pd.Series) -> pd.Series:
    """Convert date-like values safely and treat zero/blank placeholder values as missing."""
    raw = s.copy()

    # Common bad placeholders from SQLite/CSV history exports. Without this, pandas can
    # convert 0-like values into misleading epoch dates or Excel can display zero dates.
    text = raw.astype(str).str.strip()
    zero_like = (
        raw.isna()
        | text.eq("")
        | text.str.fullmatch(r"0+(?:\.0+)?", na=False)
        | text.str.fullmatch(r"0{1,2}[/\-]0{1,2}[/\-]0{2,4}", na=False)
        | text.str.lower().isin(["nan", "nat", "none", "null"])
    )

    cleaned = raw.mask(zero_like, pd.NA)
    dt = pd.to_datetime(cleaned, errors="coerce", format="mixed")

    # Guardrail: service/report dates before 1990 are almost always conversion artifacts
    # in this data set, not real equipment service dates.
    dt = dt.mask(dt.dt.year.lt(1990), pd.NaT)
    return dt


def format_for_display_export(df: pd.DataFrame) -> pd.DataFrame:
    """Format date columns as readable strings before Streamlit display and downloads."""
    if df.empty:
        return df

    out = df.copy()
    for c in out.columns:
        if c in DATE_ONLY_COLUMNS:
            dt = normalize_date_series(out[c])
            out[c] = dt.dt.strftime("%Y-%m-%d").fillna("")
        elif c in DATETIME_COLUMNS or pd.api.types.is_datetime64_any_dtype(out[c]):
            dt = normalize_date_series(out[c])
            if c in DATETIME_COLUMNS:
                out[c] = dt.dt.strftime("%Y-%m-%d %I:%M %p").fillna("")
            else:
                out[c] = dt.dt.strftime("%Y-%m-%d").fillna("")
    return out


# -----------------------------
# Utility helpers
# -----------------------------
def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def clean_text_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def format_num(x, digits: int = 1) -> str:
    try:
        if pd.isna(x):
            return "N/A"
        return f"{float(x):,.{digits}f}"
    except Exception:
        return "N/A"


def table_exists(db_path: str, table_name: str) -> bool:
    try:
        with sqlite3.connect(db_path) as conn:
            chk = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=[table_name],
            )
            return not chk.empty
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def load_mobile_history(db_path: str = DB_PATH) -> pd.DataFrame:
    """Load report history directly from SQLite. Falls back to current report table only if history is missing."""
    if table_exists(db_path, REPORT_HISTORY_TABLE):
        df = load_table(db_path, REPORT_HISTORY_TABLE)
        source = REPORT_HISTORY_TABLE
    elif table_exists(db_path, REPORT_TABLE):
        df = load_table(db_path, REPORT_TABLE)
        source = REPORT_TABLE
    else:
        return pd.DataFrame()

    if df.empty:
        return df

    df = df.copy()
    df["__source_table"] = source

    # Normalize text columns.
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].map(norm_text)

    # Build timestamp/date fields for run selection and date filters.
    ts_col = first_present(df, ["Report Run Timestamp", "Report Timestamp", "Run Timestamp", "Created On"])
    date_col = first_present(df, ["Report Run Date", "Run Date", "Date"])

    if ts_col:
        df["__run_ts"] = normalize_date_series(df[ts_col])
    elif date_col:
        df["__run_ts"] = normalize_date_series(df[date_col])
    else:
        df["__run_ts"] = pd.NaT

    if date_col:
        df["__run_date"] = normalize_date_series(df[date_col]).dt.date
    else:
        df["__run_date"] = df["__run_ts"].dt.date

    # Normalize important dates.
    for c in [
        "Current Date", "Date of Last Service", "Next Service Date", "Predicted Service Date",
        "Report Run Date", "Report Run Timestamp",
    ]:
        if c in df.columns:
            df[c] = normalize_date_series(df[c])

    # Normalize numeric columns used in KPIs.
    for c in [
        "Asset Hours", "Current Reading", "Remaining Hours", "Remaining Days", "Predicted Days to Service",
        "Expected Completion Hours", "Hours Between Last Two Services", "Avg Between Last 3 Services",
        "Days Between Last Two Services", "Avg Days Between Last 3 Services", "Open WO Count",
        "Interval Value", "Avg Hours per nDay", "Last Service Reading", "Next Service Meter",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "__run_ts" in df.columns and df["__run_ts"].notna().any():
        latest = df["__run_ts"].dropna().max()
        return df[df["__run_ts"] == latest].copy()
    if "__run_date" in df.columns and df["__run_date"].notna().any():
        latest = df["__run_date"].dropna().max()
        return df[df["__run_date"] == latest].copy()
    return df.copy()


def export_buttons(df: pd.DataFrame, base_name: str, key_prefix: str) -> None:
    if df.empty:
        return
    export_df = format_for_display_export(df)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.download_button(
            "Download CSV",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{base_name}_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            width="stretch",
            key=f"{key_prefix}_csv",
        )
    with c2:
        buffer = io.BytesIO()
        try:
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                export_df.to_excel(writer, index=False, sheet_name="Report")
            st.download_button(
                "Download XLSX",
                data=buffer.getvalue(),
                file_name=f"{base_name}_{datetime.now():%Y%m%d_%H%M}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
                key=f"{key_prefix}_xlsx",
            )
        except Exception:
            st.info("XLSX export requires xlsxwriter. CSV export is still available.")


def export_kpi_pdf(kpi_summary_df: pd.DataFrame, title: str = "Mobile Service KPI Summary") -> bytes:
    if kpi_summary_df.empty or not MATPLOTLIB_AVAILABLE:
        return b""

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        table = ax.table(
            cellText=kpi_summary_df.fillna("").astype(str).values,
            colLabels=kpi_summary_df.columns,
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def status_mask(df: pd.DataFrame, status: str) -> pd.Series:
    if "KPI Status" in df.columns:
        return clean_text_series(df["KPI Status"]).eq(status)
    # fallback to Yes/No columns if KPI Status is missing
    if status in df.columns:
        return clean_text_series(df[status]).eq("Yes")
    return pd.Series(False, index=df.index)


def service_audit_mask(df: pd.DataFrame) -> pd.Series:
    if "Service Audit" in df.columns:
        return clean_text_series(df["Service Audit"]).eq("Yes")
    if "Date of Last Service" in df.columns:
        return pd.to_datetime(df["Date of Last Service"], errors="coerce").isna()
    return pd.Series(False, index=df.index)


def open_wo_mask(df: pd.DataFrame) -> pd.Series:
    if "Open WO Count" in df.columns:
        return pd.to_numeric(df["Open WO Count"], errors="coerce").fillna(0).gt(0)
    if "Open WO Exists" in df.columns:
        return clean_text_series(df["Open WO Exists"]).str.lower().eq("yes")
    return pd.Series(False, index=df.index)


def filter_view(df: pd.DataFrame, selected_locations: list[str], selected_assets: list[str], selected_statuses: list[str], selected_interval_types: list[str]) -> pd.DataFrame:
    out = df.copy()
    if selected_locations and "Location" in out.columns:
        out = out[out["Location"].astype(str).isin(selected_locations)]
    if selected_assets and "Asset" in out.columns:
        out = out[out["Asset"].astype(str).isin(selected_assets)]
    if selected_statuses and "KPI Status" in out.columns:
        out = out[out["KPI Status"].astype(str).isin(selected_statuses)]
    if selected_interval_types and "Interval Type" in out.columns:
        out = out[out["Interval Type"].astype(str).isin(selected_interval_types)]
    return out.copy()


def kpi_detail_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "Asset", "Asset ID", "Location", "Schedule Name", "Schedule ID", "Interval Type", "Interval Unit",
        "Interval Value", "KPI Status", "Asset Hours", "Current Reading", "Meter Reset Applied", "Current Date",
        "Date of Last Service", "Last Service Type", "Last Service Reading", "Next Service Type",
        "Next Service Meter", "Remaining Hours", "Remaining Days", "Remaining Interval Display",
        "Predicted Service Date", "Expected Completion Hours", "Hours Between Last Two Services",
        "Hours Between Last Two Status", "Avg Between Last 3 Services", "Avg Between Last 3 Status",
        "Days Between Last Two Services", "Days Between Last Two Status", "Avg Days Between Last 3 Services",
        "Avg Days Between Last 3 Status", "Open WO Exists", "Open WO Count", "Open WO ID", "Open WO Title",
        "Open WO Status", "Service Audit", "Report Run Date", "Report Run Timestamp",
    ]
    return [c for c in cols if c in df.columns]


def due_detail_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "Asset", "Asset ID", "Location", "Interval Type", "Interval Unit", "Interval Value",
        "Remaining Interval Display", "Open WO ID", "Open WO Title", "Open WO Status",
        "Predicted Service Date", "Asset Hours", "Current Reading", "Meter Reset Applied",
        "Last Service Reading", "Next Service Meter", "Remaining Hours", "Current Date",
        "Date of Last Service", "Next Service Date", "Remaining Days", "Last Service Type",
        "Next Service Type", "Expected Completion Hours", "Avg Hours per nDay", "KPI Status",
    ]
    return [c for c in cols if c in df.columns]


def render_metric_row(df_view: pd.DataFrame) -> dict:
    total_rows = len(df_view)
    unique_assets = df_view["Asset"].replace("", pd.NA).dropna().nunique() if "Asset" in df_view.columns else total_rows
    coming_due_count = int(status_mask(df_view, "Coming Due").sum())
    overdue_count = int(status_mask(df_view, "Over Due").sum())
    missing_meter_count = int(status_mask(df_view, "Missing Meter").sum())
    needs_new_reading_count = int(status_mask(df_view, "Needs New Reading").sum())
    current_count = int(status_mask(df_view, "Current").sum())
    service_audit_count = int(service_audit_mask(df_view).sum())
    open_wo_assets = int(open_wo_mask(df_view).sum())

    c = st.columns(8)
    c[0].metric("Rows", f"{total_rows:,}")
    c[1].metric("Assets", f"{unique_assets:,}")
    c[2].metric("Coming Due", f"{coming_due_count:,}")
    c[3].metric("Over Due", f"{overdue_count:,}")
    c[4].metric("Missing Meter", f"{missing_meter_count:,}")
    c[5].metric("Needs New Reading", f"{needs_new_reading_count:,}")
    c[6].metric("Open WO Assets", f"{open_wo_assets:,}")
    c[7].metric("Service Audit", f"{service_audit_count:,}")

    return {
        "Rows": total_rows,
        "Assets": unique_assets,
        "Coming Due": coming_due_count,
        "Over Due": overdue_count,
        "Missing Meter": missing_meter_count,
        "Needs New Reading": needs_new_reading_count,
        "Current": current_count,
        "Open WO Assets": open_wo_assets,
        "Service Audit": service_audit_count,
    }


def interval_kpis(df_view: pd.DataFrame, interval_type: str) -> dict:
    if "Interval Type" not in df_view.columns:
        work = df_view.iloc[0:0].copy()
    else:
        work = df_view[clean_text_series(df_view["Interval Type"]).str.lower().eq(interval_type.lower())].copy()

    if interval_type.lower() in ["progressive", "repeating"]:
        avg_last_service = to_number(work.get("Hours Between Last Two Services", pd.Series(dtype=float))).dropna().mean()
        avg_last_three = to_number(work.get("Avg Between Last 3 Services", pd.Series(dtype=float))).dropna().mean()
        overdue_last_services = int(clean_text_series(work.get("Hours Between Last Two Status", pd.Series(dtype=str))).eq("Bad").sum())
        avg_overdue_last_three = int(clean_text_series(work.get("Avg Between Last 3 Status", pd.Series(dtype=str))).eq("Bad").sum())
    else:
        avg_last_service = to_number(work.get("Days Between Last Two Services", pd.Series(dtype=float))).dropna().mean()
        avg_last_three = to_number(work.get("Avg Days Between Last 3 Services", pd.Series(dtype=float))).dropna().mean()
        overdue_last_services = int(clean_text_series(work.get("Days Between Last Two Status", pd.Series(dtype=str))).eq("Bad").sum())
        avg_overdue_last_three = int(clean_text_series(work.get("Avg Days Between Last 3 Status", pd.Series(dtype=str))).eq("Bad").sum())

    return {
        f"{interval_type} - Avg Last Service": avg_last_service,
        f"{interval_type} - Avg of Last Three": avg_last_three,
        f"{interval_type} - OverDue Last Services": overdue_last_services,
        f"{interval_type} - Avg OverDue in Last Three Services": avg_overdue_last_three,
    }


def render_interval_section(df_view: pd.DataFrame, interval_type: str) -> dict:
    vals = interval_kpis(df_view, interval_type)
    st.markdown(f"### {interval_type} KPI")
    c1, c2, c3, c4 = st.columns(4)
    keys = list(vals.keys())
    c1.metric(keys[0], format_num(vals[keys[0]], 1))
    c2.metric(keys[1], format_num(vals[keys[1]], 1))
    c3.metric(keys[2], f"{int(vals[keys[2]]):,}")
    c4.metric(keys[3], f"{int(vals[keys[3]]):,}")
    return vals


def render_kpi_page(df_view: pd.DataFrame) -> None:
    st.subheader("KPI Summary")
    st.caption("This page opens first and reads directly from Mobile_Service_Report_History. No API, no work order upload, and no report build/refresh logic is included.")

    if df_view.empty:
        st.info("No rows match the current filters.")
        return

    base_metrics = render_metric_row(df_view)
    progressive = render_interval_section(df_view, "Progressive")
    repeating = render_interval_section(df_view, "Repeating")
    fixed = render_interval_section(df_view, "Fixed")

    kpi_summary_rows = []
    for k, v in base_metrics.items():
        kpi_summary_rows.append({"KPI": k, "Value": v})
    for group in [progressive, repeating, fixed]:
        for k, v in group.items():
            kpi_summary_rows.append({"KPI": k, "Value": v})
    kpi_summary = pd.DataFrame(kpi_summary_rows)

    st.markdown("### KPI Summary Table")
    display_summary = kpi_summary.copy()
    display_summary["Value"] = display_summary["Value"].map(lambda x: format_num(x, 1) if isinstance(x, float) and pd.notna(x) else x)
    st.dataframe(display_summary, width="stretch", hide_index=True)

    if MATPLOTLIB_AVAILABLE:
        pdf_data = export_kpi_pdf(display_summary, title="Mobile Service KPI Summary")
        st.download_button(
            "Download KPI Summary PDF",
            data=pdf_data,
            file_name=f"Mobile_Service_KPI_Summary_{datetime.now():%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
            width="stretch",
            key="kpi_pdf",
        )

    st.markdown("### KPI Detail")
    detail_cols = kpi_detail_columns(df_view)
    detail = df_view[detail_cols].copy() if detail_cols else df_view.copy()
    st.dataframe(format_for_display_export(detail), width="stretch", hide_index=True)
    export_buttons(detail, "Mobile_Service_KPI_Detail", "kpi_detail")


def render_status_page(df_view: pd.DataFrame, status: str, title: str, file_name: str) -> None:
    st.subheader(title)
    if df_view.empty:
        st.info("No rows match the current filters.")
        return
    work = df_view[status_mask(df_view, status)].copy()
    if work.empty:
        st.info(f"No {title.lower()} rows match the current filters.")
        return
    cols = due_detail_columns(work)
    view = work[cols].copy() if cols else work.copy()
    st.dataframe(format_for_display_export(view), width="stretch", hide_index=True)
    export_buttons(view, file_name, file_name.lower())


def render_service_audit_page(df_view: pd.DataFrame) -> None:
    st.subheader("Service Audit")
    work = df_view[service_audit_mask(df_view)].copy()
    if work.empty:
        st.info("No service audit rows match the current filters.")
        return
    cols = [c for c in [
        "Asset", "Asset ID", "Location", "Schedule Name", "Schedule ID", "Interval Type", "Interval Unit",
        "Interval Value", "Meter Required", "Meter ID", "Resolved Meter ID", "Asset Hours", "Current Reading",
        "Meter Reset Applied", "Current Date", "Date of Last Service", "Last Service Type", "Next Service Type",
        "Next Service Meter", "Next Service Date", "Predicted Service Date", "Open WO Exists", "Open WO ID",
        "Open WO Status", "KPI Status", "Service Audit",
    ] if c in work.columns]
    view = work[cols].copy() if cols else work.copy()
    st.dataframe(format_for_display_export(view), width="stretch", hide_index=True)
    export_buttons(view, "Mobile_Service_Audit", "service_audit")


def render_open_wo_page(df_view: pd.DataFrame) -> None:
    st.subheader("Open Work Orders")
    work = df_view[open_wo_mask(df_view)].copy()
    if work.empty:
        st.info("No open work order rows match the current filters.")
        return
    cols = [c for c in [
        "Open WO ID", "Open WO Title", "Open WO Status", "Asset", "Asset ID", "Location", "Interval Type",
        "Last Service Type", "Next Service Type", "Asset Hours", "Current Reading", "Meter Reset Applied",
        "Current Date", "Remaining Hours", "Remaining Days", "Predicted Service Date", "Expected Completion Hours",
    ] if c in work.columns]
    view = work[cols].copy() if cols else work.copy()
    st.dataframe(format_for_display_export(view), width="stretch", hide_index=True)
    export_buttons(view, "Mobile_Service_Open_WorkOrders", "open_wo")


def render_main_report(df_view: pd.DataFrame) -> None:
    st.subheader("Main Report")
    if df_view.empty:
        st.info("No rows match the current filters.")
        return
    cols_default = [c for c in [
        "Asset", "Asset ID", "Location", "Schedule Name", "Interval Type", "Interval Unit", "Interval Value",
        "KPI Status", "Asset Hours", "Current Reading", "Meter Reset Applied", "Current Date", "Date of Last Service",
        "Last Service Type", "Next Service Type", "Remaining Hours", "Remaining Days", "Remaining Interval Display",
        "Predicted Service Date", "Open WO Exists", "Open WO ID", "Report Run Date", "Report Run Timestamp",
    ] if c in df_view.columns]
    with st.expander("Column Display", expanded=False):
        cols = st.multiselect("Columns", list(df_view.columns), default=cols_default, key="main_cols")
    view = df_view[cols].copy() if cols else df_view.copy()
    st.dataframe(format_for_display_export(view), width="stretch", hide_index=True)
    export_buttons(view, "Mobile_Service_Main_Report", "main_report")


def render_history_page(hist_filtered: pd.DataFrame) -> None:
    st.subheader("Report History")
    if hist_filtered.empty:
        st.info("No history rows match the current filters.")
        return
    view = hist_filtered.drop(columns=["__source_table", "__run_ts", "__run_date"], errors="ignore").copy()
    st.dataframe(format_for_display_export(view), width="stretch", hide_index=True)
    export_buttons(view, "Mobile_Service_Report_History", "history")


# -----------------------------
# Page
# -----------------------------
st.title("Mobile Service Report")
st.caption("Report-only view from Mobile_Service_Report_History in maintenance_master.db")

history = load_mobile_history(DB_PATH)
locations_df = load_locations(DB_PATH)

if history.empty:
    st.warning("No Mobile Service Report history was found in maintenance_master.db.")
    st.stop()

# -----------------------------
# Sidebar filters
# -----------------------------
with st.sidebar:
    st.header("Mobile Service Filters")
    st.caption("Source")
    st.code(DB_PATH, language="text")
    st.code(history["__source_table"].iloc[0] if "__source_table" in history.columns else REPORT_HISTORY_TABLE, language="text")

    run_mode = st.radio(
        "Report Snapshot",
        ["Latest Run", "Run Date", "Custom Date Range", "All History"],
        index=0,
        help="Latest Run is recommended for current KPI review. Custom Date Range and All History are best for trend/history review.",
    )

    hist_base = history.copy()
    valid_dates = sorted([d for d in hist_base.get("__run_date", pd.Series(dtype=object)).dropna().unique().tolist()])

    selected_run_date = None
    start_date = None
    end_date = None

    if run_mode == "Latest Run":
        hist_base = latest_snapshot(hist_base)
    elif run_mode == "Run Date" and valid_dates:
        selected_run_date = st.selectbox("Run Date", valid_dates, index=len(valid_dates) - 1)
        hist_base = hist_base[hist_base["__run_date"] == selected_run_date].copy()
    elif run_mode == "Custom Date Range" and valid_dates:
        min_date = min(valid_dates)
        max_date = max(valid_dates)
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        hist_base = hist_base[
            hist_base["__run_date"].notna()
            & (hist_base["__run_date"] >= start_date)
            & (hist_base["__run_date"] <= end_date)
        ].copy()

    st.divider()

    valid_locations = get_valid_locations(locations_df)
    location_source = hist_base if hist_base is not None else history
    if not valid_locations and "Location" in location_source.columns:
        valid_locations = sorted([x for x in location_source["Location"].dropna().astype(str).unique().tolist() if x.strip()])
    selected_locations = st.multiselect("Location", valid_locations)

    asset_source = hist_base.copy()
    if selected_locations and "Location" in asset_source.columns:
        asset_source = asset_source[asset_source["Location"].astype(str).isin(selected_locations)]
    asset_options = sorted([x for x in asset_source.get("Asset", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if x.strip()])
    selected_assets = st.multiselect("Asset", asset_options)

    status_options = [s for s in KPI_STATUS_ORDER if s in set(history.get("KPI Status", pd.Series(dtype=str)).astype(str))]
    if not status_options and "KPI Status" in history.columns:
        status_options = sorted([x for x in history["KPI Status"].dropna().astype(str).unique().tolist() if x.strip()])
    selected_statuses = st.multiselect("KPI Status", status_options)

    interval_options = sorted([x for x in history.get("Interval Type", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if x.strip()])
    selected_interval_types = st.multiselect("Interval Type", interval_options)

hist_filtered = filter_view(hist_base, selected_locations, selected_assets, selected_statuses, selected_interval_types)

# Sort newest first for readability.
if "__run_ts" in hist_filtered.columns:
    hist_filtered = hist_filtered.sort_values("__run_ts", ascending=False, na_position="last")

run_label = ""
if run_mode == "Latest Run" and "__run_ts" in hist_base.columns and hist_base["__run_ts"].notna().any():
    run_label = f"Latest run: {hist_base['__run_ts'].dropna().max():%Y-%m-%d %I:%M %p}"
elif run_mode == "Run Date" and selected_run_date:
    run_label = f"Run date: {selected_run_date}"
elif run_mode == "Custom Date Range" and start_date and end_date:
    run_label = f"Run dates: {start_date} to {end_date}"
else:
    run_label = "All history"

st.caption(f"{run_label} | Filtered rows: {len(hist_filtered):,}")

# KPI tab first so the page opens on KPI Summary.
kpi_tab, main_tab, coming_due_tab, overdue_tab, missing_meter_tab, needs_reading_tab, service_audit_tab, open_wo_tab, history_tab = st.tabs([
    "KPI Summary",
    "Main Report",
    "Coming Due",
    "Over Due",
    "Missing Meter",
    "Needs New Reading",
    "Service Audit",
    "Open Work Orders",
    "Report History",
])

with kpi_tab:
    render_kpi_page(hist_filtered)

with main_tab:
    render_main_report(hist_filtered)

with coming_due_tab:
    render_status_page(hist_filtered, "Coming Due", "Coming Due Services", "Mobile_Service_Coming_Due")

with overdue_tab:
    render_status_page(hist_filtered, "Over Due", "Over Due Services", "Mobile_Service_Over_Due")

with missing_meter_tab:
    render_status_page(hist_filtered, "Missing Meter", "Missing Meter", "Mobile_Service_Missing_Meter")

with needs_reading_tab:
    render_status_page(hist_filtered, "Needs New Reading", "Needs New Reading", "Mobile_Service_Needs_New_Reading")

with service_audit_tab:
    render_service_audit_page(hist_filtered)

with open_wo_tab:
    render_open_wo_page(hist_filtered)

with history_tab:
    render_history_page(hist_filtered)
