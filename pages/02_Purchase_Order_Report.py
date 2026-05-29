# 02_Purchase_Order_Report.py
# Streamlit page for Purchase_Orders reporting

from __future__ import annotations

import io
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from auth_helper import require_login

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


try:
    from reporting_shared import (
        DB_PATH,
        PO_TABLE,
        load_table,
        load_locations,
        get_valid_locations,
        norm_text,
        money,
    )
except Exception:
    # Fallback keeps the page runnable if reporting_shared.py has not been copied yet.
    DB_PATH = str(Path(__file__).resolve().parents[1] / "maintenance_master.db")
    PO_TABLE = "Purchase_Orders"

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
            return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)

    @st.cache_data(show_spinner=False)
    def load_locations(db_path: str = DB_PATH) -> pd.DataFrame:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query('SELECT * FROM "Locations_Master"', conn)

    def get_valid_locations(locations_df: pd.DataFrame) -> list[str]:
        col = next((c for c in ["All Parents", "All Parent Locations", "Location", "Name"] if c in locations_df.columns), None)
        if not col:
            return []
        values = locations_df[col].dropna().astype(str).map(str.strip)
        return sorted(values[values.ne("")].unique().tolist())

st.set_page_config(page_title="Purchase Order Report", layout="wide")
require_login()


DATE_CANDIDATES = ["Posting Date", "Completed On", "Approved On", "Created On", "Due Date"]
PO_COST_COLUMN = "Received Cost"
COST_CANDIDATES = [PO_COST_COLUMN, "Ordered Cost", "Total Received Cost", "Total Ordered Cost"]
LOCATION_CANDIDATES = ["NS Segmentation Location", "Location", "All Parent Locations"]
WO_CANDIDATES = ["Maintenance Work Order", "WORKORDER", "Work Order", "Workorder"]
NS_ITEM_CANDIDATES = ["NS Item", "Item", "Expense Account"]


# -----------------------------
# Helpers
# -----------------------------
def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)


def is_blank_series(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().isin(["", "nan", "NaN", "None", "NONE"])


@st.cache_data(show_spinner=False)
def load_po_data() -> pd.DataFrame:
    return load_table(DB_PATH, PO_TABLE)


def prepare_po_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df.empty:
        return df, {}

    df = df.copy()

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].map(norm_text)

    date_col = first_present(df, DATE_CANDIDATES)
    # Use line-level Received Cost for totals. Do not sum Total Received Cost because it can repeat the full PO total on every line.
    cost_col = PO_COST_COLUMN if PO_COST_COLUMN in df.columns else first_present(df, COST_CANDIDATES)
    loc_col = first_present(df, LOCATION_CANDIDATES)
    wo_col = first_present(df, WO_CANDIDATES)
    ns_item_col = first_present(df, NS_ITEM_CANDIDATES)

    if date_col:
        df["Report Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
    else:
        df["Report Date"] = pd.NaT

    if cost_col:
        df["Report Cost"] = pd.to_numeric(df[cost_col], errors="coerce").fillna(0.0)
    else:
        df["Report Cost"] = 0.0

    if loc_col:
        df["Report Location"] = df[loc_col].map(norm_text)
    else:
        df["Report Location"] = ""

    if wo_col:
        df["Report Maintenance WO"] = df[wo_col].map(norm_text)
    else:
        df["Report Maintenance WO"] = ""

    if ns_item_col:
        df["Report Type"] = df[ns_item_col].map(norm_text)
    else:
        df["Report Type"] = ""

    ns_item = df["Report Type"].fillna("").astype(str).str.strip()
    wo_blank = is_blank_series(df["Report Maintenance WO"])

    is_inventory = ns_item.str.startswith("13410", na=False)
    is_capital = ns_item.str.startswith("16910", na=False)
    is_not_cmms = (~is_inventory) & (~is_capital) & wo_blank
    is_cmms_monitored = (~is_inventory) & (~is_capital) & (~wo_blank)

    df["PO Cost Category"] = "Other"
    df.loc[is_inventory, "PO Cost Category"] = "Inventory - 13410"
    df.loc[is_capital, "PO Cost Category"] = "Capital / Construction in Progress - 16910"
    df.loc[is_not_cmms, "PO Cost Category"] = "Not CMMS Monitored Cost"
    df.loc[is_cmms_monitored, "PO Cost Category"] = "CMMS Monitored Cost"

    meta = {
        "date_col": date_col,
        "cost_col": cost_col,
        "location_col": loc_col,
        "wo_col": wo_col,
        "ns_item_col": ns_item_col,
    }

    return df, meta


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


def apply_filters(
    df: pd.DataFrame,
    selected_locations: list[str],
    selected_categories: list[str],
    selected_statuses: list[str],
    selected_vendors: list[str],
    selected_types: list[str],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    out = df.copy()
    out = out[out["Report Date"].notna()]
    out = out[(out["Report Date"] >= start_dt) & (out["Report Date"] <= end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]

    if selected_locations:
        out = out[out["Report Location"].isin(selected_locations)]


    if selected_categories:
        out = out[out["PO Cost Category"].isin(selected_categories)]

    if selected_statuses and "Status" in out.columns:
        out = out[out["Status"].isin(selected_statuses)]

    if selected_vendors and "Vendor" in out.columns:
        out = out[out["Vendor"].isin(selected_vendors)]

    if selected_types:
        out = out[out["Report Type"].isin(selected_types)]

    return out.sort_values("Report Date", ascending=False)


def build_pdf(df: pd.DataFrame, filters: dict, summary: dict) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not installed.")

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

    story.append(Paragraph("Purchase Order Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now():%Y-%m-%d %I:%M %p}", styles["Normal"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<br/>".join([f"<b>{k}:</b> {v}" for k, v in filters.items()]), styles["Normal"]))
    story.append(Spacer(1, 8))

    summary_data = [["Metric", "Value"]] + [[k, v] for k, v in summary.items()]
    summary_tbl = Table(summary_data, hAlign="LEFT", colWidths=[2.7 * inch, 2.2 * inch])
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_tbl)
    story.append(Spacer(1, 10))

    export_cols = [c for c in [
        "Report Date",
        "Purchase Order #",
        "Purchase Order Title",
        "Status",
        "Vendor",
        "Report Location",
        "PO Cost Category",
        "Report Type",
        "Report Maintenance WO",
        "Capital Work Order",
        "Line Name",
        "Part Number",
        "Report Cost",
    ] if c in df.columns]

    preview = df[export_cols].head(80).copy()

    if "Report Date" in preview.columns:
        preview["Report Date"] = pd.to_datetime(preview["Report Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "Report Cost" in preview.columns:
        preview["Report Cost"] = preview["Report Cost"].map(money)

    table_data = [export_cols] + preview.fillna("").astype(str).values.tolist()
    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.2, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 5.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    story.append(Paragraph("Raw Data Preview - first 80 filtered rows", styles["Heading2"]))
    story.append(tbl)

    doc.build(story)
    return buffer.getvalue()




# -----------------------------
# Reviewed KPI Helpers
# -----------------------------
# The KPI tabs are report-only. They use the already-reviewed audit table as the
# source of truth, then enrich each reviewed PO with current Purchase_Orders fields
# such as segmentation location, Type, vendor/status, Maintenance WO, and Received Cost.
REVIEW_TABLE_CANDIDATES = [
    "mx_purchase-orders-audit-reviewed",
    "mx_purchase_orders_audit_reviewed",
]


def sql_table_exists(db_path: str, table_name: str) -> bool:
    try:
        with sqlite3.connect(db_path) as conn:
            q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            return pd.read_sql_query(q, conn, params=[table_name]).shape[0] > 0
    except Exception:
        return False


def resolve_table_name(db_path: str, candidates: list[str]) -> str | None:
    for table_name in candidates:
        if sql_table_exists(db_path, table_name):
            return table_name
    return None


@st.cache_data(show_spinner=False)
def load_optional_table(db_path: str, table_name: str | None) -> pd.DataFrame:
    if not table_name or not sql_table_exists(db_path, table_name):
        return pd.DataFrame()
    try:
        return load_table(db_path, table_name)
    except Exception:
        # Fallback for table names containing hyphens.
        try:
            with sqlite3.connect(db_path) as conn:
                return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        except Exception:
            return pd.DataFrame()


def clean_text_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df.empty:
        return None

    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]

    for cand in candidates:
        c_low = cand.lower().strip()
        for col in df.columns:
            if c_low in str(col).lower().strip():
                return col

    return None


def _safe_series(df: pd.DataFrame, col: str | None, default: str = "") -> pd.Series:
    if col and col in df.columns:
        return clean_text_series(df[col])
    return pd.Series(default, index=df.index, dtype="string")


def _first_nonblank(series: pd.Series) -> str:
    for value in series:
        if pd.notna(value) and str(value).strip() not in ["", "nan", "NaN", "None", "NONE"]:
            return str(value).strip()
    return ""


def _join_unique(series: pd.Series) -> str:
    vals = sorted({str(v).strip() for v in series if pd.notna(v) and str(v).strip() not in ["", "nan", "NaN", "None", "NONE"]})
    return ", ".join(vals)


def add_po_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized PO ID and PO number keys to any PO/review dataframe."""
    work = df.copy()

    po_id_col = find_col(work, [
        "po_id", "PO ID", "Purchase Order ID", "purchase_order_id", "ID", "id",
    ])
    po_num_col = find_col(work, [
        "po_number", "PO Number", "PO #", "Purchase Order #", "Purchase Order", "PO", "purchase_order_number",
    ])

    work["_PO_ID_KEY"] = _safe_series(work, po_id_col)
    work["_PO_NUMBER_KEY"] = _safe_series(work, po_num_col)

    # General key for display/filter fallback.
    work["_PO_KEY"] = work["_PO_ID_KEY"].where(work["_PO_ID_KEY"].ne(""), work["_PO_NUMBER_KEY"])
    fallback = pd.Series(work.index.astype(str), index=work.index, dtype="string")
    work["_PO_KEY"] = work["_PO_KEY"].replace("", pd.NA).astype("string").fillna(fallback)

    return work


def build_po_cross_reference(po_df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per PO from Purchase_Orders for enrichment of reviewed KPI rows."""
    if po_df.empty:
        return pd.DataFrame()

    work = add_po_keys(po_df)

    po_id_col = find_col(work, ["Purchase Order ID", "po_id", "PO ID"])
    po_num_col = find_col(work, ["Purchase Order #", "po_number", "PO Number", "PO #"])
    title_col = find_col(work, ["Purchase Order Title", "Title", "PO Title", "Name"])
    status_col = find_col(work, ["Status", "PO Status"])
    vendor_col = find_col(work, ["Vendor", "Vendor Name"])
    cwo_col = find_col(work, ["Capital Work Order", "Capital WO", "Capital Wok Order"])
    approved_col = find_col(work, ["Approved On", "Approved Date"])
    completed_col = find_col(work, ["Completed On", "Completed Date"])
    posting_col = find_col(work, ["Posting Date", "Report Date"])

    rows = []
    for key, g in work.groupby("_PO_KEY", dropna=False):
        approved_dt = pd.to_datetime(g[approved_col], errors="coerce").dropna().min() if approved_col else pd.NaT
        completed_dt = pd.to_datetime(g[completed_col], errors="coerce").dropna().max() if completed_col else pd.NaT
        posting_dt = pd.to_datetime(g[posting_col], errors="coerce").dropna().min() if posting_col else pd.NaT

        rows.append({
            "_PO_KEY": str(key),
            "_PO_ID_KEY": _first_nonblank(g["_PO_ID_KEY"]),
            "_PO_NUMBER_KEY": _first_nonblank(g["_PO_NUMBER_KEY"]),
            "PO": _first_nonblank(g[po_num_col]) if po_num_col else _first_nonblank(g["_PO_NUMBER_KEY"]),
            "PO ID": _first_nonblank(g[po_id_col]) if po_id_col else _first_nonblank(g["_PO_ID_KEY"]),
            "Title": _first_nonblank(g[title_col]) if title_col else "",
            "Status": _first_nonblank(g[status_col]) if status_col else "",
            "Vendor": _first_nonblank(g[vendor_col]) if vendor_col else "",
            "Location": _first_nonblank(g["Report Location"]) if "Report Location" in g.columns else "",
            "Type": _join_unique(g["Report Type"]) if "Report Type" in g.columns else "",
            "Cost Category": _join_unique(g["PO Cost Category"]) if "PO Cost Category" in g.columns else "",
            "Maintenance WO": _join_unique(g["Report Maintenance WO"]) if "Report Maintenance WO" in g.columns else "",
            "Capital WO": _join_unique(g[cwo_col]) if cwo_col else "",
            "Received Cost": float(pd.to_numeric(g.get("Report Cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
            "Posting Date": posting_dt,
            "Approved On": approved_dt,
            "Completed On": completed_dt,
        })

    return pd.DataFrame(rows)


def normalize_reviewed_table(reviewed_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the already-reviewed table from the auditing app."""
    if reviewed_df.empty:
        return pd.DataFrame()

    work = add_po_keys(reviewed_df)

    for c in work.columns:
        if work[c].dtype == "object":
            work[c] = work[c].map(norm_text)

    flag_key_col = find_col(work, ["flag_key", "Flag Key"])
    flag_col = find_col(work, ["flag_type", "Flag Type", "Flag", "Audit Flag"])
    status_col = find_col(work, ["review_status", "Review Status"])
    reason_col = find_col(work, ["reason_code", "Reason Code", "Review Reason", "reason"])
    note_col = find_col(work, ["review_note", "Review Note", "Notes", "Comment"])
    by_col = find_col(work, ["reviewed_by", "Reviewed By"])
    reviewed_ts_col = find_col(work, ["reviewed_ts", "Reviewed TS", "Reviewed On", "Reviewed At"])
    created_col = find_col(work, ["created_ts", "Created TS", "Created On"])
    updated_col = find_col(work, ["updated_ts", "Updated TS", "Updated On"])
    review_id_col = find_col(work, ["review_id", "Review ID"])

    out = pd.DataFrame(index=work.index)
    out["_PO_KEY"] = work["_PO_KEY"].astype(str)
    out["_PO_ID_KEY"] = work["_PO_ID_KEY"].astype(str)
    out["_PO_NUMBER_KEY"] = work["_PO_NUMBER_KEY"].astype(str)
    out["Review ID"] = _safe_series(work, review_id_col)
    out["Flag Key"] = _safe_series(work, flag_key_col)
    out["Flag"] = _safe_series(work, flag_col)
    out["Review Status"] = _safe_series(work, status_col, "Reviewed").replace("", "Reviewed")
    out["KPI Reason"] = _safe_series(work, reason_col)
    out["KPI Reason"] = out["KPI Reason"].where(out["KPI Reason"].ne(""), out["Flag"])
    out["KPI Reason"] = out["KPI Reason"].replace("", "Reviewed Audit Item")

    # Reason code is the KPI reporting category. If the reason says false positive,
    # treat it as reviewed with no error even if an older row kept a generic status.
    false_positive_reason = out["KPI Reason"].str.lower().str.contains("false positive|no error|no issue", na=False)
    out.loc[false_positive_reason, "Review Status"] = "VIEWED_NO_ERROR"
    out["Notes"] = _safe_series(work, note_col)
    out["Reviewed By"] = _safe_series(work, by_col)
    out["Reviewed On"] = pd.to_datetime(work[reviewed_ts_col], errors="coerce") if reviewed_ts_col else pd.NaT
    out["Created On"] = pd.to_datetime(work[created_col], errors="coerce") if created_col else pd.NaT
    out["Updated On"] = pd.to_datetime(work[updated_col], errors="coerce") if updated_col else pd.NaT
    out["Source Table"] = "Reviewed Audit Table"

    return out


def build_reviewed_kpi_detail(reviewed_df: pd.DataFrame, po_df: pd.DataFrame) -> pd.DataFrame:
    reviewed = normalize_reviewed_table(reviewed_df)
    if reviewed.empty:
        return pd.DataFrame()

    po_xref = build_po_cross_reference(po_df)

    if po_xref.empty:
        detail = reviewed.copy()
        for c in ["PO", "PO ID", "Title", "Status", "Vendor", "Location", "Type", "Cost Category", "Maintenance WO", "Capital WO", "Received Cost", "Posting Date", "Approved On", "Completed On"]:
            if c not in detail.columns:
                detail[c] = 0.0 if c == "Received Cost" else ""
        return detail

    # Prefer PO ID match. Then backfill by PO Number for any rows where the reviewed
    # table and Purchase_Orders table identify the same PO differently.
    detail = reviewed.merge(
        po_xref,
        on="_PO_ID_KEY",
        how="left",
        suffixes=("", "_poid"),
    )

    missing_po_match = detail["PO"].isna() if "PO" in detail.columns else pd.Series(True, index=detail.index)
    if missing_po_match.any():
        by_num = reviewed.loc[missing_po_match].merge(
            po_xref,
            on="_PO_NUMBER_KEY",
            how="left",
            suffixes=("", "_ponum"),
        )
        for c in ["_PO_KEY", "PO", "PO ID", "Title", "Status", "Vendor", "Location", "Type", "Cost Category", "Maintenance WO", "Capital WO", "Received Cost", "Posting Date", "Approved On", "Completed On"]:
            if c in by_num.columns:
                detail.loc[missing_po_match, c] = by_num[c].values

    # Clean display columns.
    detail["PO"] = clean_text_series(detail.get("PO", pd.Series("", index=detail.index))).where(
        clean_text_series(detail.get("PO", pd.Series("", index=detail.index))).ne(""),
        clean_text_series(detail.get("_PO_NUMBER_KEY", pd.Series("", index=detail.index))),
    )
    detail["PO ID"] = clean_text_series(detail.get("PO ID", pd.Series("", index=detail.index))).where(
        clean_text_series(detail.get("PO ID", pd.Series("", index=detail.index))).ne(""),
        clean_text_series(detail.get("_PO_ID_KEY", pd.Series("", index=detail.index))),
    )
    detail["Received Cost"] = pd.to_numeric(detail.get("Received Cost", 0), errors="coerce").fillna(0.0)

    return detail


@st.cache_data(show_spinner=False)
def load_audit_kpi_data(db_path: str) -> tuple[pd.DataFrame, dict]:
    review_table = resolve_table_name(db_path, REVIEW_TABLE_CANDIDATES)

    availability = {
        "KPI source": "Reviewed audit table only",
        "Reviewed table used": review_table or "Not found",
        "Reviewed table candidates": ", ".join(REVIEW_TABLE_CANDIDATES),
        "PO enrichment table": PO_TABLE,
    }

    reviewed_df = load_optional_table(db_path, review_table)
    po_raw = load_table(db_path, PO_TABLE)
    po_prepared, _ = prepare_po_df(po_raw)

    detail = build_reviewed_kpi_detail(reviewed_df, po_prepared)
    return detail, availability


def filtered_po_identifiers(df: pd.DataFrame) -> tuple[set[str], set[str], set[str]]:
    work = add_po_keys(df)
    po_keys = set(clean_text_series(work["_PO_KEY"]).tolist())
    po_ids = set(clean_text_series(work["_PO_ID_KEY"]).replace("", pd.NA).dropna().tolist())
    po_numbers = set(clean_text_series(work["_PO_NUMBER_KEY"]).replace("", pd.NA).dropna().tolist())
    return po_keys, po_ids, po_numbers


def apply_report_filter_to_audit(audit_detail: pd.DataFrame, report_df: pd.DataFrame) -> pd.DataFrame:
    """Restrict reviewed KPI rows to the currently filtered PO report rows."""
    if audit_detail.empty:
        return audit_detail

    po_keys, po_ids, po_numbers = filtered_po_identifiers(report_df)

    mask = pd.Series(False, index=audit_detail.index)
    if "_PO_KEY" in audit_detail.columns and po_keys:
        mask = mask | audit_detail["_PO_KEY"].astype(str).isin(po_keys)
    if "_PO_ID_KEY" in audit_detail.columns and po_ids:
        mask = mask | audit_detail["_PO_ID_KEY"].astype(str).isin(po_ids)
    if "_PO_NUMBER_KEY" in audit_detail.columns and po_numbers:
        mask = mask | audit_detail["_PO_NUMBER_KEY"].astype(str).isin(po_numbers)
    if "PO ID" in audit_detail.columns and po_ids:
        mask = mask | clean_text_series(audit_detail["PO ID"]).isin(po_ids)
    if "PO" in audit_detail.columns and po_numbers:
        mask = mask | clean_text_series(audit_detail["PO"]).isin(po_numbers)

    return audit_detail[mask].copy()


def is_otr_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    text = (
        clean_text_series(df.get("KPI Reason", pd.Series("", index=df.index))) + " " +
        clean_text_series(df.get("Flag", pd.Series("", index=df.index))) + " " +
        clean_text_series(df.get("Flag Key", pd.Series("", index=df.index)))
    ).str.lower()
    return text.str.contains("otr", na=False)


def review_status_bucket(s: pd.Series) -> pd.Series:
    txt = clean_text_series(s).str.lower()
    out = pd.Series("Reviewed", index=s.index)

    no_error_mask = txt.str.contains(
        "viewed_no_error|no error|false positive|cleared|valid no|reviewed no|no issue|no finding",
        na=False,
    )
    with_error_mask = txt.str.contains(
        "viewed_with_error|with error|confirmed|error|issue|corrective|mis|improper|failed|governance|allotment|allocation",
        na=False,
    )
    open_mask = txt.str.fullmatch("open|", na=False)

    out[no_error_mask] = "Reviewed - No Error"
    out[with_error_mask] = "Reviewed - With Error"
    out[open_mask] = "Reviewed"
    return out


def _unique_po_cost(detail: pd.DataFrame) -> float:
    if detail.empty:
        return 0.0
    po_key = "_PO_ID_KEY" if "_PO_ID_KEY" in detail.columns else "_PO_KEY"
    cost_by_po = detail[[po_key, "Received Cost"]].copy()
    cost_by_po["Received Cost"] = pd.to_numeric(cost_by_po["Received Cost"], errors="coerce").fillna(0.0)
    return float(cost_by_po.drop_duplicates(po_key)["Received Cost"].sum())


def summarize_kpi_detail(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=["KPI Reason", "Reviewed - With Error", "Reviewed - No Error", "Reviewed", "Reviewed Items", "POs", "Received Cost"])
    work = detail.copy()
    work["Review Bucket"] = review_status_bucket(work.get("Review Status", pd.Series("", index=work.index)))
    work["Received Cost"] = pd.to_numeric(work.get("Received Cost", 0), errors="coerce").fillna(0.0)
    po_col = "_PO_ID_KEY" if "_PO_ID_KEY" in work.columns else "_PO_KEY"

    rows = []
    for reason, g in work.groupby("KPI Reason", dropna=False):
        buckets = g["Review Bucket"].value_counts().to_dict()
        rows.append({
            "KPI Reason": reason,
            "Reviewed - With Error": int(buckets.get("Reviewed - With Error", 0)),
            "Reviewed - No Error": int(buckets.get("Reviewed - No Error", 0)),
            "Reviewed": int(buckets.get("Reviewed", 0)),
            "Reviewed Items": len(g),
            "POs": g[po_col].replace("", pd.NA).dropna().nunique() if po_col in g.columns else len(g),
            "Received Cost": _unique_po_cost(g),
        })
    out = pd.DataFrame(rows).sort_values(["Reviewed - With Error", "Received Cost"], ascending=False)
    return out


def format_detail_for_display(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    cols = [c for c in [
        "PO", "PO ID", "Title", "Status", "Vendor", "Location", "Type", "Cost Category",
        "Maintenance WO", "Capital WO", "KPI Reason", "Flag", "Review Status",
        "Reviewed By", "Reviewed On", "Notes", "Received Cost", "Posting Date",
        "Approved On", "Completed On",
    ] if c in detail.columns]
    view = detail[cols].copy()
    if "Received Cost" in view.columns:
        view["Received Cost"] = pd.to_numeric(view["Received Cost"], errors="coerce").fillna(0.0).map(money)
    for c in ["Reviewed On", "Posting Date", "Approved On", "Completed On"]:
        if c in view.columns:
            view[c] = pd.to_datetime(view[c], errors="coerce").dt.strftime("%Y-%m-%d")
            view[c] = view[c].fillna("").replace("NaT", "")
    return view


def render_kpi_review_tab(detail: pd.DataFrame, availability: dict):
    st.subheader("PO KPI Review")
    st.caption("Read-only report review pulled only from the existing reviewed PO audit table. Purchase_Orders is used only as a temporary cross-reference for Location, Type, Vendor/Status, and Received Cost.")

    if detail.empty:
        st.warning("No reviewed KPI detail matched the current report filters.")
        st.write("KPI source availability:", availability)
        return

    review_detail = detail[~is_otr_row(detail)].copy()
    if review_detail.empty:
        st.info("No non-OTR reviewed KPI items match the current filters.")
        return

    review_detail["Review Bucket"] = review_status_bucket(review_detail.get("Review Status", pd.Series("", index=review_detail.index)))
    total_items = len(review_detail)
    po_key = "_PO_ID_KEY" if "_PO_ID_KEY" in review_detail.columns else "_PO_KEY"
    unique_pos = review_detail[po_key].replace("", pd.NA).dropna().nunique() if po_key in review_detail.columns else 0
    with_error = int((review_detail["Review Bucket"] == "Reviewed - With Error").sum())
    no_error = int((review_detail["Review Bucket"] == "Reviewed - No Error").sum())
    reviewed_other = int((review_detail["Review Bucket"] == "Reviewed").sum())
    total_cost = _unique_po_cost(review_detail)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Reviewed Items", f"{total_items:,}")
    k2.metric("Reviewed POs", f"{unique_pos:,}")
    k3.metric("With Error", f"{with_error:,}")
    k4.metric("No Error", f"{no_error:,}")
    k5.metric("Reviewed Other", f"{reviewed_other:,}")
    st.metric("Unique PO Received Cost", money(total_cost))

    summary = summarize_kpi_detail(review_detail)
    if not summary.empty:
        summary_display = summary.copy()
        summary_display["Received Cost"] = summary_display["Received Cost"].map(money)
        st.markdown("### Review Reason Summary")
        st.dataframe(summary_display, width="stretch", hide_index=True)

        chart_df = summary.head(15).copy()
        st.markdown("### Reviewed With Error by Reason")
        st.bar_chart(chart_df, x="KPI Reason", y="Reviewed - With Error", width="stretch")

    st.markdown("### Reviewed KPI Detail")
    view = format_detail_for_display(review_detail)
    st.dataframe(view, width="stretch", hide_index=True)
    st.download_button(
        "Download KPI Review Detail CSV",
        data=view.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"purchase_order_kpi_review_detail_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        width="stretch",
    )


def render_kpi_otr_tab(detail: pd.DataFrame, availability: dict):
    st.subheader("PO KPI OTR")
    st.caption("Read-only OTR review pulled only from the existing reviewed PO audit table. Purchase_Orders is used only as a temporary cross-reference for Location, Type, Vendor/Status, and Received Cost.")

    if detail.empty:
        st.warning("No reviewed OTR detail matched the current report filters.")
        st.write("KPI source availability:", availability)
        return

    otr_detail = detail[is_otr_row(detail)].copy()
    if otr_detail.empty:
        st.info("No reviewed OTR KPI items match the current filters.")
        return

    otr_detail["Review Bucket"] = review_status_bucket(otr_detail.get("Review Status", pd.Series("", index=otr_detail.index)))
    total_items = len(otr_detail)
    po_key = "_PO_ID_KEY" if "_PO_ID_KEY" in otr_detail.columns else "_PO_KEY"
    unique_pos = otr_detail[po_key].replace("", pd.NA).dropna().nunique() if po_key in otr_detail.columns else 0
    confirmed = int((otr_detail["Review Bucket"] == "Reviewed - With Error").sum())
    no_error = int((otr_detail["Review Bucket"] == "Reviewed - No Error").sum())
    reviewed_other = int((otr_detail["Review Bucket"] == "Reviewed").sum())

    # OTR-specific KPI: reviewed OTR items/POs that are missing a Maintenance Work Order.
    # Exclude 13410 / Parts Inventory from this Missing MWO KPI because inventory POs
    # are not expected to carry a Maintenance Work Order. They remain included in
    # the full Reviewed OTR Detail table below.
    if "Maintenance WO" in otr_detail.columns:
        missing_mwo_mask = clean_text_series(otr_detail["Maintenance WO"]).eq("")
    else:
        missing_mwo_mask = pd.Series(False, index=otr_detail.index)

    type_text = clean_text_series(otr_detail.get("Type", pd.Series("", index=otr_detail.index))).str.lower()
    category_text = clean_text_series(otr_detail.get("Cost Category", pd.Series("", index=otr_detail.index))).str.lower()
    inventory_13410_mask = (
        type_text.str.startswith("13410", na=False)
        | type_text.str.contains("13410", na=False)
        | category_text.str.contains("13410|inventory", na=False)
    )

    otr_missing_mwo = otr_detail[missing_mwo_mask & ~inventory_13410_mask].copy()
    otr_missing_mwo_items = len(otr_missing_mwo)
    otr_missing_mwo_pos = (
        otr_missing_mwo[po_key].replace("", pd.NA).dropna().nunique()
        if po_key in otr_missing_mwo.columns else 0
    )
    otr_missing_mwo_cost = _unique_po_cost(otr_missing_mwo)

    total_cost = _unique_po_cost(otr_detail)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Reviewed OTR Items", f"{total_items:,}")
    k2.metric("OTR POs", f"{unique_pos:,}")
    k3.metric("OTR Missing MWO", f"{otr_missing_mwo_pos:,}")
    k4.metric("Confirmed / Error", f"{confirmed:,}")
    k5.metric("No Error", f"{no_error:,}")
    k6.metric("Reviewed Other", f"{reviewed_other:,}")

    c_cost1, c_cost2 = st.columns(2)
    c_cost1.metric("Unique OTR PO Received Cost", money(total_cost))
    c_cost2.metric("OTR Missing MWO Cost", money(otr_missing_mwo_cost))

    summary = summarize_kpi_detail(otr_detail)
    if not summary.empty:
        summary_display = summary.copy()
        summary_display["Received Cost"] = summary_display["Received Cost"].map(money)
        st.markdown("### OTR Reason Summary")
        st.dataframe(summary_display, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        if "Vendor" in otr_detail.columns:
            vendor_chart = otr_detail.groupby("Vendor", as_index=False)["Received Cost"].sum().sort_values("Received Cost", ascending=False).head(15)
            st.markdown("### OTR by Vendor")
            st.bar_chart(vendor_chart, x="Vendor", y="Received Cost", width="stretch")
    with c2:
        if "Location" in otr_detail.columns:
            loc_chart = otr_detail.groupby("Location", as_index=False)["Received Cost"].sum().sort_values("Received Cost", ascending=False).head(15)
            st.markdown("### OTR by Location")
            st.bar_chart(loc_chart, x="Location", y="Received Cost", width="stretch")

    st.markdown("### OTR Missing Maintenance Work Order")
    if not otr_missing_mwo.empty:
        missing_view = format_detail_for_display(otr_missing_mwo)
        st.dataframe(missing_view, width="stretch", hide_index=True)
        st.download_button(
            "Download OTR Missing MWO CSV",
            data=missing_view.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"purchase_order_otr_missing_mwo_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            width="stretch",
        )
    else:
        st.info("No reviewed OTR purchase orders are missing a Maintenance Work Order for the current filters.")

    st.markdown("### Reviewed OTR Detail")
    view = format_detail_for_display(otr_detail)
    st.dataframe(view, width="stretch", hide_index=True)
    st.download_button(
        "Download OTR Detail CSV",
        data=view.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"purchase_order_otr_detail_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        width="stretch",
    )



# -----------------------------
# Outstanding PO Helpers
# -----------------------------
OUTSTANDING_STATUS_EXCLUDE = {
    "COMPLETED", "CANCELLED", "CANCELED", "REJECTED", "VOID", "CLOSED", "FULFILLED"
}


def _num_col(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    col = first_present(df, candidates)
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index, dtype="float64")


def build_outstanding_po_lines(po_df: pd.DataFrame) -> pd.DataFrame:
    """Return line-level outstanding PO detail with example-style columns."""
    if po_df.empty:
        return pd.DataFrame()

    work = po_df.copy()

    ordered_qty = _num_col(work, ["Ordered Quantity", "Order Quantity", "Quantity", "Qty Ordered", "Ord"])
    received_qty = _num_col(work, ["Received Quantity", "Quantity Received", "Qty Received", "Rcvd"])
    balance_qty = ordered_qty - received_qty

    status = clean_text_series(work.get("Status", pd.Series("", index=work.index))).str.upper()
    open_status_mask = ~status.isin(OUTSTANDING_STATUS_EXCLUDE)

    # Outstanding means the PO is still open and the line has remaining quantity.
    outstanding_mask = open_status_mask & (balance_qty > 0)
    out = work.loc[outstanding_mask].copy()
    if out.empty:
        return out

    out["Ord"] = ordered_qty.loc[out.index]
    out["Rcvd"] = received_qty.loc[out.index]
    out["Bal"] = balance_qty.loc[out.index]

    created_col = first_present(out, ["Created On", "Created", "Creation Date", "Approved On", "Posting Date"])
    if created_col:
        out["Created On Display"] = pd.to_datetime(out[created_col], errors="coerce").dt.strftime("%Y-%m-%d")
        out["Created On Sort"] = pd.to_datetime(out[created_col], errors="coerce")
    else:
        out["Created On Display"] = ""
        out["Created On Sort"] = pd.NaT

    display_map = {
        "Purchase Order #": first_present(out, ["Purchase Order #", "PO #", "PO Number", "Purchase Order"]),
        "Vendor": first_present(out, ["Vendor", "Vendor Name"]),
        "Location": "Report Location" if "Report Location" in out.columns else first_present(out, LOCATION_CANDIDATES),
        "Status": first_present(out, ["Status", "PO Status"]),
        "Line Number": first_present(out, ["Line Number", "Line #", "Line"]),
        "Line Type": first_present(out, ["Line Type", "Type"]),
        "Line Name": first_present(out, ["Line Name", "Name", "Description"]),
        "Part Number": first_present(out, ["Part Number", "Part #", "Part"]),
    }

    view = pd.DataFrame(index=out.index)
    for display_col, source_col in display_map.items():
        if source_col and source_col in out.columns:
            view[display_col] = out[source_col]
        else:
            view[display_col] = ""

    view["Created On"] = out["Created On Display"]
    view["Created On Sort"] = out["Created On Sort"]
    view["Ord"] = out["Ord"]
    view["Rcvd"] = out["Rcvd"]
    view["Bal"] = out["Bal"]

    # Keep the display order close to the uploaded sample, with Location added for filtering clarity.
    ordered_cols = [
        "Purchase Order #", "Vendor", "Location", "Created On", "Status", "Line Number",
        "Line Type", "Line Name", "Part Number", "Ord", "Rcvd", "Bal", "Created On Sort"
    ]
    return view[[c for c in ordered_cols if c in view.columns]].reset_index(drop=True)


def render_outstanding_po_tab(po_df: pd.DataFrame, selected_locations: list[str]):
    st.subheader("Outstanding Purchase Orders")
    st.caption("Shows open PO line items with remaining quantity. Default display matches the uploaded Outstanding_POs example, with Location added for filtering clarity.")

    outstanding = build_outstanding_po_lines(po_df)
    if outstanding.empty:
        st.info("No outstanding PO lines found for the current data.")
        return

    if selected_locations and "Location" in outstanding.columns:
        outstanding = outstanding[outstanding["Location"].isin(selected_locations)].copy()

    c1, c2, c3 = st.columns([1.2, 1.2, 1.4])
    with c1:
        loc_options = sorted([x for x in outstanding.get("Location", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if x])
        tab_locations = st.multiselect(
            "Outstanding Location Filter",
            loc_options,
            default=selected_locations if selected_locations else [],
            key="outstanding_location_filter",
            help="Uses the main Location filter as the default when selected.",
        )
    with c2:
        line_mode = st.radio(
            "Outstanding Line Filter",
            ["All outstanding lines", "Parts outstanding only"],
            horizontal=False,
            key="outstanding_line_mode",
        )
    with c3:
        sort_mode = st.selectbox(
            "Sort / Export Order",
            ["Vendor", "Purchase Order #", "Created On", "Largest Balance", "Location"],
            key="outstanding_sort_mode",
        )

    filtered = outstanding.copy()
    if tab_locations and "Location" in filtered.columns:
        filtered = filtered[filtered["Location"].isin(tab_locations)]

    if line_mode == "Parts outstanding only" and "Line Type" in filtered.columns:
        filtered = filtered[filtered["Line Type"].fillna("").astype(str).str.upper().str.contains("PART", na=False)]

    sort_cols = {
        "Vendor": ["Vendor", "Purchase Order #", "Line Number"],
        "Purchase Order #": ["Purchase Order #", "Line Number"],
        "Created On": ["Created On Sort", "Purchase Order #", "Line Number"],
        "Largest Balance": ["Bal", "Vendor", "Purchase Order #"],
        "Location": ["Location", "Vendor", "Purchase Order #", "Line Number"],
    }.get(sort_mode, ["Vendor", "Purchase Order #", "Line Number"])
    sort_cols = [c for c in sort_cols if c in filtered.columns]
    ascending = False if sort_mode == "Largest Balance" else True
    if sort_cols:
        filtered = filtered.sort_values(sort_cols, ascending=ascending, kind="mergesort")

    po_count = filtered["Purchase Order #"].replace("", pd.NA).dropna().nunique() if "Purchase Order #" in filtered.columns else 0
    vendor_count = filtered["Vendor"].replace("", pd.NA).dropna().nunique() if "Vendor" in filtered.columns else 0
    total_bal = pd.to_numeric(filtered.get("Bal", 0), errors="coerce").fillna(0).sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Outstanding Lines", f"{len(filtered):,}")
    k2.metric("Outstanding POs", f"{po_count:,}")
    k3.metric("Vendors", f"{vendor_count:,}")
    k4.metric("Total Balance Qty", f"{total_bal:,.0f}")

    default_cols = [c for c in [
        "Purchase Order #", "Vendor", "Created On", "Status", "Line Number",
        "Line Type", "Line Name", "Part Number", "Ord", "Rcvd", "Bal"
    ] if c in filtered.columns]
    optional_cols = [c for c in filtered.columns if c not in ["Created On Sort"]]

    with st.expander("Outstanding PO Column Display", expanded=False):
        display_cols = st.multiselect(
            "Columns for Outstanding PO table/export",
            optional_cols,
            default=default_cols,
            key="outstanding_display_cols",
        )

    display = filtered[display_cols].copy() if display_cols else filtered.drop(columns=["Created On Sort"], errors="ignore").copy()
    st.dataframe(display, width="stretch", hide_index=True)

    st.download_button(
        f"Download Outstanding POs CSV - sorted by {sort_mode}",
        data=display.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"outstanding_pos_{sort_mode.lower().replace(' ', '_').replace('#', 'number')}_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        width="stretch",
    )


# -----------------------------
# Page
# -----------------------------
st.title("Purchase Order Report")
st.caption("Filtered PO reporting from maintenance_master.db / Purchase_Orders")

with st.sidebar:
    st.header("Purchase Order Reporting")
    st.caption("Connected to maintenance_master.db")
    st.code(DB_PATH, language="text")
    st.code(PO_TABLE, language="text")

raw_df = load_po_data()
locations_df = load_locations(DB_PATH)

df, meta = prepare_po_df(raw_df)

if df.empty:
    st.warning("No purchase order data loaded. Check maintenance_master.db and the Purchase_Orders table.")
    st.stop()

required = ["Report Date", "Report Location", "Report Type", "PO Cost Category"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required prepared column(s): {', '.join(missing)}")
    st.stop()

st.caption(
    f"Date column: {meta.get('date_col') or 'not found'} | "
    f"Cost column: {meta.get('cost_col') or 'not found'} | "
    f"Location column: {meta.get('location_col') or 'not found'} | "
    f"Type column: {meta.get('ns_item_col') or 'not found'}"
)
st.caption("PO totals, KPIs, and charts use line-level Received Cost. Total Received Cost remains available in the raw table but is not summed.")

# -----------------------------
# Filters
# -----------------------------
valid_locations = get_valid_locations(locations_df)

# If the master location table is unavailable, fall back only so the page can still run.
if not valid_locations:
    valid_locations = sorted([x for x in df["Report Location"].dropna().astype(str).unique().tolist() if x])

f1, f2, f3 = st.columns([1.35, 1.35, 1])

with f1:
    selected_locations = st.multiselect("Location", valid_locations)

with f2:
    type_options = sorted([x for x in df["Report Type"].dropna().astype(str).unique().tolist() if x])
    selected_types = st.multiselect(
        "Type",
        type_options,
        help="Formerly NS Item. Used to filter PO cost/account type.",
    )

with f3:
    period_mode = st.radio("Date Range", ["YTD", "Monthly", "Custom"], horizontal=False)

valid_dates = df["Report Date"].dropna()
def_min = valid_dates.min().date() if not valid_dates.empty else date.today()
def_max = valid_dates.max().date() if not valid_dates.empty else date.today()

with f3:
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

st.subheader("Additional PO Filters")

a1, a2, a3 = st.columns([1.2, 1, 1.2])

category_options = [
    "Capital / Construction in Progress - 16910",
    "Inventory - 13410",
    "Not CMMS Monitored Cost",
    "CMMS Monitored Cost",
    "Other",
]

with a1:
    selected_categories = st.multiselect(
        "Cost Category",
        category_options,
        default=[],
        help="Capital = NS Item 16910. Inventory = NS Item 13410. Not CMMS Monitored = not 13410/16910 and no Maintenance Work Order.",
    )

with a2:
    status_options = sorted([x for x in df.get("Status", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if x])
    selected_statuses = st.multiselect("Status", status_options)

with a3:
    vendor_options = sorted([x for x in df.get("Vendor", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if x])
    selected_vendors = st.multiselect("Vendor", vendor_options)


fdf = apply_filters(
    df,
    selected_locations=selected_locations,
    selected_categories=selected_categories,
    selected_statuses=selected_statuses,
    selected_vendors=selected_vendors,
    selected_types=selected_types,
    start_dt=start_dt,
    end_dt=end_dt,
)

report_tab, outstanding_po_tab, kpi_review_tab, kpi_otr_tab = st.tabs(["PO Report", "Outstanding POs", "KPI Review", "KPI OTR"])

with report_tab:
    # -----------------------------
    # KPIs
    # -----------------------------
    po_col = first_present(fdf, ["Purchase Order #", "Purchase Order ID"])
    unique_pos = fdf[po_col].replace("", pd.NA).dropna().nunique() if po_col else 0
    vendors = fdf["Vendor"].replace("", pd.NA).dropna().nunique() if "Vendor" in fdf.columns else 0
    locations = fdf["Report Location"].replace("", pd.NA).dropna().nunique()
    mwo_mask = ~is_blank_series(fdf["Report Maintenance WO"]) if "Report Maintenance WO" in fdf.columns else pd.Series(False, index=fdf.index)
    if po_col and po_col in fdf.columns:
        po_with_mwo = fdf.loc[mwo_mask, po_col].replace("", pd.NA).dropna().nunique()
        po_missing_mwo = fdf.loc[~mwo_mask, po_col].replace("", pd.NA).dropna().nunique()
    else:
        po_with_mwo = 0
        po_missing_mwo = 0

    # Separate informational count: unique WO numbers. This is not the same as POs with MWOs,
    # because multiple POs can reference the same Maintenance Work Order.
    unique_mwo_count = fdf.loc[mwo_mask, "Report Maintenance WO"].replace("", pd.NA).dropna().nunique() if "Report Maintenance WO" in fdf.columns else 0
    total_cost = float(fdf["Report Cost"].sum()) if "Report Cost" in fdf.columns else 0.0
    
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Filtered PO Cost", money(total_cost))
    k2.metric("Rows", f"{len(fdf):,}")
    k3.metric("Purchase Orders", f"{unique_pos:,}")
    k4.metric("Vendors", f"{vendors:,}")
    k5.metric("POs With MWO", f"{po_with_mwo:,}", help=f"Unique filtered purchase orders with a non-blank Maintenance Work Order. Unique MWO numbers: {unique_mwo_count:,}")
    
    st.caption(f"Showing {start_dt:%Y-%m-%d} through {end_dt:%Y-%m-%d}")
    
    # -----------------------------
    # Charts
    # -----------------------------
    st.subheader("PO Cost Trend")
    chart_df = fdf.copy()
    
    if not chart_df.empty:
        days = max((end_dt - start_dt).days, 1)
    
        if days <= 45:
            chart_df["Period"] = chart_df["Report Date"].dt.date
        else:
            chart_df["Period"] = chart_df["Report Date"].dt.to_period("M").astype(str)
    
        trend = chart_df.groupby("Period", as_index=False)["Report Cost"].sum().sort_values("Period")
        st.bar_chart(trend, x="Period", y="Report Cost", width="stretch")
    else:
        st.info("No rows match the selected filters.")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Cost by PO Category")
        if not fdf.empty:
            cat_chart = fdf.groupby("PO Cost Category", as_index=False)["Report Cost"].sum().sort_values("Report Cost", ascending=False)
            st.bar_chart(cat_chart, x="PO Cost Category", y="Report Cost", width="stretch")
        else:
            st.info("No category data available.")
    
    with c2:
        st.subheader("Top Vendors by Cost")
        if "Vendor" in fdf.columns and not fdf.empty:
            vendor_chart = fdf.groupby("Vendor", as_index=False)["Report Cost"].sum().sort_values("Report Cost", ascending=False).head(15)
            st.bar_chart(vendor_chart, x="Vendor", y="Report Cost", width="stretch")
        else:
            st.info("No vendor data available.")
    
    c3, c4 = st.columns(2)
    
    with c3:
        st.subheader("Cost by Location")
        if not fdf.empty:
            loc_chart = fdf.groupby("Report Location", as_index=False)["Report Cost"].sum().sort_values("Report Cost", ascending=False).head(20)
            st.bar_chart(loc_chart, x="Report Location", y="Report Cost", width="stretch")
        else:
            st.info("No location data available.")
    
    with c4:
        st.subheader("Top Types by Cost")
        if not fdf.empty:
            item_chart = fdf.groupby("Report Type", as_index=False)["Report Cost"].sum().sort_values("Report Cost", ascending=False).head(20)
            st.bar_chart(item_chart, x="Report Type", y="Report Cost", width="stretch")
        else:
            st.info("No Type data available.")
    
    
    # -----------------------------
    # Raw table + exports
    # -----------------------------
    st.subheader("Raw Filtered Data")
    
    show_cols_default = [c for c in [
        "Report Date",
        "Purchase Order #",
        "Purchase Order ID",
        "Purchase Order Title",
        "Status",
        "Vendor",
        "Report Location",
        "PO Cost Category",
        "Report Type",
        "Report Maintenance WO",
        "Capital Work Order",
        "Line Number",
        "Line Name",
        "Part Number",
        "Ordered Quantity",
        "Received Quantity",
        "Unit Cost",
        "Ordered Cost",
        "Received Cost",
        "Total Ordered Cost",
        "Total Received Cost",
        "Report Cost",
        "Posting Date",
        "Approved On",
        "Completed On",
        "Created On",
    ] if c in fdf.columns]
    
    with st.expander("Column Display", expanded=False):
        display_cols = st.multiselect("Columns", list(fdf.columns), default=show_cols_default)
    
    view = fdf[display_cols].copy() if display_cols else fdf.copy()
    st.dataframe(view, width="stretch", hide_index=True)
    
    csv_bytes = view.to_csv(index=False).encode("utf-8-sig")
    
    filters_for_pdf = {
        "Location": ", ".join(selected_locations) if selected_locations else "All",
            "Date Range": f"{start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}",
        "Cost Category": ", ".join(selected_categories) if selected_categories else "All",
        "Status": ", ".join(selected_statuses) if selected_statuses else "All",
        "Vendor": ", ".join(selected_vendors) if selected_vendors else "All",
        "Type": ", ".join(selected_types) if selected_types else "All",
        "Source": PO_TABLE,
    }
    
    summary_for_pdf = {
        "Filtered PO Cost": money(total_cost),
        "Rows": f"{len(fdf):,}",
        "Purchase Orders": f"{unique_pos:,}",
        "Vendors": f"{vendors:,}",
        "Locations": f"{locations:,}",
        "POs With MWO": f"{po_with_mwo:,}",
        "POs Missing MWO": f"{po_missing_mwo:,}",
        "Unique Maintenance WO Numbers": f"{unique_mwo_count:,}",
    }
    
    b1, b2 = st.columns([1, 1])
    
    with b1:
        st.download_button(
            "Download Filtered CSV",
            data=csv_bytes,
            file_name=f"purchase_orders_filtered_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            width="stretch",
        )
    
    with b2:
        if REPORTLAB_AVAILABLE:
            pdf_bytes = build_pdf(view, filters_for_pdf, summary_for_pdf)
            st.download_button(
                "Download Filtered PDF",
                data=pdf_bytes,
                file_name=f"purchase_order_report_{datetime.now():%Y%m%d_%H%M}.pdf",
                mime="application/pdf",
                width="stretch",
            )
        else:
            st.warning("PDF export requires ReportLab. Install with: pip install reportlab")


with outstanding_po_tab:
    render_outstanding_po_tab(df, selected_locations)

# Load the existing audit output once and reuse it for both KPI tabs.
audit_detail_all, audit_availability = load_audit_kpi_data(DB_PATH)
audit_detail_filtered = apply_report_filter_to_audit(audit_detail_all, fdf)

with kpi_review_tab:
    render_kpi_review_tab(audit_detail_filtered, audit_availability)

with kpi_otr_tab:
    render_kpi_otr_tab(audit_detail_filtered, audit_availability)
