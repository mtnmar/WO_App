# 04_Vendor_Report.py
# Report-only Vendor Audit / Vendor Performance page

from __future__ import annotations

import io
import re
import sqlite3
from datetime import datetime

import numpy as np
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


st.set_page_config(page_title="Vendor Report", layout="wide")
require_login()

# =========================
# DATABASE CONFIG
# =========================
try:
    from reporting_shared import DB_PATH
except Exception:
    from pathlib import Path
    DB_PATH = str(Path(__file__).resolve().parents[1] / "maintenance_master.db")
VENDORS_TABLE = "Vendors_Master"
PURCHASE_ORDERS_TABLE = "Purchase_Orders"
VENDOR_AUDIT_MANUAL_TABLE = "mx_vendor_audit_manual"
VENDOR_AUDIT_CURRENT_TABLE = "mx_vendor_audit_current"


# -----------------------------
# Basic helpers
# -----------------------------
def connect_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


@st.cache_data(show_spinner=False)
def load_table(table_name: str, db_path: str = DB_PATH) -> pd.DataFrame:
    try:
        with connect_db(db_path) as conn:
            if not table_exists(conn, table_name):
                return pd.DataFrame()
            return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    except Exception:
        return pd.DataFrame()


def clean_text(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def text_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[col].fillna("").astype(str)


def norm_id_series(series: pd.Series) -> pd.Series:
    s = series.copy().replace({np.nan: None}).astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.replace({"<NA>": None, "nan": None, "None": None, "": None})


def normalize_vendor_key(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).upper().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^A-Z0-9]+", "", s)
    return s


def money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"


def lead_time_score(days_val):
    if pd.isna(days_val):
        return np.nan
    d = float(days_val)
    if d <= 2:
        return 1
    if d <= 7:
        return 2
    if d <= 14:
        return 3
    if d <= 30:
        return 4
    return 5


def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower().strip(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower().strip() in lower:
            return lower[c.lower().strip()]
    return None


# -----------------------------
# Vendor audit logic copied into report-only form
# -----------------------------
def build_vendor_master_summary(vendors: pd.DataFrame) -> pd.DataFrame:
    if vendors is None or vendors.empty:
        return pd.DataFrame()

    work = vendors.copy()
    work["Vendor ID Key"] = norm_id_series(work["ID"]) if "ID" in work.columns else pd.Series([None] * len(work), index=work.index)
    work["Vendor Name"] = text_series(work, "Vendor").map(clean_text)
    work["Vendor Key"] = work["Vendor Name"].map(normalize_vendor_key)
    work["Contact Name Clean"] = text_series(work, "Contact Name").map(clean_text)
    work["Role Clean"] = text_series(work, "Role").map(clean_text)
    work["Email Clean"] = text_series(work, "Email").map(clean_text)
    work["Phone Clean"] = text_series(work, "Phone Number").map(clean_text)
    work["Assets Clean"] = text_series(work, "Assets").map(clean_text)
    work["Parts Clean"] = text_series(work, "Parts").map(clean_text)
    work["Locations Clean"] = text_series(work, "Locations").map(clean_text)

    rows = []
    for vendor_key, grp in work.groupby("Vendor Key", dropna=False):
        vendor_name = next((x for x in grp["Vendor Name"].dropna().astype(str).tolist() if x.strip()), "")
        vendor_id = next((x for x in norm_id_series(grp["Vendor ID Key"]).dropna().astype(str).tolist() if x.strip()), "")
        contact_names = sorted({x for x in grp["Contact Name Clean"].dropna().astype(str).tolist() if x.strip()})
        emails = sorted({x for x in grp["Email Clean"].dropna().astype(str).tolist() if x.strip()})
        phones = sorted({x for x in grp["Phone Clean"].dropna().astype(str).tolist() if x.strip()})
        roles = sorted({x for x in grp["Role Clean"].dropna().astype(str).tolist() if x.strip()})

        rows.append({
            "Vendor Key": vendor_key,
            "Vendor ID": vendor_id,
            "Vendor": vendor_name,
            "Contact Count": len(contact_names),
            "Contacts": " | ".join(contact_names),
            "Roles": " | ".join(roles),
            "Emails": " | ".join(emails),
            "Phones": " | ".join(phones),
            "Has Contact": int(len(contact_names) > 0),
            "Has Email": int(len(emails) > 0),
            "Has Phone": int(len(phones) > 0),
            "Assets Listed": int(text_series(grp, "Assets Clean").str.strip().ne("").any()),
            "Parts Listed": int(text_series(grp, "Parts Clean").str.strip().ne("").any()),
            "Locations Listed": int(text_series(grp, "Locations Clean").str.strip().ne("").any()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["Completeness Score"] = out[["Has Contact", "Has Email", "Has Phone"]].sum(axis=1)
    out["Completeness Status"] = np.select(
        [out["Completeness Score"].eq(3), out["Completeness Score"].eq(2), out["Completeness Score"].le(1)],
        ["Complete", "Review", "Incomplete"],
        default="Review",
    )

    def reasons(row):
        r = []
        if int(row.get("Has Contact", 0)) == 0:
            r.append("Missing contact")
        if int(row.get("Has Email", 0)) == 0:
            r.append("Missing email")
        if int(row.get("Has Phone", 0)) == 0:
            r.append("Missing phone")
        return " | ".join(r)

    out["Completeness Issues"] = out.apply(reasons, axis=1)
    return out.sort_values(["Completeness Status", "Vendor"], ascending=[False, True]).reset_index(drop=True)


def build_po_vendor_summary(pos: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Vendor Key", "PO Count", "Completed PO Count", "Avg Lead Time Days", "Median Lead Time Days",
        "Max Lead Time Days", "Last PO Created", "Last PO Completed", "Logic Lead Time Score Floor",
        "Logic Lead Time Basis", "Total Received Cost",
    ]
    if pos is None or pos.empty or "Vendor" not in pos.columns:
        return pd.DataFrame(columns=cols)

    po = pos.copy()
    po["Vendor"] = text_series(po, "Vendor").map(clean_text)
    po["Vendor Key"] = po["Vendor"].map(normalize_vendor_key)
    po_id_col = first_present(po, ["Purchase Order ID", "PO ID", "ID", "Purchase Order #", "PO #"])
    po["PO ID Key"] = norm_id_series(po[po_id_col]) if po_id_col else pd.Series([None] * len(po), index=po.index)

    created_col = first_present(po, ["Created On", "Created on", "Created"])
    completed_col = first_present(po, ["Completed On", "Completed on", "Completed"])
    approved_col = first_present(po, ["Approved On", "Approved on", "Approved"])
    cost_col = first_present(po, ["Received Cost", "Report Cost", "Total Received Cost"])

    po["Created DT"] = pd.to_datetime(po[created_col], errors="coerce") if created_col else pd.NaT
    po["Completed DT"] = pd.to_datetime(po[completed_col], errors="coerce") if completed_col else pd.NaT
    po["Approved DT"] = pd.to_datetime(po[approved_col], errors="coerce") if approved_col else pd.NaT
    po["Received Cost Num"] = pd.to_numeric(po[cost_col], errors="coerce").fillna(0.0) if cost_col else 0.0

    po_header = po.sort_values(["Vendor Key", "PO ID Key", "Completed DT"], ascending=[True, True, True]).drop_duplicates(
        subset=["Vendor Key", "PO ID Key"], keep="last"
    ).copy()
    po_header["Lead Time Days"] = (po_header["Completed DT"] - po_header["Created DT"]).dt.total_seconds() / 86400.0
    fallback = (po_header["Completed DT"] - po_header["Approved DT"]).dt.total_seconds() / 86400.0
    po_header["Lead Time Days"] = po_header["Lead Time Days"].where(po_header["Lead Time Days"].notna(), fallback)
    po_header.loc[po_header["Lead Time Days"].lt(0), "Lead Time Days"] = np.nan

    # Cost needs line-level totals, so aggregate from PO lines separately, not from deduplicated header rows.
    po_cost = po.groupby("Vendor Key", dropna=False).agg(Total_Received_Cost=("Received Cost Num", "sum")).reset_index()

    valid = po_header[po_header["Vendor Key"].astype(str).str.strip().ne("")].copy()
    out = valid.groupby("Vendor Key", dropna=False).agg(
        PO_Count=("PO ID Key", lambda s: s.dropna().nunique()),
        Completed_PO_Count=("Lead Time Days", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        Avg_Lead_Time_Days=("Lead Time Days", "mean"),
        Median_Lead_Time_Days=("Lead Time Days", "median"),
        Max_Lead_Time_Days=("Lead Time Days", "max"),
        Last_PO_Created=("Created DT", "max"),
        Last_PO_Completed=("Completed DT", "max"),
    ).reset_index()
    out = out.rename(columns={c: c.replace("_", " ") for c in out.columns})
    out = out.merge(po_cost.rename(columns={"Total_Received_Cost": "Total Received Cost"}), on="Vendor Key", how="left")
    out["Logic Lead Time Score Floor"] = out["Avg Lead Time Days"].map(lead_time_score)
    out["Logic Lead Time Basis"] = np.where(
        pd.to_numeric(out["Completed PO Count"], errors="coerce").fillna(0).gt(0),
        "PO Created-to-Completed average",
        "No completed PO history",
    )
    return out[[c for c in cols if c in out.columns]]


def normalize_manual_df(manual: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "Vendor Key": "",
        "Use Manual Override": False,
        "Manual Lead Time Score Floor": np.nan,
        "Vendor Override Code": "",
        "Vendor Override Notes": "",
        "Manual Updated At": pd.NaT,
    }
    out = manual.copy() if manual is not None else pd.DataFrame()
    for c, d in defaults.items():
        if c not in out.columns:
            out[c] = d
    out["Vendor Key"] = text_series(out, "Vendor Key")
    out["Use Manual Override"] = out["Use Manual Override"].map(lambda x: bool(x) if not pd.isna(x) else False)
    out["Manual Lead Time Score Floor"] = pd.to_numeric(out["Manual Lead Time Score Floor"], errors="coerce")
    return out[list(defaults.keys())].drop_duplicates("Vendor Key", keep="last")


def build_vendor_audit(vendors: pd.DataFrame, pos: pd.DataFrame, manual: pd.DataFrame, saved_current: pd.DataFrame | None = None) -> pd.DataFrame:
    # Prefer saved current table when available, because this is a reporting app.
    if saved_current is not None and not saved_current.empty:
        out = saved_current.copy()
        for c in ["Avg Lead Time Days", "Median Lead Time Days", "Max Lead Time Days", "Vendor Lead Time Score Floor", "Logic Lead Time Score Floor", "Manual Lead Time Score Floor", "Total Received Cost"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    master = build_vendor_master_summary(vendors)
    po_summary = build_po_vendor_summary(pos)
    manual = normalize_manual_df(manual)
    if master.empty:
        return pd.DataFrame()

    out = master.merge(po_summary, on="Vendor Key", how="left").merge(manual, on="Vendor Key", how="left")
    out["Use Manual Override"] = out["Use Manual Override"].fillna(False).astype(bool)
    out["Manual Lead Time Score Floor"] = pd.to_numeric(out["Manual Lead Time Score Floor"], errors="coerce")
    out["Logic Lead Time Score Floor"] = pd.to_numeric(out["Logic Lead Time Score Floor"], errors="coerce")
    out["Vendor Lead Time Score Floor"] = np.where(
        out["Use Manual Override"] & out["Manual Lead Time Score Floor"].notna(),
        out["Manual Lead Time Score Floor"],
        out["Logic Lead Time Score Floor"],
    )
    out["Vendor Override Source"] = np.where(
        out["Use Manual Override"] & out["Manual Lead Time Score Floor"].notna(),
        "Manual",
        np.where(out["Logic Lead Time Score Floor"].notna(), "Logic", "Missing"),
    )
    out["Vendor Override Code"] = text_series(out, "Vendor Override Code")
    out["Vendor Override Notes"] = text_series(out, "Vendor Override Notes")
    out["Lead Time Risk Band"] = np.select(
        [out["Vendor Lead Time Score Floor"].ge(5), out["Vendor Lead Time Score Floor"].ge(4), out["Vendor Lead Time Score Floor"].ge(3), out["Vendor Lead Time Score Floor"].ge(1)],
        ["Severe", "High", "Moderate", "Low"],
        default="Missing",
    )
    out["Audit Issues"] = out["Completeness Issues"].fillna("")
    out["Current Saved At"] = pd.Timestamp.now()
    preferred = [
        "Vendor ID", "Vendor", "Completeness Status", "Completeness Issues", "Contact Count", "Contacts", "Emails", "Phones",
        "PO Count", "Completed PO Count", "Avg Lead Time Days", "Median Lead Time Days", "Max Lead Time Days", "Total Received Cost",
        "Logic Lead Time Score Floor", "Use Manual Override", "Manual Lead Time Score Floor", "Vendor Lead Time Score Floor",
        "Vendor Override Source", "Lead Time Risk Band", "Vendor Override Code", "Vendor Override Notes",
        "Last PO Created", "Last PO Completed", "Parts Listed", "Assets Listed", "Locations Listed", "Vendor Key", "Current Saved At",
    ]
    return out[[c for c in preferred if c in out.columns]].sort_values(["Lead Time Risk Band", "Completeness Status", "Vendor"], ascending=[True, True, True]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_vendor_report_data() -> tuple[pd.DataFrame, dict]:
    vendors = load_table(VENDORS_TABLE)
    pos = load_table(PURCHASE_ORDERS_TABLE)
    manual = load_table(VENDOR_AUDIT_MANUAL_TABLE)
    saved_current = load_table(VENDOR_AUDIT_CURRENT_TABLE)
    audit = build_vendor_audit(vendors, pos, manual, saved_current=saved_current)
    meta = {
        "Vendor source": VENDOR_AUDIT_CURRENT_TABLE if not saved_current.empty else VENDORS_TABLE,
        "PO source": PURCHASE_ORDERS_TABLE,
        "Manual override source": VENDOR_AUDIT_MANUAL_TABLE if not manual.empty else "Not found / none",
        "Rows": len(audit),
    }
    return audit, meta


# -----------------------------
# Exports
# -----------------------------
def to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Vendor Report") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return buffer.getvalue()


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
    story.append(Paragraph("Vendor Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.now():%Y-%m-%d %I:%M %p}", styles["Normal"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<br/>".join([f"<b>{k}:</b> {v}" for k, v in filters.items()]), styles["Normal"]))
    story.append(Spacer(1, 8))

    summary_data = [["Metric", "Value"]] + [[k, v] for k, v in summary.items()]
    summary_tbl = Table(summary_data, hAlign="LEFT", colWidths=[2.8 * inch, 2.2 * inch])
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
        "Vendor", "Completeness Status", "Completeness Issues", "Contact Count", "PO Count",
        "Completed PO Count", "Avg Lead Time Days", "Vendor Lead Time Score Floor",
        "Vendor Override Source", "Lead Time Risk Band", "Total Received Cost",
    ] if c in df.columns]
    preview = df[export_cols].head(80).copy()
    for c in ["Avg Lead Time Days", "Vendor Lead Time Score Floor", "Total Received Cost"]:
        if c in preview.columns:
            preview[c] = pd.to_numeric(preview[c], errors="coerce").round(2)
    if "Total Received Cost" in preview.columns:
        preview["Total Received Cost"] = preview["Total Received Cost"].map(money)

    table_data = [export_cols] + preview.fillna("").astype(str).values.tolist()
    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.2, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(Paragraph("Vendor Detail Preview - first 80 rows", styles["Heading2"]))
    story.append(tbl)
    doc.build(story)
    return buffer.getvalue()


# -----------------------------
# Page
# -----------------------------
st.title("Vendor Report")
st.caption("Report-only vendor completeness, contact review, PO lead-time performance, and override visibility.")

with st.sidebar:
    st.header("Vendor Reporting")
    st.caption("Connected to maintenance_master.db")
    st.code(DB_PATH, language="text")
    st.code(VENDOR_AUDIT_CURRENT_TABLE, language="text")

report_df, source_meta = load_vendor_report_data()

if report_df.empty:
    st.warning("No vendor reporting data loaded. Run the Vendor Audit app or confirm Vendors_Master and Purchase_Orders exist in maintenance_master.db.")
    st.stop()

st.caption(" | ".join([f"{k}: {v}" for k, v in source_meta.items()]))

# Filters
f1, f2, f3, f4 = st.columns([1.7, 1.15, 1.15, 1.15])
with f1:
    vendor_options = sorted([v for v in report_df.get("Vendor", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if v.strip()])
    selected_vendors = st.multiselect("Vendor", vendor_options)
with f2:
    completeness_options = sorted([x for x in report_df.get("Completeness Status", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if x.strip()])
    selected_completeness = st.multiselect("Completeness", completeness_options)
with f3:
    risk_options = [x for x in ["Severe", "High", "Moderate", "Low", "Missing"] if x in set(report_df.get("Lead Time Risk Band", pd.Series(dtype=str)).dropna().astype(str))]
    selected_risk = st.multiselect("Lead Time Risk", risk_options)
with f4:
    source_options = sorted([x for x in report_df.get("Vendor Override Source", pd.Series(dtype=str)).dropna().astype(str).unique().tolist() if x.strip()])
    selected_sources = st.multiselect("Override Source", source_options)

fdf = report_df.copy()
if selected_vendors and "Vendor" in fdf.columns:
    fdf = fdf[fdf["Vendor"].astype(str).isin(selected_vendors)].copy()
if selected_completeness and "Completeness Status" in fdf.columns:
    fdf = fdf[fdf["Completeness Status"].astype(str).isin(selected_completeness)].copy()
if selected_risk and "Lead Time Risk Band" in fdf.columns:
    fdf = fdf[fdf["Lead Time Risk Band"].astype(str).isin(selected_risk)].copy()
if selected_sources and "Vendor Override Source" in fdf.columns:
    fdf = fdf[fdf["Vendor Override Source"].astype(str).isin(selected_sources)].copy()

summary_tab, lead_tab, detail_tab = st.tabs(["Vendor KPI Summary", "Lead Time Review", "Vendor Detail"])

with summary_tab:
    total_vendors = len(fdf)
    incomplete = int((fdf.get("Completeness Status", pd.Series(index=fdf.index)).astype(str) == "Incomplete").sum()) if not fdf.empty else 0
    review = int((fdf.get("Completeness Status", pd.Series(index=fdf.index)).astype(str) == "Review").sum()) if not fdf.empty else 0
    complete = int((fdf.get("Completeness Status", pd.Series(index=fdf.index)).astype(str) == "Complete").sum()) if not fdf.empty else 0
    manual_overrides = int(fdf.get("Use Manual Override", pd.Series(False, index=fdf.index)).fillna(False).astype(bool).sum()) if not fdf.empty else 0
    high_severe = int(fdf.get("Lead Time Risk Band", pd.Series(index=fdf.index)).astype(str).isin(["High", "Severe"]).sum()) if not fdf.empty else 0
    po_count = int(pd.to_numeric(fdf.get("PO Count", pd.Series(0, index=fdf.index)), errors="coerce").fillna(0).sum()) if not fdf.empty else 0
    total_cost = float(pd.to_numeric(fdf.get("Total Received Cost", pd.Series(0, index=fdf.index)), errors="coerce").fillna(0).sum()) if not fdf.empty else 0.0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Vendors", f"{total_vendors:,}")
    k2.metric("Complete", f"{complete:,}")
    k3.metric("Review", f"{review:,}")
    k4.metric("Incomplete", f"{incomplete:,}")
    k5.metric("Manual Overrides", f"{manual_overrides:,}")
    k6.metric("High/Severe LT", f"{high_severe:,}")
    c1, c2 = st.columns(2)
    c1.metric("PO Count", f"{po_count:,}")
    c2.metric("Total Received Cost", money(total_cost))

    st.markdown("### Completeness Summary")
    if "Completeness Status" in fdf.columns and not fdf.empty:
        comp_summary = fdf.groupby("Completeness Status", as_index=False).agg(Vendors=("Vendor", "count")).sort_values("Vendors", ascending=False)
        st.dataframe(comp_summary, width="stretch", hide_index=True)
        st.bar_chart(comp_summary, x="Completeness Status", y="Vendors", width="stretch")

    st.markdown("### Lead Time Risk Summary")
    if "Lead Time Risk Band" in fdf.columns and not fdf.empty:
        risk_summary = fdf.groupby("Lead Time Risk Band", as_index=False).agg(Vendors=("Vendor", "count")).sort_values("Vendors", ascending=False)
        st.dataframe(risk_summary, width="stretch", hide_index=True)
        st.bar_chart(risk_summary, x="Lead Time Risk Band", y="Vendors", width="stretch")

    st.markdown("### Contact Completeness Issues")
    issue_cols = [c for c in ["Vendor", "Completeness Status", "Completeness Issues", "Contacts", "Emails", "Phones"] if c in fdf.columns]
    contact_issues = fdf[fdf.get("Completeness Status", pd.Series(index=fdf.index)).astype(str).isin(["Incomplete", "Review"])].copy()
    st.dataframe(contact_issues[issue_cols], width="stretch", hide_index=True)

with lead_tab:
    st.subheader("Lead Time Review")
    lead_cols = [c for c in [
        "Vendor", "PO Count", "Completed PO Count", "Avg Lead Time Days", "Median Lead Time Days", "Max Lead Time Days",
        "Logic Lead Time Score Floor", "Use Manual Override", "Manual Lead Time Score Floor", "Vendor Lead Time Score Floor",
        "Vendor Override Source", "Lead Time Risk Band", "Vendor Override Code", "Vendor Override Notes",
        "Last PO Created", "Last PO Completed", "Total Received Cost",
    ] if c in fdf.columns]
    lead_view = fdf[lead_cols].copy()
    for c in ["Avg Lead Time Days", "Median Lead Time Days", "Max Lead Time Days"]:
        if c in lead_view.columns:
            lead_view[c] = pd.to_numeric(lead_view[c], errors="coerce").round(1)
    st.dataframe(lead_view, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        if not fdf.empty and "Vendor" in fdf.columns and "Avg Lead Time Days" in fdf.columns:
            top_lt = fdf.copy()
            top_lt["Avg Lead Time Days"] = pd.to_numeric(top_lt["Avg Lead Time Days"], errors="coerce")
            top_lt = top_lt.dropna(subset=["Avg Lead Time Days"]).sort_values("Avg Lead Time Days", ascending=False).head(15)
            st.markdown("### Highest Avg Lead Time")
            st.bar_chart(top_lt, x="Vendor", y="Avg Lead Time Days", width="stretch")
    with c2:
        if not fdf.empty and "Vendor" in fdf.columns and "Total Received Cost" in fdf.columns:
            top_cost = fdf.copy()
            top_cost["Total Received Cost"] = pd.to_numeric(top_cost["Total Received Cost"], errors="coerce").fillna(0)
            top_cost = top_cost.sort_values("Total Received Cost", ascending=False).head(15)
            st.markdown("### Highest Received Cost")
            st.bar_chart(top_cost, x="Vendor", y="Total Received Cost", width="stretch")

with detail_tab:
    st.subheader("Vendor Detail")
    default_cols = [c for c in [
        "Vendor ID", "Vendor", "Completeness Status", "Completeness Issues", "Contact Count", "Contacts", "Emails", "Phones",
        "PO Count", "Completed PO Count", "Avg Lead Time Days", "Median Lead Time Days", "Max Lead Time Days", "Total Received Cost",
        "Logic Lead Time Score Floor", "Use Manual Override", "Manual Lead Time Score Floor", "Vendor Lead Time Score Floor",
        "Vendor Override Source", "Lead Time Risk Band", "Vendor Override Code", "Vendor Override Notes",
        "Last PO Created", "Last PO Completed", "Parts Listed", "Assets Listed", "Locations Listed", "Vendor Key", "Current Saved At",
    ] if c in fdf.columns]

    with st.expander("Column Display", expanded=False):
        display_cols = st.multiselect("Columns", list(fdf.columns), default=default_cols)

    view = fdf[display_cols].copy() if display_cols else fdf.copy()
    st.dataframe(view, width="stretch", hide_index=True)

    filters_for_pdf = {
        "Vendor": ", ".join(selected_vendors) if selected_vendors else "All",
        "Completeness": ", ".join(selected_completeness) if selected_completeness else "All",
        "Lead Time Risk": ", ".join(selected_risk) if selected_risk else "All",
        "Override Source": ", ".join(selected_sources) if selected_sources else "All",
        "Source": source_meta.get("Vendor source", "Vendor report"),
    }
    summary_for_pdf = {
        "Vendors": f"{len(fdf):,}",
        "Incomplete": f"{int((fdf.get('Completeness Status', pd.Series(index=fdf.index)).astype(str) == 'Incomplete').sum()):,}",
        "Review": f"{int((fdf.get('Completeness Status', pd.Series(index=fdf.index)).astype(str) == 'Review').sum()):,}",
        "Manual Overrides": f"{int(fdf.get('Use Manual Override', pd.Series(False, index=fdf.index)).fillna(False).astype(bool).sum()):,}",
        "High/Severe LT": f"{int(fdf.get('Lead Time Risk Band', pd.Series(index=fdf.index)).astype(str).isin(['High','Severe']).sum()):,}",
        "Total Received Cost": money(float(pd.to_numeric(fdf.get('Total Received Cost', pd.Series(0, index=fdf.index)), errors='coerce').fillna(0).sum())),
    }

    b1, b2, b3 = st.columns(3)
    with b1:
        st.download_button(
            "Download Vendor CSV",
            data=view.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"vendor_report_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            width="stretch",
        )
    with b2:
        st.download_button(
            "Download Vendor XLSX",
            data=to_xlsx_bytes(view),
            file_name=f"vendor_report_{datetime.now():%Y%m%d_%H%M}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    with b3:
        if REPORTLAB_AVAILABLE:
            st.download_button(
                "Download Vendor PDF",
                data=build_pdf(view, filters_for_pdf, summary_for_pdf),
                file_name=f"vendor_report_{datetime.now():%Y%m%d_%H%M}.pdf",
                mime="application/pdf",
                width="stretch",
            )
        else:
            st.warning("PDF export requires ReportLab. Install with: pip install reportlab")
