r"""
08_MX_NS_PO_Crosscheck.py

Streamlit reporting page for comparing MaintainX purchase orders against NetSuite PO records.
Designed to live inside:
    C:\Users\Brad\Desktop\Reporting\pages\

Inputs come from the Reporting App shared database path:
    reporting_shared.DB_PATH

Fallback:
    C:\Users\Brad\Desktop\Maintenance Pipeline\maintenance_master.db

Expected SQLite tables, with flexible name detection:
    - NetSuite_Master / Netsuite_Master / NetSuite Master
    - Purchase_Orders / Purchase Orders / Purchase -Orders

This page does NOT rebuild parquet databases. It reads directly from maintenance_master.db.
Review/OK decisions are saved back into SQLite table:
    mx_ns_crosscheck_reviews
"""

from __future__ import annotations

import re
import sqlite3
from datetime import date
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st

try:
    from auth_helper import require_login
except Exception:
    def require_login():
        return None

try:
    from reporting_shared import DB_PATH as REPORTING_DB_PATH
except Exception:
    REPORTING_DB_PATH = r"C:\Users\Brad\Desktop\Maintenance Pipeline\maintenance_master.db"


# ============================================================
# CONFIG
# ============================================================
DEFAULT_DB_PATH = Path(REPORTING_DB_PATH)
DATE_FLOOR = pd.Timestamp("2025-01-01")
REVIEW_TABLE = "mx_ns_crosscheck_reviews"

NS_TABLE_CANDIDATES = [
    "Netsuite_Master",
    "NetSuite_Master",
    "NetSuite Master",
    "Netsuite Master",
    "netsuite_master",
]

MX_TABLE_CANDIDATES = [
    "Purchase_Orders",
    "Purchase Orders",
    "Purchase -Orders",
    "Purchase-Orders",
    "purchase_orders",
]


# ============================================================
# GENERAL HELPERS
# ============================================================
def _clean_name(x: object) -> str:
    return str(x or "").strip()


def _norm_table_name(name: str) -> str:
    """Normalize a table name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _col(df: pd.DataFrame, colname: str) -> pd.Series:
    if colname in df.columns:
        return df[colname]
    return pd.Series(pd.NA, index=df.index, dtype="string")


def _first_present_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        key = c.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def _first_nonblank(s: pd.Series):
    s2 = s.astype("string").fillna("").str.strip()
    for v in s2.tolist():
        if v:
            return v
    return pd.NA


def _join_unique(series: pd.Series, max_len: int = 800) -> str:
    s = series.astype("string").fillna("").str.strip()
    vals = sorted(set([v for v in s.tolist() if v]))
    out = ", ".join(vals)
    if len(out) > max_len:
        out = out[:max_len] + "..."
    return out if out else pd.NA


def _excel_serial_to_datetime(s: pd.Series) -> pd.Series:
    """Safely parse dates that may be Excel serials or normal date strings.

    Some columns can contain large IDs, timestamps, blanks, or malformed numeric text.
    Passing those directly to pd.to_datetime(..., unit="D") can overflow. Only values
    in a realistic Excel serial range are converted as Excel dates; everything else
    is parsed as text/date normally.
    """
    s_text = s.astype("string")
    dt_from_text = pd.to_datetime(s_text, errors="coerce")

    s_num = pd.to_numeric(s_text.str.replace(",", "", regex=False), errors="coerce")
    # Excel serial range roughly covers 2000-01-01 through 2100-12-31.
    # This avoids overflow from IDs or malformed numeric values.
    excel_mask = s_num.between(36526, 73415)
    dt_from_number = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if excel_mask.any():
        dt_from_number.loc[excel_mask] = pd.to_datetime(
            s_num.loc[excel_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    return dt_from_number.fillna(dt_from_text)


def _clean_dates(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce")
    dt = dt.where(dt >= DATE_FLOOR, pd.NaT)
    return dt


def _money(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def make_review_key(parts: list[object]) -> str:
    safe = []
    for p in parts:
        if p is None or (isinstance(p, float) and pd.isna(p)):
            safe.append("")
        else:
            safe.append(str(p).strip())
    return "|".join(safe)


def xlsx_bytes(df: pd.DataFrame) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    out.seek(0)
    return out.getvalue()


# ============================================================
# SQLITE HELPERS
# ============================================================
def get_conn(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path), check_same_thread=False)


def list_tables(conn: sqlite3.Connection) -> list[str]:
    q = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    return pd.read_sql_query(q, conn)["name"].astype(str).tolist()


def find_table(conn: sqlite3.Connection, candidates: list[str]) -> str | None:
    tables = list_tables(conn)
    exact = {t.lower(): t for t in tables}
    for cand in candidates:
        if cand.lower() in exact:
            return exact[cand.lower()]

    norm_lookup = {_norm_table_name(t): t for t in tables}
    for cand in candidates:
        n = _norm_table_name(cand)
        if n in norm_lookup:
            return norm_lookup[n]

    return None


@st.cache_data(show_spinner=False, ttl=300)
def load_table(db_path_str: str, table_name: str) -> pd.DataFrame:
    with sqlite3.connect(db_path_str) as conn:
        return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)


def ensure_review_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {REVIEW_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_type TEXT NOT NULL,
            review_key TEXT NOT NULL,
            status TEXT NOT NULL,
            note TEXT,
            ts TEXT NOT NULL,
            UNIQUE(review_type, review_key)
        )
        """
    )
    conn.commit()


def load_reviews(conn: sqlite3.Connection) -> pd.DataFrame:
    ensure_review_table(conn)
    return pd.read_sql_query(f"SELECT * FROM {REVIEW_TABLE}", conn)


def save_review_rows(conn: sqlite3.Connection, rows: list[dict]) -> int:
    ensure_review_table(conn)
    if not rows:
        return 0
    sql = f"""
        INSERT INTO {REVIEW_TABLE} (review_type, review_key, status, note, ts)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(review_type, review_key)
        DO UPDATE SET
            status = excluded.status,
            note = excluded.note,
            ts = excluded.ts
    """
    payload = [(r["review_type"], r["review_key"], r["status"], r.get("note", ""), r["ts"]) for r in rows]
    conn.executemany(sql, payload)
    conn.commit()
    return len(payload)


# ============================================================
# STANDARDIZE MAINTAINX PURCHASE ORDERS
# ============================================================
def standardize_mx_purchase_orders(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    po_col = _first_present_col(df, ["Purchase Order #", "PO #", "PO", "Purchase Order", "Document Number"])
    if not po_col:
        return pd.DataFrame()

    date_col = _first_present_col(df, ["Completed On", "Posting Date", "Approved On", "Created On", "Date"])
    vendor_col = _first_present_col(df, ["Vendor", "Name"])
    line_col = _first_present_col(df, ["Line Name", "Description", "Item Description"])
    ns_item_col = _first_present_col(df, ["NS Item", "Item"])
    ns_loc_col = _first_present_col(df, ["NS Segmentation Location", "NS Item Location", "NS Location", "Location"])
    wo_col = _first_present_col(df, ["Maintenance Work Order", "Work Order", "WO"])
    approver_col = _first_present_col(df, ["Approver Name", "Created By", "created_by"])
    received_col = _first_present_col(df, ["Received Cost", "Amount", "Item_Line_Amount"])
    ordered_total_col = _first_present_col(df, ["Total Ordered Cost", "PO Total", "Total Cost"])
    received_total_col = _first_present_col(df, ["Total Received Cost"])
    source_col = _first_present_col(df, ["_source_file", "Source_File", "Source File"])

    out = pd.DataFrame(index=df.index)
    out["PO_Key"] = _col(df, po_col).astype("string").fillna("").str.strip()

    if date_col:
        out["Date"] = _clean_dates(_excel_serial_to_datetime(_col(df, date_col)))
    else:
        out["Date"] = pd.NaT

    out["Vendor"] = _col(df, vendor_col).astype("string").fillna("").str.strip() if vendor_col else pd.NA
    out["Line Name"] = _col(df, line_col).astype("string").fillna("").str.strip() if line_col else pd.NA
    out["NS Item"] = _col(df, ns_item_col).astype("string").fillna("").str.strip() if ns_item_col else pd.NA
    out["NS Location"] = _col(df, ns_loc_col).astype("string").fillna("").str.strip() if ns_loc_col else pd.NA
    out["Maintenance Work Order"] = _col(df, wo_col).astype("string").fillna("").str.strip() if wo_col else pd.NA
    out["created_by"] = _col(df, approver_col).astype("string").fillna("").str.strip() if approver_col else pd.NA
    out["Source_File"] = _col(df, source_col).astype("string").fillna("").str.strip() if source_col else pd.NA

    out["Maintenance Work Order"] = out["Maintenance Work Order"].replace({"": pd.NA, "0": pd.NA, "0.0": pd.NA, "nan": pd.NA, "None": pd.NA})
    out["Has_Maint_WO"] = out["Maintenance Work Order"].astype("string").fillna("").str.strip().ne("")

    out["Received Cost"] = _money(_col(df, received_col)) if received_col else pd.NA
    out["Total Ordered Cost"] = _money(_col(df, ordered_total_col)) if ordered_total_col else pd.NA
    out["Total Received Cost"] = _money(_col(df, received_total_col)) if received_total_col else pd.NA

    out = out[out["PO_Key"].ne("")].copy()

    # ITEM rows are source rows. PO_TOTAL rows are rolled up by PO.
    items = out.copy()
    items["LineType"] = "ITEM"
    items["Amount"] = _money(items["Received Cost"])

    header = (
        out.groupby("PO_Key", dropna=False)
        .agg({
            "Date": "max",
            "Vendor": _first_nonblank,
            "NS Location": _first_nonblank,
            "created_by": _first_nonblank,
            "Maintenance Work Order": _first_nonblank,
            "Total Ordered Cost": "max",
            "Total Received Cost": "max",
            "Has_Maint_WO": "max",
            "Source_File": _first_nonblank,
        })
        .reset_index()
    )
    header["LineType"] = "PO_TOTAL"
    header["Line Name"] = header["Vendor"]
    header["Amount"] = _money(header["Total Ordered Cost"]).fillna(_money(header["Total Received Cost"]))

    item_list = out.groupby("PO_Key")["NS Item"].apply(lambda s: _join_unique(s))
    header["NS_Item_List"] = header["PO_Key"].map(item_list)
    header["NS Item"] = header["NS_Item_List"].astype("string").str.split(",").str[0].str.strip()

    items["NS_Item_List"] = pd.NA

    cols = [
        "Date", "PO_Key", "LineType", "Vendor", "Line Name", "NS Item", "NS_Item_List",
        "NS Location", "Maintenance Work Order", "Has_Maint_WO", "created_by", "Amount",
        "Total Received Cost", "Total Ordered Cost", "Source_File",
    ]
    final = pd.concat([header[cols], items[cols]], ignore_index=True)
    final["Date"] = _clean_dates(final["Date"])
    return final.sort_values(["Date", "PO_Key", "LineType"], ascending=[True, True, True])


# ============================================================
# STANDARDIZE NETSUITE MASTER
# ============================================================
def standardize_netsuite_master(raw: pd.DataFrame, mx_lookup: pd.DataFrame | None = None) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {"DocumentNumber": "Document Number", "CreatedBy": "Created By"}
    df = df.rename(columns=rename_map)

    for col in ["Date", "Document Number", "Created By", "Name", "Item", "Description", "Amount", "Location", "Department"]:
        if col not in df.columns:
            df[col] = pd.NA

    source_col = _first_present_col(df, ["_source_file", "Source_File", "Source File"])
    if not source_col:
        df["Source_File"] = pd.NA
    elif source_col != "Source_File":
        df["Source_File"] = df[source_col]

    df["Date"] = _clean_dates(_excel_serial_to_datetime(df["Date"]))
    df["Amount"] = _money(df["Amount"])
    df["PO_Key"] = df["Document Number"].astype("string").fillna("").str.strip()

    doc = df["PO_Key"].astype("string").fillna("")
    df["PO_Source"] = "Other"
    df.loc[doc.str.match(r"^\s*GI", na=False), "PO_Source"] = "MaintainX"
    df.loc[doc.str.match(r"^\s*PO", na=False), "PO_Source"] = "NetSuite"

    name_has = df["Name"].astype("string").fillna("").str.strip().ne("")
    item_has = df["Item"].astype("string").fillna("").str.strip().ne("")
    df["LineType"] = "UNKNOWN"
    df.loc[name_has & ~item_has, "LineType"] = "PO_TOTAL"
    df.loc[item_has, "LineType"] = "ITEM"

    df["PO_Total_Amount"] = pd.NA
    df["Item_Line_Amount"] = pd.NA
    df.loc[df["LineType"].eq("PO_TOTAL"), "PO_Total_Amount"] = df.loc[df["LineType"].eq("PO_TOTAL"), "Amount"]
    df.loc[df["LineType"].eq("ITEM"), "Item_Line_Amount"] = df.loc[df["LineType"].eq("ITEM"), "Amount"]

    po_total_from_header = df[df["LineType"].eq("PO_TOTAL")].groupby("PO_Key", dropna=False)["Amount"].max()
    po_total_from_items = df[df["LineType"].eq("ITEM")].groupby("PO_Key", dropna=False)["Amount"].sum()
    df["PO_GrandTotal"] = df["PO_Key"].map(po_total_from_header).fillna(df["PO_Key"].map(po_total_from_items))

    df["created_by"] = df["Created By"].astype("string")

    # For GI / MaintainX POs in NetSuite, use MaintainX approver if available.
    if mx_lookup is not None and not mx_lookup.empty and "created_by" in mx_lookup.columns:
        mx_po = mx_lookup[mx_lookup["LineType"].eq("PO_TOTAL")].copy()
        approver_map = mx_po.groupby("PO_Key")["created_by"].agg(_first_nonblank)
        is_gi = df["PO_Source"].eq("MaintainX")
        df.loc[is_gi, "created_by"] = df.loc[is_gi, "PO_Key"].map(approver_map).fillna(df.loc[is_gi, "created_by"])

    order_map = {"PO_TOTAL": 0, "ITEM": 1, "UNKNOWN": 2}
    df["_LineOrder"] = df["LineType"].map(order_map).fillna(9)
    return df.sort_values(["Date", "PO_Key", "_LineOrder"], ascending=[True, True, True]).drop(columns=["_LineOrder"])


# ============================================================
# FILTER HELPERS
# ============================================================
def date_filter_ui(df: pd.DataFrame, date_col: str, key_prefix: str):
    dt = pd.to_datetime(_col(df, date_col), errors="coerce")
    dt = dt.where(dt >= DATE_FLOOR, pd.NaT)

    min_date = dt.min()
    max_date = dt.max()
    if pd.isna(max_date):
        max_date = pd.Timestamp.today()
    if pd.isna(min_date) or min_date < DATE_FLOOR:
        min_date = DATE_FLOOR

    max_date_d = max_date.date()
    ytd_start = date(max_date_d.year, 1, 1)
    if ytd_start < DATE_FLOOR.date():
        ytd_start = DATE_FLOOR.date()

    mode = st.selectbox(
        "Date Filter Mode",
        ["YTD", "Custom Range", "Month", "Year"],
        index=0,
        key=f"{key_prefix}_date_mode",
    )

    date_start = ytd_start
    date_end = max_date_d

    if mode == "YTD":
        st.caption(f"YTD selected: {date_start} through {date_end}")

    elif mode == "Custom Range":
        d1, d2 = st.columns(2)
        with d1:
            date_start = st.date_input("Start Date", value=ytd_start, key=f"{key_prefix}_start")
        with d2:
            date_end = st.date_input("End Date", value=max_date_d, key=f"{key_prefix}_end")
    elif mode == "Month":
        tmp = df.copy()
        tmp["_dt"] = dt
        tmp = tmp.dropna(subset=["_dt"])
        opts = sorted(tmp["_dt"].dt.to_period("M").astype(str).unique().tolist())
        pick = st.selectbox("Year-Month", opts, index=len(opts) - 1 if opts else 0, key=f"{key_prefix}_month")
        if pick:
            y, m = map(int, pick.split("-"))
            date_start = date(y, m, 1)
            date_end = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    else:
        tmp = df.copy()
        tmp["_dt"] = dt
        years = sorted(tmp.dropna(subset=["_dt"])["_dt"].dt.year.unique().tolist())
        y = st.selectbox("Year", years, index=len(years) - 1 if years else 0, key=f"{key_prefix}_year")
        if y:
            date_start = date(int(y), 1, 1)
            date_end = date(int(y) + 1, 1, 1)

    return mode, date_start, date_end

def apply_date_mask(df: pd.DataFrame, date_col: str, mode: str, start: date, end: date) -> pd.Series:
    dt = pd.to_datetime(_col(df, date_col), errors="coerce")
    dt = dt.where(dt >= DATE_FLOOR, pd.NaT)
    ds = pd.to_datetime(start)
    de = pd.to_datetime(end)
    if mode in ["YTD", "Custom Range"]:
        return (dt >= ds) & (dt <= de)
    return (dt >= ds) & (dt < de)


def filter_by_po_item(df: pd.DataFrame, item_col: str, include: list[str], exclude: list[str]) -> pd.DataFrame:
    if df.empty or "PO_Key" not in df.columns:
        return df
    out = df.copy()
    po_key = _col(out, "PO_Key").astype("string").fillna("").str.strip()
    line_type = _col(out, "LineType").astype("string").fillna("").str.upper()
    items = out[line_type.eq("ITEM")].copy()
    item_s = _col(items, item_col).astype("string").fillna("").str.strip()
    item_po = _col(items, "PO_Key").astype("string").fillna("").str.strip()

    keep_keys = set(po_key.tolist())
    keep_keys.discard("")

    if include:
        include_keys = set(item_po[item_s.isin(include)].tolist())
        keep_keys &= include_keys

    if exclude:
        bad_keys = set(item_po[item_s.isin(exclude)].tolist())
        keep_keys -= bad_keys

    return out[po_key.isin(keep_keys)].copy()


def business_days_in_range(start: date, end: date, mode: str) -> int:
    """Return Monday-Friday day count for the selected date range.

    Custom Range is inclusive of the end date because apply_date_mask() is inclusive.
    Month/Year modes use an exclusive end date because apply_date_mask() is exclusive.
    """
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if pd.isna(start_ts) or pd.isna(end_ts):
        return 0

    if mode in ["YTD", "Custom Range"]:
        days = pd.bdate_range(start=start_ts, end=end_ts)
    else:
        # Month/Year date_end is the first day AFTER the selected period.
        days = pd.bdate_range(start=start_ts, end=end_ts - pd.Timedelta(days=1))
    return int(len(days))


def avg_pos_per_business_day(po_totals: pd.DataFrame, po_key_col: str, business_days: int) -> float:
    """Average POs per scheduled work day, using a 5-day work week denominator."""
    if po_totals.empty or business_days <= 0:
        return 0.0
    return float(_col(po_totals, po_key_col).nunique() / business_days)


def data_through_date(df: pd.DataFrame, date_col: str, selected_end: date) -> date | None:
    """Return the latest loaded data date inside the selected range.

    This is used for PO/workday averages because MaintainX and NetSuite are not
    refreshed on the same schedule. MaintainX may be current daily, while
    NetSuite may only be current through the prior month.
    """
    if df is None or df.empty:
        return None
    dt = pd.to_datetime(_col(df, date_col), errors="coerce")
    dt = dt.where(dt >= DATE_FLOOR, pd.NaT).dropna()
    if dt.empty:
        return None
    selected_end_ts = pd.to_datetime(selected_end)
    dt = dt[dt <= selected_end_ts]
    if dt.empty:
        return None
    return dt.max().date()


def business_days_for_loaded_data(start: date, loaded_through: date | None) -> int:
    """Business-day denominator from selected start through that source's loaded-through date."""
    if loaded_through is None:
        return 0
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(loaded_through)
    if pd.isna(start_ts) or pd.isna(end_ts) or end_ts < start_ts:
        return 0
    return int(len(pd.bdate_range(start=start_ts, end=end_ts)))


def pct_of_total(part: float | int, total: float | int) -> float:
    """Safe percentage helper for KPI cards."""
    try:
        total_f = float(total)
        part_f = float(part)
    except Exception:
        return 0.0
    if total_f == 0:
        return 0.0
    return part_f / total_f * 100.0


def avg_items_per_po(item_rows: pd.DataFrame, po_totals: pd.DataFrame) -> float:
    """Average ITEM lines per unique PO. This is not divided by business days."""
    po_count = int(_col(po_totals, "PO_Key").nunique()) if po_totals is not None and not po_totals.empty else 0
    if po_count <= 0:
        return 0.0
    return float(len(item_rows) / po_count)


def avg_po_value(po_totals: pd.DataFrame) -> float:
    """Average PO value using PO_TOTAL rows."""
    po_count = int(_col(po_totals, "PO_Key").nunique()) if po_totals is not None and not po_totals.empty else 0
    if po_count <= 0:
        return 0.0
    return float(_money(_col(po_totals, "Amount")).fillna(0).sum() / po_count)


def make_ns_gi_lookup(ns_all: pd.DataFrame) -> pd.DataFrame:
    """All NetSuite GI PO_TOTAL records, intentionally NOT date filtered.

    The selected period should be driven by MaintainX completed dates. NetSuite may have
    posted/completed the same GI in a prior month, so date-filtering this lookup can
    create false MISSING_IN_NETSUITE results.
    """
    ns_gi = ns_all[(_col(ns_all, "PO_Source").eq("MaintainX")) & (_col(ns_all, "LineType").eq("PO_TOTAL"))].copy()
    ns_gi["_POKEY"] = _col(ns_gi, "PO_Key").astype("string").fillna("").str.strip()
    ns_gi = ns_gi[ns_gi["_POKEY"].ne("")].copy()
    return ns_gi


# ============================================================
# STREAMLIT PAGE
# ============================================================
st.set_page_config(page_title="MX vs NetSuite Cross-Check", page_icon="🔎", layout="wide")

st.title("MX vs NetSuite PO Cross-Check")
st.caption("Reporting App Page 08 — reads NetSuite_Master and Purchase_Orders from the shared reporting database.")

db_path = Path(DEFAULT_DB_PATH)

with st.sidebar:
    st.subheader("Data Source")
    st.code(str(db_path), language="text")
    refresh = st.button("🔄 Clear cache / reload", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()

if not db_path.exists():
    st.error(f"Database not found: {db_path}")
    st.stop()

conn = get_conn(db_path)
all_tables = list_tables(conn)
ns_table = find_table(conn, NS_TABLE_CANDIDATES)
mx_table = find_table(conn, MX_TABLE_CANDIDATES)

with st.expander("DB table detection", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.write("**Detected NetSuite table**")
    c1.code(ns_table or "Not found")
    c2.write("**Detected MaintainX PO table**")
    c2.code(mx_table or "Not found")
    c3.write("**Available tables**")
    c3.dataframe(pd.DataFrame({"tables": all_tables}), hide_index=True, use_container_width=True)

if not ns_table or not mx_table:
    st.error("Could not find both required SQLite tables. Check table names above.")
    st.stop()

raw_mx = load_table(str(db_path), mx_table)
mx = standardize_mx_purchase_orders(raw_mx)
raw_ns = load_table(str(db_path), ns_table)
ns = standardize_netsuite_master(raw_ns, mx_lookup=mx)

if mx.empty or ns.empty:
    st.error("One of the standardized datasets came back empty. Use the table detection expander to verify inputs.")
    st.stop()

reviews = load_reviews(conn)

page = st.radio(
    "View",
    ["KPI Dashboard", "PO Cross-Check", "NS Bypass", "NetSuite PO Viewer", "MaintainX PO Viewer", "Review Overrides"],
    horizontal=True,
)

# ============================================================
# COMMON FILTERS
# ============================================================
st.subheader("Filters")
base_date_df = ns if page in ["KPI Dashboard", "NS Bypass", "NetSuite PO Viewer"] else mx[mx["LineType"].eq("PO_TOTAL")]
date_mode, date_start, date_end = date_filter_ui(base_date_df, "Date", "mxns_common")
business_days = business_days_in_range(date_start, date_end, date_mode)

person_opts = sorted(set(_col(ns, "created_by").dropna().astype("string").tolist() + _col(mx, "created_by").dropna().astype("string").tolist()))
loc_opts = sorted(set(_col(ns, "Location").dropna().astype("string").tolist() + _col(mx, "NS Location").dropna().astype("string").tolist()))
item_opts = sorted(set(_col(ns, "Item").dropna().astype("string").tolist() + _col(mx, "NS Item").dropna().astype("string").tolist()))

f1, f2, f3 = st.columns(3)
with f1:
    locations = st.multiselect("Location", loc_opts, default=[], key="common_location")
with f2:
    people = st.multiselect("Person / created_by", person_opts, default=[], key="common_people")
with f3:
    only_unreviewed = st.checkbox("Hide Reviewed OK", value=True, key="common_hide_reviewed")

with st.expander("Optional Item Filter"):
    i1, i2 = st.columns(2)
    with i1:
        include_items = st.multiselect("Include item", item_opts, default=[], key="common_include_items")
    with i2:
        exclude_items = st.multiselect("Exclude item", item_opts, default=[], key="common_exclude_items")

# Filter datasets used for viewer/KPI counts.
# Cross-check matching uses date-filtered MX records against an all-date NetSuite GI lookup.
#
# Important KPI behavior:
#   ns_f_base / mx_f_base = selected date/location/item filters BEFORE person filtering.
#   ns_f / mx_f           = same filters AFTER person filtering.
# This lets the KPI page show the selected person's % of the total workload.
ns_f_base = ns.loc[apply_date_mask(ns, "Date", date_mode, date_start, date_end)].copy()
mx_f_base = mx.loc[apply_date_mask(mx, "Date", date_mode, date_start, date_end)].copy()

if locations:
    ns_f_base = ns_f_base[_col(ns_f_base, "Location").isin(locations)].copy()
    mx_f_base = mx_f_base[_col(mx_f_base, "NS Location").isin(locations)].copy()

if include_items or exclude_items:
    ns_f_base = filter_by_po_item(ns_f_base, "Item", include_items, exclude_items)
    mx_f_base = filter_by_po_item(mx_f_base, "NS Item", include_items, exclude_items)

ns_f = ns_f_base.copy()
mx_f = mx_f_base.copy()

if people:
    ns_f = ns_f[_col(ns_f, "created_by").isin(people)].copy()
    mx_f = mx_f[_col(mx_f, "created_by").isin(people)].copy()

# ============================================================
# KPI DASHBOARD
# ============================================================
if page == "KPI Dashboard":
    # Current KPI view after all selected filters, including person.
    mx_po_tot = mx_f[_col(mx_f, "LineType").eq("PO_TOTAL")].copy()
    mx_items = mx_f[_col(mx_f, "LineType").eq("ITEM")].copy()
    ns_bypass = ns_f[_col(ns_f, "PO_Source").eq("NetSuite")].copy()
    ns_bypass_tot = ns_bypass[_col(ns_bypass, "LineType").eq("PO_TOTAL")].copy()
    ns_bypass_items = ns_bypass[_col(ns_bypass, "LineType").eq("ITEM")].copy()

    # Base KPI totals after date/location/item filters but BEFORE person filtering.
    mx_po_tot_base = mx_f_base[_col(mx_f_base, "LineType").eq("PO_TOTAL")].copy()
    ns_bypass_base = ns_f_base[_col(ns_f_base, "PO_Source").eq("NetSuite")].copy()
    ns_bypass_tot_base = ns_bypass_base[_col(ns_bypass_base, "LineType").eq("PO_TOTAL")].copy()

    mx_po_count = int(_col(mx_po_tot, "PO_Key").nunique())
    ns_po_count = int(_col(ns_bypass_tot, "PO_Key").nunique())
    mx_total_po_count = int(_col(mx_po_tot_base, "PO_Key").nunique())
    ns_total_po_count = int(_col(ns_bypass_tot_base, "PO_Key").nunique())

    mx_po_cost = float(_money(_col(mx_po_tot, "Amount")).fillna(0).sum())
    ns_po_cost = float(_money(_col(ns_bypass_tot, "Amount")).fillna(0).sum())
    mx_pct_total = pct_of_total(mx_po_count, mx_total_po_count)
    ns_pct_total = pct_of_total(ns_po_count, ns_total_po_count)

    # MaintainX and NetSuite are refreshed on different schedules.
    # Use a separate business-day denominator for each source based on that
    # source's latest loaded date inside the selected period.
    mx_loaded_through = data_through_date(mx_po_tot_base, "Date", date_end)
    ns_loaded_through = data_through_date(ns_bypass_tot_base, "Date", date_end)
    mx_business_days = business_days_for_loaded_data(date_start, mx_loaded_through)
    ns_business_days = business_days_for_loaded_data(date_start, ns_loaded_through)

    st.markdown("### PO Volume")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("MX POs", f"{mx_po_count:,}")
    k2.metric("NS Bypass POs", f"{ns_po_count:,}")
    k3.metric("MX % of Total", f"{mx_pct_total:,.1f}%")
    k4.metric("NS % of Total", f"{ns_pct_total:,.1f}%")
    k5.metric("Avg PO/workday MX", f"{avg_pos_per_business_day(mx_po_tot, 'PO_Key', mx_business_days):,.2f}")
    k6.metric("Avg PO/workday NS", f"{avg_pos_per_business_day(ns_bypass_tot, 'PO_Key', ns_business_days):,.2f}")

    mx_loaded_label = mx_loaded_through.strftime("%Y-%m-%d") if mx_loaded_through else "No data"
    ns_loaded_label = ns_loaded_through.strftime("%Y-%m-%d") if ns_loaded_through else "No data"
    st.caption(
        f"Avg PO/workday now uses separate loaded-through dates: "
        f"MX through {mx_loaded_label} = {mx_business_days:,} Mon-Fri days; "
        f"NetSuite through {ns_loaded_label} = {ns_business_days:,} Mon-Fri days. "
        f"% of Total uses the same date/location/item filters before the Person filter: "
        f"MX {mx_po_count:,}/{mx_total_po_count:,}; NS {ns_po_count:,}/{ns_total_po_count:,}."
    )

    st.markdown("### Cost and PO Detail Averages")
    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("MX PO Cost", f"${mx_po_cost:,.2f}")
    d2.metric("NS Bypass Cost", f"${ns_po_cost:,.2f}")
    d3.metric("Avg MX Items/PO", f"{avg_items_per_po(mx_items, mx_po_tot):,.2f}")
    d4.metric("Avg NS Items/PO", f"{avg_items_per_po(ns_bypass_items, ns_bypass_tot):,.2f}")
    d5.metric("Avg MX PO Value", f"${avg_po_value(mx_po_tot):,.2f}")
    d6.metric("Avg NS PO Value", f"${avg_po_value(ns_bypass_tot):,.2f}")
    st.caption("Avg Items/PO is total item-line count divided by unique POs. It is not divided by business days.")

    if people:
        person_label = ", ".join([str(p) for p in people])
        st.info(f"Person filter active: {person_label}. The % cards show this selection as a share of the total after date/location/item filters.")

    st.divider()

    # Cross-check summary
    mx_po = mx_po_tot[["PO_Key", "Date", "NS Location", "created_by", "Amount"]].copy().rename(columns={"Amount": "MX_PO_Total"})
    mx_po["_POKEY"] = _col(mx_po, "PO_Key").astype("string").fillna("").str.strip()
    # IMPORTANT: NetSuite GI lookup is NOT date filtered.
    # The selected range is driven by MX completed date. This prevents false missing
    # results when the GI exists in NetSuite but was posted/completed in a prior month.
    ns_gi_tot = make_ns_gi_lookup(ns)

    cross = mx_po.merge(
        ns_gi_tot[["_POKEY", "Date", "Amount", "Location", "created_by", "Source_File"]].drop_duplicates("_POKEY"),
        on="_POKEY",
        how="left",
        suffixes=("", "_NS"),
    ).rename(columns={"Date_NS": "NS_Date", "Amount": "NS_PO_Total", "created_by_NS": "ns_created_by", "Source_File": "NS_Source_File"})

    cross["MX_PO_Total"] = _money(_col(cross, "MX_PO_Total"))
    cross["NS_PO_Total"] = _money(_col(cross, "NS_PO_Total"))
    cross["Delta"] = cross["MX_PO_Total"].fillna(0) - cross["NS_PO_Total"].fillna(0)
    cross["MatchFlag"] = "MATCH"
    cross.loc[cross["NS_PO_Total"].isna(), "MatchFlag"] = "MISSING_IN_NETSUITE"
    cross.loc[(~cross["NS_PO_Total"].isna()) & (cross["Delta"].abs() > 0.01), "MatchFlag"] = "MISMATCH"

    mismatches = cross[cross["MatchFlag"].eq("MISMATCH")].copy()
    missing = cross[cross["MatchFlag"].eq("MISSING_IN_NETSUITE")].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Mismatched GI POs", f"{len(mismatches):,}")
    c2.metric("Missing GI in NetSuite", f"{len(missing):,}")
    c3.metric("Total Absolute Delta", f"${_money(_col(mismatches, 'Delta')).abs().fillna(0).sum():,.2f}")

    st.write("### Mismatches")
    st.dataframe(mismatches, use_container_width=True, height=280)
    st.download_button("⬇️ Download mismatches XLSX", xlsx_bytes(mismatches), "mx_ns_mismatches.xlsx")

    st.write("### Missing GI in NetSuite")
    st.dataframe(missing, use_container_width=True, height=280)
    st.download_button("⬇️ Download missing XLSX", xlsx_bytes(missing), "mx_ns_missing_in_netsuite.xlsx")

# ============================================================
# PO CROSS-CHECK REVIEW PAGE
# ============================================================
elif page == "PO Cross-Check":
    mx_po_tot = mx_f[_col(mx_f, "LineType").eq("PO_TOTAL")].copy()
    # NetSuite GI lookup intentionally uses ALL dates.
    # MX completed date controls which POs are being audited for the selected period.
    ns_gi_tot = make_ns_gi_lookup(ns)

    mx_po_tot["_POKEY"] = _col(mx_po_tot, "PO_Key").astype("string").fillna("").str.strip()

    x = mx_po_tot[["PO_Key", "_POKEY", "Date", "NS Location", "created_by", "Amount", "Vendor", "Source_File"]].copy()
    x = x.rename(columns={"Amount": "MX_PO_Total", "created_by": "mx_created_by"})
    ns_keep = ns_gi_tot[["_POKEY", "Date", "Amount", "Location", "created_by", "Source_File"]].copy()
    ns_keep = ns_keep.rename(columns={"Date": "NS_Date", "Amount": "NS_PO_Total", "created_by": "ns_created_by", "Source_File": "NS_Source_File"})
    x = x.merge(ns_keep.drop_duplicates("_POKEY"), on="_POKEY", how="left")

    x["MX_PO_Total"] = _money(_col(x, "MX_PO_Total"))
    x["NS_PO_Total"] = _money(_col(x, "NS_PO_Total"))
    x["Delta"] = x["MX_PO_Total"].fillna(0) - x["NS_PO_Total"].fillna(0)
    x["MatchFlag"] = "MATCH"
    x.loc[x["NS_PO_Total"].isna(), "MatchFlag"] = "MISSING_IN_NETSUITE"
    x.loc[(~x["NS_PO_Total"].isna()) & (x["Delta"].abs() > 0.01), "MatchFlag"] = "MISMATCH"

    flag_filter = st.multiselect("MatchFlag", ["MATCH", "MISMATCH", "MISSING_IN_NETSUITE"], default=["MISMATCH", "MISSING_IN_NETSUITE"])
    if flag_filter:
        x = x[_col(x, "MatchFlag").isin(flag_filter)].copy()

    x["_review_key"] = x.apply(
        lambda r: make_review_key(["CROSSCHECK", r.get("PO_Key"), round(float(r.get("MX_PO_Total") or 0), 2), round(float(r.get("NS_PO_Total") or 0), 2)]),
        axis=1,
    )
    ok_keys = set(reviews.loc[(reviews["review_type"].eq("CROSSCHECK")) & (reviews["status"].eq("OK")), "review_key"].astype(str).tolist()) if not reviews.empty else set()
    x["ReviewedOK"] = x["_review_key"].isin(ok_keys)
    if only_unreviewed:
        x = x[~x["ReviewedOK"]].copy()

    st.write("### Results")
    show_cols = ["PO_Key", "Date", "NS_Date", "NS Location", "mx_created_by", "Vendor", "MX_PO_Total", "NS_PO_Total", "Delta", "MatchFlag", "Location", "ns_created_by", "ReviewedOK", "Source_File", "NS_Source_File"]
    show_cols = [c for c in show_cols if c in x.columns]
    st.dataframe(x[show_cols], use_container_width=True, height=550)

    st.write("### Mark OK")
    selected = st.multiselect("Select PO_Key(s) to mark OK", sorted(_col(x, "PO_Key").dropna().astype("string").unique().tolist()))
    note = st.text_input("Optional note")
    if st.button("✅ Mark selected as OK"):
        rows = []
        now = str(pd.Timestamp.now())
        for po in selected:
            r = x[_col(x, "PO_Key").astype("string").eq(str(po))].head(1)
            if r.empty:
                continue
            rows.append({"review_type": "CROSSCHECK", "review_key": r["_review_key"].iloc[0], "status": "OK", "note": note, "ts": now})
        n = save_review_rows(conn, rows)
        st.success(f"Marked OK: {n}")
        st.cache_data.clear()
        st.rerun()

    st.download_button("⬇️ Download cross-check XLSX", xlsx_bytes(x[show_cols]), "mx_ns_crosscheck.xlsx")


# ============================================================
# NS BYPASS PAGE
# ============================================================
elif page == "NS Bypass":
    st.subheader("NS Bypass Analysis")
    st.caption("NetSuite POs with PO numbers starting with PO are treated as bypassing MaintainX/CMMS. Item filter uses the NetSuite Item column only.")

    ns_bypass = ns_f[_col(ns_f, "PO_Source").eq("NetSuite")].copy()
    ns_bypass["_POKEY"] = _col(ns_bypass, "PO_Key").astype("string").fillna("").str.strip()

    item_rows = ns_bypass[_col(ns_bypass, "LineType").eq("ITEM")].copy()
    ns_item_options = sorted(
        _col(item_rows, "Item")
        .astype("string")
        .fillna("")
        .str.strip()
        .loc[lambda s: s.ne("")]
        .unique()
        .tolist()
    )

    selected_ns_items = st.multiselect(
        "Filter bypass POs by NetSuite Item",
        options=ns_item_options,
        default=[],
        key="ns_bypass_item_filter",
        help="When selected, the page keeps any NetSuite bypass PO that contains at least one selected Item line."
    )

    ignored_ns_items = st.multiselect(
        "Ignore NetSuite Item(s) from bypass",
        options=ns_item_options,
        default=[],
        key="ns_bypass_ignore_items",
        help="Use this for items that are intentionally outside CMMS tracking, such as coal. Ignored item lines are removed from the bypass totals and detail."
    )

    item_rows_all = item_rows.copy()
    ignored_item_rows = pd.DataFrame(columns=item_rows_all.columns)
    if ignored_ns_items:
        ignored_mask = _col(item_rows_all, "Item").astype("string").fillna("").str.strip().isin(ignored_ns_items)
        ignored_item_rows = item_rows_all[ignored_mask].copy()
        item_rows = item_rows_all[~ignored_mask].copy()

        # Remove ignored ITEM rows from the detail. Keep PO_TOTAL only when the PO still has at least one non-ignored item line.
        remaining_po_keys = set(
            _col(item_rows, "PO_Key").astype("string").fillna("").str.strip().loc[lambda s: s.ne("")].tolist()
        )
        ns_bypass = ns_bypass[
            (
                _col(ns_bypass, "LineType").eq("PO_TOTAL")
                & ns_bypass["_POKEY"].isin(remaining_po_keys)
            )
            |
            (
                _col(ns_bypass, "LineType").eq("ITEM")
                & _col(ns_bypass, "PO_Key").astype("string").fillna("").str.strip().isin(remaining_po_keys)
                & ~_col(ns_bypass, "Item").astype("string").fillna("").str.strip().isin(ignored_ns_items)
            )
            |
            (
                ~_col(ns_bypass, "LineType").isin(["PO_TOTAL", "ITEM"])
                & ns_bypass["_POKEY"].isin(remaining_po_keys)
            )
        ].copy()

    if selected_ns_items:
        matching_po_keys = set(
            _col(item_rows, "PO_Key")
            .loc[_col(item_rows, "Item").astype("string").fillna("").str.strip().isin(selected_ns_items)]
            .astype("string")
            .fillna("")
            .str.strip()
            .tolist()
        )
        matching_po_keys.discard("")
        ns_bypass = ns_bypass[ns_bypass["_POKEY"].isin(matching_po_keys)].copy()
        item_rows = item_rows[_col(item_rows, "PO_Key").astype("string").fillna("").str.strip().isin(matching_po_keys)].copy()

    bypass_tot = ns_bypass[_col(ns_bypass, "LineType").eq("PO_TOTAL")].copy()
    bypass_items = ns_bypass[_col(ns_bypass, "LineType").eq("ITEM")].copy()

    po_count = int(_col(bypass_tot, "PO_Key").nunique())
    # Header totals cannot be split when a PO has both ignored and included lines, so the main bypass cost is item-line based.
    po_cost = float(_money(_col(bypass_items, "Amount")).fillna(0).sum())
    item_line_cost = po_cost
    ignored_line_cost = float(_money(_col(ignored_item_rows, "Amount")).fillna(0).sum()) if ignored_ns_items else 0.0
    ns_bypass_loaded_through = data_through_date(bypass_tot, "Date", date_end)
    ns_bypass_business_days = business_days_for_loaded_data(date_start, ns_bypass_loaded_through)
    avg_po_workday = avg_pos_per_business_day(bypass_tot, "PO_Key", ns_bypass_business_days)

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Bypass POs", f"{po_count:,}")
    k2.metric("Bypass cost", f"${po_cost:,.2f}")
    k3.metric("Ignored item cost", f"${ignored_line_cost:,.2f}")
    k4.metric("Bypass item lines", f"{len(bypass_items):,}")
    k5.metric("Ignored item lines", f"{len(ignored_item_rows):,}")
    k6.metric("Avg bypass PO/workday", f"{avg_po_workday:,.2f}")
    ns_bypass_loaded_label = ns_bypass_loaded_through.strftime("%Y-%m-%d") if ns_bypass_loaded_through else "No data"
    st.caption(
        f"Average bypass PO/workday uses NetSuite loaded-through date {ns_bypass_loaded_label} "
        f"({ns_bypass_business_days:,} Monday-Friday days from the selected start). "
        f"Bypass cost is item-line based so ignored items are excluded cleanly."
    )

    if not bypass_items.empty:
        st.write("### Bypass cost by NetSuite Item")
        item_summary = (
            bypass_items.assign(_Amount=_money(_col(bypass_items, "Amount")).fillna(0))
            .groupby("Item", dropna=False)
            .agg(
                PO_Count=("PO_Key", "nunique"),
                Line_Count=("PO_Key", "size"),
                Item_Line_Total=("_Amount", "sum"),
            )
            .reset_index()
            .sort_values("Item_Line_Total", ascending=False)
        )
        st.dataframe(item_summary, use_container_width=True, height=260)
        st.download_button("⬇️ Download bypass item summary XLSX", xlsx_bytes(item_summary), "ns_bypass_item_summary.xlsx")

    st.write("### Bypass PO detail")
    display_cols = [
        "Date", "Document Number", "PO_Key", "LineType", "created_by", "Created By",
        "Location", "Department", "Name", "Item", "Description", "Amount", "PO_GrandTotal", "Source_File"
    ]
    display_cols = [c for c in display_cols if c in ns_bypass.columns]
    st.dataframe(ns_bypass[display_cols], use_container_width=True, height=550)
    st.download_button("⬇️ Download NS bypass detail XLSX", xlsx_bytes(ns_bypass[display_cols]), "ns_bypass_detail.xlsx")

# ============================================================
# VIEWERS
# ============================================================
elif page == "NetSuite PO Viewer":
    po_source = st.multiselect("PO Source", sorted(_col(ns_f, "PO_Source").dropna().astype("string").unique().tolist()), default=sorted(_col(ns_f, "PO_Source").dropna().astype("string").unique().tolist()))
    view = ns_f[_col(ns_f, "PO_Source").isin(po_source)].copy() if po_source else ns_f.copy()
    display_cols = ["Date", "Document Number", "PO_Key", "PO_Source", "LineType", "created_by", "Created By", "Location", "Department", "Name", "Item", "Description", "Amount", "PO_GrandTotal", "Source_File"]
    display_cols = [c for c in display_cols if c in view.columns]
    st.metric("Rows", f"{len(view):,}")
    st.metric("Distinct POs", f"{_col(view, 'PO_Key').nunique():,}")
    st.dataframe(view[display_cols], use_container_width=True, height=650)
    st.download_button("⬇️ Download NetSuite view XLSX", xlsx_bytes(view[display_cols]), "netsuite_po_view.xlsx")

elif page == "MaintainX PO Viewer":
    show = st.selectbox("Show", ["PO Totals", "PO Totals + Line Items"], index=0)
    view = mx_f.copy()
    if show == "PO Totals":
        view = view[_col(view, "LineType").eq("PO_TOTAL")].copy()
    display_cols = ["Date", "PO_Key", "LineType", "Vendor", "Line Name", "NS Item", "NS_Item_List", "NS Location", "Maintenance Work Order", "Has_Maint_WO", "created_by", "Amount", "Total Received Cost", "Total Ordered Cost", "Source_File"]
    display_cols = [c for c in display_cols if c in view.columns]
    st.metric("Rows", f"{len(view):,}")
    st.metric("Distinct POs", f"{_col(view, 'PO_Key').nunique():,}")
    st.metric("Sum PO_TOTAL", f"${_money(_col(view[_col(view, 'LineType').eq('PO_TOTAL')], 'Amount')).fillna(0).sum():,.2f}")
    st.dataframe(view[display_cols], use_container_width=True, height=650)
    st.download_button("⬇️ Download MaintainX view XLSX", xlsx_bytes(view[display_cols]), "maintainx_po_view.xlsx")

else:
    st.write("### Review Overrides")
    st.caption(f"Saved in SQLite table: {REVIEW_TABLE}")
    st.dataframe(reviews, use_container_width=True, height=600)
    if not reviews.empty:
        st.download_button("⬇️ Download reviews XLSX", xlsx_bytes(reviews), "mx_ns_crosscheck_reviews.xlsx")
