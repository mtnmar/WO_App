# 05_Inventory_Analysis_Report.py
# Streamlit reporting page for inventory analysis from maintenance_master.db
# Uses Parts_Master + mx_inventory_transaction_detail_current only.

from __future__ import annotations

import io
import os
import re
import sqlite3
from datetime import date, datetime

import numpy as np
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


st.set_page_config(page_title="Inventory Analysis Report", layout="wide")

try:
    from reporting_shared import DB_PATH
except Exception:
    from pathlib import Path
    DB_PATH = str(Path(__file__).resolve().parents[1] / "maintenance_master.db")
PARTS_TABLE = "Parts_Master"
RESTOCK_TABLE = "ReStock_Master"
INVENTORY_TX_TABLE = "mx_inventory_transaction_detail_current"
LOCATIONS_TABLE = "Locations_Master"

PARTS_CSV_FALLBACK = ""
INVENTORY_TX_CSV_FALLBACK = ""

ALL = "All"

KPI_TARGETS = {
    "Inventory Growth %": "Movement based on selected period transaction net change.",
    "Avg MoM Change %": "Average monthly inventory movement as % of beginning inventory value.",
}

ALLOWED_PART_TYPES = {
    "51080 - HYDRATED LIME BAGS", "52694 - SAFETY SUPPLIES", "53635 - SMALL TOOLS",
    "53637 - PUMPS", "53638 - COMMUNICATIONS-RAD", "53665 - SACKING SUPPLIES",
    "53666 - ROCK DUST / AG BAG", "53667 - SUPER SACKS", "53668 - RD/AG SHRINK WRAP",
    "53669 - SUPERSACK WRAP", "53685 - PALLETS", "53693 - Enviromental Supplies",
    "53825 - LAB SUPPLIES", "54002 - MINE ELECTRICAL & LIGHTING",
    "54005 - HIGHWAY TRUCKS & TRAILERS", "54006 - SCALES", "54008 - FORKLIFT",
    "54010 - HIGH LIFTS-BULLDOZER", "54011 - PITMAN ELLIOTT TOWERS",
    "54015 - HEAVY DUTY HAULERS", "54020 - SHOVELS/CRANES", "54024 - END LOADERS",
    "54025 - MINE LOADERS", "54030 - DRILLS", "54031 - ROOF BOLTER", "54032 - Fans",
    "54033 - Blowers", "54034 - Rotary Airlocks", "54035 - COMPRESSORS",
    "54040 - CRUSHERS & GRINDING EQUPMENT", "54041 - HYDRAULIC HAMMERS-CRUSHERS",
    "54045 - SCREENING EQUIPMENT", "54050 - CONVEYORS", "54051 - CONVEYOR BELT FEEDERS",
    "54052 - DUST COLLECTORS", "54053 - WATER PUMPS", "54055 - KILN",
    "54060 - HYDRATING PLANT", "54065 - FG SACKING MACHINE", "54070 - ASPHALT-MECHANICAL",
    "54075 - ASPHALT-ELECTRICAL", "54090 - GRADALL", "54095 - DRYER",
    "54205 - BUILDINGS", "54215 - AUTOMOBILES", "54225 - UNCLASSIFIED",
    "55011 - ENVIROMENTAL SERVICES", "55099 - MISCELLANEOUS OUTSIDE SERVICES",
}


# -----------------------------
# Helpers
# -----------------------------
def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if s.lower() in {"nan", "none", "<na>", "nat"}:
        return ""
    return s


def money(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except Exception:
        return "$0.00"


def pct(value) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):,.2f}%"
    except Exception:
        return ""


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    exact = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in exact:
            return exact[key]
    for cand in candidates:
        key = cand.strip().lower().replace("_", " ")
        for col in df.columns:
            col_key = str(col).strip().lower().replace("_", " ")
            if key in col_key:
                return col
    return None


def to_num(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string").str.strip()
    s = s.str.replace(",", "", regex=False).str.replace("$", "", regex=False)
    s = s.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return not pd.read_sql_query(q, conn, params=[table_name]).empty


@st.cache_data(show_spinner=False)
def load_sql_table(table_name: str) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            if not table_exists(conn, table_name):
                return pd.DataFrame()
            return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    parts = load_sql_table(PARTS_TABLE)
    restock = load_sql_table(RESTOCK_TABLE)
    tx = load_sql_table(INVENTORY_TX_TABLE)
    locations = load_sql_table(LOCATIONS_TABLE)

    source = {
        "Parts Source": f"SQLite table: {PARTS_TABLE}" if not parts.empty else "CSV fallback",
        "ReStock Source": f"SQLite table: {RESTOCK_TABLE}" if not restock.empty else "Not loaded",
        "Inventory Tx Source": f"SQLite table: {INVENTORY_TX_TABLE}" if not tx.empty else "CSV fallback",
        "DB Path": DB_PATH,
    }

    if parts.empty and os.path.exists(PARTS_CSV_FALLBACK):
        parts = pd.read_csv(PARTS_CSV_FALLBACK, low_memory=False)
    if tx.empty and os.path.exists(INVENTORY_TX_CSV_FALLBACK):
        tx = pd.read_csv(INVENTORY_TX_CSV_FALLBACK, low_memory=False)

    return parts, restock, tx, locations, source


def location_options_from_locations_master(locations_df: pd.DataFrame, tx_df: pd.DataFrame) -> list[str]:
    parent_col = find_col(locations_df, ["All Parents", "All Parent Locations", "All Parent Location"]) if not locations_df.empty else None
    if parent_col:
        opts = sorted({clean_text(x) for x in locations_df[parent_col].dropna().tolist() if clean_text(x)})
        return [ALL] + opts

    # Fallback if Locations_Master is unavailable.
    if "Parent Location" in tx_df.columns:
        opts = sorted({clean_text(x) for x in tx_df["Parent Location"].dropna().tolist() if clean_text(x)})
        return [ALL] + opts

    return [ALL]


def first_present_text(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for c in candidates:
        col = find_col(df, [c])
        if col and col in df.columns:
            s = df[col].map(clean_text)
            if s.ne("").any():
                return s.replace("", np.nan)
    return pd.Series(np.nan, index=df.index, dtype="object")


def build_location_map(locations: pd.DataFrame) -> pd.DataFrame:
    if locations is None or locations.empty:
        return pd.DataFrame(columns=["Location", "Parent Location"])
    name_col = find_col(locations, ["Name"])
    parent_col = find_col(locations, ["All Parents", "All Parent Locations", "All Parent Location"])
    if not name_col or not parent_col:
        return pd.DataFrame(columns=["Location", "Parent Location"])
    out = pd.DataFrame({
        "Location": locations[name_col].map(clean_text),
        "Parent Location": locations[parent_col].map(clean_text),
    })
    out = out[(out["Location"] != "") & (out["Parent Location"] != "")]
    return out.drop_duplicates("Location", keep="first")



def norm_id_value(value) -> str:
    """Normalize ID values for joins."""
    s = clean_text(value)
    if not s:
        return ""
    return re.sub(r"\.0$", "", s)


def apply_parts_parent_location(parts_df: pd.DataFrame, locations_df: pd.DataFrame) -> pd.DataFrame:
    """Match audit app logic: Parts_Master.Location -> Locations_Master.Name -> All Parents."""
    out = parts_df.copy()
    loc_map = build_location_map(locations_df)
    if "__Part Location" not in out.columns:
        out["__Part Location"] = ""
    if not loc_map.empty:
        mapping = dict(zip(loc_map["Location"], loc_map["Parent Location"]))
        out["__Parent Location"] = out["__Part Location"].map(mapping).fillna(out["__Part Location"])
    else:
        out["__Parent Location"] = out["__Part Location"]
    out["__Parent Location"] = out["__Parent Location"].map(clean_text)
    return out


def apply_restock_cost_fallback(parts_df: pd.DataFrame, restock_df: pd.DataFrame) -> pd.DataFrame:
    """Match audit app cost fallback: if Parts_Master Total Cost is 0, use ReStock_Master Total Cost."""
    out = parts_df.copy()
    if restock_df is None or restock_df.empty or "ID" not in restock_df.columns:
        return out
    r = restock_df.copy()
    r["__Part ID Key"] = r["ID"].map(norm_id_value)
    total_col = find_col(r, ["Total Cost", "Inventory Value", "Total Value", "Extended Cost"])
    if not total_col:
        return out
    r["__ReStock Total Cost"] = to_num(r[total_col]).fillna(0.0)
    lookup = (
        r[["__Part ID Key", "__ReStock Total Cost"]]
        .drop_duplicates("__Part ID Key", keep="first")
    )
    if "__Part ID Key" not in out.columns:
        if "ID" in out.columns:
            out["__Part ID Key"] = out["ID"].map(norm_id_value)
        else:
            out["__Part ID Key"] = ""
    out = out.merge(lookup, on="__Part ID Key", how="left")
    out["__Part Value"] = np.where(
        pd.to_numeric(out["__Part Value"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(out["__ReStock Total Cost"], errors="coerce").fillna(0).ne(0),
        pd.to_numeric(out["__ReStock Total Cost"], errors="coerce").fillna(0),
        pd.to_numeric(out["__Part Value"], errors="coerce").fillna(0),
    )
    return out.drop(columns=["__ReStock Total Cost"], errors="ignore")


def build_inventory_error_audit_report(parts_df: pd.DataFrame, locations_df: pd.DataFrame) -> pd.DataFrame:
    work = parts_df.copy()
    work["Part ID Key"] = work["ID"].map(clean_text) if "ID" in work.columns else pd.Series("", index=work.index)
    work["Part Name"] = work[find_col(work, ["Name"])].map(clean_text) if find_col(work, ["Name"]) else ""
    pn_col = find_col(work, ["Part Numbers", "Part Number"])
    work["Part Numbers"] = work[pn_col].map(clean_text).replace("", np.nan) if pn_col else np.nan
    loc_col = find_col(work, ["Location"])
    work["Sub-Location"] = work[loc_col].map(clean_text) if loc_col else ""
    area_col = find_col(work, ["Area"])
    work["Area"] = work[area_col].map(clean_text).replace("", np.nan) if area_col else np.nan
    subtype_col = find_col(work, ["SUB-TYPE", "Subtype", "Part Subtype"])
    work["Part Subtype"] = work[subtype_col].map(clean_text).replace("", np.nan) if subtype_col else np.nan
    types_col = find_col(work, ["Types", "Part Types", "Type"])
    work["Part Types"] = work[types_col].map(clean_text).replace("", np.nan) if types_col else np.nan
    qty_col = find_col(work, ["Quantity in Stock", "Available Quantity", "Qty In Stock"])
    unit_col = find_col(work, ["Unit Cost"])
    total_col = find_col(work, ["Total Cost"])
    work["Qty In Stock"] = to_num(work[qty_col]).fillna(0) if qty_col else 0
    work["Unit Cost Num"] = to_num(work[unit_col]).fillna(0) if unit_col else 0
    work["Total Cost Num"] = to_num(work[total_col]).fillna(0) if total_col else 0
    work["Vendor"] = first_present_text(work, ["Vendor", "Vendors", "Preferred Vendor", "Preferred Vendors", "Vendor Name", "Supplier"])

    loc_map = build_location_map(locations_df)
    if not loc_map.empty:
        work = work.merge(loc_map, left_on="Sub-Location", right_on="Location", how="left")
        work["Parent Location"] = work["Parent Location"].fillna(work["Sub-Location"])
        work = work.drop(columns=["Location"], errors="ignore")
    else:
        work["Parent Location"] = work["Sub-Location"]

    work["Missing Part Number Flag"] = work["Part Numbers"].isna().astype(int)
    work["Missing Vendor Flag"] = work["Vendor"].isna().astype(int)
    work["Missing Location Flag"] = work["Parent Location"].isna().astype(int) | work["Parent Location"].map(clean_text).eq("")
    work["Missing Area Flag"] = work["Area"].isna().astype(int)
    work["Missing Sub-Type Flag"] = work["Part Subtype"].isna().astype(int)
    work["Missing Types Flag"] = work["Part Types"].isna().astype(int)
    work["Invalid Types Flag"] = np.where(
        work["Part Types"].fillna("").astype(str).str.strip().ne("")
        & ~work["Part Types"].isin(sorted(ALLOWED_PART_TYPES)),
        1, 0
    )
    work["Types Error Flag"] = np.where((work["Missing Types Flag"] == 1) | (work["Invalid Types Flag"] == 1), 1, 0)

    def reasons(row):
        r = []
        if int(row.get("Missing Part Number Flag", 0)): r.append("Missing Part Number")
        if int(row.get("Missing Vendor Flag", 0)): r.append("Missing Vendor")
        if int(row.get("Missing Location Flag", 0)): r.append("Missing Location")
        if int(row.get("Missing Area Flag", 0)): r.append("Missing Area")
        if int(row.get("Missing Sub-Type Flag", 0)): r.append("Missing Sub-Type")
        if int(row.get("Missing Types Flag", 0)): r.append("Missing Types")
        elif int(row.get("Invalid Types Flag", 0)): r.append("Types Not In Approved List")
        return " | ".join(r)

    work["Error Reasons"] = work.apply(reasons, axis=1)
    flag_cols = ["Missing Part Number Flag", "Missing Vendor Flag", "Missing Location Flag", "Missing Area Flag", "Missing Sub-Type Flag", "Types Error Flag"]
    work["Error Count"] = sum(pd.to_numeric(work[c], errors="coerce").fillna(0) for c in flag_cols)
    out = work[work["Error Count"] > 0].copy()
    cols = [
        "Parent Location", "Sub-Location", "Part ID Key", "Part Name", "Part Numbers", "Vendor",
        "Area", "Part Subtype", "Part Types", "Qty In Stock", "Unit Cost Num", "Total Cost Num",
        "Missing Part Number Flag", "Missing Vendor Flag", "Missing Location Flag",
        "Missing Area Flag", "Missing Sub-Type Flag", "Missing Types Flag", "Invalid Types Flag",
        "Types Error Flag", "Error Count", "Error Reasons",
    ]
    return out[[c for c in cols if c in out.columns]].sort_values(
        ["Parent Location", "Sub-Location", "Error Count", "Part Name"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def prepare_parts(parts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p = parts.copy()
    if "ID" in p.columns:
        p["__Part ID Key"] = p["ID"].map(norm_id_value)
    else:
        p["__Part ID Key"] = ""
    meta = {
        "value_col": find_col(p, ["Total Cost", "Inventory Value", "Total Value", "Extended Cost"]),
        "qty_col": find_col(p, ["Quantity in Stock", "On Hand", "Qty On Hand", "Quantity", "Available Quantity"]),
        "unit_col": find_col(p, ["Unit Cost", "Avg Cost", "Average Cost", "Last Price", "Cost"]),
        "location_col": find_col(p, ["Location"]),
        "type_col": find_col(p, ["Types", "Part Types", "Type"]),
        "vendor_col": find_col(p, ["Vendors", "Vendor"]),
    }

    if meta["value_col"]:
        p["__Part Value"] = to_num(p[meta["value_col"]]).fillna(0.0)
    elif meta["qty_col"] and meta["unit_col"]:
        p["__Part Value"] = to_num(p[meta["qty_col"]]).fillna(0.0) * to_num(p[meta["unit_col"]]).fillna(0.0)
    else:
        p["__Part Value"] = 0.0

    p["__Part Location"] = p[meta["location_col"]].map(clean_text) if meta["location_col"] else ""
    p["__Part Type"] = p[meta["type_col"]].map(clean_text) if meta["type_col"] else ""
    p["__Part Vendor"] = p[meta["vendor_col"]].map(clean_text) if meta["vendor_col"] else ""

    if meta["qty_col"]:
        p["__Qty Stock"] = to_num(p[meta["qty_col"]]).fillna(0.0)
    else:
        p["__Qty Stock"] = 0.0

    min_col = find_col(p, ["Minimum Quantity", "Min Qty", "Minimum"])
    if min_col:
        p["__Min Qty"] = to_num(p[min_col]).fillna(0.0)
    else:
        p["__Min Qty"] = 0.0

    return p, meta


def prepare_inventory_tx(tx: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    t = tx.copy()
    meta = {
        "date_col": find_col(t, ["Tx Date", "Transaction Date", "Date"]),
        "direction_col": find_col(t, ["Direction Clean", "Direction"]),
        "cost_col": find_col(t, ["Total Cost Num", "Total Cost", "Amount", "Value"]),
        "parent_location_col": find_col(t, ["Parent Location"]),
        "reason_col": find_col(t, ["Transaction Reason Clean", "Transaction Reason", "Reason"]),
        "tx_type_col": find_col(t, ["Tx Type Clean", "Transaction Type", "Type"]),
        "type_col": find_col(t, ["Part Types", "Part Types Tx", "Part Types_y", "Types"]),
        "part_col": find_col(t, ["Part Name Clean", "Part Name"]),
        "part_id_col": find_col(t, ["Part ID Key", "Part ID"]),
        "wo_col": find_col(t, ["WO ID Key", "Work Order ID"]),
        "po_col": find_col(t, ["PO ID Key", "Purchase Order ID"]),
        "manual_col": find_col(t, ["Manual Update Flag"]),
        "restock_col": find_col(t, ["Restock Flag"]),
        "direction_bucket_col": find_col(t, ["Manual Direction Bucket"]),
    }

    t["__Tx Date"] = pd.to_datetime(t[meta["date_col"]], errors="coerce") if meta["date_col"] else pd.NaT
    t["__Direction"] = t[meta["direction_col"]].map(clean_text).str.upper() if meta["direction_col"] else ""
    t["__Parent Location"] = t[meta["parent_location_col"]].map(clean_text) if meta["parent_location_col"] else ""
    t["__Reason"] = t[meta["reason_col"]].map(clean_text).str.upper() if meta["reason_col"] else ""
    t["__Tx Type"] = t[meta["tx_type_col"]].map(clean_text).str.upper() if meta.get("tx_type_col") else ""
    t["__Part Type"] = t[meta["type_col"]].map(clean_text) if meta["type_col"] else ""
    t["__Part"] = t[meta["part_col"]].map(clean_text) if meta["part_col"] else ""

    if meta["cost_col"]:
        t["__Cost"] = to_num(t[meta["cost_col"]]).fillna(0.0)
    else:
        t["__Cost"] = 0.0

    # Normalize movement: IN positive, OUT negative.
    t["__Abs Cost"] = t["__Cost"].abs()
    t["__Signed Cost"] = np.where(
        t["__Direction"].str.startswith("OUT"),
        -t["__Abs Cost"],
        np.where(t["__Direction"].str.startswith("IN"), t["__Abs Cost"], t["__Cost"])
    )

    t["__Has WO"] = False
    if meta["wo_col"]:
        t["__Has WO"] = t[meta["wo_col"]].map(clean_text).ne("")

    t["__Has PO"] = False
    if meta["po_col"]:
        t["__Has PO"] = t[meta["po_col"]].map(clean_text).ne("")

    t["__Manual Update"] = False
    if meta["manual_col"]:
        t["__Manual Update"] = to_num(t[meta["manual_col"]]).fillna(0).astype(int).eq(1)

    t["__Restock"] = False
    if meta["restock_col"]:
        t["__Restock"] = to_num(t[meta["restock_col"]]).fillna(0).astype(int).eq(1)

    t["__Non Assigned Usage"] = t["__Parent Location"].eq("")

    return t, meta


def month_start(y: int, m: int) -> pd.Timestamp:
    return pd.Timestamp(f"{y}-{m:02d}-01")


def next_month(ts: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(ts) + pd.offsets.MonthBegin(1)).normalize()


def date_window(period_mode: str, year_pick: int, month_pick: int, custom_start, custom_end):
    if period_mode == "YTD":
        start = month_start(year_pick, 1)
        end = next_month(month_start(year_pick, month_pick))
    elif period_mode == "Monthly":
        start = month_start(year_pick, month_pick)
        end = next_month(start)
    elif period_mode == "Rolling 12":
        end = next_month(month_start(year_pick, month_pick))
        start = (end - pd.DateOffset(months=12)).normalize()
    else:
        start = pd.Timestamp(custom_start or date(year_pick, 1, 1)).normalize()
        end = pd.Timestamp(custom_end or date.today()).normalize() + pd.Timedelta(days=1)
    return start, end


def build_monthly_flow(tx_window: pd.DataFrame, current_inventory_value: float, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    if tx_window.empty:
        return pd.DataFrame(columns=[
            "Period", "Beginning Inventory Value", "In $", "Out $", "Net Change $",
            "Ending Inventory Value", "%Δ Inventory vs Prior", "Transactions In", "Transactions Out",
            "PO In $", "WO Out $", "Manual In $", "Manual Out $", "Non-Assigned Usage $"
        ])

    tx_window = tx_window.copy()
    tx_window["__Month"] = tx_window["__Tx Date"].dt.to_period("M").dt.to_timestamp()

    # Back-cast beginning value from current inventory and selected period net movement.
    net_selected = float(tx_window["__Signed Cost"].sum())
    beginning_value = float(current_inventory_value - net_selected)
    running = beginning_value

    rows = []
    cursor = pd.Timestamp(year=window_start.year, month=window_start.month, day=1)
    while cursor < window_end:
        nxt = next_month(cursor)
        label = cursor.strftime("%b %Y")
        month = tx_window[(tx_window["__Tx Date"] >= cursor) & (tx_window["__Tx Date"] < nxt)].copy()

        is_in = month["__Direction"].str.startswith("IN")
        is_out = month["__Direction"].str.startswith("OUT")

        in_cost = float(month.loc[is_in, "__Abs Cost"].sum())
        out_cost = float(month.loc[is_out, "__Abs Cost"].sum())
        net = in_cost - out_cost
        ending = running + net
        mom = ((ending - running) / running * 100.0) if running else np.nan

        po_in = float(month.loc[is_in & month["__Has PO"], "__Abs Cost"].sum())
        wo_out = float(month.loc[is_out & month["__Has WO"], "__Abs Cost"].sum())
        manual_in = float(month.loc[is_in & month["__Manual Update"], "__Abs Cost"].sum())
        manual_out = float(month.loc[is_out & month["__Manual Update"], "__Abs Cost"].sum())
        non_assigned = float(month.loc[month["__Non Assigned Usage"], "__Abs Cost"].sum())

        rows.append({
            "Period": label,
            "Beginning Inventory Value": running,
            "In $": in_cost,
            "Out $": out_cost,
            "Net Change $": net,
            "Ending Inventory Value": ending,
            "%Δ Inventory vs Prior": mom,
            "Transactions In": int(is_in.sum()),
            "Transactions Out": int(is_out.sum()),
            "PO In $": po_in,
            "WO Out $": wo_out,
            "Manual In $": manual_in,
            "Manual Out $": manual_out,
            "Non-Assigned Usage $": non_assigned,
        })
        running = ending
        cursor = nxt

    return pd.DataFrame(rows)


def build_reasons(tx_window: pd.DataFrame) -> pd.DataFrame:
    if tx_window.empty:
        return pd.DataFrame()
    r = (
        tx_window.assign(**{"Transaction Reason": tx_window["__Reason"].replace("", "(blank)")})
        .groupby("Transaction Reason", dropna=False)
        .agg(
            Count=("Transaction Reason", "size"),
            Amount=("__Signed Cost", "sum"),
            Abs_Amount=("__Abs Cost", "sum"),
        )
        .reset_index()
    )
    total_count = r["Count"].sum()
    total_abs = r["Abs_Amount"].sum()
    r["% by Count"] = np.where(total_count, r["Count"] / total_count * 100.0, 0.0)
    r["% by Amount"] = np.where(total_abs, r["Abs_Amount"] / total_abs * 100.0, 0.0)
    return r.sort_values("Count", ascending=False)


def build_pdf(title: str, summary_df: pd.DataFrame, detail_df: pd.DataFrame, filters: dict) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not installed.")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=.35*inch, leftMargin=.35*inch, topMargin=.35*inch, bottomMargin=.35*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Paragraph(f"Generated: {datetime.now():%Y-%m-%d %I:%M %p}", styles["Normal"]), Spacer(1, 8)]
    story.append(Paragraph("<br/>".join([f"<b>{k}:</b> {v}" for k, v in filters.items()]), styles["Normal"]))
    story.append(Spacer(1, 8))

    for label, df in [("KPI Summary", summary_df), ("Monthly Detail Preview", detail_df.head(80))]:
        if df.empty:
            continue
        d = df.copy().astype(str)
        tbl = Table([d.columns.tolist()] + d.values.tolist(), repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), .25, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 6),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(Paragraph(label, styles["Heading2"]))
        story.append(tbl)
        story.append(Spacer(1, 10))
    doc.build(story)
    return buffer.getvalue()


def xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    b = io.BytesIO()
    with pd.ExcelWriter(b, engine="xlsxwriter") as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=str(sheet)[:31])
    return b.getvalue()


# -----------------------------
# Page
# -----------------------------
st.title("Inventory Analysis Report")
st.caption("Uses only Parts_Master and mx_inventory_transaction_detail_current.")

with st.sidebar:
    st.header("Inventory Reporting")
    st.code(DB_PATH, language="text")
    st.code(PARTS_TABLE, language="text")
    st.code(RESTOCK_TABLE, language="text")
    st.code(INVENTORY_TX_TABLE, language="text")

parts_raw, restock_raw, tx_raw, locations_df, source_info = load_inputs()
parts, parts_meta = prepare_parts(parts_raw)
parts = apply_parts_parent_location(parts, locations_df)
parts = apply_restock_cost_fallback(parts, restock_raw)
tx, tx_meta = prepare_inventory_tx(tx_raw)

if parts.empty:
    st.error("No Parts_Master data loaded.")
    st.stop()
if tx.empty:
    st.error("No mx_inventory_transaction_detail_current data loaded.")
    st.stop()

# Visible source check
with st.expander("Source / Column Check", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parts Rows", f"{len(parts):,}")
    c2.metric("Inventory Tx Rows", f"{len(tx):,}")
    c3.metric("Tx Parent Location Filled", f"{int(tx['__Parent Location'].ne('').sum()):,}")
    c4.metric("Tx Parent Location Blank", f"{int(tx['__Parent Location'].eq('').sum()):,}")
    st.caption(f"Parts parent-location mapped rows: {int(parts['__Parent Location'].ne('').sum()):,} / {len(parts):,}")
    st.caption(f"Parts source: {source_info['Parts Source']} | ReStock source: {source_info.get('ReStock Source', 'Not loaded')} | Inventory transaction source: {source_info['Inventory Tx Source']}")
    st.caption(
        f"Tx date: {tx_meta.get('date_col') or 'not found'} | "
        f"Tx cost: {tx_meta.get('cost_col') or 'not found'} | "
        f"Parent Location: {tx_meta.get('parent_location_col') or 'not found'} | "
        f"Direction: {tx_meta.get('direction_col') or 'not found'}"
    )

if not tx_meta.get("parent_location_col"):
    st.error("The table mx_inventory_transaction_detail_current must contain Parent Location.")
    st.stop()
if not tx_meta.get("date_col"):
    st.error("The table mx_inventory_transaction_detail_current must contain Tx Date or Transaction Date.")
    st.stop()
if not tx_meta.get("cost_col"):
    st.error("The table mx_inventory_transaction_detail_current must contain Total Cost Num or Total Cost.")
    st.stop()

location_options = location_options_from_locations_master(locations_df, tx)
valid_years = sorted(tx["__Tx Date"].dropna().dt.year.astype(int).unique().tolist()) if tx["__Tx Date"].notna().any() else [date.today().year]
latest_dt = tx["__Tx Date"].dropna().max() if tx["__Tx Date"].notna().any() else pd.Timestamp.today()

f1, f2, f3, f4 = st.columns([1.6, 1.25, 1, 1])
with f1:
    selected_location = st.selectbox("Location", location_options, index=0)
with f2:
    type_options = sorted({x for x in tx["__Part Type"].dropna().astype(str).tolist() if x})
    selected_types = st.multiselect("Type", type_options)
with f3:
    period_mode = st.radio("Date Range", ["YTD", "Monthly", "Rolling 12", "Custom"], horizontal=False)
with f4:
    year_pick = st.selectbox("Year", valid_years[::-1], index=0)
    month_pick = st.selectbox("Month", list(range(1, 13)), index=max(min(int(latest_dt.month) - 1, 11), 0))

custom_start = custom_end = None
if period_mode == "Custom":
    d1, d2 = st.columns(2)
    with d1:
        custom_start = st.date_input("Start", value=date(int(year_pick), 1, 1))
    with d2:
        custom_end = st.date_input("End", value=latest_dt.date())

window_start, window_end = date_window(period_mode, int(year_pick), int(month_pick), custom_start, custom_end)

# Apply filters
parts_view = parts.copy()
tx_filtered_all = tx.copy()

if selected_location != ALL:
    tx_filtered_all = tx_filtered_all[tx_filtered_all["__Parent Location"].eq(selected_location)].copy()
    if "__Parent Location" in parts_view.columns:
        parts_view = parts_view[parts_view["__Parent Location"].eq(selected_location)].copy()

if selected_types:
    tx_filtered_all = tx_filtered_all[tx_filtered_all["__Part Type"].isin(selected_types)].copy()
    parts_view = parts_view[parts_view["__Part Type"].isin(selected_types)].copy()

tx_window = tx_filtered_all[
    tx_filtered_all["__Tx Date"].notna() &
    (tx_filtered_all["__Tx Date"] >= window_start) &
    (tx_filtered_all["__Tx Date"] < window_end)
].copy()

with st.expander("Filtered Data Check", expanded=False):
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("Filtered Parts", f"{len(parts_view):,}")
    fc2.metric("Filtered Tx Rows", f"{len(tx_filtered_all):,}")
    fc3.metric("Window Tx Rows", f"{len(tx_window):,}")
    fc4.metric("Selected Location", selected_location)
    st.caption(f"Filtered Parts Current Inventory Value: {money(float(parts_view['__Part Value'].sum()) if not parts_view.empty else 0.0)}")

# Non-assigned usage = OUT transactions not associated to a work order.
# Keep this independent of selected location, but respect Type and date filters.
non_assigned_source = tx.copy()
if selected_types:
    non_assigned_source = non_assigned_source[non_assigned_source["__Part Type"].isin(selected_types)].copy()

non_assigned_window = non_assigned_source[
    non_assigned_source["__Tx Date"].notna() &
    (non_assigned_source["__Tx Date"] >= window_start) &
    (non_assigned_source["__Tx Date"] < window_end) &
    non_assigned_source["__Direction"].str.startswith("OUT") &
    (~non_assigned_source["__Has WO"])
].copy()

# Current inventory value from Parts_Master snapshot
current_inventory_value = float(parts_view["__Part Value"].sum())
active_parts = len(parts_view)

monthly = build_monthly_flow(tx_window, current_inventory_value, window_start, window_end)
reasons = build_reasons(tx_window)

is_in = tx_window["__Direction"].str.startswith("IN") if not tx_window.empty else pd.Series(dtype=bool)
is_out = tx_window["__Direction"].str.startswith("OUT") if not tx_window.empty else pd.Series(dtype=bool)

tx_in_count = int(is_in.sum()) if len(tx_window) else 0
tx_out_count = int(is_out.sum()) if len(tx_window) else 0
tx_in_cost = float(tx_window.loc[is_in, "__Abs Cost"].sum()) if len(tx_window) else 0.0
tx_out_cost = float(tx_window.loc[is_out, "__Abs Cost"].sum()) if len(tx_window) else 0.0

net_change = float(tx_window["__Signed Cost"].sum()) if not tx_window.empty else 0.0
beginning_inventory_value = current_inventory_value - net_change
inventory_growth = ((current_inventory_value - beginning_inventory_value) / beginning_inventory_value * 100.0) if beginning_inventory_value else np.nan
avg_mom = float(monthly["%Δ Inventory vs Prior"].mean()) if not monthly.empty else np.nan

reopen_mask = (
    tx_window["__Reason"].str.contains("REOPEN", na=False) |
    tx_window["__Tx Type"].str.contains("REOPEN", na=False)
) if not tx_window.empty else pd.Series(dtype=bool)
reopen_count = int(reopen_mask.sum()) if len(tx_window) else 0
reopen_cost = float(tx_window.loc[reopen_mask, "__Abs Cost"].sum()) if len(tx_window) else 0.0

non_assigned_count = int(len(non_assigned_window))
non_assigned_cost = float(non_assigned_window["__Abs Cost"].sum()) if not non_assigned_window.empty else 0.0

# Created inventory = IN transactions where the transaction type or reason is CREATED.
created_inventory_mask = (
    tx_window["__Direction"].str.startswith("IN") &
    (tx_window["__Tx Type"].str.contains("CREATED", na=False) | tx_window["__Reason"].str.contains("CREATED", na=False))
) if not tx_window.empty else pd.Series(dtype=bool)
created_inventory_count = int(created_inventory_mask.sum()) if len(tx_window) else 0
created_inventory_cost = float(tx_window.loc[created_inventory_mask, "__Abs Cost"].sum()) if len(tx_window) else 0.0

kpis = pd.DataFrame([
    {"KPI": "Inventory Growth %", "Value": inventory_growth, "Definition": KPI_TARGETS["Inventory Growth %"]},
    {"KPI": "Avg MoM Change %", "Value": avg_mom, "Definition": KPI_TARGETS["Avg MoM Change %"]},
])

kpi_tab, analysis_tab, tx_tab, audit_tab, detail_tab = st.tabs([
    "KPI Overview",
    "Inventory Analysis",
    "Transaction Review",
    "Inventory Audit",
    "Inventory Detail",
])

with kpi_tab:
    st.subheader("Inventory KPI Overview")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Current Inventory Value", money(current_inventory_value))
    a2.metric("Active Parts", f"{active_parts:,}")
    a3.metric("Transactions", f"{len(tx_window):,}")
    a4.metric("Inventory Growth", pct(inventory_growth))
    a5.metric("Avg MoM", pct(avg_mom))

    st.markdown("### Transaction Flow")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Transactions In", f"{tx_in_count:,}")
    b2.metric("Transactions In Cost", money(tx_in_cost))
    b3.metric("Transactions Out", f"{tx_out_count:,}")
    b4.metric("Transactions Out Cost", money(tx_out_cost))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Reopened PO/WO Returns", f"{reopen_count:,}")
    c2.metric("Reopened PO/WO Return Cost", money(reopen_cost))
    c3.metric("Non-Assigned Usage Rows", f"{non_assigned_count:,}")
    c4.metric("Non-Assigned Usage Cost", money(non_assigned_cost))
    c5.metric("Created Inventory Rows", f"{created_inventory_count:,}")
    c6.metric("Created Inventory Cost", money(created_inventory_cost))

    st.caption(f"Window: {window_start:%Y-%m-%d} through {(window_end - pd.Timedelta(days=1)):%Y-%m-%d}")
    st.dataframe(kpis.assign(Value=kpis["Value"].map(pct)), use_container_width=True, hide_index=True)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download KPI CSV",
            data=kpis.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"inventory_kpi_summary_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        if REPORTLAB_AVAILABLE:
            pdf = build_pdf(
                "Inventory KPI Summary",
                kpis.assign(Value=kpis["Value"].map(pct)),
                monthly,
                {
                    "Location": selected_location,
                    "Types": ", ".join(selected_types) if selected_types else "All",
                    "Date Range": f"{window_start:%Y-%m-%d} to {(window_end - pd.Timedelta(days=1)):%Y-%m-%d}",
                    "Source": f"{PARTS_TABLE} + {INVENTORY_TX_TABLE}",
                }
            )
            st.download_button("Download KPI PDF", data=pdf, file_name=f"inventory_kpi_summary_{datetime.now():%Y%m%d_%H%M}.pdf", mime="application/pdf", use_container_width=True)

with analysis_tab:
    st.subheader("Monthly Inventory Analysis")
    st.caption(f"Filtered by Location: {selected_location}; Type: {', '.join(selected_types) if selected_types else 'All'}; Window Tx Rows: {len(tx_window):,}")
    if monthly.empty:
        st.info("No monthly transaction rows match the selected filters.")
    else:
        st.line_chart(monthly.set_index("Period")[["Ending Inventory Value"]], use_container_width=True)
        st.bar_chart(monthly.set_index("Period")[["In $", "Out $", "Net Change $"]], use_container_width=True)

        display = monthly.copy()
        for c in ["Beginning Inventory Value", "In $", "Out $", "Net Change $", "Ending Inventory Value", "PO In $", "WO Out $", "Manual In $", "Manual Out $", "Non-Assigned Usage $"]:
            if c in display.columns:
                display[c] = display[c].map(money)
        if "%Δ Inventory vs Prior" in display.columns:
            display["%Δ Inventory vs Prior"] = display["%Δ Inventory vs Prior"].map(pct)
        st.dataframe(display, use_container_width=True, hide_index=True)

with tx_tab:
    st.subheader("Transaction Review")
    st.markdown("### Location Assignment Summary")
    loc_summary = (
        tx_window.assign(Location_Source=np.where(tx_window["__Parent Location"].eq(""), "Non-Assigned Part Usage", "Parent Location"))
        .groupby("Location_Source", dropna=False)
        .agg(Rows=("Location_Source", "size"), Cost=("__Abs Cost", "sum"))
        .reset_index()
        if not tx_window.empty else pd.DataFrame(columns=["Location_Source", "Rows", "Cost"])
    )
    if not non_assigned_window.empty and selected_location != ALL:
        extra = pd.DataFrame([{
            "Location_Source": "Non-Assigned Part Usage",
            "Rows": non_assigned_count,
            "Cost": non_assigned_cost,
        }])
        loc_summary = pd.concat([loc_summary, extra], ignore_index=True)
    loc_display = loc_summary.copy()
    if not loc_display.empty:
        loc_display["Cost"] = loc_display["Cost"].map(money)
    st.dataframe(loc_display, use_container_width=True, hide_index=True)

    st.markdown("### Transaction Reasons")
    if reasons.empty:
        st.info("No transaction reason data for current filters.")
    else:
        r = reasons.copy()
        r["Amount"] = r["Amount"].map(money)
        r["Abs_Amount"] = r["Abs_Amount"].map(money)
        r["% by Count"] = r["% by Count"].map(pct)
        r["% by Amount"] = r["% by Amount"].map(pct)
        st.dataframe(r, use_container_width=True, hide_index=True)

    created_inventory_window = tx_window.loc[created_inventory_mask].copy() if len(tx_window) else pd.DataFrame()
    if not created_inventory_window.empty:
        st.markdown("### Created Parts Inventory")
        created_cols = [c for c in [
            "Transaction ID", "Tx Date", "Transaction Date", "Direction", "Direction Clean",
            "Part ID", "Part Name", "Parent Location", "Part Location", "Total Cost Num", "Total Cost",
            "Transaction Type", "Tx Type Clean", "Transaction Reason", "Transaction Reason Clean",
            "Work Order ID", "Purchase Order ID", "Transaction Initiator"
        ] if c in created_inventory_window.columns]
        st.dataframe(created_inventory_window[created_cols] if created_cols else created_inventory_window, use_container_width=True, hide_index=True)

    if not non_assigned_window.empty:
        st.markdown("### Non-Assigned Part Usage")
        cols = [c for c in [
            "Transaction ID", "Tx Date", "Transaction Date", "Direction", "Direction Clean",
            "Part ID", "Part Name", "Part Location", "Parent Location", "Total Cost Num", "Total Cost",
            "Transaction Type", "Tx Type Clean", "Transaction Reason", "Transaction Reason Clean",
            "Work Order ID", "Purchase Order ID", "Transaction Initiator"
        ] if c in non_assigned_window.columns]
        st.dataframe(non_assigned_window[cols] if cols else non_assigned_window, use_container_width=True, hide_index=True)

with audit_tab:
    st.subheader("Inventory Audit - Inventory Error Review")
    error_view = build_inventory_error_audit_report(parts_raw, locations_df)
    if selected_location != ALL and "Parent Location" in error_view.columns:
        error_view = error_view[error_view["Parent Location"].eq(selected_location)].copy()
    if selected_types and "Part Types" in error_view.columns:
        error_view = error_view[error_view["Part Types"].isin(selected_types)].copy()

    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Inventory Error Parts", f"{len(error_view):,}")
    e2.metric("Missing Part #", f"{int(pd.to_numeric(error_view.get('Missing Part Number Flag', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()):,}")
    e3.metric("Missing Vendor", f"{int(pd.to_numeric(error_view.get('Missing Vendor Flag', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()):,}")
    e4.metric("Missing Location", f"{int(pd.to_numeric(error_view.get('Missing Location Flag', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()):,}")
    e5.metric("Types Errors", f"{int(pd.to_numeric(error_view.get('Types Error Flag', pd.Series(dtype=float)), errors='coerce').fillna(0).sum()):,}")

    if error_view.empty:
        st.info("No inventory error rows match the current filters.")
    else:
        reason_summary = (
            error_view.assign(Error_Reason_Split=error_view["Error Reasons"].fillna("").astype(str).str.split(" | "))
            .explode("Error_Reason_Split")
        )
        reason_summary = reason_summary[reason_summary["Error_Reason_Split"].fillna("").astype(str).str.strip().ne("")]
        if not reason_summary.empty:
            rs = reason_summary.groupby("Error_Reason_Split", as_index=False).agg(Parts=("Part ID Key", "count"), Total_Cost=("Total Cost Num", "sum"))
            rs = rs.rename(columns={"Error_Reason_Split": "Error Reason"}).sort_values("Parts", ascending=False)
            rs_display = rs.copy()
            rs_display["Total_Cost"] = rs_display["Total_Cost"].map(money)
            st.markdown("### Error Reason Summary")
            st.dataframe(rs_display, use_container_width=True, hide_index=True)

        st.markdown("### Inventory Error Detail")
        st.dataframe(error_view, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Inventory Error Detail CSV",
            data=error_view.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"inventory_error_detail_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            use_container_width=True,
        )


with detail_tab:
    st.subheader("Inventory Detail")
    d1, d2 = st.tabs(["Parts_Master", "mx_inventory_transaction_detail_current"])
    with d1:
        cols = [c for c in ["ID", "Name", "Location", "__Parent Location", "Types", "Quantity in Stock", "Minimum Quantity", "Unit Cost", "Total Cost", "Vendors"] if c in parts_view.columns]
        st.dataframe(parts_view[cols] if cols else parts_view, use_container_width=True, hide_index=True)
    with d2:
        default_cols = [c for c in [
            "Transaction ID", "Tx Date", "Transaction Date", "Direction Clean", "Direction",
            "Part ID", "Part Name", "Parent Location", "Part Location", "Total Cost Num", "Total Cost",
            "Tx Type Clean", "Transaction Type", "Transaction Reason Clean", "Transaction Reason", "WO ID Key", "Work Order ID", "PO ID Key", "Purchase Order ID"
        ] if c in tx_window.columns]
        st.dataframe(tx_window[default_cols] if default_cols else tx_window, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Inventory Analysis XLSX",
        data=xlsx_bytes({
            "KPI Summary": kpis,
            "Monthly Rollup": monthly,
            "Transaction Reasons": reasons,
            "Parts Detail": parts_view.head(100000),
            "Inventory Tx Detail": tx_window.head(100000),
            "Created Inventory": tx_window.loc[created_inventory_mask].head(100000) if len(tx_window) else pd.DataFrame(),
            "Non-Assigned Usage": non_assigned_window.head(100000),
            "Inventory Error Detail": build_inventory_error_audit_report(parts_raw, locations_df).head(100000),
        }),
        file_name=f"inventory_analysis_report_{datetime.now():%Y%m%d_%H%M}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
