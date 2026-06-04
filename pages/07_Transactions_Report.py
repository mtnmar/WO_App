# 07_Straight_Transactions_Report.py
# Reporting app page: straight inventory transaction report for Finance.
#
# Place this file in:
#   C:\Users\Brad\Desktop\Reporting\pages\07_Straight_Transactions_Report.py
#
# Reads the same DB path used by the reporting app when reporting_shared.DB_PATH exists,
# otherwise falls back to:
#   C:\Users\Brad\Desktop\Maintenance Pipeline\maintenance_master.db
#
# Required table:
#   Transactions
#
# Optional enrichment tables:
#   Workorders
#   Purchase_Orders
#   Locations_Master
#   Assets_Master

from __future__ import annotations

import sqlite3
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from auth_helper import require_login
except Exception:
    def require_login():
        return None

try:
    from reporting_shared import DB_PATH
except Exception:
    DB_PATH = r"C:\Users\Brad\Desktop\Maintenance Pipeline\maintenance_master.db"

TRANSACTIONS_TABLE = "Transactions"
WORKORDERS_TABLE = "Workorders"
PURCHASE_ORDERS_TABLE = "Purchase_Orders"
LOCATIONS_TABLE = "Locations_Master"
ASSETS_TABLE = "Assets_Master"
PARTS_TABLE = "Parts_Master"


BAG_USAGE_TARGETS = [
    {"Name": "ROCK DUST STRETCH WRAP", "Type": "53669 - SUPERSACK WRAP", "Department": "726 - Fine Grinding - Bagging", "Location": "335 - Fine Grinding"},
    {"Name": "HYDRATE SHRINK WRAP", "Type": "53665 - SACKING SUPPLIES", "Department": "728 - General Plant", "Location": "320 - Hydrate"},
    {"Name": "HYDRATE STRETCH WRAP", "Type": "53665 - SACKING SUPPLIES", "Department": "728 - General Plant", "Location": "320 - Hydrate"},
    {"Name": "SUPER SACK SHRINK WRAP", "Type": "53668 - RD/AG SHRINK WRAP", "Department": "726 - Fine Grinding - Bagging", "Location": "335 - Fine Grinding"},
    {"Name": "50LB ROCK DUST BAGS", "Type": "53666 - ROCK DUST / AG BAG", "Department": "726 - Fine Grinding - Bagging", "Location": "335 - Fine Grinding"},
    {"Name": "AG-LIME BAGS", "Type": "53666 - ROCK DUST / AG BAG", "Department": "726 - Fine Grinding - Bagging", "Location": "335 - Fine Grinding"},
    {"Name": '55" SUPER SACK BAGS', "Type": "53667 - SUPER SACKS", "Department": "726 - Fine Grinding - Bagging", "Location": "335 - Fine Grinding"},
    {"Name": '45" SUPER SACK BAGS', "Type": "53667 - SUPER SACKS", "Department": "726 - Fine Grinding - Bagging", "Location": "335 - Fine Grinding"},
    {"Name": "HYDRATE BAGS", "Type": "51080 - HYDRATED LIME BAGS", "Department": "728 - General Plant", "Location": "320 - Hydrate"},
    {"Name": "HYDRATE PALLET LINERS", "Type": "53665 - SACKING SUPPLIES", "Department": "728 - General Plant", "Location": "320 - Hydrate"},
]


BAG_USAGE_EXPORT_COLUMNS = ['Tx Date', 'Name', 'Qty Change', 'Unit Cost Num', 'Total Cost Num', 'Billing Type', 'Billing Department', 'Billing Location', 'Work Order ID', 'Reason', 'Part ID Key', 'PART TYPE', 'Transaction Type', 'Purchase Order ID', 'PO Number', 'WO Asset', 'Asset Type', 'Transaction ID', 'Initiator', 'DIRECTION']


# ============================================================
# BASIC HELPERS
# ============================================================
def connect_db(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path), check_same_thread=False)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


@st.cache_data(show_spinner=False)
def read_table(db_path: str, table_name: str) -> pd.DataFrame:
    with connect_db(db_path) as conn:
        if not table_exists(conn, table_name):
            return pd.DataFrame()
        return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)


def text_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series("", index=df.index if isinstance(df, pd.DataFrame) else None, dtype="object")
    return df[col].fillna("").astype(str)


def clean_text(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    return s if s else None


def norm_id_series(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return s


def parse_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string").str.strip()
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("$", "", regex=False)
    s = s.str.replace("(", "-", regex=False)
    s = s.str.replace(")", "", regex=False)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def first_present_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        key = c.strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def first_nonblank(series: pd.Series):
    for value in series.astype("string").fillna("").str.strip().tolist():
        if value:
            return value
    return pd.NA


def join_unique(series: pd.Series, max_len: int = 500):
    vals = sorted(set(v for v in series.astype("string").fillna("").str.strip().tolist() if v))
    out = ", ".join(vals)
    if len(out) > max_len:
        out = out[:max_len] + "..."
    return out if out else pd.NA


def xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Financial Export") -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    out.seek(0)
    return out.getvalue()


# ============================================================
# LOOKUPS / DATA PREP
# ============================================================
def build_location_map(locations: pd.DataFrame) -> pd.DataFrame:
    if locations is None or locations.empty:
        return pd.DataFrame(columns=["Part Location Clean", "Parent Location"])

    loc_name_col = first_present_col(locations, ["Name", "Location"])
    parent_col = first_present_col(locations, ["All Parents", "All Parent Locations", "Parent Location"])
    if not loc_name_col:
        return pd.DataFrame(columns=["Part Location Clean", "Parent Location"])

    out = pd.DataFrame()
    out["Part Location Clean"] = text_series(locations, loc_name_col).map(clean_text)
    out["Parent Location"] = text_series(locations, parent_col).map(clean_text) if parent_col else out["Part Location Clean"]
    out["Parent Location"] = out["Parent Location"].fillna(out["Part Location Clean"])
    return out.dropna(subset=["Part Location Clean"]).drop_duplicates("Part Location Clean", keep="first")


def build_workorder_lookup(workorders: pd.DataFrame) -> pd.DataFrame:
    """Return WO accounting/location fields only.

    Asset type is intentionally NOT pulled from the work order. It is added later
    from Assets_Master using the work order asset ID/name.
    """
    cols = [
        "WO ID Key", "WO Asset", "WO Asset ID", "WO Sublocation / Department",
        "WO Parent Location / Location", "WO Title",
    ]
    if workorders is None or workorders.empty:
        return pd.DataFrame(columns=cols)

    wo = workorders.copy()
    wo.columns = [str(c).strip() for c in wo.columns]

    id_col = first_present_col(wo, ["ID", "Work Order ID", "WO ID"])
    if not id_col:
        return pd.DataFrame(columns=cols)

    asset_col = first_present_col(wo, ["Asset", "Work Order Asset"])
    asset_id_col = first_present_col(wo, ["Asset ID", "Work Order Asset ID"])
    dept_col = first_present_col(wo, ["NS Department", "Department", "Location", "Location2"])
    loc_col = first_present_col(wo, ["NS Location", "All Parent Locations", "Parent Location", "Location2", "Location"])
    title_col = first_present_col(wo, ["Title", "Description"])

    out = pd.DataFrame()
    out["WO ID Key"] = norm_id_series(wo[id_col])
    out["WO Asset"] = text_series(wo, asset_col).map(clean_text) if asset_col else None
    out["WO Asset ID"] = norm_id_series(wo[asset_id_col]) if asset_id_col else pd.Series(pd.NA, index=wo.index, dtype="string")
    out["WO Sublocation / Department"] = text_series(wo, dept_col).map(clean_text) if dept_col else None
    out["WO Parent Location / Location"] = text_series(wo, loc_col).map(clean_text) if loc_col else None
    out["WO Title"] = text_series(wo, title_col).map(clean_text) if title_col else None
    out = out.dropna(subset=["WO ID Key"]).drop_duplicates("WO ID Key", keep="first")
    return out


def build_asset_lookup(assets: pd.DataFrame) -> pd.DataFrame:
    """Lookup asset type/location from Assets_Master.

    Required user correction: use asset type from Assets_Master[Types], not work order type.
    """
    cols = [
        "Asset ID Key", "Asset Name Key", "Asset Type",
        "Asset Sublocation / Department", "Asset Parent Location / Location",
    ]
    if assets is None or assets.empty:
        return pd.DataFrame(columns=cols)

    a = assets.copy()
    a.columns = [str(c).strip() for c in a.columns]

    id_col = first_present_col(a, ["ID", "Asset ID"])
    name_col = first_present_col(a, ["Name", "Asset", "Asset Name"])
    type_col = first_present_col(a, ["Types", "Asset Type", "Type"])
    dept_col = first_present_col(a, ["Location", "Department", "NS Department"])
    parent_col = first_present_col(a, ["All Parent Locations", "All Parents", "Parent Location", "NS Location"])

    out = pd.DataFrame(index=a.index)
    out["Asset ID Key"] = norm_id_series(a[id_col]) if id_col else pd.Series(pd.NA, index=a.index, dtype="string")
    asset_name = text_series(a, name_col).map(clean_text) if name_col else pd.Series([None] * len(a), index=a.index)
    out["Asset Name Key"] = asset_name.fillna("").astype(str).str.strip().str.upper()
    out["Asset Type"] = text_series(a, type_col).map(clean_text) if type_col else None
    out["Asset Sublocation / Department"] = text_series(a, dept_col).map(clean_text) if dept_col else None
    out["Asset Parent Location / Location"] = text_series(a, parent_col).map(clean_text) if parent_col else None
    out["Asset Parent Location / Location"] = out["Asset Parent Location / Location"].fillna(out["Asset Sublocation / Department"])

    out = out[(out["Asset ID Key"].notna()) | (out["Asset Name Key"].astype(str).str.strip().ne(""))].copy()
    # Prefer ID matches, but keep a name key too for fallback.
    return out.drop_duplicates(["Asset ID Key", "Asset Name Key"], keep="first")

def build_po_lookup(purchase_orders: pd.DataFrame) -> pd.DataFrame:
    if purchase_orders is None or purchase_orders.empty:
        return pd.DataFrame(columns=["PO ID Key", "PO Number", "PO NS Item", "PO NS Department", "PO NS Location", "PO Vendor"])

    po = purchase_orders.copy()
    po.columns = [str(c).strip() for c in po.columns]

    id_col = first_present_col(po, ["Purchase Order ID", "ID", "PO ID"])
    if not id_col:
        return pd.DataFrame(columns=["PO ID Key", "PO Number", "PO NS Item", "PO NS Department", "PO NS Location", "PO Vendor"])

    po_num_col = first_present_col(po, ["Purchase Order #", "PO #", "Document Number"])
    ns_item_col = first_present_col(po, ["NS Item", "Item"])
    dept_col = first_present_col(po, ["Department", "NS Department"])
    loc_col = first_present_col(po, ["NS Segmentation Location", "NS Item Location", "NS Location", "Location"])
    vendor_col = first_present_col(po, ["Vendor", "Name"])

    work = pd.DataFrame()
    work["PO ID Key"] = norm_id_series(po[id_col])
    work["PO Number"] = text_series(po, po_num_col).map(clean_text) if po_num_col else None
    work["PO NS Item"] = text_series(po, ns_item_col).map(clean_text) if ns_item_col else None
    work["PO NS Department"] = text_series(po, dept_col).map(clean_text) if dept_col else None
    work["PO NS Location"] = text_series(po, loc_col).map(clean_text) if loc_col else None
    work["PO Vendor"] = text_series(po, vendor_col).map(clean_text) if vendor_col else None
    work = work.dropna(subset=["PO ID Key"])

    if work.empty:
        return pd.DataFrame(columns=["PO ID Key", "PO Number", "PO NS Item", "PO NS Department", "PO NS Location", "PO Vendor"])

    out = work.groupby("PO ID Key", dropna=False).agg(
        **{
            "PO Number": ("PO Number", first_nonblank),
            "PO NS Item": ("PO NS Item", join_unique),
            "PO NS Department": ("PO NS Department", join_unique),
            "PO NS Location": ("PO NS Location", join_unique),
            "PO Vendor": ("PO Vendor", first_nonblank),
        }
    ).reset_index()
    return out


def prep_transactions(raw: pd.DataFrame, locations: pd.DataFrame, workorders: pd.DataFrame, purchase_orders: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    tx = raw.copy()
    tx.columns = [str(c).strip() for c in tx.columns]

    date_col = first_present_col(tx, ["Transaction Date", "Date", "Created On", "Created"])
    direction_col = first_present_col(tx, ["Direction"])
    type_col = first_present_col(tx, ["Transaction Type", "Type"])
    part_id_col = first_present_col(tx, ["Part ID", "Part Id", "Item ID"])
    part_name_col = first_present_col(tx, ["Part Name", "Name", "Item"])
    part_type_col = first_present_col(tx, ["Part Types", "Part Type", "Type", "SUB-TYPE"])
    part_area_col = first_present_col(tx, ["Part Area", "Area"])
    part_location_col = first_present_col(tx, ["Part Location", "Location"])
    po_col = first_present_col(tx, ["Purchase Order ID", "PO ID", "Purchase Order", "Purchase Order #"])
    wo_col = first_present_col(tx, ["Work Order ID", "WO ID", "Work Order", "Work Order #"])
    qty_before_col = first_present_col(tx, ["Quantity Before"])
    qty_col = first_present_col(tx, ["Quantity Added to Inventory", "Qty Change", "Quantity", "Qty"])
    qty_after_col = first_present_col(tx, ["Quantity After"])
    unit_cost_col = first_present_col(tx, ["Unit Cost"])
    total_cost_col = first_present_col(tx, ["Total Cost", "Amount"])
    initiator_col = first_present_col(tx, ["Transaction Initiator", "Created By", "User"])
    reason_col = first_present_col(tx, ["Transaction Reason", "Reason"])
    txid_col = first_present_col(tx, ["Transaction ID", "ID"])

    out = pd.DataFrame(index=tx.index)
    out["Transaction ID"] = norm_id_series(tx[txid_col]) if txid_col else pd.Series(pd.NA, index=tx.index, dtype="string")
    out["DATE"] = pd.to_datetime(tx[date_col], errors="coerce") if date_col else pd.NaT
    out["DIRECTION"] = text_series(tx, direction_col).str.upper().str.strip() if direction_col else ""
    out["Transaction Type"] = text_series(tx, type_col).str.upper().str.strip() if type_col else ""
    out["Part ID"] = norm_id_series(tx[part_id_col]) if part_id_col else pd.Series(pd.NA, index=tx.index, dtype="string")
    out["PART NAME"] = text_series(tx, part_name_col).map(clean_text) if part_name_col else None
    out["PART TYPE"] = text_series(tx, part_type_col).map(clean_text) if part_type_col else None
    out["Part Area"] = text_series(tx, part_area_col).map(clean_text) if part_area_col else None
    out["Part Location"] = text_series(tx, part_location_col).map(clean_text) if part_location_col else None
    out["Purchase Order ID"] = norm_id_series(tx[po_col]) if po_col else pd.Series(pd.NA, index=tx.index, dtype="string")
    out["Work Order ID"] = norm_id_series(tx[wo_col]) if wo_col else pd.Series(pd.NA, index=tx.index, dtype="string")
    tx_wo_asset_col = first_present_col(tx, ["Work Order Asset", "WO Asset"])
    tx_wo_asset_id_col = first_present_col(tx, ["Work Order Asset ID", "Asset ID", "WO Asset ID"])
    out["Tx WO Asset"] = text_series(tx, tx_wo_asset_col).map(clean_text) if tx_wo_asset_col else None
    out["Tx WO Asset ID"] = norm_id_series(tx[tx_wo_asset_id_col]) if tx_wo_asset_id_col else pd.Series(pd.NA, index=tx.index, dtype="string")
    out["QUANTITY BEFORE"] = parse_numeric_series(tx[qty_before_col]) if qty_before_col else pd.NA
    out["QUANTITY ADDED"] = parse_numeric_series(tx[qty_col]).fillna(0) if qty_col else 0.0
    out["QUANTITY AFTER"] = parse_numeric_series(tx[qty_after_col]) if qty_after_col else pd.NA
    out["UNIT COST"] = parse_numeric_series(tx[unit_cost_col]).fillna(0) if unit_cost_col else 0.0
    if total_cost_col:
        out["TOTAL COST"] = parse_numeric_series(tx[total_cost_col]).fillna(0)
    else:
        out["TOTAL COST"] = out["QUANTITY ADDED"].abs() * out["UNIT COST"].fillna(0)
    out["Initiator"] = text_series(tx, initiator_col).map(clean_text) if initiator_col else None
    out["Reason"] = text_series(tx, reason_col).map(clean_text) if reason_col else None

    out["Has PO"] = out["Purchase Order ID"].astype("string").fillna("").str.strip().ne("")
    out["Has WO"] = out["Work Order ID"].astype("string").fillna("").str.strip().ne("")

    loc_map = build_location_map(locations)
    if not loc_map.empty:
        out = out.merge(loc_map, left_on="Part Location", right_on="Part Location Clean", how="left")
        out = out.drop(columns=["Part Location Clean"], errors="ignore")
    else:
        out["Parent Location"] = out["Part Location"]
    out["Parent Location"] = out["Parent Location"].fillna(out["Part Location"])

    wo_lookup = build_workorder_lookup(workorders)
    if not wo_lookup.empty:
        out = out.merge(wo_lookup, left_on="Work Order ID", right_on="WO ID Key", how="left").drop(columns=["WO ID Key"], errors="ignore")
    else:
        for c in ["WO Asset", "WO Asset ID", "WO Sublocation / Department", "WO Parent Location / Location", "WO Title"]:
            out[c] = pd.NA

    # Prefer the work order table, then fallback to the transaction's own work-order asset fields.
    out["WO Asset"] = out["WO Asset"].fillna(out["Tx WO Asset"])
    out["WO Asset ID"] = norm_id_series(out["WO Asset ID"]).fillna(norm_id_series(out["Tx WO Asset ID"]))

    asset_lookup = build_asset_lookup(assets)
    if not asset_lookup.empty:
        # Primary join by asset ID. Do not let blank asset IDs match each other.
        id_lookup = asset_lookup[
            asset_lookup["Asset ID Key"].astype("string").fillna("").str.strip().ne("")
        ][["Asset ID Key", "Asset Type", "Asset Sublocation / Department", "Asset Parent Location / Location"]].drop_duplicates("Asset ID Key")
        out = out.merge(
            id_lookup,
            left_on="WO Asset ID",
            right_on="Asset ID Key",
            how="left",
        ).drop(columns=["Asset ID Key"], errors="ignore")

        # Fallback join by asset name when an asset ID was not available.
        out["_Asset Name Key"] = out["WO Asset"].fillna("").astype(str).str.strip().str.upper()
        name_lookup = asset_lookup[
            asset_lookup["Asset Name Key"].astype(str).str.strip().ne("")
        ][["Asset Name Key", "Asset Type", "Asset Sublocation / Department", "Asset Parent Location / Location"]].drop_duplicates("Asset Name Key")
        out = out.merge(name_lookup, left_on="_Asset Name Key", right_on="Asset Name Key", how="left", suffixes=("", "_by_name"))
        for c in ["Asset Type", "Asset Sublocation / Department", "Asset Parent Location / Location"]:
            out[c] = out[c].fillna(out[f"{c}_by_name"])
        out = out.drop(columns=["_Asset Name Key", "Asset Name Key", "Asset Type_by_name", "Asset Sublocation / Department_by_name", "Asset Parent Location / Location_by_name"], errors="ignore")
    else:
        for c in ["Asset Type", "Asset Sublocation / Department", "Asset Parent Location / Location"]:
            out[c] = pd.NA

    po_lookup = build_po_lookup(purchase_orders)
    if not po_lookup.empty:
        out = out.merge(po_lookup, left_on="Purchase Order ID", right_on="PO ID Key", how="left").drop(columns=["PO ID Key"], errors="ignore")
    else:
        for c in ["PO Number", "PO NS Item", "PO NS Department", "PO NS Location", "PO Vendor"]:
            out[c] = pd.NA

    # Financial location defaults. For OUT rows, asset master / WO values should override transaction part location.
    out["WO Sublocation / Department"] = out["WO Sublocation / Department"].fillna(out["Asset Sublocation / Department"]).fillna(out["Part Location"])
    out["WO Parent Location / Location"] = out["WO Parent Location / Location"].fillna(out["Asset Parent Location / Location"]).fillna(out["Parent Location"])
    out["Financial Department / Sublocation"] = out["WO Sublocation / Department"]
    out["Financial Location / Parent"] = out["WO Parent Location / Location"]
    out["Financial NS Item"] = out["PO NS Item"]
    out["Financial NS Department"] = out["PO NS Department"]
    out["Financial NS Location"] = out["PO NS Location"]

    out["Month"] = out["DATE"].dt.to_period("M").astype("string")
    return out



def build_bag_usage_report(parts: pd.DataFrame, tx_detail: pd.DataFrame, targets: list[dict]) -> pd.DataFrame:
    """Financial bag usage report copied from the inventory audit logic.

    Bag parts are matched to the fixed name mapping. When Parts_Master is available,
    the primary match is Parts_Master[Name] -> transaction Part ID. If Parts_Master
    is unavailable or a row has no part ID match, it falls back to matching the
    transaction PART NAME directly to the configured bag names.
    """
    target_df = pd.DataFrame(targets)
    if target_df.empty or tx_detail is None or tx_detail.empty:
        return pd.DataFrame()

    target_df["Name"] = target_df["Name"].map(clean_text)
    target_df["Type"] = target_df["Type"].map(clean_text)
    target_df["Department"] = target_df["Department"].map(clean_text)
    target_df["Location"] = target_df["Location"].map(clean_text)

    tx = tx_detail.copy()
    tx["Part ID Key"] = norm_id_series(tx["Part ID"]) if "Part ID" in tx.columns else pd.Series(pd.NA, index=tx.index, dtype="string")
    tx["PART NAME"] = text_series(tx, "PART NAME").map(clean_text)
    tx["Tx Date"] = pd.to_datetime(tx.get("DATE"), errors="coerce")
    tx["Qty Change"] = pd.to_numeric(tx.get("QUANTITY ADDED"), errors="coerce").fillna(0)
    tx["Unit Cost Num"] = pd.to_numeric(tx.get("UNIT COST"), errors="coerce").fillna(0)
    tx["Total Cost Num"] = pd.to_numeric(tx.get("TOTAL COST"), errors="coerce").fillna(0)

    matched_by_id = pd.DataFrame()
    if parts is not None and not parts.empty:
        p = parts.copy()
        p.columns = [str(c).strip() for c in p.columns]
        id_col = first_present_col(p, ["ID", "Part ID"])
        name_col = first_present_col(p, ["Name", "Part Name"])
        if id_col and name_col:
            part_work = pd.DataFrame()
            part_work["Part ID Key"] = norm_id_series(p[id_col])
            part_work["Name"] = text_series(p, name_col).map(clean_text)
            matched_parts = part_work.merge(
                target_df[["Name", "Type", "Department", "Location"]],
                on="Name",
                how="inner",
            ).drop_duplicates(["Part ID Key", "Name"], keep="first")

            if not matched_parts.empty:
                matched_by_id = tx.merge(
                    matched_parts[["Part ID Key", "Name", "Type", "Department", "Location"]],
                    on="Part ID Key",
                    how="inner",
                )

    # Fallback direct transaction-name match for any bag rows not found through Parts_Master.
    direct = tx.rename(columns={"PART NAME": "Name"}).merge(
        target_df[["Name", "Type", "Department", "Location"]],
        on="Name",
        how="inner",
    )

    bag_tx = pd.concat([matched_by_id, direct], ignore_index=True, sort=False)
    if bag_tx.empty:
        return pd.DataFrame(columns=["Tx Date", "Name", "Billing Type", "Billing Department", "Billing Location"])

    bag_tx = bag_tx.drop_duplicates(
        subset=[c for c in ["Transaction ID", "Part ID Key", "Tx Date", "Name", "DIRECTION", "Qty Change"] if c in bag_tx.columns],
        keep="first",
    )

    bag_tx = bag_tx.rename(columns={
        "Type": "Billing Type",
        "Department": "Billing Department",
        "Location": "Billing Location",
    })

    # Bag usage means bags USED, so only OUT transactions belong in this report.
    # IN transactions are receipts/stock additions and should not appear in Bag Usage.
    if "DIRECTION" in bag_tx.columns:
        bag_tx = bag_tx[text_series(bag_tx, "DIRECTION").str.upper().str.strip().eq("OUT")].copy()

    preferred_cols = [
        "Tx Date", "Name", "Billing Type", "Billing Department", "Billing Location",
        "Part ID Key", "PART NAME", "PART TYPE", "Part Location", "Parent Location",
        "DIRECTION", "Transaction Type", "Purchase Order ID", "PO Number",
        "Work Order ID", "WO Asset", "Asset Type", "WO Sublocation / Department",
        "WO Parent Location / Location", "Qty Change", "Unit Cost Num", "Total Cost Num",
        "Transaction ID", "Initiator", "Reason",
    ]
    preferred_cols = [c for c in preferred_cols if c in bag_tx.columns]
    return bag_tx[preferred_cols].sort_values(["Tx Date", "Name"], ascending=[False, True]).reset_index(drop=True)


# ============================================================
# DATE FILTER
# ============================================================
def date_filter_ui(df: pd.DataFrame, date_col: str, key_prefix: str):
    """Date filter with YTD default, Monthly preset, and a true editable Custom range.

    Custom uses one date-range picker so the start/end selector always appears
    immediately after Custom is selected.
    """
    dt = pd.to_datetime(df[date_col], errors="coerce") if date_col in df.columns else pd.Series(pd.NaT, index=df.index)
    valid = dt.dropna()
    today = pd.Timestamp.today().normalize().date()

    if valid.empty:
        max_date = today
        min_date = date(today.year, 1, 1)
    else:
        max_date = valid.max().date()
        min_date = valid.min().date()

    ytd_start = date(max_date.year, 1, 1)
    if ytd_start < min_date:
        ytd_start = min_date

    mode = st.selectbox(
        "Date filter",
        ["YTD", "Monthly", "Custom"],
        index=0,
        key=f"{key_prefix}_date_mode",
        help="YTD and Monthly are preset ranges. Choose Custom to manually pick a start and end date.",
    )

    start_date = ytd_start
    end_date = max_date

    if mode == "YTD":
        st.info(f"YTD selected: {start_date:%m/%d/%Y} through {end_date:%m/%d/%Y}. Choose Custom to pick a different date range.")

    elif mode == "Monthly":
        month_source = df.loc[dt.notna()].copy()
        if not month_source.empty:
            month_opts = sorted(pd.to_datetime(month_source[date_col], errors="coerce").dt.to_period("M").astype(str).unique().tolist())
        else:
            month_opts = [pd.Timestamp(max_date).to_period("M").strftime("%Y-%m")]

        selected_month = st.selectbox(
            "Month",
            month_opts,
            index=len(month_opts) - 1 if month_opts else 0,
            key=f"{key_prefix}_month",
        )
        p = pd.Period(selected_month, freq="M")
        start_date = p.start_time.date()
        end_date = p.end_time.date()
        st.info(f"Monthly selected: {start_date:%m/%d/%Y} through {end_date:%m/%d/%Y}.")

    else:
        picked_range = st.date_input(
            "Custom date range",
            value=(ytd_start, max_date),
            min_value=min_date,
            max_value=max_date,
            key=f"{key_prefix}_custom_range",
            help="Pick the start date and end date for the report.",
        )

        if isinstance(picked_range, tuple) and len(picked_range) == 2:
            start_date, end_date = picked_range
        elif isinstance(picked_range, list) and len(picked_range) == 2:
            start_date, end_date = picked_range
        else:
            st.warning("Select both a start date and an end date.")
            start_date, end_date = ytd_start, max_date

        if start_date > end_date:
            start_date, end_date = end_date, start_date

        st.info(f"Custom selected: {start_date:%m/%d/%Y} through {end_date:%m/%d/%Y}.")

    return mode, start_date, end_date

def apply_date_filter(df: pd.DataFrame, date_col: str, start_date, end_date) -> pd.DataFrame:
    dt = pd.to_datetime(df[date_col], errors="coerce")
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return df.loc[dt.notna() & dt.ge(start_ts) & dt.le(end_ts)].copy()





def format_bag_usage_export(df: pd.DataFrame) -> pd.DataFrame:
    """Return bag usage columns in the same order as the uploaded finance CSV."""
    if df is None or df.empty:
        return pd.DataFrame(columns=BAG_USAGE_EXPORT_COLUMNS)
    out = df.copy()
    for col in BAG_USAGE_EXPORT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    if "Tx Date" in out.columns:
        dt = pd.to_datetime(out["Tx Date"], errors="coerce")
        try:
            out["Tx Date"] = dt.dt.strftime("%-m/%-d/%Y %H:%M")
        except Exception:
            out["Tx Date"] = dt.dt.strftime("%m/%d/%Y %H:%M")
    return out[BAG_USAGE_EXPORT_COLUMNS].copy()


@st.cache_data(show_spinner=False)
def prep_transactions_cached(raw, locations, workorders, purchase_orders, assets) -> pd.DataFrame:
    return prep_transactions(raw, locations, workorders, purchase_orders, assets)


@st.cache_data(show_spinner=False)
def build_bag_usage_report_cached(parts, tx) -> pd.DataFrame:
    return build_bag_usage_report(parts, tx, BAG_USAGE_TARGETS)

def render_bag_usage_view(bag_usage_master: pd.DataFrame):
    """Top-level Bag Usage report view."""
    st.subheader("Bag Usage")
    st.caption("Financial bag-usage report for OUT transactions only. Usage means bags used/issued out of inventory.")

    bag_df = bag_usage_master.copy()
    if bag_df.empty:
        st.info("No bag usage items matched the configured bag-name mapping.")
        return

    st.caption("Heavy database reads and transaction preparation are cached, so filter changes should be much faster.")

    bag_mode, bag_start_date, bag_end_date = date_filter_ui(bag_df, "Tx Date", "bag_usage_top")
    bag_view = apply_date_filter(bag_df, "Tx Date", bag_start_date, bag_end_date)

    b1, b2, b3, b4 = st.columns([2, 2, 2, 3])
    bag_name_options = ["All"] + sorted(bag_view["Name"].dropna().astype(str).unique().tolist()) if not bag_view.empty else ["All"]
    bag_type_options = ["All"] + sorted(bag_view["Billing Type"].dropna().astype(str).unique().tolist()) if not bag_view.empty else ["All"]
    bag_dept_options = ["All"] + sorted(bag_view["Billing Department"].dropna().astype(str).unique().tolist()) if not bag_view.empty else ["All"]

    bag_name_pick = b1.selectbox("Name", bag_name_options, index=0, key="bag_top_name_pick")
    bag_type_pick = b2.selectbox("Billing Type", bag_type_options, index=0, key="bag_top_type_pick")
    bag_dept_pick = b3.selectbox("Billing Department", bag_dept_options, index=0, key="bag_top_dept_pick")
    bag_search = b4.text_input("Search", key="bag_top_search_pick")

    if bag_name_pick != "All":
        bag_view = bag_view[text_series(bag_view, "Name") == bag_name_pick].copy()
    if bag_type_pick != "All":
        bag_view = bag_view[text_series(bag_view, "Billing Type") == bag_type_pick].copy()
    if bag_dept_pick != "All":
        bag_view = bag_view[text_series(bag_view, "Billing Department") == bag_dept_pick].copy()
    if bag_search.strip():
        needle = bag_search.strip().lower()
        mask = (
            text_series(bag_view, "Name").str.lower().str.contains(needle, na=False)
            | text_series(bag_view, "Billing Type").str.lower().str.contains(needle, na=False)
            | text_series(bag_view, "Billing Department").str.lower().str.contains(needle, na=False)
            | text_series(bag_view, "Billing Location").str.lower().str.contains(needle, na=False)
            | text_series(bag_view, "Part ID Key").str.lower().str.contains(needle, na=False)
        )
        bag_view = bag_view[mask].copy()

    # Safety filter: Bag Usage is OUT-only even if upstream data changes.
    bag_view = bag_view[text_series(bag_view, "DIRECTION").str.upper().str.strip().eq("OUT")].copy()
    bag_view_export = format_bag_usage_export(bag_view)

    bk1, bk2, bk3, bk4 = st.columns(4)
    bk1.metric("Usage Rows", f"{len(bag_view):,}")
    bk2.metric("Unique bag items", f"{bag_view['Name'].dropna().nunique():,}" if "Name" in bag_view.columns else "0")
    bk3.metric("Usage Qty", f"{pd.to_numeric(bag_view.get('Qty Change'), errors='coerce').fillna(0).sum():,.0f}")
    bk4.metric("Usage Cost", f"${pd.to_numeric(bag_view.get('Total Cost Num'), errors='coerce').fillna(0).sum():,.2f}")

    st.markdown("### Bag Usage Summary")
    bag_summary = (
        bag_view.groupby(["Name", "Billing Type", "Billing Department", "Billing Location", "DIRECTION"], dropna=False)
        .agg(Rows=("Name", "size"), Qty=("Qty Change", "sum"), Total_Cost=("Total Cost Num", "sum"))
        .reset_index()
        .sort_values(["Billing Location", "Name", "DIRECTION"], ascending=[True, True, True])
    )
    st.dataframe(bag_summary, use_container_width=True, height=300)

    st.markdown("### Bag Usage OUT Detail")
    st.dataframe(bag_view_export, use_container_width=True, height=520)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇️ Download Bag Usage OUT CSV",
            bag_view_export.to_csv(index=False).encode("utf-8"),
            "bag_usage_out_report.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_bag_top_out_report",
        )
    with d2:
        st.download_button(
            "⬇️ Download Bag Usage OUT XLSX",
            xlsx_bytes(bag_view_export, sheet_name="Bag Usage OUT"),
            "bag_usage_out_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="dl_bag_top_xlsx",
        )


# ============================================================
# STREAMLIT PAGE
# ============================================================
st.set_page_config(page_title="Straight Transactions Report", page_icon="🔁", layout="wide")

st.title("Straight Transactions Report")
st.caption("Finance-ready inventory transactions with Work Order and PO accounting enrichment.")

with st.sidebar:
    st.subheader("Data Source")
    st.code(str(DB_PATH), language="text")
    if st.button("🔄 Reload data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if not Path(str(DB_PATH)).exists():
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

raw_tx = read_table(str(DB_PATH), TRANSACTIONS_TABLE)
workorders = read_table(str(DB_PATH), WORKORDERS_TABLE)
purchase_orders = read_table(str(DB_PATH), PURCHASE_ORDERS_TABLE)
locations = read_table(str(DB_PATH), LOCATIONS_TABLE)
assets = read_table(str(DB_PATH), ASSETS_TABLE)
parts = read_table(str(DB_PATH), PARTS_TABLE)

if raw_tx.empty:
    st.error(f'Table not found or empty: "{TRANSACTIONS_TABLE}"')
    st.stop()

with st.expander("Loaded source tables", expanded=False):
    st.write({
        "Transactions rows": len(raw_tx),
        "Workorders rows": len(workorders),
        "Purchase_Orders rows": len(purchase_orders),
        "Locations_Master rows": len(locations),
        "Assets_Master rows": len(assets),
        "Parts_Master rows": len(parts),
    })

tx = prep_transactions_cached(raw_tx, locations, workorders, purchase_orders, assets)
if tx.empty:
    st.warning("No transaction rows were found after preparation.")
    st.stop()

bag_usage_master = build_bag_usage_report_cached(parts, tx)


report_view = st.radio(
    "Report View",
    ["Straight Transactions", "Bag Usage"],
    horizontal=True,
    key="straight_trx_report_view",
)

if report_view == "Bag Usage":
    render_bag_usage_view(bag_usage_master)
    st.stop()

# ============================================================
# FILTERS
# ============================================================
st.subheader("Filters")

f1, f2, f3 = st.columns([1.2, 1, 1])
with f1:
    location_options = sorted(set(tx["Parent Location"].dropna().astype(str).str.strip().loc[lambda s: s.ne("")].tolist()))
    selected_locations = st.multiselect(
        "Location",
        options=location_options,
        default=[],
        help="Uses parent location from Locations_Master; otherwise uses Part Location.",
    )
with f2:
    direction_filter = st.multiselect(
        "Direction",
        options=sorted(tx["DIRECTION"].dropna().astype(str).loc[lambda s: s.ne("")].unique().tolist()),
        default=[d for d in ["IN", "OUT"] if d in set(tx["DIRECTION"].unique())],
    )
with f3:
    type_options = sorted(tx["Transaction Type"].dropna().astype(str).loc[lambda s: s.ne("")].unique().tolist())
    selected_types = st.multiselect("Transaction type", options=type_options, default=[])

date_mode, start_date, end_date = date_filter_ui(tx, "DATE", "straight_trx")

c1, c2 = st.columns(2)
with c1:
    in_po_filter = st.radio(
        "IN transactions — PO linkage",
        ["All IN", "With PO", "Without PO"],
        index=0,
        horizontal=True,
        help="Applies only to IN transactions.",
    )
with c2:
    out_wo_filter = st.radio(
        "OUT transactions — WO linkage",
        ["All OUT", "With WO", "Without WO"],
        index=0,
        horizontal=True,
        help="Applies only to OUT transactions.",
    )

view = apply_date_filter(tx, "DATE", start_date, end_date)

if selected_locations:
    view = view[view["Parent Location"].isin(selected_locations)].copy()
if direction_filter:
    view = view[view["DIRECTION"].isin(direction_filter)].copy()
if selected_types:
    view = view[view["Transaction Type"].isin(selected_types)].copy()

if in_po_filter == "With PO":
    view = view[(view["DIRECTION"].ne("IN")) | (view["Has PO"])].copy()
elif in_po_filter == "Without PO":
    view = view[(view["DIRECTION"].ne("IN")) | (~view["Has PO"])].copy()

if out_wo_filter == "With WO":
    view = view[(view["DIRECTION"].ne("OUT")) | (view["Has WO"])].copy()
elif out_wo_filter == "Without WO":
    view = view[(view["DIRECTION"].ne("OUT")) | (~view["Has WO"])].copy()

# ============================================================
# FINANCIAL OUTPUT
# ============================================================
st.divider()
st.subheader("Finance Filtered Output")
st.caption("This table keeps the requested transaction format and adds Work Order / PO accounting fields for review and export.")

in_view = view[view["DIRECTION"].eq("IN")].copy()
out_view = view[view["DIRECTION"].eq("OUT")].copy()

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Rows", f"{len(view):,}")
m2.metric("Unique parts", f"{view['Part ID'].dropna().nunique():,}")
m3.metric("IN rows", f"{len(in_view):,}")
m4.metric("OUT rows", f"{len(out_view):,}")
m5.metric("IN without PO", f"{len(in_view[~in_view['Has PO']]):,}")
m6.metric("OUT without WO", f"{len(out_view[~out_view['Has WO']]):,}")

c1, c2, c3 = st.columns(3)
c1.metric("Total cost", f"${view['TOTAL COST'].fillna(0).sum():,.2f}")
c2.metric("IN cost", f"${in_view['TOTAL COST'].fillna(0).sum():,.2f}")
c3.metric("OUT cost", f"${out_view['TOTAL COST'].fillna(0).sum():,.2f}")

financial_cols = [
    "DIRECTION", "DATE", "PART NAME", "PART TYPE", "QUANTITY BEFORE", "QUANTITY ADDED", "QUANTITY AFTER",
    "UNIT COST", "TOTAL COST",
    "Work Order ID", "WO Asset", "Asset Type", "WO Sublocation / Department", "WO Parent Location / Location",
    "Transaction ID", "Transaction Type", "Part ID", "Part Location", "Parent Location",
    "Purchase Order ID", "PO Number", "PO Vendor", "PO NS Item", "PO NS Department", "PO NS Location",
    "WO Title", "Initiator", "Reason",
]
financial_cols = [c for c in financial_cols if c in view.columns]
financial = view[financial_cols].sort_values(["DATE", "DIRECTION", "PART NAME"], ascending=[False, True, True]).copy()

st.dataframe(financial, use_container_width=True, height=650)

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "⬇️ Download finance table CSV",
        financial.to_csv(index=False).encode("utf-8"),
        "straight_transactions_finance.csv",
        mime="text/csv",
        use_container_width=True,
    )
with d2:
    st.download_button(
        "⬇️ Download finance table XLSX",
        xlsx_bytes(financial),
        "straight_transactions_finance.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# ============================================================
# REVIEW SUMMARIES
# ============================================================
tabs = st.tabs(["By Location", "By Transaction Type", "By Linkage", "Missing Accounting Links"])

with tabs[0]:
    by_location = (
        view.groupby(["Parent Location", "DIRECTION"], dropna=False)
        .agg(
            Rows=("Transaction ID", "size"),
            Unique_Parts=("Part ID", lambda s: s.dropna().nunique()),
            Qty=("QUANTITY ADDED", "sum"),
            Total_Cost=("TOTAL COST", "sum"),
            IN_Without_PO=("Has PO", lambda s: int((~s).sum())),
            OUT_Without_WO=("Has WO", lambda s: int((~s).sum())),
        )
        .reset_index()
        .sort_values(["Total_Cost", "Rows"], ascending=[False, False])
    )
    st.dataframe(by_location, use_container_width=True, height=420)

with tabs[1]:
    by_type = (
        view.groupby(["Transaction Type", "DIRECTION"], dropna=False)
        .agg(
            Rows=("Transaction ID", "size"),
            Unique_Parts=("Part ID", lambda s: s.dropna().nunique()),
            Qty=("QUANTITY ADDED", "sum"),
            Total_Cost=("TOTAL COST", "sum"),
        )
        .reset_index()
        .sort_values(["Rows", "Total_Cost"], ascending=[False, False])
    )
    st.dataframe(by_type, use_container_width=True, height=420)

with tabs[2]:
    linkage = view.copy()
    linkage["Linkage Bucket"] = linkage.apply(
        lambda r: (
            "IN with PO" if r["DIRECTION"] == "IN" and r["Has PO"]
            else "IN without PO" if r["DIRECTION"] == "IN" and not r["Has PO"]
            else "OUT with WO" if r["DIRECTION"] == "OUT" and r["Has WO"]
            else "OUT without WO" if r["DIRECTION"] == "OUT" and not r["Has WO"]
            else "Other"
        ),
        axis=1,
    )
    by_link = (
        linkage.groupby("Linkage Bucket", dropna=False)
        .agg(
            Rows=("Transaction ID", "size"),
            Unique_Parts=("Part ID", lambda s: s.dropna().nunique()),
            Qty=("QUANTITY ADDED", "sum"),
            Total_Cost=("TOTAL COST", "sum"),
        )
        .reset_index()
        .sort_values("Rows", ascending=False)
    )
    st.dataframe(by_link, use_container_width=True, height=360)

with tabs[3]:
    missing = view[
        ((view["DIRECTION"].eq("IN")) & (~view["Has PO"]))
        | ((view["DIRECTION"].eq("OUT")) & (~view["Has WO"]))
    ].copy()
    missing_financial = missing[[c for c in financial_cols if c in missing.columns]].sort_values(["DATE", "DIRECTION"], ascending=[False, True])
    st.dataframe(missing_financial, use_container_width=True, height=420)
    st.download_button(
        "⬇️ Download missing accounting links CSV",
        missing_financial.to_csv(index=False).encode("utf-8"),
        "straight_transactions_missing_accounting_links.csv",
        mime="text/csv",
        use_container_width=True,
    )

