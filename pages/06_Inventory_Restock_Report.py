
# 06_Inventory_Restock_Report.py
# Report-only Re-Stock page for the Reporting App.
# Includes: Re-Stock w/ Cart and Req / RFQ export only.
# Excludes: API upload, PO creation, saved flow persistence.

from __future__ import annotations

import io
import re
import sqlite3
import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, LongTable, Image
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

try:
    from reporting_shared import DB_PATH
except Exception:
    from pathlib import Path
    DB_PATH = str(Path(__file__).resolve().parents[1] / "maintenance_master.db")
LOGO_PATH = Path(DB_PATH).parent / "greer_logo.png"

RESTOCK_TABLE = "ReStock_Master"
VENDORS_TABLE = "Vendors_Master"
LOCATIONS_TABLE = "Locations_Master"
USERS_TABLE = "Users_Master"
ADDRESS_BOOK_TABLE = "address_book"
PARTS_CRITICALITY_TABLE = "mx_parts_criticality_current"

ALL = "All"
ORDERING_REFERENCE_COLUMN_TEMPLATE = ['Name', 'Part Numbers', 'InStk', 'MinStk', 'OrdQty', 'MinOrdQty', 'Recommended Stock Min', 'Usage Event Count', 'Usage Count 90D', 'Usage Count 12M', 'Usage Qty 12M', 'Stock Priority', 'Lead Time', 'Inventory Risk Index', 'Effective Lead Time Source', 'Vendor', 'Location', 'All Parent', 'ID', 'Types', 'Unit Cost', 'Restock Recommendation', 'Effective Lead Time Score', 'Asset Criticality', 'Likelihood of Failure', 'Wear Part Flag', 'Critical Part Flag']


st.set_page_config(page_title="Inventory Re-Stock Report", layout="wide")
st.markdown(
    """
    <style>
    .block-container{max-width:98%!important;padding-top:1rem;padding-right:1rem;padding-left:1rem;padding-bottom:1rem;}
    [data-testid="stDataFrame"], [data-testid="stDataEditor"] {width:100%!important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def norm_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "<na>", "nat"} else s


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def find_col(df: pd.DataFrame, *names: str) -> str | None:
    if df is None or df.empty:
        return None
    norm_map = {_norm(c): c for c in df.columns}
    for n in names:
        nn = _norm(n)
        if nn in norm_map:
            return norm_map[nn]
    for c in df.columns:
        cn = _norm(c)
        for n in names:
            nn = _norm(n)
            if nn and (nn in cn or cn in nn):
                return c
    return None


def to_num(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string").str.strip()
    s = s.str.replace(",", "", regex=False).str.replace("$", "", regex=False)
    s = s.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def money(v) -> str:
    try:
        if pd.isna(v):
            return "$0.00"
        return f"${float(v):,.2f}"
    except Exception:
        return "$0.00"


def split_multi_values(val):
    s = norm_str(val)
    if not s:
        return []
    parts = re.split(r"\s*[;,|/]\s*", s)
    return [p.strip() for p in parts if p.strip()]


def row_has_vendor(cell_value, selected_vendor):
    if not selected_vendor or selected_vendor == ALL:
        return True
    return selected_vendor in split_multi_values(cell_value)


def safe_options(df: pd.DataFrame, col: str | None, all_label: str = ALL) -> list[str]:
    if not col or col not in df.columns or df.empty:
        return [all_label]
    vals = df[col].dropna().astype(str).str.strip()
    vals = vals[vals.ne("") & vals.str.lower().ne("nan")]
    return [all_label] + sorted(vals.unique().tolist())


def build_vendor_options(df_in: pd.DataFrame, vendor_col: str | None) -> list[str]:
    if not vendor_col or vendor_col not in df_in.columns or df_in.empty:
        return [ALL]
    vals = []
    for v in df_in[vendor_col].tolist():
        vals.extend(split_multi_values(v))
    vals = sorted(pd.Series(vals, dtype="object").dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist())
    return [ALL] + vals


def get_logo_path() -> str | None:
    return str(LOGO_PATH) if LOGO_PATH.exists() else None


@st.cache_data(show_spinner=False)
def read_table(table_name: str) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            exists = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                conn,
                params=[table_name],
            )
            if exists.empty:
                return pd.DataFrame()
            return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        read_table(RESTOCK_TABLE),
        read_table(VENDORS_TABLE),
        read_table(LOCATIONS_TABLE),
        read_table(USERS_TABLE),
        read_table(ADDRESS_BOOK_TABLE),
    )


def map_locations(restock: pd.DataFrame, locations: pd.DataFrame, loc_col: str) -> pd.DataFrame:
    out = restock.copy()
    loc_name_col = find_col(locations, "Name", "Location", "Location Name")
    loc_parent_col = find_col(locations, "All Parents", "All Parent Locations", "AllParents")
    if loc_name_col and loc_parent_col and loc_col in out.columns:
        loc_map = locations[[loc_name_col, loc_parent_col]].drop_duplicates().copy()
        loc_map.columns = ["_loc_name", "_all_parents"]
        loc_map["_loc_name"] = loc_map["_loc_name"].map(norm_str)
        loc_map["_all_parents"] = loc_map["_all_parents"].map(norm_str)
        out["_loc_name_join"] = out[loc_col].map(norm_str)
        out = out.merge(loc_map, how="left", left_on="_loc_name_join", right_on="_loc_name")
        out["Location_Filter"] = out["_all_parents"].where(out["_all_parents"].map(norm_str).ne(""), out[loc_col].map(norm_str))
    else:
        out["Location_Filter"] = out[loc_col].map(norm_str) if loc_col in out.columns else ""
    return out


def get_ship_to(all_parent_locations: str, addresses: pd.DataFrame) -> str:
    if addresses.empty or not all_parent_locations:
        return ""
    col_parent = find_col(addresses, "all_parent_locations", "All Parent Locations", "All Parents")
    if not col_parent:
        return ""
    sub = addresses.loc[addresses[col_parent].astype(str).str.strip().eq(str(all_parent_locations).strip())]
    if sub.empty:
        return ""
    row = sub.iloc[0]
    maintainx_col = find_col(addresses, "maintainx_address")
    if maintainx_col:
        maintainx = norm_str(row.get(maintainx_col))
        if maintainx:
            return maintainx
    addr_col = find_col(addresses, "address")
    city_col = find_col(addresses, "city")
    state_col = find_col(addresses, "state")
    zip_col = find_col(addresses, "zip")
    city = norm_str(row.get(city_col)) if city_col else ""
    state = norm_str(row.get(state_col)) if state_col else ""
    zip_code = norm_str(row.get(zip_col)) if zip_col else ""
    city_state_zip = " ".join([x for x in [f"{city}," if city else "", state, zip_code] if x]).strip()
    parts = [norm_str(row.get(col_parent)), norm_str(row.get(addr_col)) if addr_col else "", city_state_zip]
    return "\n".join([x for x in parts if x]).strip()


def get_vendor_ship_to(all_parent_locations: str, addresses: pd.DataFrame) -> str:
    ship = get_ship_to(all_parent_locations, addresses)
    lines = [x for x in ship.splitlines() if norm_str(x)]
    if len(lines) >= 2:
        return "\n".join(lines[1:])
    return ship


def get_vendor_contact_details(vendors_df: pd.DataFrame, vendor_col: str, contact_col: str, email_col: str | None, phone_col: str | None, vendor_name: str, contact_name: str) -> tuple[str, str, str]:
    if vendors_df.empty or not vendor_col:
        return "", "", ""
    sub = vendors_df[vendors_df[vendor_col].astype(str).str.strip().eq(str(vendor_name).strip())].copy() if vendor_name else vendors_df.iloc[0:0].copy()
    vendor_id_col = find_col(vendors_df, "ID", "Vendor ID")
    vendor_id = norm_str(sub[vendor_id_col].iloc[0]) if vendor_id_col and not sub.empty else ""
    chosen = sub.iloc[0:0].copy()
    if contact_name and contact_col and not sub.empty:
        chosen = sub[sub[contact_col].astype(str).str.strip().eq(str(contact_name).strip())].copy()
    if chosen.empty and not sub.empty:
        chosen = sub.head(1).copy()
    email = norm_str(chosen[email_col].iloc[0]) if email_col and not chosen.empty else ""
    phone = norm_str(chosen[phone_col].iloc[0]) if phone_col and not chosen.empty else ""
    return vendor_id, email, phone


def normalize_cart(df: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["ID", "Vendor", "Part Number", "Description", "Qty", "UOM", "Unit Price", "Extended", "Notes"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    qty = to_num(out["Qty"]).fillna(0)
    price = to_num(out["Unit Price"])
    out["Extended"] = ""
    mask = qty.notna() & price.notna()
    out.loc[mask, "Extended"] = (qty[mask] * price[mask]).round(2)
    return out[cols]


def sort_cart_for_export(cart_df: pd.DataFrame, sort_col: str, ascending: bool) -> pd.DataFrame:
    """Return cart in the same order the user wants exported."""
    out = normalize_cart(cart_df)
    if out.empty:
        return out
    if sort_col and sort_col in out.columns:
        return out.sort_values(sort_col, ascending=ascending, kind="mergesort").reset_index(drop=True)
    return out.reset_index(drop=True)


def build_export_lines(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_cart(df)
    qty = to_num(out["Qty"])
    price = to_num(out["Unit Price"])
    out["Extended"] = ""
    mask = qty.notna() & price.notna()
    out.loc[mask, "Extended"] = (qty[mask] * price[mask]).round(2)
    out["Line #"] = range(1, len(out) + 1)
    out = out.rename(columns={"Description": "Description", "Unit Price": "Unit Cost"})
    return out[["Line #", "Part Number", "Description", "Qty", "UOM", "Unit Cost", "Extended", "Notes", "ID"]]


def requisition_header_export(h: dict) -> dict:
    return {
        "Requisition #": norm_str(h.get("doc_no")),
        "Date": str(pd.Timestamp.today().date()),
        "Requester": norm_str(h.get("requestor_name")),
        "Priority": norm_str(h.get("priority") or "Medium"),
        "Vendor": norm_str(h.get("vendor_name")),
        "Vendor Contact": norm_str(h.get("vendor_contact_name")),
        "Department": "000 - Overhead",
        "Item": "13410 - PARTS INVENTORY",
        "All Parent Locations": norm_str(h.get("all_parent_locations")),
        "Needed By": norm_str(h.get("needed_by")),
        "Ship To": norm_str(h.get("ship_to")),
        "Vendor Ship To": norm_str(h.get("vendor_ship_to")),
        "Notes": norm_str(h.get("remarks")),
    }


def rfq_header_export(h: dict) -> dict:
    ship_to = norm_str(h.get("vendor_ship_to") or h.get("ship_to"))
    ship_lines = ship_to.splitlines()
    ship_company = ship_lines[0] if len(ship_lines) > 0 else ""
    ship_addr = ship_lines[1] if len(ship_lines) > 1 else ""
    ship_csz = ship_lines[2] if len(ship_lines) > 2 else ""
    address_pretty = "\n\n".join([
        "\n".join([x for x in ["Shipping To:", ship_company, ship_addr, ship_csz, norm_str(h.get("requestor_name"))] if x]),
        "\n".join([
            "Billing To:",
            "Greer Industries, Inc",
            "P.O. Box 1900",
            "Morgantown, WV, 26507",
            "Heather Page, Accounts Payable",
            "Phone: (304) 594-1768",
            "hpage@greerindustries.com",
        ]),
    ])
    return {
        "created_ts": int(dt.datetime.now().timestamp()),
        "vendor": norm_str(h.get("vendor_name")),
        "vendor_email": norm_str(h.get("vendor_contact_email")),
        "vendor_phone": norm_str(h.get("vendor_contact_phone")),
        "contact_name": norm_str(h.get("vendor_contact_name")),
        "contact_email": norm_str(h.get("vendor_contact_email")),
        "contact_phone": norm_str(h.get("vendor_contact_phone")),
        "user": norm_str(h.get("requestor_name")),
        "address": ship_addr,
        "company": ship_company,
        "location": norm_str(h.get("all_parent_locations")),
        "address_pretty": address_pretty,
    }


def requisition_pdf_bytes(header: dict, lines: pd.DataFrame) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not installed.")
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=landscape(letter))
    w, h = landscape(letter)
    left = 0.6 * inch
    top = h - 0.55 * inch
    logo_path = get_logo_path()
    if logo_path:
        try:
            c.drawImage(logo_path, left, top - 0.10 * inch, width=1.8 * inch, height=0.7 * inch, preserveAspectRatio=True, mask="auto")
        except Exception:
            pass
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left + 2.0 * inch, top, "Requisition")
    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(w - left, top, str(header.get("Requisition #", "")))
    y = top - 0.35 * inch
    line_h = 0.22 * inch
    xL = left
    xR = w / 2 + 0.2 * inch
    y0 = y
    fields_left = [
        ("Date", header.get("Date", "")),
        ("Requester", header.get("Requester", "")),
        ("Priority", header.get("Priority", "")),
        ("Vendor", header.get("Vendor", "")),
        ("Vendor Contact", header.get("Vendor Contact", "")),
        ("Item", header.get("Item", "")),
    ]
    fields_right = [
        ("Department", header.get("Department", "")),
        ("Location", header.get("All Parent Locations", "")),
        ("Needed By", header.get("Needed By", "")),
        ("Ship To", header.get("Ship To", "")),
    ]
    for k, v in fields_left:
        c.setFont("Helvetica-Bold", 10); c.drawString(xL, y, f"{k}:")
        c.setFont("Helvetica", 10); c.drawString(xL + 1.45 * inch, y, str(v)[:78])
        y -= line_h
    y = y0
    for k, v in fields_right:
        c.setFont("Helvetica-Bold", 10); c.drawString(xR, y, f"{k}:")
        c.setFont("Helvetica", 10)
        if k == "Ship To":
            ship_lines = [line.strip() for line in str(v).splitlines() if line.strip()] or [""]
            c.drawString(xR + 1.75 * inch, y, ship_lines[0][:40])
            for extra_line in ship_lines[1:]:
                y -= line_h
                c.drawString(xR + 1.75 * inch, y, extra_line[:40])
            y -= line_h
        else:
            c.drawString(xR + 1.75 * inch, y, str(v)[:60])
            y -= line_h
    y = min(y, y0 - len(fields_left) * line_h) - 0.05 * inch
    notes = str(header.get("Notes", "") or "")
    if notes.strip():
        c.setFont("Helvetica-Bold", 10); c.drawString(left, y, "Notes:")
        c.setFont("Helvetica", 10); c.drawString(left + 0.8 * inch, y, notes[:160])
        y -= 0.25 * inch
    y -= 0.05 * inch
    c.setFont("Helvetica-Bold", 11); c.drawString(left, y, "Line Items")
    y -= 0.22 * inch
    lines_out = build_export_lines(lines)
    col_x = {"Line #": left, "Part Number": left + 0.55 * inch, "Description": left + 2.05 * inch, "Qty": left + 7.10 * inch, "UOM": left + 7.70 * inch, "Unit Cost": left + 8.25 * inch, "Extended": left + 9.10 * inch}
    headers = ["Line #", "Part Number", "Description", "Qty", "UOM", "Unit Cost", "Extended"]
    c.setFont("Helvetica-Bold", 9)
    for hname in headers:
        c.drawString(col_x[hname], y, hname)
    y -= 0.14 * inch; c.line(left, y, w - left, y); y -= 0.10 * inch
    c.setFont("Helvetica", 8.7)
    for _, row in lines_out.iterrows():
        if y < 0.8 * inch:
            c.showPage(); y = h - 0.7 * inch; c.setFont("Helvetica", 8.7)
        c.drawString(col_x["Line #"], y, str(row.get("Line #", ""))[:6])
        c.drawString(col_x["Part Number"], y, str(row.get("Part Number", ""))[:20])
        c.drawString(col_x["Description"], y, str(row.get("Description", ""))[:70])
        c.drawString(col_x["Qty"], y, str(row.get("Qty", ""))[:8])
        c.drawString(col_x["UOM"], y, str(row.get("UOM", ""))[:8])
        c.drawRightString(col_x["Unit Cost"] + 0.7 * inch, y, str(row.get("Unit Cost", ""))[:12])
        c.drawRightString(col_x["Extended"] + 0.75 * inch, y, str(row.get("Extended", ""))[:12])
        y -= 0.16 * inch
    total = to_num(lines_out.get("Extended", pd.Series(dtype=float))).fillna(0).sum()
    c.setFont("Helvetica-Bold", 11); c.drawRightString(w - left, 0.6 * inch, f"Total: {total:,.2f}")
    c.showPage(); c.save()
    return buf.getvalue()


def rfq_pdf_bytes(doc_no: str, lines_df: pd.DataFrame, header: dict) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not installed.")
    lines = normalize_cart(lines_df)
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    styleN = styles["Normal"]
    story = []
    logo_path = get_logo_path()
    if logo_path:
        try:
            story.append(Image(logo_path, width=2.0 * inch, height=0.7 * inch)); story.append(Spacer(1, 8))
        except Exception:
            pass
    story.append(Paragraph("<b>REQUEST FOR QUOTE</b>", styles["Title"]))
    story.append(Paragraph(str(header.get("vendor", "")), styleN))
    story.append(Spacer(1, 12))
    created = dt.datetime.fromtimestamp(int(header.get("created_ts", int(dt.datetime.now().timestamp())))).strftime("%m/%d/%Y")
    info_rows = [["Date:", created], ["Contact Name:", str(header.get("contact_name", ""))], ["Phone #:", str(header.get("contact_phone", ""))], ["Email Address:", str(header.get("contact_email", ""))], ["Quote Request #:", str(doc_no)]]
    info_tbl = Table(info_rows, colWidths=[1.4 * inch, doc.width - 1.4 * inch])
    info_tbl.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 10)]))
    story.append(info_tbl); story.append(Spacer(1, 10))
    story.append(Paragraph(str(header.get("address_pretty", "")).replace("\n", "<br/>"), styleN))
    story.append(Spacer(1, 12)); story.append(Paragraph("<b>Please quote the following:</b>", styleN)); story.append(Spacer(1, 6))
    table_data = [[Paragraph("<b>QTY</b>", styleN), Paragraph("<b>PART #</b>", styleN), Paragraph("<b>DESCRIPTION</b>", styleN), Paragraph("<b>AMOUNT (EA)</b>", styleN)]]
    for _, r in lines.iterrows():
        amt = ""
        v = pd.to_numeric(r.get("Unit Price", ""), errors="coerce")
        if pd.notna(v):
            amt = f"{float(v):,.2f}"
        table_data.append([str(r.get("Qty", "")).strip(), str(r.get("Part Number", "")).strip(), str(r.get("Description", "")).strip(), amt])
    tbl = LongTable(table_data, colWidths=[0.6 * inch, 1.4 * inch, 3.7 * inch, 1.2 * inch], repeatRows=1)
    tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.75, colors.black), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (0, 0), (-1, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 1), (0, -1), "CENTER"), ("ALIGN", (3, 1), (3, -1), "RIGHT")]))
    story.append(tbl)
    doc.build(story)
    return bio.getvalue()


def xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=sheet[:31])
    return bio.getvalue()


def empty_cart_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["ID", "Vendor", "Part Number", "Description", "Qty", "UOM", "Unit Price", "Extended", "Notes"])


def clear_restock_report_flow():
    st.session_state["restock_report_cart"] = empty_cart_df()
    for k, v in {
        "restock_report_vendor": "",
        "restock_report_contact": "",
        "restock_report_requestor": "",
        "restock_report_needed_by": "",
        "restock_report_priority": "Medium",
        "restock_report_remarks": "",
        "rsr_req_doc_no": next_doc_no("REQ"),
        "rsr_rfq_doc_no": next_doc_no("RFQ"),
        "rsr_req_location": ALL,
    }.items():
        st.session_state[k] = v


def norm_id_value(x) -> str:
    s = norm_str(x)
    return re.sub(r"\.0$", "", s)


def pick_first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    return find_col(df, *names) if df is not None and not df.empty else None


def build_criticality_view(criticality: pd.DataFrame) -> pd.DataFrame:
    if criticality is None or criticality.empty:
        return pd.DataFrame()
    df = criticality.copy()
    id_col = pick_first_col(df, ["Part ID Key", "Part ID", "ID"])
    if id_col:
        df["__Part ID Key"] = df[id_col].map(norm_id_value)
    else:
        df["__Part ID Key"] = ""
    return df


def filter_criticality(criticality: pd.DataFrame, location=ALL, part_type=ALL, search="") -> pd.DataFrame:
    df = build_criticality_view(criticality)
    if df.empty:
        return df
    loc_col = pick_first_col(df, ["Parent Location", "All Parent", "All Parents", "Location_Filter"])
    type_col = pick_first_col(df, ["Part Types", "Types", "Type"])
    if location != ALL and loc_col:
        df = df[df[loc_col].astype(str).str.strip().eq(location)].copy()
    if part_type != ALL and type_col:
        df = df[df[type_col].astype(str).str.strip().eq(part_type)].copy()
    if search.strip():
        needle = search.strip().lower()
        search_cols = [c for c in [
            pick_first_col(df, ["Part Name", "Name", "Description"]),
            pick_first_col(df, ["Part Numbers", "Part Number"]),
            pick_first_col(df, ["Part ID Key", "Part ID", "ID"]),
            pick_first_col(df, ["Vendor", "Vendors"]),
            pick_first_col(df, ["Area"]),
        ] if c]
        if search_cols:
            blob = df[search_cols].astype(str).agg(" | ".join, axis=1).str.lower()
            df = df[blob.str.contains(needle, na=False)].copy()
    return df


def display_criticality_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    preferred = [
        "Parent Location", "Sub-Location", "Part ID Key", "Part Name", "Part Numbers",
        "Part Types", "Part Subtype", "Area", "Vendor",
        "Qty In Stock", "Minimum Qty", "Recommended Stock Min",
        "Lead Time", "Calc Lead Time", "Avg Lead Time", "Manual Lead Time",
        "Lead Time Score", "Effective Lead Time Score", "Effective Lead Time Source",
        "Asset Criticality", "Asset Criticality Factor", "Asset Criticality Source",
        "Likelihood of Failure", "Calculated Likelihood", "Likelihood Manual Override", "Likelihood Basis",
        "Usage Event Count", "Usage Count 90D", "Usage Count 12M", "Usage Qty 12M",
        "Inventory Risk Index", "Stock Priority", "Restock Recommendation",
        "Wear Part Flag", "Critical Part Flag", "Part Manual Match Flag",
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        cols = list(df.columns)[:30]
    out = df[cols].copy()
    sort_cols = [c for c in ["Parent Location", "Part Types", "Inventory Risk Index", "Part Name"] if c in out.columns]
    if sort_cols:
        ascending = [True] * len(sort_cols)
        if "Inventory Risk Index" in sort_cols:
            ascending[sort_cols.index("Inventory Risk Index")] = False
        out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return out.reset_index(drop=True)



def format_report_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Format display/export columns without blanking text identifiers.

    Rules:
    - Text/identifier columns remain text even when they contain numbers.
    - Index/score/risk/factor columns display 2 decimals.
    - Count/quantity/stock/cost/lead-time numeric columns display whole numbers.
    """
    out = df.copy()

    text_keywords = [
        "name", "number", "numbers", "part number", "part numbers",
        "id", "vendor", "location", "parent", "type", "types",
        "priority", "recommendation", "source", "basis", "area",
        "sub-location", "subtype", "description", "flag"
    ]
    decimal_keywords = [
        "index", "score", "factor", "likelihood", "criticality", "risk"
    ]
    whole_number_keywords = [
        "count", "qty", "quantity", "stock", "min", "ord", "lead time",
        "usage", "cost", "in stk", "instk"
    ]

    for col in out.columns:
        col_l = str(col).strip().lower()

        # Never coerce obvious text / identifier fields.
        if any(k in col_l for k in text_keywords):
            out[col] = out[col].map(norm_str)
            continue

        raw = out[col]
        s_num = pd.to_numeric(raw, errors="coerce")
        nonblank = raw.map(norm_str).ne("").sum() if hasattr(raw, "map") else len(raw)
        numeric_count = int(s_num.notna().sum())

        # Avoid turning mixed text columns blank just because a few rows contain numbers.
        if nonblank == 0 or numeric_count == 0 or numeric_count < max(1, int(nonblank * 0.80)):
            continue

        if any(k in col_l for k in decimal_keywords):
            out[col] = s_num.map(lambda x: "" if pd.isna(x) else f"{float(x):,.2f}")
        elif any(k in col_l for k in whole_number_keywords):
            out[col] = s_num.map(lambda x: "" if pd.isna(x) else f"{float(x):,.0f}")
        else:
            out[col] = s_num.map(lambda x: "" if pd.isna(x) else f"{float(x):,.0f}")

    return out


def apply_column_selector(df: pd.DataFrame, key: str, default_cols: list[str] | None = None, label: str = "Column Selector") -> pd.DataFrame:
    """Collapsed column selector used before display/export."""
    if df is None or df.empty:
        return pd.DataFrame()
    available = list(df.columns)
    if default_cols:
        default = [c for c in default_cols if c in available]
    else:
        default = available
    if not default:
        default = available
    with st.expander(label, expanded=False):
        selected_cols = st.multiselect(
            "Choose columns to display/export",
            options=available,
            default=default,
            key=key,
        )
    if not selected_cols:
        selected_cols = default
    return df[selected_cols].copy()


def order_columns_by_template(df: pd.DataFrame, template_cols: list[str]) -> pd.DataFrame:
    """Order columns by uploaded/reference template first, then append extras."""
    if df is None or df.empty:
        return pd.DataFrame()
    preferred = [c for c in template_cols if c in df.columns]
    extras = [c for c in df.columns if c not in preferred]
    return df[preferred + extras].copy()

def build_ordering_reference(restock_display: pd.DataFrame, criticality: pd.DataFrame) -> pd.DataFrame:
    if restock_display is None or restock_display.empty:
        return pd.DataFrame()
    r = restock_display.copy()
    r["__Part ID Key"] = r.get("ID", "").map(norm_id_value) if "ID" in r.columns else ""
    crit = build_criticality_view(criticality)
    if crit.empty or "__Part ID Key" not in crit.columns:
        return r.drop(columns=["__Part ID Key"], errors="ignore")
    crit_cols = ["__Part ID Key"] + [c for c in [
        "Inventory Risk Index", "Stock Priority", "Restock Recommendation",
        "Recommended Stock Min", "Lead Time", "Effective Lead Time Score",
        "Effective Lead Time Source", "Asset Criticality", "Likelihood of Failure",
        "Usage Event Count", "Usage Count 90D", "Usage Count 12M", "Usage Qty 12M",
        "Wear Part Flag", "Critical Part Flag"
    ] if c in crit.columns]
    crit_small = crit[crit_cols].drop_duplicates("__Part ID Key", keep="first")
    out = r.merge(crit_small, on="__Part ID Key", how="left")
    out = out.drop(columns=["__Part ID Key"], errors="ignore")

    # Match uploaded ordering-reference column order first, append any extra columns afterward.
    out = order_columns_by_template(out, ORDERING_REFERENCE_COLUMN_TEMPLATE)

    sort_cols = [c for c in ["Stock Priority", "Inventory Risk Index", "Vendor", "All Parent", "Name"] if c in out.columns]
    if sort_cols:
        ascending = [True] * len(sort_cols)
        if "Inventory Risk Index" in sort_cols:
            ascending[sort_cols.index("Inventory Risk Index")] = False
        out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return out.reset_index(drop=True)


def next_doc_no(prefix: str):
    year = dt.date.today().year
    key = f"restock_report_next_{prefix}_{year}"
    current = int(st.session_state.get(key, 0)) + 1
    st.session_state[key] = current
    return f"{prefix}-{year}-{current:04d}"


# -----------------------------
# Load Data
# -----------------------------
st.title("Inventory Re-Stock Report")
st.caption("Report-only re-stock cart and Req/RFQ exports. No API upload, PO creation, or saved-flow writes.")

restock, vendors, locations, users, addresses = load_all()
criticality = read_table(PARTS_CRITICALITY_TABLE)
if restock.empty:
    st.error(f"No data loaded from {RESTOCK_TABLE}.")
    st.stop()
if locations.empty:
    st.warning(f"{LOCATIONS_TABLE} is empty or unavailable. Location filter will use raw re-stock location values.")

col_loc_raw = find_col(restock, "Location", "Location Name", "Site", "Storeroom")
col_vendor = find_col(restock, "Vendor", "Preferred Vendor", "Vendor Name", "Supplier", "Vendors")
col_name = find_col(restock, "Name", "Description", "Part Name", "Item Name")
col_types = find_col(restock, "Types", "Type")
col_partnums = find_col(restock, "Part Numbers", "Part Number", "Part #", "Item Number", "PartNumber")
col_id = find_col(restock, "ID", "Part ID", "PartID")
col_instock = find_col(restock, "Quantity in Stock", "Qty in Stock", "In Stock", "Stock Qty")
col_minstock = find_col(restock, "Minimum Quantity", "Min Quantity", "Minimum Qty", "Min")
col_orderedqty = find_col(restock, "Ordered Quantity", "Qty Ordered", "On Order", "OrdQty")
col_uom = find_col(restock, "UOM", "Unit", "Unit of Measure")
col_unit_cost = find_col(restock, "Unit Cost", "Average Cost", "Avg Cost", "Last Price", "Cost")

required = {
    "ReStock Location": col_loc_raw,
    "Vendor": col_vendor,
    "Name": col_name,
    "Part Numbers": col_partnums,
    "ID": col_id,
    "Quantity in Stock": col_instock,
    "Minimum Quantity": col_minstock,
    "Ordered Quantity": col_orderedqty,
    "Types": col_types,
}
missing = [k for k, v in required.items() if not v]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.write("ReStock columns:", list(restock.columns))
    st.stop()

restock = map_locations(restock, locations, col_loc_raw)

ven_vendor_col = find_col(vendors, "Vendor") if not vendors.empty else None
ven_id_col = find_col(vendors, "ID", "Vendor ID") if not vendors.empty else None
ven_contact_col = find_col(vendors, "Contact Name", "Contact", "Name") if not vendors.empty else None
ven_email_col = find_col(vendors, "Email", "Contact Email", "Email Address") if not vendors.empty else None
ven_phone_col = find_col(vendors, "Phone Number", "Phone", "Telephone", "Contact Phone") if not vendors.empty else None
user_name_col = find_col(users, "Name", "Display", "Display Name", "Full Name", "User Name") if not users.empty else None

if "restock_report_cart" not in st.session_state:
    st.session_state["restock_report_cart"] = empty_cart_df()
for k, v in {
    "restock_report_vendor": "",
    "restock_report_contact": "",
    "restock_report_requestor": "",
    "restock_report_needed_by": "",
    "restock_report_priority": "Medium",
    "restock_report_remarks": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Flow Controls
# -----------------------------
ctrl1, ctrl2 = st.columns([1.2, 4.8])
with ctrl1:
    if st.button("🆕 Clear / New Cart + Req/RFQ", key="rsr_clear_new_flow", use_container_width=True):
        clear_restock_report_flow()
        st.success("Started a new cart and Req/RFQ flow.")
        st.rerun()
with ctrl2:
    cart_count = len(st.session_state.get("restock_report_cart", empty_cart_df()))
    st.caption(f"Current cart lines: {cart_count:,}. Use Clear / New Cart + Req/RFQ to reset cart, header fields, and document numbers.")

# -----------------------------
# Tabs
# -----------------------------
tab_restock, tab_req, tab_crit, tab_order_ref = st.tabs(["Re-Stock w/ Cart", "Req / RFQ", "Parts Criticality Index", "Ordering Reference"])

with tab_restock:
    st.subheader("Re-Stock w/ Cart")
    o1, o2, o3 = st.columns([1.2, 1.4, 2.4])
    with o1:
        show_complete = st.toggle("Show Complete Re-Stock", value=False, key="rsr_show_complete", help="Off = only parts with MinOrdQty > 0.")
    with o2:
        include_zero_min = st.toggle("Include Zero Min", value=True, key="rsr_include_zero_min")

    base = restock.copy()
    base["InStk"] = to_num(base[col_instock]).fillna(0)
    base["MinStk"] = to_num(base[col_minstock]).fillna(0)
    base["OrdQty"] = to_num(base[col_orderedqty]).fillna(0)
    base["MinOrdQty"] = (base["MinStk"] - (base["InStk"] + base["OrdQty"])).clip(lower=0)
    base["Vendor_Display"] = base[col_vendor].apply(lambda x: ", ".join(split_multi_values(x)))
    base["Unit Cost"] = to_num(base[col_unit_cost]).fillna(0) if col_unit_cost else 0.0

    if not include_zero_min:
        base = base[base["MinStk"] > 0].copy()
    if not show_complete:
        base = base[base["MinOrdQty"] > 0].copy()

    f1, f2, f3, f4 = st.columns([1.4, 1.4, 1.2, 1.7])
    with f1:
        loc_pick = st.selectbox("Location", safe_options(base, "Location_Filter"), key="rsr_loc")
    base_for_vendor = base.copy()
    if loc_pick != ALL:
        base_for_vendor = base_for_vendor[base_for_vendor["Location_Filter"].astype(str).str.strip().eq(loc_pick)].copy()
    with f2:
        vendor_pick = st.selectbox("Vendor", build_vendor_options(base_for_vendor, col_vendor), key="rsr_vendor")
    base_for_type = base_for_vendor.copy()
    if vendor_pick != ALL:
        base_for_type = base_for_type[base_for_type[col_vendor].apply(lambda x: row_has_vendor(x, vendor_pick))].copy()
    with f3:
        type_pick = st.selectbox("Types", safe_options(base_for_type, col_types), key="rsr_types")
    with f4:
        search_txt = st.text_input("Search", placeholder="Search part name, number, vendor, location...", key="rsr_search")

    view = base.copy()
    if loc_pick != ALL:
        view = view[view["Location_Filter"].astype(str).str.strip().eq(loc_pick)].copy()
    if vendor_pick != ALL:
        view = view[view[col_vendor].apply(lambda x: row_has_vendor(x, vendor_pick))].copy()
    if type_pick != ALL:
        view = view[view[col_types].astype(str).str.strip().eq(type_pick)].copy()
    if search_txt.strip():
        blob_cols = [col_name, col_partnums, col_vendor, col_loc_raw, "Location_Filter", col_types]
        blob = view[[c for c in blob_cols if c in view.columns]].astype(str).agg(" | ".join, axis=1).str.lower()
        view = view[blob.str.contains(search_txt.strip().lower(), na=False)].copy()

    table_view = view[[col_name, col_partnums, "InStk", "MinStk", "OrdQty", "MinOrdQty", "Vendor_Display", col_loc_raw, "Location_Filter", col_id, col_types, "Unit Cost"]].copy()
    table_view = table_view.rename(columns={
        col_name: "Name",
        col_partnums: "Part Numbers",
        "Vendor_Display": "Vendor",
        col_loc_raw: "Location",
        "Location_Filter": "All Parent",
        col_id: "ID",
        col_types: "Types",
    })
    table_view["Select"] = False
    table_view = table_view[["Select", "Name", "Part Numbers", "InStk", "MinStk", "OrdQty", "MinOrdQty", "Vendor", "Location", "All Parent", "ID", "Types", "Unit Cost"]]

    st.markdown("### Display Table")
    s1, s2, s3 = st.columns([1.2, 1.2, 2.8])
    with s1:
        sort_col = st.selectbox("Export / Display Sort", ["Name", "Vendor", "All Parent", "Location", "Types", "MinOrdQty", "InStk"], index=0, key="rsr_sort_col")
    with s2:
        sort_asc = st.toggle("Ascending", value=True, key="rsr_sort_asc")
    sorted_view = table_view.sort_values(sort_col, ascending=sort_asc, kind="mergesort").reset_index(drop=True) if sort_col in table_view.columns else table_view.reset_index(drop=True)

    csv_cols = [c for c in sorted_view.columns if c != "Select"]
    st.download_button(
        "Download Filtered Display Table CSV",
        data=sorted_view[csv_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name=f"restock_filtered_display_{dt.datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        use_container_width=True,
        key="rsr_display_csv",
    )

    edited = st.data_editor(
        sorted_view,
        use_container_width=True,
        height=500,
        key="rsr_table_editor",
        hide_index=True,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select"),
            "InStk": st.column_config.NumberColumn("InStk", format="%.0f"),
            "MinStk": st.column_config.NumberColumn("MinStk", format="%.0f"),
            "OrdQty": st.column_config.NumberColumn("OrdQty", format="%.0f"),
            "MinOrdQty": st.column_config.NumberColumn("MinOrdQty", format="%.0f"),
            "Unit Cost": st.column_config.NumberColumn("Unit Cost", format="%.2f"),
            "ID": st.column_config.TextColumn("ID", disabled=True),
        },
    )

    b1, b2, b3 = st.columns([1.4, 1.1, 3.5])
    with b1:
        add_clicked = st.button("Add Selected to Cart", key="rsr_add_cart", use_container_width=True)
    with b2:
        clear_clicked = st.button("Clear Cart", key="rsr_clear_cart", use_container_width=True)

    if clear_clicked:
        st.session_state["restock_report_cart"] = empty_cart_df()
        st.success("Cart cleared.")

    if add_clicked:
        selected = edited[edited["Select"] == True].copy()
        if selected.empty:
            st.warning("No rows selected.")
        else:
            # Prices intentionally start blank because these lines still need quoted.
            cart_add = pd.DataFrame({
                "ID": selected["ID"].astype(str),
                "Vendor": selected.get("Vendor", pd.Series("", index=selected.index)).astype(str),
                "Part Number": selected["Part Numbers"].astype(str),
                "Description": selected["Name"].astype(str),
                "Qty": pd.to_numeric(selected["MinOrdQty"], errors="coerce").fillna(1),
                "UOM": "",
                "Unit Price": "",
                "Extended": "",
                "Notes": "",
            })
            st.session_state["restock_report_cart"] = normalize_cart(pd.concat([st.session_state["restock_report_cart"], cart_add], ignore_index=True))
            if vendor_pick != ALL and not st.session_state.get("restock_report_vendor"):
                st.session_state["restock_report_vendor"] = vendor_pick
            st.success(f"Added {len(selected):,} line(s) to cart.")

    st.markdown("### Cart")
    cart_df = normalize_cart(st.session_state["restock_report_cart"])
    if cart_df.empty:
        st.info("Cart is empty.")
    else:
        cs1, cs2, cs3 = st.columns([1.3, 1.0, 2.7])
        with cs1:
            cart_sort_col = st.selectbox(
                "Cart Export Sort",
                ["Current Order", "Vendor", "Description", "Part Number", "Qty", "ID"],
                index=0,
                key="rsr_cart_sort_col",
                help="Use Vendor to group all cart lines by vendor before exporting.",
            )
        with cs2:
            cart_sort_asc = st.toggle("Cart Ascending", value=True, key="rsr_cart_sort_asc")

        edited_cart = st.data_editor(
            cart_df,
            use_container_width=True,
            height=260,
            key="rsr_cart_editor",
            hide_index=True,
            column_config={
                "ID": st.column_config.TextColumn("ID", disabled=True),
                "Qty": st.column_config.NumberColumn("Qty", format="%.0f"),
                "Unit Price": st.column_config.NumberColumn("Unit Price", format="%.2f"),
                "Extended": st.column_config.NumberColumn("Extended", format="%.2f", disabled=True),
            },
        )
        st.session_state["restock_report_cart"] = normalize_cart(edited_cart)

        cart_export_df = st.session_state["restock_report_cart"].copy()
        if cart_sort_col != "Current Order":
            # Vendor is kept in Notes when added from a vendor-filtered table in some workflows;
            # if a Vendor column is later added to the cart, this will use it automatically.
            if cart_sort_col not in cart_export_df.columns and cart_sort_col == "Vendor":
                cart_export_df["Vendor"] = ""
            cart_export_df = sort_cart_for_export(cart_export_df, cart_sort_col, cart_sort_asc)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download Cart CSV", data=cart_export_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"restock_cart_{dt.datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("Download Cart XLSX", data=xlsx_bytes({"Cart": cart_export_df}), file_name=f"restock_cart_{dt.datetime.now():%Y%m%d_%H%M}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

with tab_req:
    st.subheader("Req / RFQ")
    cart = normalize_cart(st.session_state["restock_report_cart"])
    req_sort_col = st.session_state.get("rsr_cart_sort_col", "Current Order")
    req_sort_asc = bool(st.session_state.get("rsr_cart_sort_asc", True))
    if req_sort_col != "Current Order":
        if req_sort_col not in cart.columns and req_sort_col == "Vendor":
            cart["Vendor"] = ""
        cart = sort_cart_for_export(cart, req_sort_col, req_sort_asc)
    if cart.empty:
        st.info("Cart is empty. Add lines from the Re-Stock w/ Cart tab first.")

    vendor_options = sorted(vendors[ven_vendor_col].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()) if ven_vendor_col and not vendors.empty else []
    requester_options = sorted(users[user_name_col].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()) if user_name_col and not users.empty else []
    loc_options = safe_options(restock, "Location_Filter")

    h1, h2, h3 = st.columns(3)
    with h1:
        selected_parent = st.selectbox("All Parent Location", loc_options, key="rsr_req_location")
    with h2:
        flow_vendor = st.selectbox("Vendor", [""] + vendor_options, key="restock_report_vendor")
    with h3:
        vendor_sub = vendors[vendors[ven_vendor_col].astype(str).str.strip().eq(str(flow_vendor).strip())].copy() if flow_vendor and ven_vendor_col else vendors.iloc[0:0].copy()
        contact_options = sorted(vendor_sub[ven_contact_col].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist()) if ven_contact_col and not vendor_sub.empty else []
        flow_contact = st.selectbox("Contact Name", [""] + contact_options, key="restock_report_contact")

    vendor_id, contact_email, contact_phone = get_vendor_contact_details(vendors, ven_vendor_col, ven_contact_col, ven_email_col, ven_phone_col, flow_vendor, flow_contact) if ven_vendor_col and ven_contact_col else ("", "", "")

    h4, h5, h6 = st.columns(3)
    with h4:
        if requester_options:
            requestor = st.selectbox("Requestor", [""] + requester_options, key="restock_report_requestor")
        else:
            requestor = st.text_input("Requestor", key="restock_report_requestor")
    with h5:
        st.text_input("Vendor ID", value=vendor_id, disabled=True)
    with h6:
        st.text_input("Needed By", key="restock_report_needed_by")

    h7, h8, h9 = st.columns(3)
    with h7:
        st.text_input("Contact Email", value=contact_email, disabled=True)
    with h8:
        st.text_input("Contact Phone", value=contact_phone, disabled=True)
    with h9:
        st.selectbox("Priority", ["Low", "Medium", "High", "Urgent"], index=["Low", "Medium", "High", "Urgent"].index(st.session_state.get("restock_report_priority", "Medium")) if st.session_state.get("restock_report_priority", "Medium") in ["Low", "Medium", "High", "Urgent"] else 1, key="restock_report_priority")

    ship_to = get_ship_to(selected_parent if selected_parent != ALL else "", addresses)
    vendor_ship_to = get_vendor_ship_to(selected_parent if selected_parent != ALL else "", addresses)
    a1, a2 = st.columns(2)
    with a1:
        st.text_area("Ship To", value=ship_to, height=100, disabled=True)
    with a2:
        st.text_area("Vendor Ship To", value=vendor_ship_to, height=100, disabled=True)

    remarks = st.text_area("Remarks / Notes", key="restock_report_remarks", height=90)

    st.markdown("### Req / RFQ Lines")
    if not cart.empty:
        req_lines = st.data_editor(
            cart.copy(),
            use_container_width=True,
            hide_index=True,
            key="rsr_req_lines_editor",
            column_config={
                "ID": st.column_config.TextColumn("ID", disabled=True),
                "Qty": st.column_config.NumberColumn("Qty", format="%.0f"),
                "Unit Price": st.column_config.NumberColumn("Unit Price", format="%.2f"),
                "Extended": st.column_config.NumberColumn("Extended", format="%.2f", disabled=True),
            },
        )
        cart = normalize_cart(req_lines)
        st.session_state["restock_report_cart"] = cart

    header_common = {
        "vendor_name": flow_vendor,
        "vendor_id": vendor_id,
        "vendor_contact_name": flow_contact,
        "vendor_contact_email": contact_email,
        "vendor_contact_phone": contact_phone,
        "requestor_name": requestor,
        "needed_by": st.session_state.get("restock_report_needed_by", ""),
        "priority": st.session_state.get("restock_report_priority", "Medium"),
        "remarks": remarks,
        "all_parent_locations": selected_parent if selected_parent != ALL else "",
        "ship_to": ship_to,
        "vendor_ship_to": vendor_ship_to,
    }

    st.markdown("### Export Requisition")
    req_no = st.text_input("Requisition #", value=next_doc_no("REQ"), key="rsr_req_doc_no")
    req_header = requisition_header_export({**header_common, "doc_no": req_no})
    r1, r2, r3 = st.columns(3)
    with r1:
        st.download_button("Download Requisition CSV", data=pd.DataFrame([req_header]).to_csv(index=False).encode("utf-8-sig"), file_name=f"{req_no}_header.csv", mime="text/csv", use_container_width=True)
    with r2:
        st.download_button("Download Requisition Lines CSV", data=build_export_lines(cart).to_csv(index=False).encode("utf-8-sig"), file_name=f"{req_no}_lines.csv", mime="text/csv", use_container_width=True)
    with r3:
        if REPORTLAB_AVAILABLE:
            st.download_button("Download Requisition PDF", data=requisition_pdf_bytes(req_header, cart), file_name=f"{req_no}.pdf", mime="application/pdf", use_container_width=True)
        else:
            st.warning("ReportLab not available for PDF export.")

    st.markdown("### Export RFQ")
    rfq_no = st.text_input("RFQ #", value=next_doc_no("RFQ"), key="rsr_rfq_doc_no")
    rfq_header = rfq_header_export(header_common)
    q1, q2, q3 = st.columns(3)
    with q1:
        st.download_button("Download RFQ CSV", data=pd.DataFrame([rfq_header]).to_csv(index=False).encode("utf-8-sig"), file_name=f"{rfq_no}_header.csv", mime="text/csv", use_container_width=True)
    with q2:
        st.download_button("Download RFQ Lines CSV", data=build_export_lines(cart).to_csv(index=False).encode("utf-8-sig"), file_name=f"{rfq_no}_lines.csv", mime="text/csv", use_container_width=True)
    with q3:
        if REPORTLAB_AVAILABLE:
            st.download_button("Download RFQ PDF", data=rfq_pdf_bytes(rfq_no, cart, rfq_header), file_name=f"{rfq_no}.pdf", mime="application/pdf", use_container_width=True)
        else:
            st.warning("ReportLab not available for PDF export.")


with tab_crit:
    st.subheader("Parts Criticality Index")
    st.caption(f"Read-only report view from SQLite table: {PARTS_CRITICALITY_TABLE}")
    if criticality.empty:
        st.warning(f"No data found in {PARTS_CRITICALITY_TABLE}. Run/update the inventory audit first.")
    else:
        cf1, cf2, cf3 = st.columns([1.4, 1.4, 2.2])
        crit_base = build_criticality_view(criticality)
        crit_loc_col = pick_first_col(crit_base, ["Parent Location", "All Parent", "All Parents", "Location_Filter"])
        crit_type_col = pick_first_col(crit_base, ["Part Types", "Types", "Type"])
        with cf1:
            crit_loc = st.selectbox("Location", safe_options(crit_base, crit_loc_col), key="rsr_crit_loc")
        with cf2:
            crit_type = st.selectbox("Part Types", safe_options(crit_base, crit_type_col), key="rsr_crit_type")
        with cf3:
            crit_search = st.text_input("Search criticality", placeholder="Part, part number, vendor, area...", key="rsr_crit_search")

        crit_view = filter_criticality(criticality, crit_loc, crit_type, crit_search)
        crit_display = display_criticality_table(crit_view)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Parts", f"{len(crit_view):,}")
        if not crit_view.empty:
            risk_col = pick_first_col(crit_view, ["Inventory Risk Index"] )
            rec_col = pick_first_col(crit_view, ["Recommended Stock Min"] )
            wear_col = pick_first_col(crit_view, ["Wear Part Flag"] )
            critical_col = pick_first_col(crit_view, ["Critical Part Flag"] )
            avg_risk = pd.to_numeric(crit_view[risk_col], errors="coerce").mean() if risk_col else 0
            rec_count = int((pd.to_numeric(crit_view[rec_col], errors="coerce").fillna(0) > 0).sum()) if rec_col else 0
            if wear_col or critical_col:
                wear_mask = pd.to_numeric(crit_view[wear_col], errors="coerce").fillna(0).gt(0) if wear_col else pd.Series(False, index=crit_view.index)
                critical_mask = pd.to_numeric(crit_view[critical_col], errors="coerce").fillna(0).gt(0) if critical_col else pd.Series(False, index=crit_view.index)
                wc_count = int((wear_mask | critical_mask).sum())
            else:
                wc_count = 0
        else:
            avg_risk = 0
            rec_count = 0
            wc_count = 0
        k2.metric("Avg Risk Index", f"{avg_risk:,.2f}")
        k3.metric("Recommended Stock", f"{rec_count:,}")
        k4.metric("Wear / Critical", f"{wc_count:,}")

        crit_selected = apply_column_selector(
            crit_display,
            key="rsr_crit_column_selector",
            default_cols=list(crit_display.columns),
            label="Column Selector - Parts Criticality Index",
        )
        crit_selected_fmt = format_report_numbers(crit_selected)

        st.download_button(
            "Download Part Criticality Index CSV",
            data=crit_selected_fmt.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"parts_criticality_index_{dt.datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            use_container_width=True,
            key="rsr_crit_csv",
        )
        st.dataframe(crit_selected_fmt, use_container_width=True, hide_index=True, height=650)

with tab_order_ref:
    st.subheader("Ordering Reference: Re-Stock + Parts Criticality")
    st.caption("Combines the current filtered re-stock table with the saved parts criticality index for ordering priority review.")
    try:
        ordering_ref = build_ordering_reference(sorted_view[csv_cols].copy(), criticality)
    except Exception:
        ordering_ref = pd.DataFrame()

    if ordering_ref.empty:
        st.info("No ordering reference rows available. Check the Re-Stock filters and confirm the parts criticality table exists.")
    else:
        of1, of2 = st.columns([1.4, 4])
        with of1:
            ord_sort = st.selectbox(
                "Reference Sort",
                [c for c in ["Stock Priority", "Inventory Risk Index", "Vendor", "All Parent", "Name", "MinOrdQty"] if c in ordering_ref.columns] or list(ordering_ref.columns[:1]),
                key="rsr_order_ref_sort",
            )
        if ord_sort in ordering_ref.columns:
            ascending = False if ord_sort == "Inventory Risk Index" else True
            ordering_ref = ordering_ref.sort_values(ord_sort, ascending=ascending, kind="mergesort").reset_index(drop=True)
        ordering_ref = order_columns_by_template(ordering_ref, ORDERING_REFERENCE_COLUMN_TEMPLATE)
        ordering_selected = apply_column_selector(
            ordering_ref,
            key="rsr_order_ref_column_selector",
            default_cols=[c for c in ORDERING_REFERENCE_COLUMN_TEMPLATE if c in ordering_ref.columns],
            label="Column Selector - Ordering Reference",
        )
        ordering_selected_fmt = format_report_numbers(ordering_selected)

        st.download_button(
            "Download Ordering Reference CSV",
            data=ordering_selected_fmt.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"restock_ordering_reference_{dt.datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv",
            use_container_width=True,
            key="rsr_order_ref_csv",
        )
        st.dataframe(ordering_selected_fmt, use_container_width=True, hide_index=True, height=650)
