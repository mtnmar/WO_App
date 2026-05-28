# reporting_shared.py
# Shared configuration and cached database loaders for the Reporting app.

from __future__ import annotations

import os
import sqlite3
from typing import Iterable

import pandas as pd
import streamlit as st


DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "maintenance_master.db"))

LOCATIONS_TABLE = "Locations_Master"
ASSETS_TABLE = "Assets_Master"
PO_TABLE = "Purchase_Orders"
ASSET_HISTORY_TABLE = "Asset_History_Merged"


def norm_text(x) -> str:
    """Normalize text values for filters and joins."""
    if pd.isna(x):
        return ""
    return str(x).strip()


def money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"


@st.cache_data(show_spinner=False)
def table_exists(db_path: str, table_name: str) -> bool:
    if not db_path or not os.path.exists(db_path):
        return False

    with sqlite3.connect(db_path) as conn:
        q = """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND lower(name)=lower(?)
            LIMIT 1
        """
        result = pd.read_sql_query(q, conn, params=[table_name])

    return not result.empty


@st.cache_data(show_spinner=False)
def load_table(db_path: str, table_name: str) -> pd.DataFrame:
    """Load a SQLite table from the shared maintenance database."""
    if not db_path or not os.path.exists(db_path):
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        q = f'SELECT * FROM "{table_name}"'
        return pd.read_sql_query(q, conn)


@st.cache_data(show_spinner=False)
def load_locations(db_path: str = DB_PATH) -> pd.DataFrame:
    """Load the master location table used to control valid Location filters."""
    if not table_exists(db_path, LOCATIONS_TABLE):
        return pd.DataFrame()

    df = load_table(db_path, LOCATIONS_TABLE).copy()

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].map(norm_text)

    return df


@st.cache_data(show_spinner=False)
def load_assets(db_path: str = DB_PATH) -> pd.DataFrame:
    """Load the master asset table for current/future page filters."""
    if not table_exists(db_path, ASSETS_TABLE):
        return pd.DataFrame()

    df = load_table(db_path, ASSETS_TABLE).copy()

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].map(norm_text)

    return df


def get_valid_locations(locations_df: pd.DataFrame) -> list[str]:
    """
    Return the proper reporting location list.

    Primary source:
        Locations_Master[All Parents]

    Falls back to common alternatives only if All Parents is unavailable.
    """
    if locations_df.empty:
        return []

    candidate_cols = [
        "All Parents",
        "All Parent Locations",
        "Location",
        "Name",
    ]

    col = next((c for c in candidate_cols if c in locations_df.columns), None)
    if not col:
        return []

    values = (
        locations_df[col]
        .dropna()
        .astype(str)
        .map(str.strip)
    )

    values = values[values.ne("")]
    return sorted(values.unique().tolist())


def get_asset_options(assets_df: pd.DataFrame, selected_locations: Iterable[str] | None = None) -> list[str]:
    """
    Return asset options from Assets_Master.

    Location matching is attempted against All Parent Locations when available.
    """
    if assets_df.empty:
        return []

    df = assets_df.copy()
    selected_locations = list(selected_locations or [])

    if selected_locations and "All Parent Locations" in df.columns:
        df = df[df["All Parent Locations"].isin(selected_locations)]

    asset_col = next((c for c in ["Name", "ASSET", "Asset", "Asset Name"] if c in df.columns), None)
    if not asset_col:
        return []

    values = df[asset_col].dropna().astype(str).map(str.strip)
    values = values[values.ne("")]
    return sorted(values.unique().tolist())
