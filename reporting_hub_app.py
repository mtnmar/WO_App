import streamlit as st
from pathlib import Path
import sys

st.set_page_config(page_title="Maintenance Reporting Test", layout="wide")

st.title("Maintenance Reporting - Startup Test")
st.success("Main app loaded successfully.")

st.subheader("Environment Check")
st.write("Python version:", sys.version)
st.write("Current working directory:", str(Path.cwd()))
st.write("App file:", str(Path(__file__).resolve()))

st.subheader("Repo Root Files")
root = Path(__file__).resolve().parent
files = sorted([p.name for p in root.iterdir() if p.is_file()])
st.write(files)

st.info("This version intentionally has no pages and does not connect to the database yet.")
