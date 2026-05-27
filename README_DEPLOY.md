# Streamlit Cloud deployment

Upload these files to the repo root:

- app.py
- reporting_shared.py
- requirements.txt
- runtime.txt
- .gitattributes
- .streamlit/config.toml
- pages/ folder
- maintenance_master.db

Streamlit Cloud settings:

- Main file path: app.py
- Python: controlled by runtime.txt = python-3.11

Important:

- The database must be named exactly maintenance_master.db.
- It must be in the same folder as app.py.
- If GitHub refuses the upload because the database is over 100 MB, the DB cannot be committed normally. Use Git LFS or a smaller DB copy.
- This package uses minimal requirements only. The previous package included extra packages that can cause Streamlit Cloud build failures.
