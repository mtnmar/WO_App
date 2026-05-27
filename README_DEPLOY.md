# Maintenance Reporting App - Streamlit Cloud Deploy

Upload this structure to the Streamlit app repo:

```text
app.py
reporting_shared.py
requirements.txt
.gitattributes
maintenance_master.db
pages/
```

Important Streamlit Cloud setting:

```text
Main file path: app.py
```

The database must be named exactly:

```text
maintenance_master.db
```

and it must be in the repo root beside `app.py`.

If Streamlit says only "Error running app," open **Manage app > Logs**. The updated `app.py` also checks whether the database exists and whether SQLite can open it.
