# Maintenance Reporting App - Streamlit Cloud Deployment

This package is the PC reporting app converted for online deployment.

## Required repo layout

Place `maintenance_master.db` in the repository root beside `app.py`.

```text
app.py
reporting_shared.py
maintenance_master.db
requirements.txt
.gitattributes
pages/
  01_Asset_History_Report.py
  02_Purchase_Order_Report.py
  03_Mobile_Service_Report.py
  04_Vendor_Report.py
  05_Inventory_Analysis_Report.py
  06_Inventory_Restock_Report.py
```

## Streamlit Cloud

Set the main file path to:

```text
app.py
```

No secrets are required if `maintenance_master.db` is committed in the same repo root.

## Local override

If running locally against a different database path, set:

```text
MAINTENANCE_DB_PATH=C:\Users\Brad\Desktop\Maintenance Pipeline\maintenance_master.db
```

Otherwise the app uses:

```text
./maintenance_master.db
```

## Large DB note

If `maintenance_master.db` is larger than GitHub's normal file limit, use Git LFS for `*.db`.
