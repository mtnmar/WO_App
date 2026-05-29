# Reporting app with login

Main file path in Streamlit Cloud:
reporting_hub_app.py

Required repo root files:
- reporting_hub_app.py
- auth_helper.py
- reporting_shared.py
- requirements.txt
- runtime.txt
- maintenance_master.db
- pages/

Streamlit secrets example:

[app_config.access]
admin_usernames = ["brad"]

[app_config.credentials.usernames.brad]
name = "Brad"
password = "your-password"
