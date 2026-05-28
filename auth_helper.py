# auth_helper.py
from __future__ import annotations

import hmac
import streamlit as st

try:
    import bcrypt  # type: ignore
except Exception:
    bcrypt = None


def _get_nested(obj, keys, default=None):
    cur = obj
    for key in keys:
        try:
            cur = cur[key]
        except Exception:
            try:
                cur = getattr(cur, key)
            except Exception:
                return default
    return cur


def _to_plain_dict(x):
    try:
        return dict(x)
    except Exception:
        return x or {}


def _load_users() -> dict:
    """
    Supported secrets formats:

    Preferred:
    [app_config.credentials.usernames.brad]
    name = "Brad"
    password = "plain password OR bcrypt hash"

    Also supported:
    [credentials.usernames.brad]
    name = "Brad"
    password = "plain password OR bcrypt hash"

    Simple fallback:
    [passwords]
    brad = "plain password OR bcrypt hash"
    """
    users = _get_nested(st.secrets, ["app_config", "credentials", "usernames"], None)
    if users is None:
        users = _get_nested(st.secrets, ["credentials", "usernames"], None)

    if users is not None:
        return {str(k): _to_plain_dict(v) for k, v in _to_plain_dict(users).items()}

    passwords = _get_nested(st.secrets, ["passwords"], None)
    if passwords is not None:
        return {
            str(k): {"name": str(k), "password": str(v)}
            for k, v in _to_plain_dict(passwords).items()
        }

    return {}


def _allowed_admins() -> list[str]:
    admins = _get_nested(st.secrets, ["app_config", "access", "admin_usernames"], [])
    try:
        return [str(x) for x in list(admins)]
    except Exception:
        return []


def _check_password(candidate: str, stored: str) -> bool:
    stored = str(stored or "")
    candidate = str(candidate or "")

    if stored.startswith(("$2a$", "$2b$", "$2y$")):
        if bcrypt is None:
            return False
        try:
            return bool(bcrypt.checkpw(candidate.encode("utf-8"), stored.encode("utf-8")))
        except Exception:
            return False

    return hmac.compare_digest(candidate, stored)


def require_login() -> None:
    """Gate a Streamlit page behind username/password authentication."""
    users = _load_users()

    if not users:
        st.error("Login is enabled, but no user credentials were found in Streamlit secrets.")
        st.code('''[app_config.credentials.usernames.brad]
name = "Brad"
password = "your-password"''', language="toml")
        st.stop()

    if st.session_state.get("authenticated") is True:
        with st.sidebar:
            user = st.session_state.get("username", "")
            display = st.session_state.get("name", user)
            st.success(f"Signed in: {display}")
            if st.button("Sign out", use_container_width=True):
                for key in ["authenticated", "username", "name", "is_admin"]:
                    st.session_state.pop(key, None)
                st.rerun()
        return

    st.title("Maintenance Reporting Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        user_cfg = users.get(username)
        stored_password = None
        if user_cfg:
            stored_password = user_cfg.get("password_hash") or user_cfg.get("password")

        if user_cfg and stored_password and _check_password(password, stored_password):
            admins = _allowed_admins()
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["name"] = user_cfg.get("name", username)
            st.session_state["is_admin"] = username in admins if admins else True
            st.rerun()

        st.error("Invalid username or password.")

    st.stop()
