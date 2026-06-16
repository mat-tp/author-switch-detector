"""
    src/auth/auth.py — User authentication backed by a JSON file.

    Users are stored in  users/users.json  as:
        {
          "username": {
            "username":    str,
            "password":    str  (SHA-256 hex digest — NOT plaintext),
            "role":        str  ("admin" | "user"),
            "created_at":  str  (ISO-8601 UTC)
          },
          ...
        }

    Public API
    ----------
    register(username, password, role="user")  -> (True, None) | (False, error_msg)
    login(username, password)                  -> (True, None) | (False, error_msg)
    get_current_user()                         -> dict | None
    logout()
    login_required(f)                          -> decorator that redirects to /login
    admin_required(f)                          -> decorator that redirects if not admin
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from functools import wraps

from flask import flash, redirect, request, session, url_for

# ── Storage path ──────────────────────────────────────────────────────────────

_USERS_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "dataset", "users")
USERS_FILE  = os.path.join(_USERS_DIR, "users.json")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    """Return the SHA-256 hex digest of *password*."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _load_users() -> dict:
    """Load the user store from disk; return an empty dict if missing."""
    os.makedirs(_USERS_DIR, exist_ok=True)
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict) -> None:
    """Persist the user store to disk."""
    os.makedirs(_USERS_DIR, exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as fh:
        json.dump(users, fh, indent=2)


def _validate_inputs(username: str, password: str):
    """
    Basic input validation.
    Returns (True, None) or (False, error_message).
    """
    username = username.strip()
    if not username:
        return False, "Username cannot be empty."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(username) > 32:
        return False, "Username must be at most 32 characters."
    if not username.replace("_", "").replace("-", "").isalnum():
        return False, "Username may only contain letters, digits, hyphens, and underscores."
    if not password:
        return False, "Password cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    return True, None


# ── Public auth functions ─────────────────────────────────────────────────────

def register(username: str, password: str, role: str = "user"):
    """
    Create a new user account.

    Return : 
    (True, None)          on success
    (False, error_msg)    if validation fails or username is taken
    """
    username = username.strip().lower()
    ok, err = _validate_inputs(username, password)
    if not ok:
        return False, err

    users = _load_users()
    if username in users:
        return False, "Username already taken."

    users[username] = {
        "username":   username,
        "password":   _hash_password(password),
        "role":       role if role in ("admin", "user") else "user",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_users(users)
    return True, None


def login(username: str, password: str):
    """
    Verify credentials and write the username into the Flask session.

    Return :
    (True, None)          on success
    (False, error_msg)    if credentials are wrong
    """
    username = username.strip().lower()
    if not username or not password:
        return False, "Please enter both username and password."

    users = _load_users()
    user  = users.get(username)
    if user is None or user["password"] != _hash_password(password):
        return False, "Invalid username or password."

    session["username"] = username
    session["role"]     = user.get("role", "user")
    return True, None


def logout() -> None:
    """Remove the current user from the session."""
    session.pop("username", None)
    session.pop("role",     None)


def get_current_user() -> dict | None:
    """
    Return the full user record for the logged-in user, or None.
    """
    username = session.get("username")
    if not username:
        return None
    return _load_users().get(username)


def get_all_users() -> list[dict]:
    """Return a list of all user records (passwords excluded) — admin use only."""
    users = _load_users()
    return [
        {k: v for k, v in u.items() if k != "password"}
        for u in users.values()
    ]


# ── Decorators ────────────────────────────────────────────────────────────────

def login_required(f):
    """
    Route decorator: redirect to /login if the user is not authenticated.
    Preserves the original URL so the user is sent back after logging in.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("username"):
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("auth_login", next=request.path))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """
    Route decorator: redirect to home if the user is not an admin.
    Implies login_required — unauthenticated users are also redirected.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("username"):
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("auth_login", next=request.path))
        if session.get("role") != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated
