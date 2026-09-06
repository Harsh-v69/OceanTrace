"""Role-based access control on protected routes."""
from __future__ import annotations

import pytest

from backend.core.security import require_min_role, require_roles
from backend.models.user import UserRole
from backend.tests._helpers import API, auth_header, make_user


# --------------------------------------------------------------------------- #
# require_roles  ->  GET /api/v1/system/settings  (NATIONAL only)
# --------------------------------------------------------------------------- #
def test_national_can_read_effective_settings(client):
    make_user(client, email="nat@example.com", role="NATIONAL")
    resp = client.get(
        f"{API}/system/settings", headers=auth_header(client, "nat@example.com")
    )
    assert resp.status_code == 200
    assert resp.json()["env"] == "test"


def test_pilot_is_forbidden_from_effective_settings(client):
    make_user(client, email="pil@example.com", role="PILOT")
    resp = client.get(
        f"{API}/system/settings", headers=auth_header(client, "pil@example.com")
    )
    assert resp.status_code == 403


def test_regional_is_forbidden_from_effective_settings(client):
    make_user(client, email="reg@example.com", role="REGIONAL")
    resp = client.get(
        f"{API}/system/settings", headers=auth_header(client, "reg@example.com")
    )
    assert resp.status_code == 403


def test_settings_requires_authentication(client):
    assert client.get(f"{API}/system/settings").status_code == 401


# --------------------------------------------------------------------------- #
# require_min_role  ->  GET /api/v1/alerts  (REGIONAL and above)
# --------------------------------------------------------------------------- #
def test_alert_feed_min_role(client):
    make_user(client, email="p@example.com", role="PILOT")
    make_user(client, email="r@example.com", role="REGIONAL")
    make_user(client, email="n@example.com", role="NATIONAL")

    assert (
        client.get(f"{API}/alerts", headers=auth_header(client, "p@example.com")).status_code
        == 403
    )
    assert (
        client.get(f"{API}/alerts", headers=auth_header(client, "r@example.com")).status_code
        == 200
    )
    assert (
        client.get(f"{API}/alerts", headers=auth_header(client, "n@example.com")).status_code
        == 200
    )


# --------------------------------------------------------------------------- #
# dependency factories in isolation
# --------------------------------------------------------------------------- #
class _FakeUser:
    def __init__(self, role: UserRole):
        self.role = role


def test_require_roles_dependency_logic():
    dep = require_roles(UserRole.NATIONAL)
    assert dep(user=_FakeUser(UserRole.NATIONAL)).role is UserRole.NATIONAL
    with pytest.raises(Exception):
        dep(user=_FakeUser(UserRole.PILOT))


def test_require_min_role_dependency_logic():
    dep = require_min_role(UserRole.REGIONAL)
    assert dep(user=_FakeUser(UserRole.REGIONAL))
    assert dep(user=_FakeUser(UserRole.NATIONAL))
    with pytest.raises(Exception):
        dep(user=_FakeUser(UserRole.PILOT))
