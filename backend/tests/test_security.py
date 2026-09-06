"""Password hashing and JWT primitives."""
from __future__ import annotations

import time

import jwt
import pytest

from backend.core.config import settings
from backend.core.security import (
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_hash_is_not_plaintext_and_verifies():
    digest = hash_password("correct horse battery staple")
    assert digest != "correct horse battery staple"
    assert digest.startswith("$2")  # bcrypt marker
    assert verify_password("correct horse battery staple", digest) is True
    assert verify_password("wrong password", digest) is False


def test_hash_is_salted():
    a = hash_password("same-input")
    b = hash_password("same-input")
    assert a != b
    assert verify_password("same-input", a)
    assert verify_password("same-input", b)


def test_verify_tolerates_garbage_hash():
    assert verify_password("anything", "not-a-real-hash") is False
    assert verify_password("anything", "") is False


def test_long_password_is_accepted():
    long_pw = "p" * 200
    digest = hash_password(long_pw)
    assert verify_password(long_pw, digest) is True


def test_access_token_round_trip():
    token = create_access_token(42, extra_claims={"role": "NATIONAL"})
    payload = decode_token(token)
    assert payload["sub"] == "42"
    assert payload["type"] == "access"
    assert payload["role"] == "NATIONAL"
    assert payload["exp"] > payload["iat"]


def test_expired_token_is_rejected():
    token = create_access_token(1, expires_minutes=-1)
    time.sleep(1)
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_token(token)


def test_tampered_token_is_rejected():
    token = create_access_token(1)
    wrong_secret = "the-wrong-secret-but-long-enough-to-avoid-a-length-warning"
    with pytest.raises(jwt.InvalidTokenError):
        jwt.decode(token, wrong_secret, algorithms=[settings.JWT_ALGORITHM])
