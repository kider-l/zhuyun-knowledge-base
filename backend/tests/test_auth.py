from app.auth import create_access_token, verify_access_token


def test_token_roundtrip() -> None:
    token = create_access_token("admin")
    assert verify_access_token(token) == "admin"

