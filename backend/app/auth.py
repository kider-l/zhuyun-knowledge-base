import base64
import hashlib
import hmac
import json
import time
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.config import get_settings


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def create_access_token(username: str, ttl_seconds: int = 60 * 60 * 12) -> str:
    settings = get_settings()
    payload = {"sub": username, "exp": int(time.time()) + ttl_seconds}
    payload_part = _b64encode(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(settings.secret_key.encode("utf-8"), payload_part.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_part}.{_b64encode(signature)}"


def verify_access_token(token: str) -> str:
    settings = get_settings()
    try:
        payload_part, signature_part = token.split(".", 1)
        expected = hmac.new(settings.secret_key.encode("utf-8"), payload_part.encode("ascii"), hashlib.sha256).digest()
        actual = _b64decode(signature_part)
        if not hmac.compare_digest(expected, actual):
            raise ValueError("bad signature")
        payload = json.loads(_b64decode(payload_part))
        if int(payload["exp"]) < int(time.time()):
            raise ValueError("expired")
        return str(payload["sub"])
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已过期，请重新登录") from exc


def require_admin(authorization: Annotated[str | None, Header()] = None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="需要管理员登录")
    username = verify_access_token(authorization.split(" ", 1)[1].strip())
    settings = get_settings()
    if username != settings.admin_username:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="没有管理员权限")
    return username


AdminUser = Annotated[str, Depends(require_admin)]

