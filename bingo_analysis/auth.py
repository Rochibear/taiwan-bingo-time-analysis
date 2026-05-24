from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Mapping


class AuthConfigError(RuntimeError):
    """Raised when authentication is enabled but required config is missing."""


@dataclass(frozen=True)
class AuthSettings:
    enabled: bool = False
    allowed_emails: tuple[str, ...] = ()
    admin_emails: tuple[str, ...] = ()
    admin_pin: str = ""
    otp_minutes: int = 10
    session_hours: float = 24.0
    resend_cooldown_seconds: int = 60
    debug_otp: bool = False
    otp_hash_secret: str = ""
    smtp_host: str = ""
    smtp_port: int = 465
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_ssl: bool = True
    smtp_starttls: bool = False


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def normalize_email(email: str) -> str:
    return email.strip().lower()


def parse_email_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_values = value.replace("\n", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]

    emails = sorted(
        {
            normalize_email(str(item))
            for item in raw_values
            if normalize_email(str(item))
        }
    )
    return tuple(emails)


def int_value(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def settings_from_secrets(secrets_mapping: Mapping[str, Any]) -> AuthSettings:
    allowed = secrets_mapping.get(
        "AUTH_ALLOWED_EMAILS",
        secrets_mapping.get("ALLOWED_EMAILS"),
    )
    admin_emails = secrets_mapping.get("AUTH_ADMIN_EMAILS", allowed)
    admin_pin = str(secrets_mapping.get("ADMIN_PIN", "")).strip()
    return AuthSettings(
        enabled=bool_value(secrets_mapping.get("AUTH_ENABLED"), True),
        allowed_emails=parse_email_list(allowed),
        admin_emails=parse_email_list(admin_emails),
        admin_pin=admin_pin,
        otp_minutes=max(1, int_value(secrets_mapping.get("AUTH_OTP_MINUTES"), 10)),
        session_hours=max(
            0.25,
            float_value(secrets_mapping.get("AUTH_SESSION_HOURS"), 24.0),
        ),
        resend_cooldown_seconds=max(
            10,
            int_value(secrets_mapping.get("AUTH_RESEND_COOLDOWN_SECONDS"), 60),
        ),
        debug_otp=bool_value(secrets_mapping.get("AUTH_DEBUG_OTP"), False),
        otp_hash_secret=str(
            secrets_mapping.get("AUTH_OTP_SECRET", admin_pin or "streamlit-otp")
        ),
        smtp_host=str(secrets_mapping.get("SMTP_HOST", "")).strip(),
        smtp_port=int_value(secrets_mapping.get("SMTP_PORT"), 465),
        smtp_username=str(secrets_mapping.get("SMTP_USERNAME", "")).strip(),
        smtp_password=str(secrets_mapping.get("SMTP_PASSWORD", "")),
        smtp_from=str(secrets_mapping.get("SMTP_FROM", "")).strip(),
        smtp_ssl=bool_value(secrets_mapping.get("SMTP_SSL"), True),
        smtp_starttls=bool_value(secrets_mapping.get("SMTP_STARTTLS"), False),
    )


def load_dynamic_emails(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return set(parse_email_list(payload.get("allowed_emails", [])))


def save_dynamic_emails(path: Path, emails: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "allowed_emails": sorted(normalize_email(email) for email in emails if email),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def allowed_emails(settings: AuthSettings, dynamic_path: Path) -> set[str]:
    return set(settings.allowed_emails) | load_dynamic_emails(dynamic_path)


def generate_otp(length: int = 6) -> str:
    upper_bound = 10**length
    return f"{secrets.randbelow(upper_bound):0{length}d}"


def hash_otp(email: str, code: str, secret: str) -> str:
    message = f"{normalize_email(email)}:{code}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def verify_otp(email: str, code: str, expected_hash: str, secret: str) -> bool:
    actual_hash = hash_otp(email, code, secret)
    return hmac.compare_digest(actual_hash, expected_hash)


def smtp_configured(settings: AuthSettings) -> bool:
    return bool(
        settings.smtp_host
        and settings.smtp_port
        and settings.smtp_username
        and settings.smtp_password
    )


def send_otp_email(settings: AuthSettings, email: str, code: str) -> None:
    if not smtp_configured(settings):
        raise AuthConfigError("SMTP is not configured")

    sender = settings.smtp_from or settings.smtp_username
    message = EmailMessage()
    message["Subject"] = "賓果賓果登入驗證碼"
    message["From"] = sender
    message["To"] = email
    message.set_content(
        "\n".join(
            [
                f"你的登入驗證碼是：{code}",
                "",
                f"此驗證碼 {settings.otp_minutes} 分鐘內有效。",
                "如果不是你本人操作，請忽略這封信。",
            ]
        )
    )

    if settings.smtp_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(
            settings.smtp_host,
            settings.smtp_port,
            timeout=20,
            context=context,
        ) as server:
            server.login(settings.smtp_username, settings.smtp_password)
            server.send_message(message)
        return

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as server:
        if settings.smtp_starttls:
            server.starttls(context=ssl.create_default_context())
        server.login(settings.smtp_username, settings.smtp_password)
        server.send_message(message)
