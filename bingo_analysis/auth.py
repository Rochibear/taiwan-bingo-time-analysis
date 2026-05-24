from __future__ import annotations

import hashlib
import hmac
import json
import re
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


DEFAULT_ADMIN_EMAILS = ("killpmite@gmail.com",)
EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)


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


def extract_emails(text: str) -> set[str]:
    return {normalize_email(match.group(0)) for match in EMAIL_PATTERN.finditer(text)}


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


def mapping_get(mapping: Mapping[str, Any], key: str) -> Any:
    try:
        return mapping.get(key)
    except Exception:
        return None


def nested_mapping(
    secrets_mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any] | None:
    value = mapping_get(secrets_mapping, key)
    return value if isinstance(value, Mapping) else None


def secret_value(secrets_mapping: Mapping[str, Any], key: str) -> Any:
    direct_keys = {key, key.upper(), key.lower()}
    for candidate in direct_keys:
        value = mapping_get(secrets_mapping, candidate)
        if value not in (None, ""):
            return value

    for section_name in ("auth", "AUTH", "smtp", "SMTP", "email", "EMAIL"):
        section = nested_mapping(secrets_mapping, section_name)
        if section is None:
            continue
        section_keys = set(direct_keys)
        for prefix in ("AUTH_", "SMTP_", "EMAIL_"):
            if key.upper().startswith(prefix):
                stripped = key[len(prefix) :]
                section_keys.update({stripped, stripped.upper(), stripped.lower()})
        for candidate in section_keys:
            value = mapping_get(section, candidate)
            if value not in (None, ""):
                return value
    return None


def first_present(secrets_mapping: Mapping[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = secret_value(secrets_mapping, key)
        if value not in (None, ""):
            return value
    return None


def settings_from_secrets(secrets_mapping: Mapping[str, Any]) -> AuthSettings:
    allowed = first_present(
        secrets_mapping,
        [
            "AUTH_ALLOWED_EMAILS",
            "AUTH_ALLOWED_EMAIL",
            "ALLOWED_EMAILS",
            "ALLOWED_EMAIL",
        ],
    )
    smtp_username = str(
        first_present(secrets_mapping, ["SMTP_USERNAME", "EMAIL_USERNAME"]) or ""
    ).strip()
    admin_emails = first_present(
        secrets_mapping,
        [
            "AUTH_ADMIN_EMAILS",
            "AUTH_ADMIN_EMAIL",
            "ADMIN_EMAILS",
            "ADMIN_EMAIL",
        ],
    )
    if admin_emails is None:
        admin_emails = allowed
    if admin_emails is None and "@" in smtp_username:
        admin_emails = smtp_username
    admin_email_values = set(DEFAULT_ADMIN_EMAILS) | set(parse_email_list(admin_emails))
    admin_pin = str(
        first_present(secrets_mapping, ["ADMIN_PIN", "AUTH_ADMIN_PIN"]) or ""
    ).strip()
    return AuthSettings(
        enabled=bool_value(
            first_present(secrets_mapping, ["AUTH_ENABLED", "ENABLED"]),
            True,
        ),
        allowed_emails=parse_email_list(allowed),
        admin_emails=tuple(sorted(admin_email_values)),
        admin_pin=admin_pin,
        otp_minutes=max(
            1,
            int_value(
                first_present(secrets_mapping, ["AUTH_OTP_MINUTES", "OTP_MINUTES"]),
                10,
            ),
        ),
        session_hours=max(
            0.25,
            float_value(
                first_present(
                    secrets_mapping,
                    ["AUTH_SESSION_HOURS", "SESSION_HOURS"],
                ),
                24.0,
            ),
        ),
        resend_cooldown_seconds=max(
            10,
            int_value(
                first_present(
                    secrets_mapping,
                    ["AUTH_RESEND_COOLDOWN_SECONDS", "RESEND_COOLDOWN_SECONDS"],
                ),
                60,
            ),
        ),
        debug_otp=bool_value(
            first_present(secrets_mapping, ["AUTH_DEBUG_OTP", "DEBUG_OTP"]),
            False,
        ),
        otp_hash_secret=str(
            first_present(secrets_mapping, ["AUTH_OTP_SECRET", "OTP_SECRET"])
            or admin_pin
            or "streamlit-otp"
        ),
        smtp_host=str(
            first_present(secrets_mapping, ["SMTP_HOST", "EMAIL_HOST"]) or ""
        ).strip(),
        smtp_port=int_value(
            first_present(secrets_mapping, ["SMTP_PORT", "EMAIL_PORT"]),
            465,
        ),
        smtp_username=smtp_username,
        smtp_password=str(
            first_present(secrets_mapping, ["SMTP_PASSWORD", "EMAIL_PASSWORD"]) or ""
        ),
        smtp_from=str(
            first_present(secrets_mapping, ["SMTP_FROM", "EMAIL_FROM"]) or ""
        ).strip(),
        smtp_ssl=bool_value(
            first_present(secrets_mapping, ["SMTP_SSL", "EMAIL_SSL"]),
            True,
        ),
        smtp_starttls=bool_value(
            first_present(secrets_mapping, ["SMTP_STARTTLS", "EMAIL_STARTTLS"]),
            False,
        ),
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
    return (
        set(settings.allowed_emails)
        | set(settings.admin_emails)
        | load_dynamic_emails(dynamic_path)
    )


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
