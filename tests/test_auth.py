from pathlib import Path

from bingo_analysis.auth import (
    DEFAULT_ADMIN_EMAILS,
    allowed_emails,
    generate_otp,
    hash_otp,
    load_dynamic_emails,
    normalize_email,
    parse_email_list,
    save_dynamic_emails,
    settings_from_secrets,
    verify_otp,
)


def test_parse_email_list_normalizes_values() -> None:
    assert parse_email_list(" A@Example.com, b@example.com\n") == (
        "a@example.com",
        "b@example.com",
    )


def test_dynamic_email_store_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "auth_users.json"

    save_dynamic_emails(path, {"Friend@Example.com"})

    assert load_dynamic_emails(path) == {"friend@example.com"}


def test_allowed_emails_merges_secrets_and_dynamic_file(tmp_path: Path) -> None:
    path = tmp_path / "auth_users.json"
    save_dynamic_emails(path, {"friend@example.com"})
    settings = settings_from_secrets(
        {
            "AUTH_ENABLED": True,
            "AUTH_ALLOWED_EMAILS": ["Owner@Example.com"],
            "AUTH_ADMIN_EMAILS": ["Owner@Example.com"],
            "ADMIN_PIN": "1234",
        }
    )

    assert settings.admin_emails == tuple(
        sorted({*DEFAULT_ADMIN_EMAILS, "owner@example.com"})
    )
    assert allowed_emails(settings, path) == {
        *DEFAULT_ADMIN_EMAILS,
        "owner@example.com",
        "friend@example.com",
    }


def test_admin_emails_are_allowed_to_login_without_general_whitelist(
    tmp_path: Path,
) -> None:
    settings = settings_from_secrets(
        {
            "AUTH_ADMIN_EMAILS": ["Admin@Example.com"],
            "ADMIN_PIN": "1234",
        }
    )

    assert allowed_emails(settings, tmp_path / "auth_users.json") == {
        *DEFAULT_ADMIN_EMAILS,
        "admin@example.com",
    }


def test_admin_email_aliases_are_supported(tmp_path: Path) -> None:
    settings = settings_from_secrets(
        {
            "AUTH_ADMIN_EMAIL": "Admin@Example.com",
            "ADMIN_PIN": "1234",
        }
    )

    assert settings.admin_emails == tuple(
        sorted({*DEFAULT_ADMIN_EMAILS, "admin@example.com"})
    )
    assert allowed_emails(settings, tmp_path / "auth_users.json") == {
        *DEFAULT_ADMIN_EMAILS,
        "admin@example.com",
    }


def test_nested_auth_secrets_are_supported(tmp_path: Path) -> None:
    settings = settings_from_secrets(
        {
            "auth": {
                "admin_email": "Admin@Example.com",
                "debug_otp": True,
            },
            "smtp": {
                "host": "smtp.example.com",
                "username": "mail@example.com",
                "password": "secret",
            },
        }
    )

    assert settings.admin_emails == tuple(
        sorted({*DEFAULT_ADMIN_EMAILS, "admin@example.com"})
    )
    assert settings.debug_otp is True
    assert settings.smtp_host == "smtp.example.com"
    assert settings.smtp_username == "mail@example.com"
    assert allowed_emails(settings, tmp_path / "auth_users.json") == {
        *DEFAULT_ADMIN_EMAILS,
        "admin@example.com",
    }


def test_smtp_username_bootstraps_admin_email(tmp_path: Path) -> None:
    settings = settings_from_secrets(
        {
            "SMTP_USERNAME": "Admin@Example.com",
            "ADMIN_PIN": "1234",
        }
    )

    assert settings.admin_emails == tuple(
        sorted({*DEFAULT_ADMIN_EMAILS, "admin@example.com"})
    )
    assert allowed_emails(settings, tmp_path / "auth_users.json") == {
        *DEFAULT_ADMIN_EMAILS,
        "admin@example.com",
    }


def test_default_admin_is_always_highest_admin(tmp_path: Path) -> None:
    settings = settings_from_secrets({})

    assert "killpmite@gmail.com" in settings.admin_emails
    assert allowed_emails(settings, tmp_path / "auth_users.json") == {
        "killpmite@gmail.com",
    }


def test_auth_enabled_defaults_to_true_unless_disabled() -> None:
    assert settings_from_secrets({}).enabled is True
    assert settings_from_secrets({"AUTH_ENABLED": False}).enabled is False


def test_otp_hash_verification() -> None:
    email = normalize_email("Owner@Example.com")
    code = generate_otp()
    digest = hash_otp(email, code, "secret")

    assert verify_otp(email, code, digest, "secret")
    assert not verify_otp(email, "000000", digest, "secret")
