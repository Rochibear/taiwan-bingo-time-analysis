from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from bingo_analysis import forecast as forecast_module
from bingo_analysis.analysis import (
    CHART_FILENAMES,
    NUMBERS,
    build_appearance_matrix,
    load_history,
)
from bingo_analysis.auth import (
    AuthConfigError,
    AuthSettings,
    DEFAULT_ADMIN_EMAILS,
    allowed_emails,
    generate_otp,
    hash_otp,
    load_dynamic_emails,
    normalize_email,
    save_dynamic_emails,
    send_otp_email,
    settings_from_secrets,
    smtp_configured,
    verify_otp,
)
from bingo_analysis.official import (
    OFFICIAL_NOTE,
    OfficialConfig,
    verify_history_with_official,
)
from bingo_analysis.pipeline import analyze_existing, run_pipeline
from bingo_analysis.scraper import ScrapeConfig

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("BINGO_DATA_DIR", PROJECT_ROOT))
OUTPUT_DIR = DATA_DIR / "output"
SUMMARY_PATH = OUTPUT_DIR / "analysis_summary.json"
OFFICIAL_SUMMARY_PATH = OUTPUT_DIR / "official_verification_summary.json"
OFFICIAL_DETAILS_PATH = OUTPUT_DIR / "official_verification.csv"
AUTH_USERS_PATH = DATA_DIR / "auth_users.json"
ANALYZE_COOLDOWN_KEY = "analyze_cooldown_until"
ANALYZE_COOLDOWN_SECONDS = 3
TAIPEI_TZ = ZoneInfo("Asia/Taipei")
STAR_MIN = getattr(forecast_module, "STAR_MIN", 1)
STAR_MAX = getattr(forecast_module, "STAR_MAX", 10)
build_forecast = forecast_module.build_forecast
backtest_star_selection = getattr(forecast_module, "backtest_star_selection", None)

CHART_TITLES = {
    "number_frequency.png": "1-80 號碼出現次數",
    "overlap_distribution.png": "相鄰兩期重複球數",
    "gap_distribution.png": "號碼再出現 gap 分布",
    "hourly_heatmap.png": "小時別偏號",
    "weekday_heatmap.png": "星期別偏號",
    "autocorrelation.png": "自相關",
    "fft_periodogram.png": "FFT 週期分析",
}

STAR_LABELS = {
    10: "十星（10 個號碼）",
    9: "九星（9 個號碼）",
    8: "八星（8 個號碼）",
    7: "七星（7 個號碼）",
    6: "六星（6 個號碼）",
    5: "五星（5 個號碼）",
    4: "四星（4 個號碼）",
    3: "三星（3 個號碼）",
    2: "二星（2 個號碼）",
    1: "一星（1 個號碼）",
}


def load_summary() -> dict[str, Any] | None:
    if not SUMMARY_PATH.exists():
        return None
    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_official_summary() -> dict[str, Any] | None:
    if not OFFICIAL_SUMMARY_PATH.exists():
        return None
    with OFFICIAL_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_auth_settings() -> AuthSettings:
    try:
        return settings_from_secrets(st.secrets)
    except Exception:
        return AuthSettings(enabled=True, admin_emails=DEFAULT_ADMIN_EMAILS)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes:
        return f"{minutes} 分 {remaining_seconds} 秒"
    return f"{remaining_seconds} 秒"


def current_auth_email() -> str | None:
    if not st.session_state.get("auth_authenticated"):
        return None
    expires_at = float(st.session_state.get("auth_expires_at", 0.0))
    if expires_at <= time.time():
        for key in ["auth_authenticated", "auth_email", "auth_expires_at"]:
            st.session_state.pop(key, None)
        return None
    return str(st.session_state.get("auth_email", ""))


def show_auth_admin_panel(settings: AuthSettings) -> None:
    if not settings.enabled:
        return
    current_email = current_auth_email()
    if not current_email or current_email not in set(settings.admin_emails):
        return

    with st.sidebar.expander("使用者管理", expanded=False):
        if not settings.admin_pin:
            st.info("管理區尚未完成設定。")
            return

        if not st.session_state.get("auth_admin_unlocked"):
            pin = st.text_input("管理 PIN", type="password", key="auth_admin_pin")
            if st.button("進入管理區", use_container_width=True):
                if pin == settings.admin_pin:
                    st.session_state["auth_admin_unlocked"] = True
                    st.rerun()
                st.error("管理 PIN 不正確。")
            return

        st.success("管理區已解鎖")
        if st.button("鎖定管理區", use_container_width=True):
            st.session_state.pop("auth_admin_unlocked", None)
            st.rerun()

        dynamic_emails = load_dynamic_emails(AUTH_USERS_PATH)
        initial_emails = set(settings.allowed_emails)
        admin_emails = set(settings.admin_emails)
        combined_emails = sorted(initial_emails | admin_emails | dynamic_emails)
        st.caption(f"目前允許 {len(combined_emails)} 個 Email。")
        if combined_emails:
            source_rows = [
                {
                    "Email": email,
                    "權限": "最高管理員" if email in admin_emails else "用戶",
                    "來源": (
                        "內建 / Secrets"
                        if email in admin_emails
                        else "Secrets"
                        if email in initial_emails
                        else "管理區"
                    ),
                }
                for email in combined_emails
            ]
            st.dataframe(pd.DataFrame(source_rows), hide_index=True)

        new_email = st.text_input("新增用戶 Email", key="auth_new_email")
        if st.button("新增用戶", use_container_width=True):
            email = normalize_email(new_email)
            if not email or "@" not in email:
                st.error("請輸入有效 Email。")
            elif email in admin_emails:
                st.info("此 Email 已是最高管理員。")
            else:
                dynamic_emails.add(email)
                save_dynamic_emails(AUTH_USERS_PATH, dynamic_emails)
                st.success(f"已新增用戶 {email}")
                st.rerun()

        removable = sorted(dynamic_emails - admin_emails)
        if removable:
            remove_email = st.selectbox("移除用戶 Email", removable)
            if st.button("移除用戶", use_container_width=True):
                dynamic_emails.discard(remove_email)
                save_dynamic_emails(AUTH_USERS_PATH, dynamic_emails)
                if st.session_state.get("auth_email") == remove_email:
                    for key in ["auth_authenticated", "auth_email", "auth_expires_at"]:
                        st.session_state.pop(key, None)
                st.success(f"已移除用戶 {remove_email}")
                st.rerun()
        else:
            st.caption("目前沒有可移除的一般用戶。")

        st.divider()
        st.caption("SMTP 測試")
        if not smtp_configured(settings):
            st.info("尚未讀到完整 SMTP 設定。")
        elif st.button("寄送測試信給我", use_container_width=True):
            try:
                send_otp_email(settings, current_email, "000000")
                st.success("測試信已寄出，請查看信箱。")
            except Exception as exc:
                st.error(
                    "SMTP 測試失敗："
                    f"{type(exc).__name__}: {str(exc)[:180]}"
                )


def store_pending_otp(settings: AuthSettings, email: str, code: str) -> None:
    now = time.time()
    st.session_state["auth_pending_email"] = email
    st.session_state["auth_otp_hash"] = hash_otp(
        email,
        code,
        settings.otp_hash_secret,
    )
    st.session_state["auth_otp_expires_at"] = now + settings.otp_minutes * 60
    st.session_state["auth_last_sent_at"] = now


def render_auth_gate(settings: AuthSettings) -> None:
    if not settings.enabled:
        return

    user_email = current_auth_email()
    if user_email:
        st.sidebar.success(f"已登入：{user_email}")
        if st.sidebar.button("登出", use_container_width=True):
            for key in ["auth_authenticated", "auth_email", "auth_expires_at"]:
                st.session_state.pop(key, None)
            st.rerun()
        return

    st.markdown("### Email OTP 登入")
    st.caption("請使用測試白名單內的 Email 取得一次性驗證碼。")

    allowed = allowed_emails(settings, AUTH_USERS_PATH)
    if not allowed:
        st.warning("登入設定尚未完成，請聯絡管理員。")

    email = normalize_email(st.text_input("Email", key="auth_login_email"))
    now = time.time()
    remaining_cooldown = (
        float(st.session_state.get("auth_last_sent_at", 0.0))
        + settings.resend_cooldown_seconds
        - now
    )
    send_disabled = remaining_cooldown > 0
    send_label = (
        f"寄送驗證碼（{format_seconds(remaining_cooldown)}）"
        if send_disabled
        else "寄送驗證碼"
    )

    if st.button("重新整理白名單", use_container_width=True):
        st.rerun()

    if st.button(send_label, disabled=send_disabled, use_container_width=True):
        if not email or "@" not in email:
            st.error("請先輸入有效 Email。")
        elif email not in allowed:
            st.error("此 Email 無法登入。")
        else:
            code = generate_otp()
            sent_or_debug = False
            try:
                if smtp_configured(settings):
                    send_otp_email(settings, email, code)
                    st.success("驗證碼已寄出，請查看信箱。")
                    sent_or_debug = True
                elif not settings.debug_otp:
                    raise AuthConfigError("SMTP 尚未設定，無法寄送驗證碼。")
            except AuthConfigError:
                st.error("驗證碼寄送失敗，請聯絡管理員。")
            except Exception:
                st.error("驗證碼寄送失敗，請聯絡管理員。")

            if settings.debug_otp:
                st.info(f"測試模式驗證碼：{code}")
                sent_or_debug = True
            if sent_or_debug:
                store_pending_otp(settings, email, code)

    pending_email = st.session_state.get("auth_pending_email")
    expires_at = float(st.session_state.get("auth_otp_expires_at", 0.0))
    if pending_email and expires_at > now:
        st.caption(f"驗證碼已送出給 {pending_email}，{format_seconds(expires_at - now)} 後失效。")
        code_input = st.text_input("6 位數驗證碼", max_chars=6, key="auth_code")
        if st.button("驗證登入", use_container_width=True):
            expected_hash = str(st.session_state.get("auth_otp_hash", ""))
            if verify_otp(
                str(pending_email),
                code_input.strip(),
                expected_hash,
                settings.otp_hash_secret,
            ):
                st.session_state["auth_authenticated"] = True
                st.session_state["auth_email"] = str(pending_email)
                st.session_state["auth_expires_at"] = (
                    time.time() + settings.session_hours * 3600
                )
                for key in ["auth_pending_email", "auth_otp_hash", "auth_otp_expires_at"]:
                    st.session_state.pop(key, None)
                st.rerun()
            st.error("驗證碼不正確。")
    elif pending_email:
        st.warning("驗證碼已過期，請重新寄送。")

    st.stop()


def cooldown_remaining_seconds(key: str) -> int:
    remaining = float(st.session_state.get(key, 0.0)) - time.time()
    return max(0, math.ceil(remaining))


def taipei_today():
    return datetime.now(TAIPEI_TZ).date()


def inject_submit_countdown(base_label: str, remaining_seconds: int) -> None:
    components.html(
        f"""
        <script>
        (() => {{
          const baseLabel = {json.dumps(base_label)};
          const storageKey = "bingoAnalyzeCooldownUntil";
          const serverCooldownUntil = Date.now() + {int(remaining_seconds)} * 1000;
          const buttons = Array.from(window.parent.document.querySelectorAll("button"));
          const button = buttons.find((candidate) => {{
            const text = (candidate.textContent || "").trim();
            return text === baseLabel || /^抓取並分析 \\(\\d+\\)$/.test(text);
          }});
          if (!button) {{
            return;
          }}
          if (button.dataset.bingoCountdownBound !== "1") {{
            button.dataset.bingoCountdownBound = "1";
            button.addEventListener("click", () => {{
              const cooldownUntil = Date.now() + 3000;
              window.parent.localStorage.setItem(storageKey, String(cooldownUntil));
              startCountdown(cooldownUntil);
            }}, true);
          }}

          function activeCooldownUntil() {{
            const localCooldownUntil = Number(
              window.parent.localStorage.getItem(storageKey) || 0
            );
            return Math.max(localCooldownUntil, serverCooldownUntil);
          }}

          const setLabel = () => {{
            const remaining = Math.ceil((activeCooldownUntil() - Date.now()) / 1000);
            if (remaining > 0) {{
              button.textContent = `${{baseLabel}} (${{remaining}})`;
              button.disabled = true;
              button.setAttribute("aria-disabled", "true");
            }} else {{
              button.textContent = baseLabel;
              button.disabled = false;
              button.removeAttribute("aria-disabled");
              window.parent.localStorage.removeItem(storageKey);
            }}
          }};

          function startCountdown(cooldownUntil) {{
            window.parent.localStorage.setItem(storageKey, String(cooldownUntil));
            setLabel();
            if (button.dataset.bingoCountdownTimer) {{
              window.parent.clearInterval(Number(button.dataset.bingoCountdownTimer));
            }}
            const timer = window.parent.setInterval(() => {{
              setLabel();
              if (Date.now() >= activeCooldownUntil()) {{
                window.parent.clearInterval(timer);
                delete button.dataset.bingoCountdownTimer;
                setLabel();
              }}
            }}, 250);
            button.dataset.bingoCountdownTimer = String(timer);
          }}

          const cooldownUntil = activeCooldownUntil();
          if (cooldownUntil > Date.now()) {{
            startCountdown(cooldownUntil);
          }} else {{
            setLabel();
          }}
        }})();
        </script>
        """,
        height=0,
    )


def number_table(items: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(items)
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "number": "號碼",
            "appearances": "出現次數",
            "appearance_rate": "出現率",
        }
    )
    frame["號碼"] = frame["號碼"].map(lambda value: f"{int(value):02d}")
    frame["出現率"] = frame["出現率"].map(lambda value: f"{value:.2%}")
    return frame


def chip_list(numbers: list[int], accent: str = "#0ea5a3") -> str:
    chips = "".join(
        f"<span class='number-chip' style='border-color:{accent}'>{number:02d}</span>"
        for number in numbers
    )
    return f"<div class='chip-row'>{chips}</div>"


def label_chip_list(labels: list[str], accent: str = "#f59e0b") -> str:
    chips = "".join(
        f"<span class='number-chip' style='border-color:{accent}'>{label}</span>"
        for label in labels
    )
    return f"<div class='chip-row'>{chips}</div>"


def pair_table(items: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(items)
    if frame.empty:
        return frame
    keep_columns = ["label", "history_count", "score"]
    frame = frame[keep_columns].rename(
        columns={
            "label": "連號",
            "history_count": "歷史同開次數",
            "score": "模型分數",
        }
    )
    frame["模型分數"] = frame["模型分數"].map(lambda value: f"{value:.4f}")
    return frame


def star_selection_table(items: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(items)
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "number": "號碼",
            "score": "綜合分數",
            "global_rate": "全期率",
            "recent_rate": "近期率",
            "hourly_rate": "同小時率",
            "current_gap": "目前 gap",
        }
    )
    frame["號碼"] = frame["號碼"].map(lambda value: f"{int(value):02d}")
    frame["綜合分數"] = frame["綜合分數"].map(lambda value: f"{value:.5f}")
    for column in ["全期率", "近期率", "同小時率"]:
        frame[column] = frame[column].map(lambda value: f"{value:.2%}")
    return frame


def backtest_table(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return details
    frame = details.copy()
    frame["selected_numbers"] = frame["selected_numbers"].map(
        lambda numbers: " ".join(f"{int(number):02d}" for number in numbers)
    )
    frame["hit_numbers"] = frame["hit_numbers"].map(
        lambda numbers: " ".join(f"{int(number):02d}" for number in numbers) or "－"
    )
    return frame.rename(
        columns={
            "draw_id": "期別",
            "date": "開獎日期",
            "time": "開獎時間",
            "selected_numbers": "建議號碼",
            "hit_numbers": "命中號碼",
            "hit_count": "命中數",
        }
    )


def normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0:
        return np.ones_like(values, dtype=float) / len(values)
    return values / total


def as_taipei_datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.to_pydatetime().replace(tzinfo=TAIPEI_TZ)
    return timestamp.tz_convert(TAIPEI_TZ).to_pydatetime()


def fallback_ranked_numbers(
    history: pd.DataFrame,
    matrix: pd.DataFrame,
    target_draw_at: datetime,
) -> pd.DataFrame:
    global_counts = matrix.sum(axis=0).reindex(NUMBERS).to_numpy(dtype=float) + 1.0
    recent_window = getattr(forecast_module, "RECENT_WINDOW_DRAWS", 300)
    recent = matrix.tail(min(recent_window, len(matrix)))
    recent_counts = recent.sum(axis=0).reindex(NUMBERS).to_numpy(dtype=float) + 1.0

    hour_mask = history["datetime"].dt.hour == target_draw_at.hour
    if hour_mask.any():
        hourly_counts = (
            matrix.loc[hour_mask].sum(axis=0).reindex(NUMBERS).to_numpy(dtype=float)
            + 1.0
        )
        hourly_draws = int(hour_mask.sum())
    else:
        hourly_counts = global_counts.copy()
        hourly_draws = len(history)

    current_gaps: list[int] = []
    for number in NUMBERS:
        appearances = np.flatnonzero(matrix[number].to_numpy())
        current_gaps.append(
            len(matrix)
            if len(appearances) == 0
            else len(matrix) - 1 - int(appearances[-1])
        )
    current_gap_values = np.asarray(current_gaps, dtype=float)
    gap_score = np.log1p(current_gap_values) + 1.0
    score = (
        0.45 * normalize_scores(global_counts)
        + 0.30 * normalize_scores(recent_counts)
        + 0.15 * normalize_scores(hourly_counts)
        + 0.10 * normalize_scores(gap_score)
    )

    return pd.DataFrame(
        {
            "number": NUMBERS,
            "score": score,
            "global_rate": (global_counts - 1.0) / max(len(history), 1),
            "recent_rate": (recent_counts - 1.0) / max(len(recent), 1),
            "hourly_rate": (hourly_counts - 1.0) / max(hourly_draws, 1),
            "current_gap": current_gap_values.astype(int),
        }
    ).sort_values(["score", "number"], ascending=[False, True])


def fallback_star_selection(history: pd.DataFrame, stars: int) -> dict[str, object]:
    if not STAR_MIN <= stars <= STAR_MAX:
        raise ValueError("stars must be between 1 and 10")

    next_draw_at = forecast_module.next_draw_datetime()
    frame = history.reset_index(drop=True)
    matrix = build_appearance_matrix(frame)
    ranked = fallback_ranked_numbers(frame, matrix, next_draw_at)
    selected = ranked.head(stars).copy()
    return {
        "stars": stars,
        "next_draw_label": next_draw_at.strftime("%Y-%m-%d %H:%M"),
        "selected_numbers": sorted(
            int(number) for number in selected["number"].tolist()
        ),
        "selected_details": selected.to_dict(orient="records"),
        "model_note": "全期頻率 45% + 近期頻率 30% + 同小時偏號 15% + 近期 gap 10%",
    }


def build_star_selection(history: pd.DataFrame, stars: int) -> dict[str, object]:
    builder = getattr(forecast_module, "build_star_selection", None)
    if builder:
        return builder(history, stars)
    return fallback_star_selection(history, stars)


def fallback_backtest_star_selection(
    history: pd.DataFrame,
    stars: int,
    evaluation_draws: int = 300,
    min_training_draws: int = 300,
    verified_draw_ids: set[str] | None = None,
) -> dict[str, object]:
    if not STAR_MIN <= stars <= STAR_MAX:
        raise ValueError("stars must be between 1 and 10")
    if evaluation_draws < 1:
        raise ValueError("evaluation_draws must be positive")

    ordered = history.sort_values(["datetime", "draw_id"]).reset_index(drop=True)
    if len(ordered) <= min_training_draws:
        raise ValueError(f"need more than {min_training_draws} draws for backtesting")

    candidate_indices = list(range(min_training_draws, len(ordered)))
    if verified_draw_ids is not None:
        verified = {str(draw_id) for draw_id in verified_draw_ids}
        candidate_indices = [
            index
            for index in candidate_indices
            if str(ordered.at[index, "draw_id"]) in verified
        ]
    candidate_indices = candidate_indices[-evaluation_draws:]

    full_matrix = build_appearance_matrix(ordered)
    rows: list[dict[str, object]] = []
    for index in candidate_indices:
        training = ordered.iloc[:index]
        training_matrix = full_matrix.iloc[:index]
        target = ordered.iloc[index]
        ranked = fallback_ranked_numbers(
            training,
            training_matrix,
            as_taipei_datetime(target["datetime"]),
        )
        selected = sorted(int(number) for number in ranked.head(stars)["number"])
        actual = {int(number) for number in target["numbers"]}
        hits = sorted(number for number in selected if number in actual)
        rows.append(
            {
                "draw_id": str(target["draw_id"]),
                "date": str(target["date"]),
                "time": str(target["time"]),
                "selected_numbers": selected,
                "hit_numbers": hits,
                "hit_count": len(hits),
            }
        )

    details = pd.DataFrame(rows)
    if details.empty:
        summary = {
            "stars": stars,
            "checked_count": 0,
            "mean_hits": 0.0,
            "hit_rate": 0.0,
            "at_least_four_hit_rate": 0.0,
            "zero_hit_rate": 0.0,
            "full_hit_rate": 0.0,
            "random_mean_hits": stars * 0.25,
            "lift_vs_random": None,
        }
        return {"summary": summary, "details": details}

    hit_counts = details["hit_count"].astype(int)
    random_mean_hits = stars * 20 / 80
    mean_hits = float(hit_counts.mean())
    summary = {
        "stars": stars,
        "checked_count": int(len(details)),
        "mean_hits": mean_hits,
        "hit_rate": float((hit_counts > 0).mean()),
        "at_least_four_hit_rate": float((hit_counts >= 4).mean()),
        "zero_hit_rate": float((hit_counts == 0).mean()),
        "full_hit_rate": float((hit_counts == stars).mean()),
        "random_mean_hits": float(random_mean_hits),
        "lift_vs_random": mean_hits / random_mean_hits if random_mean_hits else None,
    }
    return {"summary": summary, "details": details}


def history_draw_table(limit: int = 50) -> pd.DataFrame:
    history = load_history(DATA_DIR / "bingo_history.csv")
    frame = history.sort_values(["datetime", "draw_id"], ascending=False).head(limit)
    frame = frame[
        [
            "draw_id",
            "date",
            "time",
            "numbers",
            "super_number",
            "big_small",
            "odd_even",
        ]
    ].copy()
    frame["numbers"] = frame["numbers"].map(
        lambda numbers: " ".join(f"{int(number):02d}" for number in numbers)
    )
    frame["super_number"] = frame["super_number"].map(lambda value: f"{int(value):02d}")
    return frame.rename(
        columns={
            "draw_id": "期別",
            "date": "開獎日期",
            "time": "開獎時間",
            "numbers": "20 個號碼",
            "super_number": "超級獎號",
            "big_small": "猜大小",
            "odd_even": "猜單雙",
        }
    )


def show_history_draws() -> None:
    st.caption("完整資料仍會存成 bingo_history.csv；下方先顯示最新 50 期。")
    try:
        st.dataframe(
            history_draw_table(),
            hide_index=True,
            use_container_width=True,
        )
        csv_path = DATA_DIR / "bingo_history.csv"
        if csv_path.exists():
            st.download_button(
                "下載完整 bingo_history.csv",
                data=csv_path.read_bytes(),
                file_name="bingo_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception as exc:
        st.info(f"尚無可顯示的過往開獎解析：{exc}")


def official_status_table(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return details
    frame = details.copy()
    frame["status"] = frame["status"].map(
        {
            "verified": "通過",
            "mismatch": "不一致",
            "pending_official": "官方尚未更新",
        }
    ).fillna(frame["status"])
    return frame.rename(
        columns={
            "draw_id": "期別",
            "date": "開獎日期",
            "time": "開獎時間",
            "status": "校驗狀態",
            "mismatch_fields": "不一致欄位",
        }
    )


def show_official_verification() -> None:
    st.caption(OFFICIAL_NOTE)
    force_refresh = st.checkbox("重新下載官方年度檔", value=False)
    if st.button("執行官方資料校驗", use_container_width=True):
        try:
            with st.spinner("下載官方年度資料並比對中..."):
                result = verify_history_with_official(
                    DATA_DIR,
                    config=OfficialConfig(delay_seconds=1.0),
                    force_refresh=force_refresh,
                )
            st.success(
                "官方校驗完成："
                f"{result.summary['verified_count']} 通過，"
                f"{result.summary['mismatch_count']} 不一致，"
                f"{result.summary['pending_official_count']} 尚待官方更新。"
            )
        except Exception as exc:
            st.error(str(exc))

    summary = load_official_summary()
    if not summary:
        st.info("尚未執行官方資料校驗。")
        return

    metrics = st.columns(4)
    metrics[0].metric("官方最新日期", summary.get("latest_official_date") or "－")
    metrics[1].metric("校驗通過", summary.get("verified_count", 0))
    metrics[2].metric("不一致", summary.get("mismatch_count", 0))
    metrics[3].metric("官方尚未更新", summary.get("pending_official_count", 0))

    verification_rate = summary.get("verification_rate")
    if verification_rate is not None:
        st.caption(f"可比對資料通過率：{verification_rate:.2%}")

    if OFFICIAL_DETAILS_PATH.exists():
        details = pd.read_csv(OFFICIAL_DETAILS_PATH, dtype={"draw_id": "string"})
        focus = details[details["status"] != "verified"].tail(100)
        if focus.empty:
            focus = details.tail(100)
        st.dataframe(
            official_status_table(focus),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "下載官方校驗明細 CSV",
            data=OFFICIAL_DETAILS_PATH.read_bytes(),
            file_name="official_verification.csv",
            mime="text/csv",
            use_container_width=True,
        )


def verified_draw_ids_from_details() -> set[str] | None:
    if not OFFICIAL_DETAILS_PATH.exists():
        return None
    details = pd.read_csv(OFFICIAL_DETAILS_PATH, dtype={"draw_id": "string"})
    verified = details.loc[details["status"] == "verified", "draw_id"].dropna()
    if verified.empty:
        return None
    return set(verified.astype(str))


def adaptive_backtest_plan(history_count: int) -> dict[str, int]:
    if history_count < 60:
        return {
            "min_training_draws": 0,
            "max_evaluation_draws": 0,
            "default_evaluation_draws": 0,
            "slider_min": 0,
            "slider_step": 1,
        }

    min_training_draws = min(300, max(30, history_count // 2))
    max_evaluation_draws = max(1, history_count - min_training_draws)
    slider_step = 50 if max_evaluation_draws >= 100 else 10
    slider_min = 50 if max_evaluation_draws >= 100 else 10
    default_evaluation_draws = min(300, max_evaluation_draws)
    default_evaluation_draws = (
        slider_min
        + ((default_evaluation_draws - slider_min) // slider_step) * slider_step
    )
    return {
        "min_training_draws": min_training_draws,
        "max_evaluation_draws": max_evaluation_draws,
        "default_evaluation_draws": default_evaluation_draws,
        "slider_min": slider_min,
        "slider_step": slider_step,
    }


def show_prediction_backtest() -> None:
    backtest_builder = backtest_star_selection or fallback_backtest_star_selection
    if backtest_star_selection is None:
        st.caption("使用內建備援回測模組。")

    try:
        history = load_history(DATA_DIR / "bingo_history.csv")
    except Exception:
        st.info("預測回測需要先有 bingo_history.csv。請先按「抓取並分析」。")
        return

    plan = adaptive_backtest_plan(len(history))
    if plan["max_evaluation_draws"] <= 0:
        st.info(
            f"目前只有 {len(history)} 期資料，至少累積 60 期後才能做基本回測。"
        )
        return
    st.caption(
        f"目前歷史資料 {len(history)} 期；前 {plan['min_training_draws']} 期作為訓練，"
        f"最多可回測最近 {plan['max_evaluation_draws']} 期。"
    )
    if plan["min_training_draws"] < 300:
        st.warning("資料量尚少，回測結果波動會很大，請先當成粗略健康檢查。")

    controls = st.columns(2)
    with controls[0]:
        star_options = list(range(STAR_MAX, STAR_MIN - 1, -1))
        selected_stars = st.selectbox(
            "回測星數",
            options=star_options,
            format_func=lambda stars: STAR_LABELS[stars],
            key="backtest_star_count",
        )
    with controls[1]:
        evaluation_draws = st.slider(
            "回測最近期數",
            min_value=plan["slider_min"],
            max_value=plan["max_evaluation_draws"],
            value=plan["default_evaluation_draws"],
            step=plan["slider_step"],
        )

    verified_ids = verified_draw_ids_from_details()
    use_verified_only = False
    if verified_ids:
        use_verified_only = st.checkbox(
            "只回測官方已通過校驗的期數",
            value=True,
        )
    else:
        st.caption("尚未有官方校驗明細；此處先使用本機歷史資料回測。")

    try:
        result = backtest_builder(
            history,
            stars=selected_stars,
            evaluation_draws=evaluation_draws,
            min_training_draws=plan["min_training_draws"],
            verified_draw_ids=verified_ids if use_verified_only else None,
        )
    except Exception as exc:
        st.info(f"目前無法產生回測：{exc}")
        return

    summary = result["summary"]
    metrics = st.columns(5)
    metrics[0].metric("回測期數", summary["checked_count"])
    metrics[1].metric("平均命中", f"{summary['mean_hits']:.2f}")
    metrics[2].metric("至少中 1 號", f"{summary['hit_rate']:.2%}")
    metrics[3].metric(
        "至少中 4 號",
        f"{summary.get('at_least_four_hit_rate', 0.0):.2%}",
    )
    metrics[4].metric("零命中", f"{summary['zero_hit_rate']:.2%}")

    lift = summary.get("lift_vs_random")
    lift_label = f"{lift:.2f}x" if lift is not None else "－"
    st.caption(
        f"同星數隨機平均命中約 {summary['random_mean_hits']:.2f}；"
        f"模型相對隨機平均命中：{lift_label}。"
    )

    details = result["details"]
    st.dataframe(
        backtest_table(details.tail(50)),
        hide_index=True,
        use_container_width=True,
    )


def show_star_selection(history: pd.DataFrame) -> None:
    st.markdown("#### 建議選號區")
    star_options = list(range(STAR_MAX, STAR_MIN - 1, -1))
    selected_stars = st.selectbox(
        "選擇星數",
        options=star_options,
        format_func=lambda stars: STAR_LABELS[stars],
        key="star_selection_count",
    )
    selection = build_star_selection(history, selected_stars)
    st.caption(
        f"80 選 {selected_stars}｜推定下一期時間：{selection['next_draw_label']}"
    )
    st.markdown(
        chip_list(selection["selected_numbers"], "#6366f1"),
        unsafe_allow_html=True,
    )
    st.caption(f"模型：{selection['model_note']}")
    st.dataframe(
        star_selection_table(selection["selected_details"]),
        hide_index=True,
        use_container_width=True,
    )


def show_forecast() -> None:
    try:
        history = load_history(DATA_DIR / "bingo_history.csv")
        forecast = build_forecast(history)
    except Exception:
        st.info("預告區需要先有 bingo_history.csv。請先按「抓取並分析」。")
        return

    st.warning(forecast["disclaimer"])
    area_one, area_two = st.columns(2)

    with area_one:
        st.markdown("#### 下一期候選號碼")
        st.caption(f"推定下一期時間：{forecast['next_draw_label']}")
        st.markdown(
            chip_list(forecast["predicted_numbers"], "#0ea5a3"),
            unsafe_allow_html=True,
        )
        st.caption(f"模型：{forecast['model_note']}")

    with area_two:
        st.markdown("#### 預測連號")
        predicted_pairs = forecast["consecutive_in_prediction"]
        if predicted_pairs:
            st.caption("候選號碼內形成的連號")
            st.markdown(
                label_chip_list([pair["label"] for pair in predicted_pairs]),
                unsafe_allow_html=True,
            )
        else:
            st.caption("本次候選號碼沒有形成連號，改看模型連號候選。")
        st.dataframe(
            pair_table(forecast["consecutive_candidates"]),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    show_star_selection(history)


def show_summary(summary: dict[str, Any]) -> None:
    metrics = st.columns(4)
    metrics[0].metric("期數", summary["draw_count"])
    with metrics[1]:
        st.caption("日期")
        st.markdown(f"**{summary['date_start']}**  \n**{summary['date_end']}**")
    metrics[2].metric("相鄰重複平均", f"{summary.get('mean_overlap') or 0:.2f}")
    metrics[3].metric("中位 gap", f"{summary.get('median_gap') or 0:.1f}")

    with st.expander("預告區", expanded=True):
        show_forecast()

    with st.expander("過往開獎號碼解析", expanded=False):
        show_history_draws()

    with st.expander("官方資料校驗", expanded=False):
        show_official_verification()

    with st.expander("預測回測", expanded=False):
        show_prediction_backtest()

    with st.expander("熱號 / 冷號", expanded=False):
        hot, cold = st.columns(2)
        with hot:
            st.markdown("#### 熱號")
            st.dataframe(
                number_table(summary["hot_numbers"]),
                hide_index=True,
                use_container_width=True,
            )
        with cold:
            st.markdown("#### 冷號")
            st.dataframe(
                number_table(summary["cold_numbers"]),
                hide_index=True,
                use_container_width=True,
            )

    with st.expander("圖表", expanded=False):
        for left_index in range(0, len(CHART_FILENAMES), 2):
            chart_columns = st.columns(2)
            for column, filename in zip(
                chart_columns,
                CHART_FILENAMES[left_index : left_index + 2],
            ):
                chart_path = OUTPUT_DIR / filename
                if chart_path.exists():
                    column.image(str(chart_path), caption=CHART_TITLES[filename])

    with st.expander("FFT 高能量週期", expanded=False):
        period_frame = pd.DataFrame(summary.get("dominant_periods", []))
        if not period_frame.empty:
            period_frame = period_frame.rename(
                columns={"period_draws": "週期期數", "mean_power": "平均能量"}
            )
            st.dataframe(period_frame, hide_index=True, use_container_width=True)


st.set_page_config(
    page_title="賓果賓果時間分析",
    page_icon=":bar_chart:",
    layout="wide",
)
st.markdown(
    """
    <style>
    html {
        scroll-behavior: smooth;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.65rem 0 0.85rem;
    }
    .number-chip {
        border: 1px solid;
        border-radius: 999px;
        display: inline-flex;
        font-weight: 800;
        justify-content: center;
        min-width: 2.55rem;
        padding: 0.28rem 0.6rem;
    }
    div[data-testid="stExpander"] {
        border-radius: 8px;
        margin-bottom: 0.65rem;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 1rem;
        font-weight: 800;
    }
    .back-to-top {
        align-items: center;
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 999px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
        color: #ffffff !important;
        display: inline-flex;
        font-size: 0.78rem;
        font-weight: 800;
        height: 2.75rem;
        justify-content: center;
        letter-spacing: 0;
        position: fixed;
        right: 18px;
        text-decoration: none !important;
        top: 33vh;
        width: 2.75rem;
        z-index: 9999;
    }
    .back-to-top:hover {
        background: #1f2937;
        color: #ffffff !important;
        text-decoration: none !important;
    }
    @media (max-width: 640px) {
        .back-to-top {
            height: 2.5rem;
            right: 12px;
            top: 34vh;
            width: 2.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("賓果賓果時間分析")
st.caption("台灣 BINGO BINGO 歷史資料探索。圖表是診斷工具，不是未來開獎保證。")

auth_settings = load_auth_settings()
render_auth_gate(auth_settings)
show_auth_admin_panel(auth_settings)

st.markdown(
    '<div id="page-top"></div><a class="back-to-top" href="#page-top">TOP</a>',
    unsafe_allow_html=True,
)

choose_range = st.checkbox("指定日期範圍", key="choose_date_range")

with st.form("refresh"):
    st.subheader("抓取與分析")
    request_delay = st.number_input(
        "每日頁面抓取延遲（秒）",
        min_value=0.5,
        value=1.0,
        step=0.1,
    )
    if choose_range:
        today = taipei_today()
        start_column, end_column = st.columns(2)
        start_date = start_column.date_input(
            "起日",
            value=today - timedelta(days=7),
            max_value=today,
        )
        end_date = end_column.date_input("迄日", value=today, max_value=today)
        days = None
    else:
        days = int(st.number_input("來源頁面最近天數", min_value=1, value=30, step=1))
        start_date = None
        end_date = None
    submitted = st.form_submit_button(
        "抓取並分析",
        use_container_width=True,
    )

if submitted:
    remaining_cooldown = cooldown_remaining_seconds(ANALYZE_COOLDOWN_KEY)
    if remaining_cooldown > 0:
        st.session_state["refresh_notice"] = None
        st.session_state["refresh_warnings"] = []
        st.session_state["refresh_error"] = f"請再等 {remaining_cooldown} 秒後再抓取。"
        st.rerun()
    try:
        with st.spinner("抓取每日歷史資料並產生圖表中..."):
            result = run_pipeline(
                DATA_DIR,
                days=days,
                start_date=start_date,
                end_date=end_date,
                config=ScrapeConfig(delay_seconds=float(request_delay)),
            )
        st.session_state["refresh_notice"] = (
            f"已完成 {result.summary['draw_count']} 期分析。"
        )
        st.session_state["refresh_warnings"] = result.scrape_warnings[:3]
        st.session_state["refresh_error"] = None
    except Exception as exc:
        st.session_state["refresh_notice"] = None
        st.session_state["refresh_warnings"] = []
        st.session_state["refresh_error"] = str(exc)
    finally:
        st.session_state[ANALYZE_COOLDOWN_KEY] = time.time() + ANALYZE_COOLDOWN_SECONDS
        st.rerun()

if st.session_state.get("refresh_notice"):
    st.success(st.session_state["refresh_notice"])
for warning in st.session_state.get("refresh_warnings", []):
    st.warning(warning)
if st.session_state.get("refresh_error"):
    st.error(st.session_state["refresh_error"])

summary = load_summary()
if summary:
    if st.button("以現有 CSV 重算圖表", use_container_width=True):
        try:
            with st.spinner("重建圖表中..."):
                summary = analyze_existing(DATA_DIR)
            st.success(f"已重建 {summary['draw_count']} 期圖表。")
        except Exception as exc:
            st.error(str(exc))
    show_summary(summary)
else:
    st.info("還沒有雲端分析結果。先按上方的「抓取並分析」。")

inject_submit_countdown("抓取並分析", cooldown_remaining_seconds(ANALYZE_COOLDOWN_KEY))
