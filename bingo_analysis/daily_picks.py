from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

MAX_PICK_SETS = 3
MIN_PICK_NUMBERS = 5
MAX_PICK_NUMBERS = 10
MIN_NOTIFY_THRESHOLD = 5
MAX_NOTIFY_THRESHOLD = 10
TAIPEI_TIMEZONE = "Asia/Taipei"


class DailyPickError(ValueError):
    """Raised when a daily pick cannot be saved."""


def user_key(email: str) -> str:
    return str(email or "local-user").strip().lower() or "local-user"


def empty_store(day: object) -> dict[str, Any]:
    return {"date": str(day), "users": {}}


def parse_number_text(text: str) -> list[int]:
    tokens = [token for token in re.split(r"\D+", text or "") if token]
    return normalize_pick_numbers(tokens)


def normalize_pick_numbers(numbers: Iterable[object]) -> list[int]:
    cleaned: set[int] = set()
    for item in numbers:
        try:
            number = int(str(item).strip())
        except (TypeError, ValueError) as exc:
            raise DailyPickError(f"無法解析號碼：{item!r}") from exc
        if not 1 <= number <= 80:
            raise DailyPickError("號碼必須介於 1 到 80。")
        cleaned.add(number)
    return sorted(cleaned)


def validate_pick(numbers: Iterable[object], threshold: int) -> tuple[list[int], int]:
    selected = normalize_pick_numbers(numbers)
    if not MIN_PICK_NUMBERS <= len(selected) <= MAX_PICK_NUMBERS:
        raise DailyPickError("每組鎖號請選 5 到 10 個號碼。")
    threshold = int(threshold)
    if not MIN_NOTIFY_THRESHOLD <= threshold <= MAX_NOTIFY_THRESHOLD:
        raise DailyPickError("通知門檻必須介於 5 到 10。")
    if threshold > len(selected):
        raise DailyPickError("通知門檻不能大於本組號碼數。")
    return selected, threshold


def load_daily_pick_store(path: Path, day: object) -> dict[str, Any]:
    if not path.exists():
        return empty_store(day)
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return empty_store(day)
    if str(payload.get("date")) != str(day):
        return empty_store(day)
    users = payload.get("users")
    if not isinstance(users, dict):
        return empty_store(day)
    return {"date": str(day), "users": users}


def save_daily_pick_store(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def get_user_picks(payload: dict[str, Any], email: str) -> list[dict[str, Any]]:
    picks = payload.setdefault("users", {}).get(user_key(email), [])
    return [pick for pick in picks if isinstance(pick, dict)][:MAX_PICK_SETS]


def set_user_picks(
    payload: dict[str, Any],
    email: str,
    picks: list[dict[str, Any]],
) -> None:
    payload.setdefault("users", {})[user_key(email)] = picks[:MAX_PICK_SETS]


def add_daily_pick(
    payload: dict[str, Any],
    email: str,
    numbers: Iterable[object],
    threshold: int,
    source: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    picks = get_user_picks(payload, email)
    if len(picks) >= MAX_PICK_SETS:
        raise DailyPickError("每天最多只能鎖 3 組。")
    selected, threshold = validate_pick(numbers, threshold)
    pick = {
        "numbers": selected,
        "threshold": threshold,
        "source": source,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "notified_draw_ids": [],
    }
    picks.append(pick)
    set_user_picks(payload, email, picks)
    return pick


def remove_daily_pick(payload: dict[str, Any], email: str, index: int) -> None:
    picks = get_user_picks(payload, email)
    if 0 <= index < len(picks):
        del picks[index]
    set_user_picks(payload, email, picks)


def _taipei_timestamp(value: object) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(TAIPEI_TIMEZONE)
    return timestamp.tz_convert(TAIPEI_TIMEZONE)


def _is_after_pick_created(pick: dict[str, Any], draw: pd.Series) -> bool:
    created_at = _taipei_timestamp(pick.get("created_at"))
    if created_at is None:
        return True
    draw_value = draw["datetime"] if "datetime" in draw else f"{draw['date']} {draw['time']}"
    draw_at = _taipei_timestamp(draw_value)
    if draw_at is None:
        return True
    return draw_at >= created_at


def daily_match_rows(
    picks: list[dict[str, Any]],
    history: pd.DataFrame,
    day: object,
) -> list[dict[str, Any]]:
    if history.empty or not picks:
        return []
    frame = history.loc[history["date"].astype(str) == str(day)].copy()
    if frame.empty:
        return []
    sort_columns = ["datetime", "draw_id"] if "datetime" in frame else ["date", "time"]
    frame = frame.sort_values(sort_columns, ascending=False)
    rows: list[dict[str, Any]] = []
    for pick_index, pick in enumerate(picks):
        selected = set(normalize_pick_numbers(pick.get("numbers", [])))
        threshold = int(pick.get("threshold", MIN_NOTIFY_THRESHOLD))
        for _, draw in frame.iterrows():
            if not _is_after_pick_created(pick, draw):
                continue
            actual_numbers = normalize_pick_numbers(draw["numbers"])
            hits = sorted(selected & set(actual_numbers))
            rows.append(
                {
                    "pick_index": pick_index,
                    "draw_id": str(draw["draw_id"]),
                    "date": str(draw["date"]),
                    "time": str(draw["time"]),
                    "selected_numbers": sorted(selected),
                    "hit_numbers": hits,
                    "hit_count": len(hits),
                    "threshold": threshold,
                    "reached": len(hits) >= threshold,
                }
            )
    return rows


def pending_notification_rows(
    picks: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for row in match_rows:
        if not row["reached"]:
            continue
        pick = picks[int(row["pick_index"])]
        notified = {str(draw_id) for draw_id in pick.get("notified_draw_ids", [])}
        if str(row["draw_id"]) not in notified:
            pending.append(row)
    return pending


def mark_notified(picks: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    for row in rows:
        pick = picks[int(row["pick_index"])]
        notified = {str(draw_id) for draw_id in pick.get("notified_draw_ids", [])}
        notified.add(str(row["draw_id"]))
        pick["notified_draw_ids"] = sorted(notified)
