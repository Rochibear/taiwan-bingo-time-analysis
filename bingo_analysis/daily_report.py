from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .analysis import build_appearance_matrix
from .forecast import DEFAULT_STRATEGY, choose_adaptive_strategy, scored_numbers

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
LAST_DRAW_TIME = "23:55"
TEN_STAR_COUNT = 10
CANDIDATE_COUNT = 20

REPORT_COLUMNS = [
    "draw_id",
    "date",
    "time",
    "actual_numbers",
    "super_number",
    "candidate20_numbers",
    "candidate20_hit_numbers",
    "candidate20_hit_count",
    "ten_star_numbers",
    "ten_star_hit_numbers",
    "ten_star_hit_count",
    "candidate20_strategy",
    "ten_star_strategy",
]


def day_key(day: object) -> str:
    if isinstance(day, datetime):
        return day.astimezone(TAIPEI_TZ).date().isoformat()
    if isinstance(day, date):
        return day.isoformat()
    return str(day)


def daily_report_path(output_dir: Path, day: object) -> Path:
    return output_dir / f"daily_prediction_log_{day_key(day)}.csv"


def _number_text(numbers: Any) -> str:
    return " ".join(f"{int(number):02d}" for number in sorted(numbers))


def _as_taipei_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(TAIPEI_TZ)
    return timestamp.tz_convert(TAIPEI_TZ)


def _ordered_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history.copy()
    frame = history.copy()
    if "datetime" not in frame:
        frame["datetime"] = pd.to_datetime(
            frame["date"].astype(str) + " " + frame["time"].astype(str),
            errors="raise",
        )
    frame["datetime"] = frame["datetime"].map(_as_taipei_timestamp)
    return frame.sort_values(["datetime", "draw_id"]).reset_index(drop=True)


def day_draws(history: pd.DataFrame, day: object) -> pd.DataFrame:
    frame = _ordered_history(history)
    if frame.empty:
        return frame
    return frame.loc[frame["date"].astype(str) == day_key(day)].copy()


def has_final_draw(history: pd.DataFrame, day: object) -> bool:
    frame = day_draws(history, day)
    if frame.empty:
        return False
    return bool((frame["time"].astype(str).str.slice(0, 5) == LAST_DRAW_TIME).any())


def _strategy_for_training(history: pd.DataFrame, pick_count: int):
    if history.empty:
        return DEFAULT_STRATEGY, []
    matrix = build_appearance_matrix(history)
    return choose_adaptive_strategy(history, matrix, pick_count)


def build_daily_prediction_log(
    history: pd.DataFrame,
    day: object,
    *,
    ten_star_count: int = TEN_STAR_COUNT,
) -> pd.DataFrame:
    if ten_star_count < 1:
        raise ValueError("ten_star_count must be positive")

    ordered = _ordered_history(history)
    if ordered.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    day_value = day_key(day)
    day_positions = ordered.index[ordered["date"].astype(str) == day_value].tolist()
    if not day_positions:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    first_day_position = int(day_positions[0])
    pre_day = ordered.iloc[:first_day_position]
    candidate_strategy, _ = _strategy_for_training(pre_day, CANDIDATE_COUNT)
    ten_star_strategy, _ = _strategy_for_training(pre_day, ten_star_count)
    full_matrix = build_appearance_matrix(ordered)

    rows: list[dict[str, object]] = []
    for position in day_positions:
        training = ordered.iloc[:position]
        training_matrix = full_matrix.iloc[:position]
        target = ordered.iloc[position]
        target_at = _as_taipei_timestamp(target["datetime"]).to_pydatetime()

        candidate_ranked = scored_numbers(
            training,
            training_matrix,
            target_at,
            candidate_strategy,
        )
        ten_star_ranked = scored_numbers(
            training,
            training_matrix,
            target_at,
            ten_star_strategy,
        )
        candidate_numbers = sorted(
            int(number) for number in candidate_ranked.head(CANDIDATE_COUNT)["number"]
        )
        ten_star_numbers = sorted(
            int(number) for number in ten_star_ranked.head(ten_star_count)["number"]
        )
        actual_numbers = sorted(int(number) for number in target["numbers"])
        actual_set = set(actual_numbers)
        candidate_hits = sorted(number for number in candidate_numbers if number in actual_set)
        ten_star_hits = sorted(number for number in ten_star_numbers if number in actual_set)

        rows.append(
            {
                "draw_id": str(target["draw_id"]),
                "date": str(target["date"]),
                "time": str(target["time"])[:5],
                "actual_numbers": _number_text(actual_numbers),
                "super_number": int(target["super_number"]),
                "candidate20_numbers": _number_text(candidate_numbers),
                "candidate20_hit_numbers": _number_text(candidate_hits),
                "candidate20_hit_count": len(candidate_hits),
                "ten_star_numbers": _number_text(ten_star_numbers),
                "ten_star_hit_numbers": _number_text(ten_star_hits),
                "ten_star_hit_count": len(ten_star_hits),
                "candidate20_strategy": candidate_strategy.label,
                "ten_star_strategy": ten_star_strategy.label,
            }
        )

    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def save_daily_prediction_log(path: Path, report: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False, encoding="utf-8-sig")


def daily_report_summary(report: pd.DataFrame) -> dict[str, object]:
    if report.empty:
        return {
            "draw_count": 0,
            "candidate20_mean_hits": 0.0,
            "ten_star_mean_hits": 0.0,
            "candidate20_best_hit_count": 0,
            "candidate20_best_time": "",
            "ten_star_best_hit_count": 0,
            "ten_star_best_time": "",
            "at_least_four_ten_star_rate": 0.0,
        }

    candidate_hits = report["candidate20_hit_count"].astype(int)
    ten_star_hits = report["ten_star_hit_count"].astype(int)
    candidate_best = report.loc[candidate_hits.idxmax()]
    ten_star_best = report.loc[ten_star_hits.idxmax()]
    return {
        "draw_count": int(len(report)),
        "candidate20_mean_hits": float(candidate_hits.mean()),
        "ten_star_mean_hits": float(ten_star_hits.mean()),
        "candidate20_best_hit_count": int(candidate_best["candidate20_hit_count"]),
        "candidate20_best_time": str(candidate_best["time"]),
        "ten_star_best_hit_count": int(ten_star_best["ten_star_hit_count"]),
        "ten_star_best_time": str(ten_star_best["time"]),
        "at_least_four_ten_star_rate": float((ten_star_hits >= 4).mean()),
    }


def load_daily_report_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"sent_reports": {}}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"sent_reports": {}}
    if not isinstance(payload, dict):
        return {"sent_reports": {}}
    sent_reports = payload.get("sent_reports")
    if not isinstance(sent_reports, dict):
        payload["sent_reports"] = {}
    return payload


def save_daily_report_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def report_sent_record(
    payload: dict[str, Any],
    day: object,
    email: str,
) -> dict[str, Any] | None:
    sent_reports = payload.setdefault("sent_reports", {})
    day_reports = sent_reports.get(day_key(day), {})
    if not isinstance(day_reports, dict):
        return None
    record = day_reports.get(str(email).strip().lower())
    return record if isinstance(record, dict) else None


def mark_report_sent(
    payload: dict[str, Any],
    day: object,
    email: str,
    report_path: Path,
    row_count: int,
) -> None:
    sent_reports = payload.setdefault("sent_reports", {})
    day_reports = sent_reports.setdefault(day_key(day), {})
    day_reports[str(email).strip().lower()] = {
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "path": str(report_path),
        "row_count": int(row_count),
    }


def last_draw_deadline(day: object) -> datetime:
    parsed_day = date.fromisoformat(day_key(day))
    return datetime.combine(parsed_day, time(23, 55), tzinfo=TAIPEI_TZ)
