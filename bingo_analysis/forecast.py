from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .analysis import NUMBERS, build_appearance_matrix

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
DRAW_START_MINUTE = 7 * 60 + 5
DRAW_END_MINUTE = 23 * 60 + 55
DRAW_INTERVAL_MINUTES = 5
RECENT_WINDOW_DRAWS = 300

DISCLAIMER = (
    "免責聲明：本預告區只用歷史資料做統計與娛樂性候選，"
    "不代表、保證或暗示未來開獎結果。請勿把它當成投注建議。"
)


def draw_minutes() -> list[int]:
    return list(
        range(
            DRAW_START_MINUTE,
            DRAW_END_MINUTE + 1,
            DRAW_INTERVAL_MINUTES,
        )
    )


def next_draw_datetime(now: datetime | None = None) -> datetime:
    current = now or datetime.now(TAIPEI_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=TAIPEI_TZ)
    else:
        current = current.astimezone(TAIPEI_TZ)

    for minute in draw_minutes():
        candidate = current.replace(
            hour=minute // 60,
            minute=minute % 60,
            second=0,
            microsecond=0,
        )
        if candidate > current:
            return candidate

    tomorrow = current + timedelta(days=1)
    return tomorrow.replace(hour=7, minute=5, second=0, microsecond=0)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0:
        return np.ones_like(values, dtype=float) / len(values)
    return values / total


def _stable_seed(history: pd.DataFrame, next_draw_at: datetime) -> int:
    latest_draw = str(history["draw_id"].iloc[-1]) if "draw_id" in history else ""
    latest_time = str(history["datetime"].iloc[-1]) if "datetime" in history else ""
    seed_source = f"{next_draw_at.isoformat()}|{len(history)}|{latest_draw}|{latest_time}"
    digest = hashlib.sha256(seed_source.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _weighted_prediction(
    history: pd.DataFrame,
    matrix: pd.DataFrame,
    next_draw_at: datetime,
) -> list[int]:
    global_counts = matrix.sum(axis=0).reindex(NUMBERS).to_numpy(dtype=float) + 1.0
    recent = matrix.tail(min(RECENT_WINDOW_DRAWS, len(matrix)))
    recent_counts = recent.sum(axis=0).reindex(NUMBERS).to_numpy(dtype=float) + 1.0

    hour_mask = history["datetime"].dt.hour == next_draw_at.hour
    if hour_mask.any():
        hourly_counts = matrix.loc[hour_mask].sum(axis=0).reindex(NUMBERS).to_numpy(dtype=float) + 1.0
    else:
        hourly_counts = global_counts.copy()

    weights = (
        0.50 * _normalize(global_counts)
        + 0.35 * _normalize(recent_counts)
        + 0.15 * _normalize(hourly_counts)
    )
    weights = weights / weights.sum()

    rng = np.random.default_rng(_stable_seed(history, next_draw_at))
    selected = rng.choice(NUMBERS, size=20, replace=False, p=weights)
    return sorted(int(number) for number in selected)


def consecutive_pairs(numbers: list[int]) -> list[dict[str, object]]:
    number_set = set(numbers)
    pairs: list[dict[str, object]] = []
    for number in range(1, 80):
        if number in number_set and number + 1 in number_set:
            pairs.append(
                {
                    "numbers": [number, number + 1],
                    "label": f"{number:02d}-{number + 1:02d}",
                }
            )
    return pairs


def _pair_counts(rows: pd.Series) -> dict[int, int]:
    counts = {number: 0 for number in range(1, 80)}
    for numbers in rows:
        number_set = set(numbers)
        for number in range(1, 80):
            if number in number_set and number + 1 in number_set:
                counts[number] += 1
    return counts


def consecutive_candidates(
    history: pd.DataFrame,
    predicted_numbers: list[int],
    limit: int = 8,
) -> list[dict[str, object]]:
    all_counts = _pair_counts(history["numbers"])
    recent_history = history.tail(min(RECENT_WINDOW_DRAWS, len(history)))
    recent_counts = _pair_counts(recent_history["numbers"])
    predicted = set(predicted_numbers)

    candidates: list[dict[str, object]] = []
    for number in range(1, 80):
        global_rate = (all_counts[number] + 1) / (len(history) + 2)
        recent_rate = (recent_counts[number] + 1) / (len(recent_history) + 2)
        score = 0.65 * global_rate + 0.35 * recent_rate
        if number in predicted and number + 1 in predicted:
            score *= 1.35
        candidates.append(
            {
                "numbers": [number, number + 1],
                "label": f"{number:02d}-{number + 1:02d}",
                "history_count": all_counts[number],
                "score": float(score),
            }
        )

    return sorted(candidates, key=lambda row: row["score"], reverse=True)[:limit]


def build_forecast(
    history: pd.DataFrame,
    now: datetime | None = None,
) -> dict[str, object]:
    if history.empty:
        raise ValueError("history is empty")

    next_draw_at = next_draw_datetime(now)
    matrix = build_appearance_matrix(history)
    predicted_numbers = _weighted_prediction(history, matrix, next_draw_at)
    return {
        "next_draw_at": next_draw_at.isoformat(),
        "next_draw_label": next_draw_at.strftime("%Y-%m-%d %H:%M"),
        "predicted_numbers": predicted_numbers,
        "consecutive_in_prediction": consecutive_pairs(predicted_numbers),
        "consecutive_candidates": consecutive_candidates(history, predicted_numbers),
        "model_note": "全期頻率 50% + 近期頻率 35% + 同小時偏號 15%",
        "disclaimer": DISCLAIMER,
    }
