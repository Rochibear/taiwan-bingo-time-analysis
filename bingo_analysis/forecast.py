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
STAR_MIN = 1
STAR_MAX = 10

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


def _current_gaps(matrix: pd.DataFrame) -> np.ndarray:
    gaps: list[int] = []
    for number in NUMBERS:
        appearances = np.flatnonzero(matrix[number].to_numpy())
        if len(appearances) == 0:
            gaps.append(len(matrix))
        else:
            gaps.append(len(matrix) - 1 - int(appearances[-1]))
    return np.asarray(gaps, dtype=float)


def scored_numbers(
    history: pd.DataFrame,
    matrix: pd.DataFrame,
    next_draw_at: datetime,
) -> pd.DataFrame:
    global_counts = matrix.sum(axis=0).reindex(NUMBERS).to_numpy(dtype=float) + 1.0
    recent = matrix.tail(min(RECENT_WINDOW_DRAWS, len(matrix)))
    recent_counts = recent.sum(axis=0).reindex(NUMBERS).to_numpy(dtype=float) + 1.0

    hour_mask = history["datetime"].dt.hour == next_draw_at.hour
    if hour_mask.any():
        hourly_counts = (
            matrix.loc[hour_mask].sum(axis=0).reindex(NUMBERS).to_numpy(dtype=float)
            + 1.0
        )
        hourly_draws = int(hour_mask.sum())
    else:
        hourly_counts = global_counts.copy()
        hourly_draws = len(history)

    current_gaps = _current_gaps(matrix)
    gap_score = np.log1p(current_gaps) + 1.0
    score = (
        0.45 * _normalize(global_counts)
        + 0.30 * _normalize(recent_counts)
        + 0.15 * _normalize(hourly_counts)
        + 0.10 * _normalize(gap_score)
    )

    frame = pd.DataFrame(
        {
            "number": NUMBERS,
            "score": score,
            "global_rate": (global_counts - 1.0) / max(len(history), 1),
            "recent_rate": (recent_counts - 1.0) / max(len(recent), 1),
            "hourly_rate": (hourly_counts - 1.0) / max(hourly_draws, 1),
            "current_gap": current_gaps.astype(int),
        }
    )
    return frame.sort_values(["score", "number"], ascending=[False, True]).reset_index(
        drop=True
    )


def build_star_selection(
    history: pd.DataFrame,
    stars: int,
    now: datetime | None = None,
) -> dict[str, object]:
    if history.empty:
        raise ValueError("history is empty")
    if not STAR_MIN <= stars <= STAR_MAX:
        raise ValueError("stars must be between 1 and 10")

    next_draw_at = next_draw_datetime(now)
    matrix = build_appearance_matrix(history)
    ranked = scored_numbers(history, matrix, next_draw_at)
    selected = ranked.head(stars).copy()
    selected_numbers = sorted(int(number) for number in selected["number"].tolist())

    return {
        "stars": stars,
        "next_draw_at": next_draw_at.isoformat(),
        "next_draw_label": next_draw_at.strftime("%Y-%m-%d %H:%M"),
        "selected_numbers": selected_numbers,
        "selected_details": selected.to_dict(orient="records"),
        "model_note": "全期頻率 45% + 近期頻率 30% + 同小時偏號 15% + 近期 gap 10%",
        "disclaimer": DISCLAIMER,
    }


def _as_taipei_datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.to_pydatetime().replace(tzinfo=TAIPEI_TZ)
    return timestamp.tz_convert(TAIPEI_TZ).to_pydatetime()


def backtest_star_selection(
    history: pd.DataFrame,
    stars: int,
    evaluation_draws: int = 300,
    min_training_draws: int = 300,
    verified_draw_ids: set[str] | None = None,
) -> dict[str, object]:
    if history.empty:
        raise ValueError("history is empty")
    if not STAR_MIN <= stars <= STAR_MAX:
        raise ValueError("stars must be between 1 and 10")
    if evaluation_draws < 1:
        raise ValueError("evaluation_draws must be positive")

    ordered = history.sort_values(["datetime", "draw_id"]).reset_index(drop=True)
    if len(ordered) <= min_training_draws:
        raise ValueError(
            f"need more than {min_training_draws} draws for backtesting"
        )

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
        ranked = scored_numbers(
            training,
            training_matrix,
            _as_taipei_datetime(target["datetime"]),
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
