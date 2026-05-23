from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from bingo_analysis.forecast import build_forecast, next_draw_datetime


def sample_history(draws: int = 40) -> pd.DataFrame:
    rows = []
    start = datetime(2026, 5, 21, 7, 5)
    for index in range(draws):
        numbers = [((index + offset) % 80) + 1 for offset in range(20)]
        draw_at = start + timedelta(minutes=5 * index)
        rows.append(
            {
                "draw_id": str(1000 + index),
                "date": draw_at.date().isoformat(),
                "time": draw_at.strftime("%H:%M"),
                "numbers": numbers,
                "super_number": numbers[0],
                "big_small": "－",
                "odd_even": "－",
            }
        )
    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["date"] + " " + frame["time"])
    return frame


def test_next_draw_datetime_uses_five_minute_schedule() -> None:
    now = datetime(2026, 5, 23, 7, 6, tzinfo=ZoneInfo("Asia/Taipei"))

    assert next_draw_datetime(now).strftime("%H:%M") == "07:10"


def test_build_forecast_returns_prediction_and_pairs() -> None:
    forecast = build_forecast(
        sample_history(),
        now=datetime(2026, 5, 23, 10, 0, tzinfo=ZoneInfo("Asia/Taipei")),
    )

    assert len(forecast["predicted_numbers"]) == 20
    assert len(set(forecast["predicted_numbers"])) == 20
    assert len(forecast["consecutive_candidates"]) > 0
    assert "免責聲明" in forecast["disclaimer"]
