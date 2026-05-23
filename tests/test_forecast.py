from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
import pandas as pd

from bingo_analysis.forecast import (
    backtest_star_selection,
    build_forecast,
    build_star_selection,
    next_draw_datetime,
)


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


def test_build_star_selection_uses_requested_star_count() -> None:
    selection = build_star_selection(
        sample_history(),
        stars=7,
        now=datetime(2026, 5, 23, 10, 0, tzinfo=ZoneInfo("Asia/Taipei")),
    )

    assert selection["stars"] == 7
    assert len(selection["selected_numbers"]) == 7
    assert len(set(selection["selected_numbers"])) == 7
    assert len(selection["selected_details"]) == 7
    assert all(1 <= number <= 80 for number in selection["selected_numbers"])


def test_build_star_selection_rejects_out_of_range_stars() -> None:
    with pytest.raises(ValueError, match="between 1 and 10"):
        build_star_selection(sample_history(), stars=11)


def test_backtest_star_selection_reports_hit_metrics() -> None:
    result = backtest_star_selection(
        sample_history(80),
        stars=5,
        evaluation_draws=10,
        min_training_draws=20,
    )

    assert result["summary"]["checked_count"] == 10
    assert result["summary"]["stars"] == 5
    assert result["summary"]["random_mean_hits"] == pytest.approx(1.25)
    assert set(result["details"].columns) == {
        "draw_id",
        "date",
        "time",
        "selected_numbers",
        "hit_numbers",
        "hit_count",
    }
