from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from bingo_analysis.daily_report import (
    build_daily_prediction_log,
    daily_report_path,
    daily_report_summary,
    has_final_draw,
    load_daily_report_state,
    mark_report_sent,
    report_sent_record,
    save_daily_prediction_log,
    save_daily_report_state,
)


def sample_history() -> pd.DataFrame:
    rows = []
    start = datetime(2026, 5, 23, 7, 5)
    for index in range(50):
        draw_at = start + timedelta(minutes=5 * index)
        numbers = [((index + offset) % 80) + 1 for offset in range(20)]
        rows.append(
            {
                "draw_id": str(1000 + index),
                "date": draw_at.date().isoformat(),
                "time": draw_at.strftime("%H:%M"),
                "numbers": numbers,
                "super_number": numbers[0],
                "big_small": "",
                "odd_even": "",
            }
        )

    for offset, draw_time in enumerate(["07:05", "07:10", "23:55"]):
        draw_at = datetime.fromisoformat(f"2026-05-24 {draw_time}")
        numbers = [((30 + offset + number) % 80) + 1 for number in range(20)]
        rows.append(
            {
                "draw_id": str(2000 + offset),
                "date": draw_at.date().isoformat(),
                "time": draw_at.strftime("%H:%M"),
                "numbers": numbers,
                "super_number": numbers[0],
                "big_small": "",
                "odd_even": "",
            }
        )

    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["date"] + " " + frame["time"])
    return frame


def test_has_final_draw_detects_today_last_draw() -> None:
    history = sample_history()

    assert has_final_draw(history, "2026-05-24")
    assert not has_final_draw(history, "2026-05-23")


def test_build_daily_prediction_log_contains_each_today_draw() -> None:
    report = build_daily_prediction_log(sample_history(), "2026-05-24")

    assert list(report["draw_id"]) == ["2000", "2001", "2002"]
    assert report["candidate20_hit_count"].between(0, 20).all()
    assert report["ten_star_hit_count"].between(0, 10).all()
    assert report["actual_numbers"].str.contains(" ").all()
    assert report["candidate20_numbers"].str.split().map(len).eq(20).all()
    assert report["ten_star_numbers"].str.split().map(len).eq(10).all()


def test_daily_report_summary_and_state(tmp_path: Path) -> None:
    report = build_daily_prediction_log(sample_history(), "2026-05-24")
    summary = daily_report_summary(report)
    report_path = daily_report_path(tmp_path, "2026-05-24")
    state_path = tmp_path / "daily_report_state.json"

    save_daily_prediction_log(report_path, report)
    state = load_daily_report_state(state_path)
    mark_report_sent(state, "2026-05-24", "User@Example.com", report_path, len(report))
    save_daily_report_state(state_path, state)
    reloaded = load_daily_report_state(state_path)

    assert report_path.exists()
    assert summary["draw_count"] == 3
    assert report_sent_record(reloaded, "2026-05-24", "user@example.com") is not None
