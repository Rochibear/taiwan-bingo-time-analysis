from pathlib import Path

import pandas as pd
import pytest

from bingo_analysis.daily_picks import (
    DailyPickError,
    add_daily_pick,
    daily_match_rows,
    empty_store,
    load_daily_pick_store,
    mark_notified,
    parse_number_text,
    pending_notification_rows,
    save_daily_pick_store,
)


def test_parse_number_text_normalizes_and_deduplicates() -> None:
    assert parse_number_text("01, 5 05 / 80") == [1, 5, 80]


def test_load_daily_pick_store_resets_when_date_changes(tmp_path: Path) -> None:
    path = tmp_path / "daily_picks.json"
    payload = empty_store("2026-05-24")
    add_daily_pick(payload, "user@example.com", [1, 2, 3, 4, 5], 5, "自選")
    save_daily_pick_store(path, payload)

    assert load_daily_pick_store(path, "2026-05-25") == empty_store("2026-05-25")


def test_daily_pick_requires_five_to_ten_numbers() -> None:
    payload = empty_store("2026-05-24")

    with pytest.raises(DailyPickError):
        add_daily_pick(payload, "user@example.com", [1, 2, 3, 4], 4, "自選")

    with pytest.raises(DailyPickError):
        add_daily_pick(payload, "user@example.com", [1, 2, 3, 4, 5], 6, "自選")


def test_daily_pick_limits_each_user_to_three_sets() -> None:
    payload = empty_store("2026-05-24")
    for index in range(3):
        numbers = [index * 10 + number for number in [1, 2, 3, 4, 5]]
        add_daily_pick(payload, "user@example.com", numbers, 5, "自選")

    with pytest.raises(DailyPickError):
        add_daily_pick(payload, "user@example.com", [31, 32, 33, 34, 35], 5, "自選")


def test_daily_matches_and_notifications_are_deduplicated() -> None:
    payload = empty_store("2026-05-24")
    pick = add_daily_pick(
        payload,
        "user@example.com",
        [1, 2, 3, 4, 5, 6],
        5,
        "自選",
        created_at="2026-05-24T09:55:00+08:00",
    )
    history = pd.DataFrame(
        [
            {
                "draw_id": "1001",
                "date": "2026-05-24",
                "time": "10:00",
                "datetime": pd.Timestamp("2026-05-24 10:00"),
                "numbers": [1, 2, 3, 4, 5, 20, 21, 22, 23, 24],
            },
            {
                "draw_id": "1002",
                "date": "2026-05-24",
                "time": "10:05",
                "datetime": pd.Timestamp("2026-05-24 10:05"),
                "numbers": [1, 2, 30, 31, 32, 33, 34, 35, 36, 37],
            },
        ]
    )
    rows = daily_match_rows([pick], history, "2026-05-24")
    pending = pending_notification_rows([pick], rows)

    assert [row["draw_id"] for row in pending] == ["1001"]

    mark_notified([pick], pending)

    assert pending_notification_rows([pick], rows) == []


def test_daily_matches_ignore_draws_before_pick_was_locked() -> None:
    payload = empty_store("2026-05-24")
    pick = add_daily_pick(
        payload,
        "user@example.com",
        [1, 2, 3, 4, 5],
        5,
        "自選",
        created_at="2026-05-24T10:03:00+08:00",
    )
    history = pd.DataFrame(
        [
            {
                "draw_id": "1001",
                "date": "2026-05-24",
                "time": "10:00",
                "datetime": pd.Timestamp("2026-05-24 10:00"),
                "numbers": [1, 2, 3, 4, 5],
            },
            {
                "draw_id": "1002",
                "date": "2026-05-24",
                "time": "10:05",
                "datetime": pd.Timestamp("2026-05-24 10:05"),
                "numbers": [1, 2, 3, 4, 5],
            },
        ]
    )

    rows = daily_match_rows([pick], history, "2026-05-24")

    assert [row["draw_id"] for row in rows] == ["1002"]
