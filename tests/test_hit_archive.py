import pandas as pd

from bingo_analysis.hit_archive import archive_backtest_hits, format_numbers


def test_format_numbers_sorts_and_pads() -> None:
    assert format_numbers([7, 52, 1]) == "01 07 52"
    assert format_numbers("52 07 1") == "01 07 52"


def test_archive_backtest_hits_keeps_high_hits_and_deduplicates(tmp_path) -> None:
    path = tmp_path / "hit_archive.csv"
    details = pd.DataFrame(
        [
            {
                "draw_id": "1001",
                "date": "2026-05-24",
                "time": "16:10",
                "selected_numbers": [1, 2, 3, 4, 5, 6, 7],
                "hit_numbers": [1, 2, 3, 4, 5, 6],
                "hit_count": 6,
            },
            {
                "draw_id": "1002",
                "date": "2026-05-24",
                "time": "16:15",
                "selected_numbers": [11, 12, 13, 14, 15],
                "hit_numbers": [11, 12, 13, 14],
                "hit_count": 4,
            },
        ]
    )

    archive = archive_backtest_hits(
        path,
        10,
        details,
        min_hits=5,
        archived_at="2026-05-24T08:00:00+00:00",
    )
    archive = archive_backtest_hits(
        path,
        10,
        details,
        min_hits=5,
        archived_at="2026-05-24T08:05:00+00:00",
    )

    assert len(archive) == 1
    assert archive.loc[0, "hit_count"] == 6
    assert archive.loc[0, "hit_numbers"] == "01 02 03 04 05 06"
