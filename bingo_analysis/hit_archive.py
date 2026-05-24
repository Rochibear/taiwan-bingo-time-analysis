from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ARCHIVE_COLUMNS = [
    "stars",
    "draw_id",
    "date",
    "time",
    "selected_numbers",
    "hit_numbers",
    "hit_count",
    "source",
    "archived_at",
]
DEDUP_COLUMNS = [
    "stars",
    "draw_id",
    "selected_numbers",
    "hit_numbers",
    "hit_count",
    "source",
]


def format_numbers(value: Any) -> str:
    if isinstance(value, str):
        tokens = [token for token in value.replace(";", " ").split() if token]
        numbers = [int(token) for token in tokens]
    else:
        numbers = [int(number) for number in value]
    return " ".join(f"{number:02d}" for number in sorted(numbers))


def load_hit_archive(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=ARCHIVE_COLUMNS)
    frame = pd.read_csv(path, dtype={"draw_id": "string"})
    for column in ARCHIVE_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[ARCHIVE_COLUMNS]


def archive_backtest_hits(
    path: Path,
    stars: int,
    details: pd.DataFrame,
    *,
    min_hits: int = 5,
    source: str = "prediction_backtest",
    archived_at: str | None = None,
) -> pd.DataFrame:
    existing = load_hit_archive(path)
    if details.empty:
        return existing

    high_hits = details.loc[details["hit_count"].astype(int) >= int(min_hits)].copy()
    if high_hits.empty:
        return existing

    timestamp = archived_at or datetime.now(timezone.utc).isoformat()
    rows = []
    for row in high_hits.itertuples(index=False):
        rows.append(
            {
                "stars": int(stars),
                "draw_id": str(row.draw_id),
                "date": str(row.date),
                "time": str(row.time),
                "selected_numbers": format_numbers(row.selected_numbers),
                "hit_numbers": format_numbers(row.hit_numbers),
                "hit_count": int(row.hit_count),
                "source": source,
                "archived_at": timestamp,
            }
        )

    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(DEDUP_COLUMNS, keep="last")
    combined["hit_count"] = combined["hit_count"].astype(int)
    combined["stars"] = combined["stars"].astype(int)
    combined = combined.sort_values(
        ["hit_count", "date", "time", "draw_id"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return combined[ARCHIVE_COLUMNS]
