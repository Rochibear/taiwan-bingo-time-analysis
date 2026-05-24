from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .analysis import load_history

TWLOTTERY_BINGO_URL = "https://twlottery.in/lotteryBingo/list"
SOURCE_NOTE = (
    "民間來源校驗只作交叉參考，不取代官方校驗。備援來源沒有開獎時間欄，"
    "時間仍以 Pilio 原始資料為準。"
)
NUMBER_TOKEN = r"(80|[1-7]\d|0?[1-9])"
NUMBER_LINE_RE = re.compile(rf"^{NUMBER_TOKEN}$")
DRAW_LINE_RE = re.compile(r"^(\d{6,})\s*期$")
DATE_RE = re.compile(r"(\d{4})/(\d{1,2})/(\d{1,2})")
SUPER_RE = re.compile(rf"超級獎號\s*[：:]\s*{NUMBER_TOKEN}")
BIG_SMALL_RE = re.compile(r"大小\s*[：:]\s*([大小\-－])")
ODD_EVEN_RE = re.compile(r"單雙\s*[：:]\s*([單雙\-－])")


class CrosscheckError(RuntimeError):
    """Raised when a民間來源校驗 cannot be completed."""


@dataclass(frozen=True)
class CrosscheckConfig:
    source_name: str = "twlottery.in"
    source_url: str = TWLOTTERY_BINGO_URL
    timeout_seconds: float = 20.0
    max_retries: int = 3
    backoff_seconds: float = 1.5
    delay_seconds: float = 1.0
    user_agent: str = (
        "TaiwanBingoTimeAnalysis/0.1 "
        "(low-frequency public page cross-check)"
    )


@dataclass(frozen=True)
class CrosscheckResult:
    summary: dict[str, Any]
    details: pd.DataFrame
    details_path: Path
    summary_path: Path


def _normalize_marker(value: object) -> str:
    text = str(value or "").strip()
    return "－" if text in {"", "-"} else text


def _numbers_key(numbers: Iterable[object]) -> str:
    return ";".join(f"{int(number):02d}" for number in sorted(int(n) for n in numbers))


def _parse_header_date(lines: list[str]) -> date | None:
    for line in lines[:80]:
        match = DATE_RE.search(line)
        if match:
            year, month, day = (int(part) for part in match.groups())
            return date(year, month, day)
    return None


def parse_twlottery_bingo_page(
    html: str,
    requested_date: date | None = None,
) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    lines = [
        line.strip()
        for line in soup.get_text("\n", strip=True).splitlines()
        if line.strip()
    ]
    page_date = _parse_header_date(lines) or requested_date
    if page_date is None:
        raise CrosscheckError("backup source page has no usable date")
    if requested_date and page_date != requested_date:
        raise CrosscheckError(
            f"backup source returned {page_date.isoformat()} for "
            f"{requested_date.isoformat()}"
        )

    rows: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        draw_match = DRAW_LINE_RE.match(lines[index])
        if not draw_match:
            index += 1
            continue

        draw_id = draw_match.group(1)
        index += 1
        numbers: list[int] = []
        while index < len(lines):
            super_match = SUPER_RE.search(lines[index])
            if super_match:
                break
            number_match = NUMBER_LINE_RE.match(lines[index])
            if number_match:
                numbers.append(int(number_match.group(1)))
            index += 1

        if index >= len(lines):
            break
        super_match = SUPER_RE.search(lines[index])
        if not super_match:
            index += 1
            continue
        super_number = int(super_match.group(1))

        big_small = "－"
        odd_even = "－"
        scan_index = index + 1
        while scan_index < len(lines):
            if DRAW_LINE_RE.match(lines[scan_index]):
                break
            big_small_match = BIG_SMALL_RE.search(lines[scan_index])
            odd_even_match = ODD_EVEN_RE.search(lines[scan_index])
            if big_small_match:
                big_small = _normalize_marker(big_small_match.group(1))
            if odd_even_match:
                odd_even = _normalize_marker(odd_even_match.group(1))
                scan_index += 1
                break
            scan_index += 1

        if len(numbers) == 20 and len(set(numbers)) == 20:
            rows.append(
                {
                    "draw_id": draw_id,
                    "date": page_date.isoformat(),
                    "numbers": sorted(numbers),
                    "numbers_key": _numbers_key(numbers),
                    "super_number": super_number,
                    "big_small": big_small,
                    "odd_even": odd_even,
                    "backup_source": "twlottery.in",
                }
            )
        index = scan_index

    if not rows:
        raise CrosscheckError(f"backup source had no usable draw rows for {page_date}")
    return pd.DataFrame(rows).drop_duplicates("draw_id")


def crosscheck_session(config: CrosscheckConfig) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": config.user_agent,
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.7",
        }
    )
    return session


def fetch_with_retries(
    session: requests.Session,
    config: CrosscheckConfig,
    day: date,
) -> str:
    attempts = config.max_retries + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(
                config.source_url,
                params={"date": day.isoformat()},
                timeout=config.timeout_seconds,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"transient HTTP {response.status_code}",
                    response=response,
                )
            response.raise_for_status()
            response.encoding = "utf-8"
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(config.backoff_seconds * attempt)
    raise CrosscheckError(f"backup source request failed for {day}: {last_error}")


def fetch_backup_history(
    dates: list[date],
    config: CrosscheckConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    config = config or CrosscheckConfig()
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    with crosscheck_session(config) as session:
        for index, day in enumerate(sorted(set(dates))):
            try:
                html = fetch_with_retries(session, config, day)
                frames.append(parse_twlottery_bingo_page(html, day))
            except CrosscheckError as exc:
                warnings.append(str(exc))
            if index < len(dates) - 1 and config.delay_seconds > 0:
                time.sleep(config.delay_seconds)
    if not frames:
        raise CrosscheckError("; ".join(warnings) or "backup source had no records")
    return pd.concat(frames, ignore_index=True).drop_duplicates("draw_id"), warnings


def local_history_for_dates(history: pd.DataFrame, dates: list[date]) -> pd.DataFrame:
    selected_dates = {day.isoformat() for day in dates}
    frame = history.loc[history["date"].astype(str).isin(selected_dates)].copy()
    frame["numbers_key"] = frame["numbers"].map(_numbers_key)
    frame["super_number"] = frame["super_number"].astype(int)
    frame["big_small"] = frame["big_small"].map(_normalize_marker)
    frame["odd_even"] = frame["odd_even"].map(_normalize_marker)
    frame["draw_id"] = frame["draw_id"].astype(str)
    return frame


def compare_with_backup(local: pd.DataFrame, backup: pd.DataFrame) -> pd.DataFrame:
    backup = backup.copy()
    backup["draw_id"] = backup["draw_id"].astype(str)
    backup_by_draw = backup.set_index("draw_id", drop=False)
    rows: list[dict[str, object]] = []
    for row in local.sort_values(["date", "time", "draw_id"]).itertuples(index=False):
        backup_row = (
            backup_by_draw.loc[row.draw_id]
            if row.draw_id in backup_by_draw.index
            else None
        )
        if backup_row is None:
            rows.append(
                {
                    "draw_id": row.draw_id,
                    "date": row.date,
                    "time": row.time,
                    "status": "missing_backup",
                    "mismatch_fields": "draw_id",
                }
            )
            continue

        mismatches: list[str] = []
        for field in ["date", "numbers_key", "super_number", "big_small", "odd_even"]:
            if str(getattr(row, field)) != str(backup_row[field]):
                mismatches.append(field)
        rows.append(
            {
                "draw_id": row.draw_id,
                "date": row.date,
                "time": row.time,
                "status": "verified" if not mismatches else "mismatch",
                "mismatch_fields": ",".join(mismatches),
            }
        )
    return pd.DataFrame(rows)


def summarize_crosscheck(
    details: pd.DataFrame,
    backup: pd.DataFrame,
    warnings: list[str],
) -> dict[str, Any]:
    counts = details["status"].value_counts() if not details.empty else pd.Series()
    comparable_count = int(counts.get("verified", 0) + counts.get("mismatch", 0))
    return {
        "source": "twlottery.in",
        "checked_count": int(len(details)),
        "backup_count": int(len(backup)),
        "verified_count": int(counts.get("verified", 0)),
        "mismatch_count": int(counts.get("mismatch", 0)),
        "missing_backup_count": int(counts.get("missing_backup", 0)),
        "verification_rate": (
            float(counts.get("verified", 0)) / comparable_count
            if comparable_count
            else None
        ),
        "latest_backup_date": str(backup["date"].max()) if not backup.empty else None,
        "warnings": warnings,
    }


def recent_history_dates(history: pd.DataFrame, days: int) -> list[date]:
    unique_dates = sorted(pd.to_datetime(history["date"]).dt.date.unique())
    return [day for day in unique_dates[-days:]]


def verify_history_with_backup_source(
    project_root: Path,
    *,
    days: int = 2,
    config: CrosscheckConfig | None = None,
) -> CrosscheckResult:
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "source_crosscheck.csv"
    summary_path = output_dir / "source_crosscheck_summary.json"

    history = load_history(project_root / "bingo_history.csv")
    selected_dates = recent_history_dates(history, days)
    if not selected_dates:
        raise CrosscheckError("local history has no dates to cross-check")

    backup, warnings = fetch_backup_history(selected_dates, config)
    local = local_history_for_dates(history, selected_dates)
    details = compare_with_backup(local, backup)
    summary = summarize_crosscheck(details, backup, warnings)
    summary["selected_dates"] = [day.isoformat() for day in selected_dates]

    details.to_csv(details_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return CrosscheckResult(summary, details, details_path, summary_path)
