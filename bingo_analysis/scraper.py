from __future__ import annotations

import csv
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)

DEFAULT_SOURCE_URL = "https://www.pilio.idv.tw/bingo/list_history.asp"
CSV_COLUMNS = [
    "draw_id",
    "date",
    "time",
    "numbers",
    "super_number",
    "big_small",
    "odd_even",
]
NUMBER_RE = re.compile(r"(?<!\d)(0?[1-9]|[1-7]\d|80)(?!\d)")
DRAW_RE = re.compile(r"期別\s*[:：]\s*(\d+)")
SUPER_RE = re.compile(r"超級獎號\s*[:：]\s*(?<!\d)(0?[1-9]|[1-7]\d|80)(?!\d)")
BIG_SMALL_RE = re.compile(r"猜大小\s*[:：]\s*([大小\-－])")
ODD_EVEN_RE = re.compile(r"猜單雙\s*[:：]\s*([單雙\-－])")
TIME_RE = re.compile(r"\((\d{1,2}:\d{2})\)")
SITE_DATE_RE = re.compile(r"(\d{4}/\d{1,2}/\d{1,2})")


class ScrapeError(RuntimeError):
    """Raised when a source page cannot be downloaded or parsed."""


@dataclass(frozen=True)
class ScrapeConfig:
    source_url: str = DEFAULT_SOURCE_URL
    timeout_seconds: float = 20.0
    delay_seconds: float = 1.0
    max_retries: int = 3
    backoff_seconds: float = 1.5
    user_agent: str = (
        "TaiwanBingoTimeAnalysis/0.1 "
        "(educational analysis; one daily history page at a time)"
    )


def create_session(config: ScrapeConfig) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": config.user_agent,
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.7",
        }
    )
    return session


def format_site_date(day: date) -> str:
    return f"{day.year}/{day.month}/{day.day}"


def parse_site_date(raw: str) -> date:
    return datetime.strptime(raw.strip(), "%Y/%m/%d").date()


def _normalize_marker(value: str | None) -> str:
    if not value or value == "-":
        return "－"
    return value


def fetch_history_html(
    session: requests.Session,
    config: ScrapeConfig,
    requested_date: date | None = None,
) -> str:
    params = {"indate": format_site_date(requested_date)} if requested_date else None
    attempts = config.max_retries + 1
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = session.get(
                config.source_url,
                params=params,
                timeout=config.timeout_seconds,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"transient HTTP {response.status_code}",
                    response=response,
                )
            response.raise_for_status()
            # The source HTML declares UTF-8, while the HTTP response can leave
            # requests with a fallback encoding that breaks Chinese field labels.
            response.encoding = "utf-8"
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            if attempt == attempts:
                break
            wait_seconds = config.backoff_seconds * attempt
            LOGGER.warning(
                "Fetch failed for %s on attempt %s/%s; retrying in %.1fs: %s",
                requested_date or "date index",
                attempt,
                attempts,
                wait_seconds,
                exc,
            )
            time.sleep(wait_seconds)

    raise ScrapeError(f"failed to fetch {requested_date or config.source_url}: {last_error}")


def discover_available_dates(
    session: requests.Session,
    config: ScrapeConfig,
) -> list[date]:
    soup = BeautifulSoup(fetch_history_html(session, config), "html.parser")
    select = soup.select_one("select[name='indate']")
    if not select:
        raise ScrapeError("source page has no date selector named 'indate'")

    discovered: list[date] = []
    for option in select.select("option[value]"):
        value = option.get("value", "").strip()
        if not value:
            continue
        try:
            discovered.append(parse_site_date(value))
        except ValueError:
            LOGGER.debug("Skipping unrecognized date option %r", value)

    unique_dates = sorted(set(discovered))
    if not unique_dates:
        raise ScrapeError("date selector did not contain any usable dates")
    return unique_dates


def _page_date(soup: BeautifulSoup, requested_date: date | None) -> date:
    if requested_date:
        return requested_date

    header = soup.select_one("#ltotable tr td")
    match = SITE_DATE_RE.search(header.get_text(" ", strip=True) if header else "")
    if not match:
        raise ScrapeError("could not determine draw date from history page")
    return parse_site_date(match.group(1))


def parse_history_page(html: str, requested_date: date | None = None) -> list[dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("#ltotable")
    if not table:
        raise ScrapeError("history page has no table with id 'ltotable'")

    draw_date = _page_date(soup, requested_date)
    records: list[dict[str, object]] = []

    for cell in table.select("tr td"):
        text = " ".join(cell.get_text(" ", strip=True).split())
        draw_match = DRAW_RE.search(text)
        if not draw_match:
            continue

        before_super = text.split("超級獎號", maxsplit=1)[0]
        numbers = [int(value) for value in NUMBER_RE.findall(before_super)]
        super_match = SUPER_RE.search(text)
        big_small_match = BIG_SMALL_RE.search(text)
        odd_even_match = ODD_EVEN_RE.search(text)
        time_match = TIME_RE.search(text)

        if len(numbers) != 20 or len(set(numbers)) != 20:
            raise ScrapeError(
                f"draw {draw_match.group(1)} contained {len(numbers)} parsed numbers"
            )
        if not super_match or not time_match:
            raise ScrapeError(f"draw {draw_match.group(1)} missed super number or time")

        records.append(
            {
                "draw_id": draw_match.group(1),
                "date": draw_date.isoformat(),
                "time": time_match.group(1).zfill(5),
                "numbers": numbers,
                "super_number": int(super_match.group(1)),
                "big_small": _normalize_marker(
                    big_small_match.group(1) if big_small_match else None
                ),
                "odd_even": _normalize_marker(
                    odd_even_match.group(1) if odd_even_match else None
                ),
            }
        )

    if not records:
        raise ScrapeError(f"no draw rows were parsed for {draw_date.isoformat()}")
    return records


def scrape_dates(
    dates: Iterable[date],
    config: ScrapeConfig,
    session: requests.Session | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    ordered_dates = sorted(set(dates))
    if not ordered_dates:
        raise ScrapeError("no dates were selected for scraping")

    owns_session = session is None
    session = session or create_session(config)
    records: list[dict[str, object]] = []
    warnings: list[str] = []

    try:
        for index, day in enumerate(ordered_dates):
            try:
                html = fetch_history_html(session, config, day)
                records.extend(parse_history_page(html, day))
                LOGGER.info("Parsed %s daily draws for %s", len(records), day)
            except ScrapeError as exc:
                warning = f"{day.isoformat()}: {exc}"
                warnings.append(warning)
                LOGGER.warning("Skipping %s", warning)
            if index < len(ordered_dates) - 1 and config.delay_seconds > 0:
                time.sleep(config.delay_seconds)
    finally:
        if owns_session:
            session.close()

    if not records:
        detail = "; ".join(warnings) if warnings else "all pages were empty"
        raise ScrapeError(f"no draw records were scraped: {detail}")
    return records, warnings


def save_history_csv(records: Iterable[dict[str, object]], csv_path: Path) -> Path:
    rows_by_draw: dict[str, dict[str, object]] = {}
    for record in records:
        row = dict(record)
        numbers = row["numbers"]
        if not isinstance(numbers, list):
            raise ScrapeError("record numbers must be a list before CSV export")
        row["numbers"] = ";".join(f"{int(number):02d}" for number in numbers)
        rows_by_draw[str(row["draw_id"])] = row

    rows = sorted(
        rows_by_draw.values(),
        key=lambda row: (str(row["date"]), str(row["time"]), str(row["draw_id"])),
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def selected_dates_from_form(
    available_dates: list[date],
    days: int | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[date]:
    if start_date or end_date:
        lower = start_date or min(available_dates)
        upper = end_date or max(available_dates)
        if upper < lower:
            raise ValueError("end date must be on or after start date")
        return [day for day in available_dates if lower <= day <= upper]

    if days is None:
        return available_dates
    if days < 1:
        raise ValueError("days must be at least 1")
    return available_dates[-days:]


def resolve_action_url(config: ScrapeConfig) -> str:
    return urljoin(config.source_url, "list_history.asp")
