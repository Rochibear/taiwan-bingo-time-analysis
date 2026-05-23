from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .analysis import analyze_history
from .scraper import (
    ScrapeConfig,
    ScrapeError,
    create_session,
    discover_available_dates,
    fetch_live_html,
    parse_history_page,
    save_history_csv,
    scrape_dates,
    selected_dates_from_form,
)


@dataclass(frozen=True)
class PipelineResult:
    csv_path: Path
    output_dir: Path
    selected_dates: list[date]
    scrape_warnings: list[str]
    summary: dict[str, object]


def run_pipeline(
    project_root: Path,
    days: int | None = 30,
    start_date: date | None = None,
    end_date: date | None = None,
    config: ScrapeConfig | None = None,
) -> PipelineResult:
    config = config or ScrapeConfig()
    csv_path = project_root / "bingo_history.csv"
    output_dir = project_root / "output"

    with create_session(config) as session:
        available_dates = discover_available_dates(session, config)
        live_records: list[dict[str, object]] = []
        live_date: date | None = None
        live_extra_date: date | None = None

        try:
            live_records = parse_history_page(fetch_live_html(session, config))
            live_date = date.fromisoformat(str(live_records[0]["date"]))
        except Exception:
            live_records = []
            live_date = None

        if live_date and live_date not in available_dates:
            live_extra_date = live_date
            available_dates = sorted([*available_dates, live_date])

        selected_dates = selected_dates_from_form(
            available_dates,
            days=days,
            start_date=start_date,
            end_date=end_date,
        )
        if not selected_dates:
            raise ScrapeError("no dates were selected for scraping")

        history_dates = [
            selected_date
            for selected_date in selected_dates
            if selected_date != live_extra_date
        ]
        records: list[dict[str, object]] = []
        warnings: list[str] = []
        if history_dates:
            records, warnings = scrape_dates(history_dates, config, session=session)
        if live_extra_date and live_extra_date in selected_dates:
            records.extend(live_records)

    save_history_csv(records, csv_path)
    summary = analyze_history(csv_path, output_dir)
    return PipelineResult(csv_path, output_dir, selected_dates, warnings, summary)


def analyze_existing(project_root: Path) -> dict[str, object]:
    return analyze_history(project_root / "bingo_history.csv", project_root / "output")
