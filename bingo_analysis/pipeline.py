from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .analysis import analyze_history
from .scraper import (
    ScrapeConfig,
    create_session,
    discover_available_dates,
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
        selected_dates = selected_dates_from_form(
            available_dates,
            days=days,
            start_date=start_date,
            end_date=end_date,
        )
        records, warnings = scrape_dates(selected_dates, config, session=session)

    save_history_csv(records, csv_path)
    summary = analyze_history(csv_path, output_dir)
    return PipelineResult(csv_path, output_dir, selected_dates, warnings, summary)


def analyze_existing(project_root: Path) -> dict[str, object]:
    return analyze_history(project_root / "bingo_history.csv", project_root / "output")

