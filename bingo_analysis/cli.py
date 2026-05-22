from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from .pipeline import analyze_existing, run_pipeline
from .scraper import ScrapeConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_iso_date(raw: str | None) -> date | None:
    return date.fromisoformat(raw) if raw else None


def add_range_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of source-page dates to scrape when no date range is given.",
    )
    parser.add_argument("--start-date", help="ISO start date, for example 2026-05-01.")
    parser.add_argument("--end-date", help="ISO end date, for example 2026-05-21.")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Sleep seconds between daily history page requests.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Taiwan BINGO BINGO time analysis.")
    parser.add_argument("--verbose", action="store_true", help="Show progress logs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Scrape history and build charts.")
    add_range_arguments(run_parser)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Rebuild charts from the existing bingo_history.csv.",
    )
    analyze_parser.set_defaults(command="analyze")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command == "analyze":
        summary = analyze_existing(PROJECT_ROOT)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    result = run_pipeline(
        PROJECT_ROOT,
        days=None if args.start_date or args.end_date else args.days,
        start_date=parse_iso_date(args.start_date),
        end_date=parse_iso_date(args.end_date),
        config=ScrapeConfig(delay_seconds=args.delay),
    )
    response = {
        "csv": str(result.csv_path),
        "output": str(result.output_dir),
        "selected_dates": [day.isoformat() for day in result.selected_dates],
        "warnings": result.scrape_warnings,
        "summary": result.summary,
    }
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

