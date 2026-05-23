from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for

from bingo_analysis.analysis import CHART_FILENAMES, load_history
from bingo_analysis.forecast import build_forecast
from bingo_analysis.pipeline import analyze_existing, run_pipeline
from bingo_analysis.scraper import ScrapeConfig

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("BINGO_DATA_DIR", PROJECT_ROOT))
OUTPUT_DIR = DATA_DIR / "output"
SUMMARY_PATH = OUTPUT_DIR / "analysis_summary.json"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-bingo-dashboard")


def load_summary() -> dict[str, Any] | None:
    if not SUMMARY_PATH.exists():
        return None
    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_optional_date(raw: str) -> date | None:
    return date.fromisoformat(raw) if raw else None


@app.get("/")
def dashboard() -> str:
    summary = load_summary()
    forecast = None
    charts = [
        filename
        for filename in CHART_FILENAMES
        if (OUTPUT_DIR / filename).exists()
    ]
    if summary and (DATA_DIR / "bingo_history.csv").exists():
        try:
            forecast = build_forecast(load_history(DATA_DIR / "bingo_history.csv"))
        except Exception:
            forecast = None
    return render_template(
        "index.html",
        summary=summary,
        charts=charts,
        forecast=forecast,
    )


@app.post("/refresh")
def refresh() -> object:
    try:
        start_date = parse_optional_date(request.form.get("start_date", ""))
        end_date = parse_optional_date(request.form.get("end_date", ""))
        delay = max(float(request.form.get("delay", "1.0") or 1.0), 0.5)
        raw_days = request.form.get("days", "30")
        days = max(int(raw_days or 30), 1)
        result = run_pipeline(
            DATA_DIR,
            days=None if start_date or end_date else days,
            start_date=start_date,
            end_date=end_date,
            config=ScrapeConfig(delay_seconds=delay),
        )
        flash(f"完成 {result.summary['draw_count']} 期分析。", "success")
        for warning in result.scrape_warnings[:3]:
            flash(warning, "warning")
    except Exception as exc:
        flash(str(exc), "error")
    return redirect(url_for("dashboard"))


@app.post("/reanalyze")
def reanalyze() -> object:
    try:
        summary = analyze_existing(DATA_DIR)
        flash(f"已重建 {summary['draw_count']} 期圖表。", "success")
    except Exception as exc:
        flash(str(exc), "error")
    return redirect(url_for("dashboard"))


@app.get("/output/<path:filename>")
def output_file(filename: str) -> object:
    return send_from_directory(OUTPUT_DIR, filename)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
