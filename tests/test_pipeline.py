from datetime import date

import bingo_analysis.pipeline as pipeline


class DummySession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_run_pipeline_merges_live_day_when_history_selector_lags(
    monkeypatch,
    tmp_path,
) -> None:
    live_record = {
        "draw_id": "live23",
        "date": "2026-05-23",
        "time": "07:05",
        "numbers": list(range(1, 21)),
        "super_number": 1,
        "big_small": "－",
        "odd_even": "－",
    }
    history_record = {
        **live_record,
        "draw_id": "hist22",
        "date": "2026-05-22",
    }
    saved_records: list[dict[str, object]] = []

    monkeypatch.setattr(pipeline, "create_session", lambda config: DummySession())
    monkeypatch.setattr(
        pipeline,
        "discover_available_dates",
        lambda session, config: [date(2026, 5, 21), date(2026, 5, 22)],
    )
    monkeypatch.setattr(pipeline, "fetch_live_html", lambda session, config: "<html>")
    monkeypatch.setattr(pipeline, "parse_history_page", lambda html: [live_record])

    def fake_scrape_dates(dates, config, session=None):
        assert dates == [date(2026, 5, 22)]
        return [history_record], []

    monkeypatch.setattr(pipeline, "scrape_dates", fake_scrape_dates)

    def fake_save_history_csv(records, csv_path):
        saved_records.extend(records)
        return csv_path

    monkeypatch.setattr(pipeline, "save_history_csv", fake_save_history_csv)
    monkeypatch.setattr(
        pipeline,
        "analyze_history",
        lambda csv_path, output_dir: {"draw_count": len(saved_records)},
    )

    result = pipeline.run_pipeline(tmp_path, days=2)

    assert result.selected_dates == [date(2026, 5, 22), date(2026, 5, 23)]
    assert [record["draw_id"] for record in saved_records] == ["hist22", "live23"]
