from __future__ import annotations

import csv
import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning

from .analysis import load_history

OFFICIAL_API_URL = "https://api.taiwanlottery.com/TLCAPIWeB/Lottery/ResultDownload"
OFFICIAL_NOTE = (
    "官方年度檔每月 5 日更新至前一個月資料；官方檔無開獎時間欄，"
    "時間欄仍以原始抓取來源為準。"
)
NUMBER_COLUMNS = [f"獎號{index}" for index in range(1, 21)]
OFFICIAL_HOST_SUFFIXES = ("taiwanlottery.com", "taiwanlottery.com.tw")


class OfficialDataError(RuntimeError):
    """Raised when official Taiwan Lottery data cannot be loaded or compared."""


@dataclass(frozen=True)
class OfficialConfig:
    api_url: str = OFFICIAL_API_URL
    timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_seconds: float = 1.5
    delay_seconds: float = 0.5
    allow_tls_fallback: bool = True
    user_agent: str = (
        "TaiwanBingoTimeAnalysis/0.1 "
        "(official public result verification; low-frequency cache)"
    )


@dataclass(frozen=True)
class OfficialVerificationResult:
    summary: dict[str, Any]
    details: pd.DataFrame
    details_path: Path
    summary_path: Path


def official_session(config: OfficialConfig) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": config.user_agent,
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.7",
        }
    )
    return session


def is_official_download_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(
        host == suffix or host.endswith(f".{suffix}")
        for suffix in OFFICIAL_HOST_SUFFIXES
    )


def validate_response(response: requests.Response) -> requests.Response:
    if response.status_code in {429, 500, 502, 503, 504}:
        raise requests.HTTPError(
            f"transient HTTP {response.status_code}",
            response=response,
        )
    response.raise_for_status()
    return response


def request_with_retries(
    session: requests.Session,
    url: str,
    config: OfficialConfig,
    **kwargs: Any,
) -> requests.Response:
    attempts = config.max_retries + 1
    last_error: Exception | None = None
    verify_tls = bool(kwargs.pop("verify", True))
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(
                url,
                timeout=config.timeout_seconds,
                verify=verify_tls,
                **kwargs,
            )
            return validate_response(response)
        except requests.exceptions.SSLError as exc:
            last_error = exc
            if (
                config.allow_tls_fallback
                and verify_tls
                and is_official_download_host(url)
            ):
                # Taiwan Lottery's public result endpoints can fail strict OpenSSL
                # checks in Python while still opening in browsers. The fallback is
                # limited to these official, read-only download hosts.
                urllib3.disable_warnings(category=InsecureRequestWarning)
                try:
                    response = session.get(
                        url,
                        timeout=config.timeout_seconds,
                        verify=False,
                        **kwargs,
                    )
                    return validate_response(response)
                except requests.RequestException as fallback_exc:
                    last_error = fallback_exc
        except requests.RequestException as exc:
            last_error = exc
        if attempt == attempts:
            break
        time.sleep(config.backoff_seconds * attempt)
    raise OfficialDataError(f"official request failed: {last_error}")


def fetch_official_zip(
    year: int,
    cache_dir: Path,
    config: OfficialConfig | None = None,
    force_refresh: bool = False,
) -> Path:
    config = config or OfficialConfig()
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"taiwan_lottery_{year}.zip"
    if zip_path.exists() and not force_refresh:
        return zip_path

    with official_session(config) as session:
        response = request_with_retries(
            session,
            config.api_url,
            config,
            params={"year": year},
        )
        payload = response.json()
        content = payload.get("content") or {}
        download_url = content.get("path")
        if payload.get("rtCode") != 0 or not download_url:
            raise OfficialDataError(f"official API returned no download path for {year}")

        time.sleep(config.delay_seconds)
        file_response = request_with_retries(session, download_url, config)
        zip_path.write_bytes(file_response.content)
    return zip_path


def parse_official_bingo_zip(zip_path: Path) -> pd.DataFrame:
    if not zip_path.exists():
        raise OfficialDataError(f"official ZIP does not exist: {zip_path}")

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as archive:
        bingo_names = [
            name for name in archive.namelist() if "賓果賓果" in Path(name).name
        ]
        if not bingo_names:
            raise OfficialDataError(f"official ZIP has no Bingo Bingo CSV: {zip_path}")

        with archive.open(bingo_names[0]) as raw_handle:
            text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text_handle)
            for row in reader:
                numbers = [int(row[column]) for column in NUMBER_COLUMNS]
                rows.append(
                    {
                        "draw_id": str(row["期別"]).strip(),
                        "date": pd.to_datetime(row["開獎日期"]).date().isoformat(),
                        "numbers": numbers,
                        "numbers_key": ";".join(f"{number:02d}" for number in numbers),
                        "super_number": int(row["超級獎號"]),
                        "big_small": row.get("猜大小") or "－",
                        "odd_even": row.get("猜單雙") or "－",
                    }
                )

    if not rows:
        raise OfficialDataError(f"official Bingo CSV has no rows: {zip_path}")
    return pd.DataFrame(rows).sort_values(["date", "draw_id"]).reset_index(drop=True)


def official_years_for_history(history: pd.DataFrame) -> list[int]:
    years = sorted(pd.to_datetime(history["date"]).dt.year.unique())
    return [int(year) for year in years]


def load_official_history(
    history: pd.DataFrame,
    cache_dir: Path,
    config: OfficialConfig | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in official_years_for_history(history):
        zip_path = fetch_official_zip(year, cache_dir, config, force_refresh)
        frames.append(parse_official_bingo_zip(zip_path))
    if not frames:
        raise OfficialDataError("no official years were selected")
    return pd.concat(frames, ignore_index=True).drop_duplicates("draw_id")


def _history_for_comparison(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    frame["numbers_key"] = frame["numbers"].map(
        lambda numbers: ";".join(f"{int(number):02d}" for number in numbers)
    )
    frame["super_number"] = frame["super_number"].astype(int)
    frame["draw_id"] = frame["draw_id"].astype(str)
    return frame


def compare_with_official(history: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    local = _history_for_comparison(history)
    official = official.copy()
    official["draw_id"] = official["draw_id"].astype(str)
    official_by_draw = official.set_index("draw_id", drop=False)

    rows: list[dict[str, object]] = []
    for row in local.itertuples(index=False):
        official_row = (
            official_by_draw.loc[row.draw_id]
            if row.draw_id in official_by_draw.index
            else None
        )
        if official_row is None:
            rows.append(
                {
                    "draw_id": row.draw_id,
                    "date": row.date,
                    "time": row.time,
                    "status": "pending_official",
                    "mismatch_fields": "",
                }
            )
            continue

        mismatches: list[str] = []
        if str(row.date) != str(official_row["date"]):
            mismatches.append("date")
        if row.numbers_key != official_row["numbers_key"]:
            mismatches.append("numbers")
        if int(row.super_number) != int(official_row["super_number"]):
            mismatches.append("super_number")
        if str(row.big_small) != str(official_row["big_small"]):
            mismatches.append("big_small")
        if str(row.odd_even) != str(official_row["odd_even"]):
            mismatches.append("odd_even")

        rows.append(
            {
                "draw_id": row.draw_id,
                "date": row.date,
                "time": row.time,
                "status": "mismatch" if mismatches else "verified",
                "mismatch_fields": ",".join(mismatches),
            }
        )

    return pd.DataFrame(rows).sort_values(["date", "time", "draw_id"]).reset_index(
        drop=True
    )


def summarize_verification(details: pd.DataFrame, official: pd.DataFrame) -> dict[str, Any]:
    counts = details["status"].value_counts().to_dict()
    verified = int(counts.get("verified", 0))
    comparable = int(verified + counts.get("mismatch", 0))
    latest_official_date = str(official["date"].max()) if not official.empty else None
    return {
        "note": OFFICIAL_NOTE,
        "checked_count": int(len(details)),
        "official_count": int(len(official)),
        "verified_count": verified,
        "mismatch_count": int(counts.get("mismatch", 0)),
        "pending_official_count": int(counts.get("pending_official", 0)),
        "verification_rate": verified / comparable if comparable else None,
        "latest_official_date": latest_official_date,
        "status_counts": {str(key): int(value) for key, value in counts.items()},
    }


def verify_history_with_official(
    project_root: Path,
    config: OfficialConfig | None = None,
    force_refresh: bool = False,
) -> OfficialVerificationResult:
    history = load_history(project_root / "bingo_history.csv")
    output_dir = project_root / "output"
    cache_dir = project_root / "official_cache"
    details_path = output_dir / "official_verification.csv"
    summary_path = output_dir / "official_verification_summary.json"

    official = load_official_history(history, cache_dir, config, force_refresh)
    details = compare_with_official(history, official)
    summary = summarize_verification(details, official)

    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(details_path, index=False, encoding="utf-8-sig")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return OfficialVerificationResult(summary, details, details_path, summary_path)
