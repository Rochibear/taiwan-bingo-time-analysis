from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from bingo_analysis.analysis import CHART_FILENAMES, load_history
from bingo_analysis.forecast import build_forecast
from bingo_analysis.pipeline import analyze_existing, run_pipeline
from bingo_analysis.scraper import ScrapeConfig

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("BINGO_DATA_DIR", PROJECT_ROOT))
OUTPUT_DIR = DATA_DIR / "output"
SUMMARY_PATH = OUTPUT_DIR / "analysis_summary.json"
ANALYZE_COOLDOWN_KEY = "analyze_cooldown_until"

CHART_TITLES = {
    "number_frequency.png": "1-80 號碼出現次數",
    "overlap_distribution.png": "相鄰兩期重複球數",
    "gap_distribution.png": "號碼再出現 gap 分布",
    "hourly_heatmap.png": "小時別偏號",
    "weekday_heatmap.png": "星期別偏號",
    "autocorrelation.png": "自相關",
    "fft_periodogram.png": "FFT 週期分析",
}


def load_summary() -> dict[str, Any] | None:
    if not SUMMARY_PATH.exists():
        return None
    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cooldown_remaining_seconds(key: str) -> int:
    remaining = float(st.session_state.get(key, 0.0)) - time.time()
    return max(0, math.ceil(remaining))


def number_table(items: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(items)
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "number": "號碼",
            "appearances": "出現次數",
            "appearance_rate": "出現率",
        }
    )
    frame["號碼"] = frame["號碼"].map(lambda value: f"{int(value):02d}")
    frame["出現率"] = frame["出現率"].map(lambda value: f"{value:.2%}")
    return frame


def chip_list(numbers: list[int], accent: str = "#0ea5a3") -> str:
    chips = "".join(
        f"<span class='number-chip' style='border-color:{accent}'>{number:02d}</span>"
        for number in numbers
    )
    return f"<div class='chip-row'>{chips}</div>"


def label_chip_list(labels: list[str], accent: str = "#f59e0b") -> str:
    chips = "".join(
        f"<span class='number-chip' style='border-color:{accent}'>{label}</span>"
        for label in labels
    )
    return f"<div class='chip-row'>{chips}</div>"


def pair_table(items: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(items)
    if frame.empty:
        return frame
    keep_columns = ["label", "history_count", "score"]
    frame = frame[keep_columns].rename(
        columns={
            "label": "連號",
            "history_count": "歷史同開次數",
            "score": "模型分數",
        }
    )
    frame["模型分數"] = frame["模型分數"].map(lambda value: f"{value:.4f}")
    return frame


def show_forecast() -> None:
    try:
        forecast = build_forecast(load_history(DATA_DIR / "bingo_history.csv"))
    except Exception:
        st.info("預告區需要先有 bingo_history.csv。請先按「抓取並分析」。")
        return

    st.subheader("預告區")
    st.warning(forecast["disclaimer"])
    area_one, area_two = st.columns(2)

    with area_one:
        st.markdown("#### 下一期候選號碼")
        st.caption(f"推定下一期時間：{forecast['next_draw_label']}")
        st.markdown(
            chip_list(forecast["predicted_numbers"], "#0ea5a3"),
            unsafe_allow_html=True,
        )
        st.caption(f"模型：{forecast['model_note']}")

    with area_two:
        st.markdown("#### 預測連號")
        predicted_pairs = forecast["consecutive_in_prediction"]
        if predicted_pairs:
            st.caption("候選號碼內形成的連號")
            st.markdown(
                label_chip_list([pair["label"] for pair in predicted_pairs]),
                unsafe_allow_html=True,
            )
        else:
            st.caption("本次候選號碼沒有形成連號，改看模型連號候選。")
        st.dataframe(
            pair_table(forecast["consecutive_candidates"]),
            hide_index=True,
            use_container_width=True,
        )


def show_summary(summary: dict[str, Any]) -> None:
    metrics = st.columns(4)
    metrics[0].metric("期數", summary["draw_count"])
    with metrics[1]:
        st.caption("日期")
        st.markdown(f"**{summary['date_start']}**  \n**{summary['date_end']}**")
    metrics[2].metric("相鄰重複平均", f"{summary.get('mean_overlap') or 0:.2f}")
    metrics[3].metric("中位 gap", f"{summary.get('median_gap') or 0:.1f}")

    show_forecast()

    hot, cold = st.columns(2)
    with hot:
        st.subheader("熱號")
        st.dataframe(
            number_table(summary["hot_numbers"]),
            hide_index=True,
            use_container_width=True,
        )
    with cold:
        st.subheader("冷號")
        st.dataframe(
            number_table(summary["cold_numbers"]),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("圖表")
    for left_index in range(0, len(CHART_FILENAMES), 2):
        chart_columns = st.columns(2)
        for column, filename in zip(
            chart_columns,
            CHART_FILENAMES[left_index : left_index + 2],
        ):
            chart_path = OUTPUT_DIR / filename
            if chart_path.exists():
                column.image(str(chart_path), caption=CHART_TITLES[filename])

    st.subheader("FFT 高能量週期")
    period_frame = pd.DataFrame(summary.get("dominant_periods", []))
    if not period_frame.empty:
        period_frame = period_frame.rename(
            columns={"period_draws": "週期期數", "mean_power": "平均能量"}
        )
        st.dataframe(period_frame, hide_index=True, use_container_width=True)


st.set_page_config(
    page_title="賓果賓果時間分析",
    page_icon=":bar_chart:",
    layout="wide",
)
st.markdown(
    """
    <style>
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.65rem 0 0.85rem;
    }
    .number-chip {
        border: 1px solid;
        border-radius: 999px;
        display: inline-flex;
        font-weight: 800;
        justify-content: center;
        min-width: 2.55rem;
        padding: 0.28rem 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("賓果賓果時間分析")
st.caption("台灣 BINGO BINGO 歷史資料探索。圖表是診斷工具，不是未來開獎保證。")

with st.form("refresh"):
    st.subheader("抓取與分析")
    analyze_cooldown = cooldown_remaining_seconds(ANALYZE_COOLDOWN_KEY)
    analyze_label = (
        "抓取並分析"
        if analyze_cooldown == 0
        else f"抓取並分析 ({analyze_cooldown})"
    )
    choose_range = st.checkbox("指定日期範圍")
    request_delay = st.number_input(
        "每日頁面抓取延遲（秒）",
        min_value=0.5,
        value=1.0,
        step=0.1,
    )
    if choose_range:
        start_column, end_column = st.columns(2)
        start_date = start_column.date_input("起日")
        end_date = end_column.date_input("迄日")
        days = None
    else:
        days = int(st.number_input("來源頁面最近天數", min_value=1, value=30, step=1))
        start_date = None
        end_date = None
    submitted = st.form_submit_button(
        analyze_label,
        disabled=analyze_cooldown > 0,
        use_container_width=True,
    )

if submitted:
    try:
        with st.spinner("抓取每日歷史資料並產生圖表中..."):
            result = run_pipeline(
                DATA_DIR,
                days=days,
                start_date=start_date,
                end_date=end_date,
                config=ScrapeConfig(delay_seconds=float(request_delay)),
            )
        st.session_state["refresh_notice"] = (
            f"已完成 {result.summary['draw_count']} 期分析。"
        )
        st.session_state["refresh_warnings"] = result.scrape_warnings[:3]
        st.session_state["refresh_error"] = None
    except Exception as exc:
        st.session_state["refresh_notice"] = None
        st.session_state["refresh_warnings"] = []
        st.session_state["refresh_error"] = str(exc)
    finally:
        st.session_state[ANALYZE_COOLDOWN_KEY] = time.time() + 3
        st.rerun()

if st.session_state.get("refresh_notice"):
    st.success(st.session_state["refresh_notice"])
for warning in st.session_state.get("refresh_warnings", []):
    st.warning(warning)
if st.session_state.get("refresh_error"):
    st.error(st.session_state["refresh_error"])

summary = load_summary()
if summary:
    if st.button("以現有 CSV 重算圖表", use_container_width=True):
        try:
            with st.spinner("重建圖表中..."):
                summary = analyze_existing(DATA_DIR)
            st.success(f"已重建 {summary['draw_count']} 期圖表。")
        except Exception as exc:
            st.error(str(exc))
    show_summary(summary)
else:
    st.info("還沒有雲端分析結果。先按上方的「抓取並分析」。")

if cooldown_remaining_seconds(ANALYZE_COOLDOWN_KEY) > 0:
    time.sleep(1)
    st.rerun()
