from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from bingo_analysis.analysis import CHART_FILENAMES
from bingo_analysis.pipeline import analyze_existing, run_pipeline
from bingo_analysis.scraper import ScrapeConfig

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("BINGO_DATA_DIR", PROJECT_ROOT))
OUTPUT_DIR = DATA_DIR / "output"
SUMMARY_PATH = OUTPUT_DIR / "analysis_summary.json"

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


def show_summary(summary: dict[str, Any]) -> None:
    metrics = st.columns(4)
    metrics[0].metric("期數", summary["draw_count"])
    with metrics[1]:
        st.caption("日期")
        st.markdown(f"**{summary['date_start']}**  \n**{summary['date_end']}**")
    metrics[2].metric("相鄰重複平均", f"{summary.get('mean_overlap') or 0:.2f}")
    metrics[3].metric("中位 gap", f"{summary.get('median_gap') or 0:.1f}")

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
st.title("賓果賓果時間分析")
st.caption("台灣 BINGO BINGO 歷史資料探索。圖表是診斷工具，不是未來開獎保證。")

with st.form("refresh"):
    st.subheader("抓取與分析")
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
    submitted = st.form_submit_button("抓取並分析", use_container_width=True)

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
        st.success(f"已完成 {result.summary['draw_count']} 期分析。")
        for warning in result.scrape_warnings[:3]:
            st.warning(warning)
    except Exception as exc:
        st.error(str(exc))

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
