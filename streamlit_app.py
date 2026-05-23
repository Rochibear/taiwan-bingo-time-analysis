from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from bingo_analysis import forecast as forecast_module
from bingo_analysis.analysis import (
    CHART_FILENAMES,
    NUMBERS,
    build_appearance_matrix,
    load_history,
)
from bingo_analysis.official import (
    OFFICIAL_NOTE,
    OfficialConfig,
    verify_history_with_official,
)
from bingo_analysis.pipeline import analyze_existing, run_pipeline
from bingo_analysis.scraper import ScrapeConfig

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("BINGO_DATA_DIR", PROJECT_ROOT))
OUTPUT_DIR = DATA_DIR / "output"
SUMMARY_PATH = OUTPUT_DIR / "analysis_summary.json"
OFFICIAL_SUMMARY_PATH = OUTPUT_DIR / "official_verification_summary.json"
OFFICIAL_DETAILS_PATH = OUTPUT_DIR / "official_verification.csv"
ANALYZE_COOLDOWN_KEY = "analyze_cooldown_until"
ANALYZE_COOLDOWN_SECONDS = 3
TAIPEI_TZ = ZoneInfo("Asia/Taipei")
STAR_MIN = getattr(forecast_module, "STAR_MIN", 1)
STAR_MAX = getattr(forecast_module, "STAR_MAX", 10)
build_forecast = forecast_module.build_forecast
backtest_star_selection = getattr(forecast_module, "backtest_star_selection", None)

CHART_TITLES = {
    "number_frequency.png": "1-80 號碼出現次數",
    "overlap_distribution.png": "相鄰兩期重複球數",
    "gap_distribution.png": "號碼再出現 gap 分布",
    "hourly_heatmap.png": "小時別偏號",
    "weekday_heatmap.png": "星期別偏號",
    "autocorrelation.png": "自相關",
    "fft_periodogram.png": "FFT 週期分析",
}

STAR_LABELS = {
    10: "十星（10 個號碼）",
    9: "九星（9 個號碼）",
    8: "八星（8 個號碼）",
    7: "七星（7 個號碼）",
    6: "六星（6 個號碼）",
    5: "五星（5 個號碼）",
    4: "四星（4 個號碼）",
    3: "三星（3 個號碼）",
    2: "二星（2 個號碼）",
    1: "一星（1 個號碼）",
}


def load_summary() -> dict[str, Any] | None:
    if not SUMMARY_PATH.exists():
        return None
    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_official_summary() -> dict[str, Any] | None:
    if not OFFICIAL_SUMMARY_PATH.exists():
        return None
    with OFFICIAL_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def cooldown_remaining_seconds(key: str) -> int:
    remaining = float(st.session_state.get(key, 0.0)) - time.time()
    return max(0, math.ceil(remaining))


def taipei_today():
    return datetime.now(TAIPEI_TZ).date()


def inject_submit_countdown(base_label: str, remaining_seconds: int) -> None:
    components.html(
        f"""
        <script>
        (() => {{
          const baseLabel = {json.dumps(base_label)};
          const storageKey = "bingoAnalyzeCooldownUntil";
          const serverCooldownUntil = Date.now() + {int(remaining_seconds)} * 1000;
          const buttons = Array.from(window.parent.document.querySelectorAll("button"));
          const button = buttons.find((candidate) => {{
            const text = (candidate.textContent || "").trim();
            return text === baseLabel || /^抓取並分析 \\(\\d+\\)$/.test(text);
          }});
          if (!button) {{
            return;
          }}
          if (button.dataset.bingoCountdownBound !== "1") {{
            button.dataset.bingoCountdownBound = "1";
            button.addEventListener("click", () => {{
              const cooldownUntil = Date.now() + 3000;
              window.parent.localStorage.setItem(storageKey, String(cooldownUntil));
              startCountdown(cooldownUntil);
            }}, true);
          }}

          function activeCooldownUntil() {{
            const localCooldownUntil = Number(
              window.parent.localStorage.getItem(storageKey) || 0
            );
            return Math.max(localCooldownUntil, serverCooldownUntil);
          }}

          const setLabel = () => {{
            const remaining = Math.ceil((activeCooldownUntil() - Date.now()) / 1000);
            if (remaining > 0) {{
              button.textContent = `${{baseLabel}} (${{remaining}})`;
              button.disabled = true;
              button.setAttribute("aria-disabled", "true");
            }} else {{
              button.textContent = baseLabel;
              button.disabled = false;
              button.removeAttribute("aria-disabled");
              window.parent.localStorage.removeItem(storageKey);
            }}
          }};

          function startCountdown(cooldownUntil) {{
            window.parent.localStorage.setItem(storageKey, String(cooldownUntil));
            setLabel();
            if (button.dataset.bingoCountdownTimer) {{
              window.parent.clearInterval(Number(button.dataset.bingoCountdownTimer));
            }}
            const timer = window.parent.setInterval(() => {{
              setLabel();
              if (Date.now() >= activeCooldownUntil()) {{
                window.parent.clearInterval(timer);
                delete button.dataset.bingoCountdownTimer;
                setLabel();
              }}
            }}, 250);
            button.dataset.bingoCountdownTimer = String(timer);
          }}

          const cooldownUntil = activeCooldownUntil();
          if (cooldownUntil > Date.now()) {{
            startCountdown(cooldownUntil);
          }} else {{
            setLabel();
          }}
        }})();
        </script>
        """,
        height=0,
    )


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


def star_selection_table(items: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(items)
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "number": "號碼",
            "score": "綜合分數",
            "global_rate": "全期率",
            "recent_rate": "近期率",
            "hourly_rate": "同小時率",
            "current_gap": "目前 gap",
        }
    )
    frame["號碼"] = frame["號碼"].map(lambda value: f"{int(value):02d}")
    frame["綜合分數"] = frame["綜合分數"].map(lambda value: f"{value:.5f}")
    for column in ["全期率", "近期率", "同小時率"]:
        frame[column] = frame[column].map(lambda value: f"{value:.2%}")
    return frame


def backtest_table(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return details
    frame = details.copy()
    frame["selected_numbers"] = frame["selected_numbers"].map(
        lambda numbers: " ".join(f"{int(number):02d}" for number in numbers)
    )
    frame["hit_numbers"] = frame["hit_numbers"].map(
        lambda numbers: " ".join(f"{int(number):02d}" for number in numbers) or "－"
    )
    return frame.rename(
        columns={
            "draw_id": "期別",
            "date": "開獎日期",
            "time": "開獎時間",
            "selected_numbers": "建議號碼",
            "hit_numbers": "命中號碼",
            "hit_count": "命中數",
        }
    )


def normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total <= 0:
        return np.ones_like(values, dtype=float) / len(values)
    return values / total


def fallback_star_selection(history: pd.DataFrame, stars: int) -> dict[str, object]:
    if not STAR_MIN <= stars <= STAR_MAX:
        raise ValueError("stars must be between 1 and 10")

    next_draw_at = forecast_module.next_draw_datetime()
    matrix = build_appearance_matrix(history)
    global_counts = matrix.sum(axis=0).reindex(NUMBERS).to_numpy(dtype=float) + 1.0
    recent_window = getattr(forecast_module, "RECENT_WINDOW_DRAWS", 300)
    recent = matrix.tail(min(recent_window, len(matrix)))
    recent_counts = recent.sum(axis=0).reindex(NUMBERS).to_numpy(dtype=float) + 1.0

    hour_mask = history["datetime"].dt.hour == next_draw_at.hour
    if hour_mask.any():
        hourly_counts = (
            matrix.loc[hour_mask].sum(axis=0).reindex(NUMBERS).to_numpy(dtype=float)
            + 1.0
        )
        hourly_draws = int(hour_mask.sum())
    else:
        hourly_counts = global_counts.copy()
        hourly_draws = len(history)

    current_gaps: list[int] = []
    for number in NUMBERS:
        appearances = np.flatnonzero(matrix[number].to_numpy())
        current_gaps.append(
            len(matrix)
            if len(appearances) == 0
            else len(matrix) - 1 - int(appearances[-1])
        )
    current_gap_values = np.asarray(current_gaps, dtype=float)
    gap_score = np.log1p(current_gap_values) + 1.0
    score = (
        0.45 * normalize_scores(global_counts)
        + 0.30 * normalize_scores(recent_counts)
        + 0.15 * normalize_scores(hourly_counts)
        + 0.10 * normalize_scores(gap_score)
    )

    ranked = pd.DataFrame(
        {
            "number": NUMBERS,
            "score": score,
            "global_rate": (global_counts - 1.0) / max(len(history), 1),
            "recent_rate": (recent_counts - 1.0) / max(len(recent), 1),
            "hourly_rate": (hourly_counts - 1.0) / max(hourly_draws, 1),
            "current_gap": current_gap_values.astype(int),
        }
    ).sort_values(["score", "number"], ascending=[False, True])
    selected = ranked.head(stars).copy()
    return {
        "stars": stars,
        "next_draw_label": next_draw_at.strftime("%Y-%m-%d %H:%M"),
        "selected_numbers": sorted(
            int(number) for number in selected["number"].tolist()
        ),
        "selected_details": selected.to_dict(orient="records"),
        "model_note": "全期頻率 45% + 近期頻率 30% + 同小時偏號 15% + 近期 gap 10%",
    }


def build_star_selection(history: pd.DataFrame, stars: int) -> dict[str, object]:
    builder = getattr(forecast_module, "build_star_selection", None)
    if builder:
        return builder(history, stars)
    return fallback_star_selection(history, stars)


def history_draw_table(limit: int = 50) -> pd.DataFrame:
    history = load_history(DATA_DIR / "bingo_history.csv")
    frame = history.sort_values(["datetime", "draw_id"], ascending=False).head(limit)
    frame = frame[
        [
            "draw_id",
            "date",
            "time",
            "numbers",
            "super_number",
            "big_small",
            "odd_even",
        ]
    ].copy()
    frame["numbers"] = frame["numbers"].map(
        lambda numbers: " ".join(f"{int(number):02d}" for number in numbers)
    )
    frame["super_number"] = frame["super_number"].map(lambda value: f"{int(value):02d}")
    return frame.rename(
        columns={
            "draw_id": "期別",
            "date": "開獎日期",
            "time": "開獎時間",
            "numbers": "20 個號碼",
            "super_number": "超級獎號",
            "big_small": "猜大小",
            "odd_even": "猜單雙",
        }
    )


def show_history_draws() -> None:
    st.caption("完整資料仍會存成 bingo_history.csv；下方先顯示最新 50 期。")
    try:
        st.dataframe(
            history_draw_table(),
            hide_index=True,
            use_container_width=True,
        )
        csv_path = DATA_DIR / "bingo_history.csv"
        if csv_path.exists():
            st.download_button(
                "下載完整 bingo_history.csv",
                data=csv_path.read_bytes(),
                file_name="bingo_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
    except Exception as exc:
        st.info(f"尚無可顯示的過往開獎解析：{exc}")


def official_status_table(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return details
    frame = details.copy()
    frame["status"] = frame["status"].map(
        {
            "verified": "通過",
            "mismatch": "不一致",
            "pending_official": "官方尚未更新",
        }
    ).fillna(frame["status"])
    return frame.rename(
        columns={
            "draw_id": "期別",
            "date": "開獎日期",
            "time": "開獎時間",
            "status": "校驗狀態",
            "mismatch_fields": "不一致欄位",
        }
    )


def show_official_verification() -> None:
    st.caption(OFFICIAL_NOTE)
    force_refresh = st.checkbox("重新下載官方年度檔", value=False)
    if st.button("執行官方資料校驗", use_container_width=True):
        try:
            with st.spinner("下載官方年度資料並比對中..."):
                result = verify_history_with_official(
                    DATA_DIR,
                    config=OfficialConfig(delay_seconds=1.0),
                    force_refresh=force_refresh,
                )
            st.success(
                "官方校驗完成："
                f"{result.summary['verified_count']} 通過，"
                f"{result.summary['mismatch_count']} 不一致，"
                f"{result.summary['pending_official_count']} 尚待官方更新。"
            )
        except Exception as exc:
            st.error(str(exc))

    summary = load_official_summary()
    if not summary:
        st.info("尚未執行官方資料校驗。")
        return

    metrics = st.columns(4)
    metrics[0].metric("官方最新日期", summary.get("latest_official_date") or "－")
    metrics[1].metric("校驗通過", summary.get("verified_count", 0))
    metrics[2].metric("不一致", summary.get("mismatch_count", 0))
    metrics[3].metric("官方尚未更新", summary.get("pending_official_count", 0))

    verification_rate = summary.get("verification_rate")
    if verification_rate is not None:
        st.caption(f"可比對資料通過率：{verification_rate:.2%}")

    if OFFICIAL_DETAILS_PATH.exists():
        details = pd.read_csv(OFFICIAL_DETAILS_PATH, dtype={"draw_id": "string"})
        focus = details[details["status"] != "verified"].tail(100)
        if focus.empty:
            focus = details.tail(100)
        st.dataframe(
            official_status_table(focus),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "下載官方校驗明細 CSV",
            data=OFFICIAL_DETAILS_PATH.read_bytes(),
            file_name="official_verification.csv",
            mime="text/csv",
            use_container_width=True,
        )


def verified_draw_ids_from_details() -> set[str] | None:
    if not OFFICIAL_DETAILS_PATH.exists():
        return None
    details = pd.read_csv(OFFICIAL_DETAILS_PATH, dtype={"draw_id": "string"})
    verified = details.loc[details["status"] == "verified", "draw_id"].dropna()
    if verified.empty:
        return None
    return set(verified.astype(str))


def show_prediction_backtest() -> None:
    if backtest_star_selection is None:
        st.info("目前版本尚未載入回測模組，請等待 Streamlit Cloud 重新部署。")
        return

    try:
        history = load_history(DATA_DIR / "bingo_history.csv")
    except Exception:
        st.info("預測回測需要先有 bingo_history.csv。請先按「抓取並分析」。")
        return

    controls = st.columns(2)
    with controls[0]:
        star_options = list(range(STAR_MAX, STAR_MIN - 1, -1))
        selected_stars = st.selectbox(
            "回測星數",
            options=star_options,
            format_func=lambda stars: STAR_LABELS[stars],
            key="backtest_star_count",
        )
    with controls[1]:
        evaluation_draws = st.slider(
            "回測最近期數",
            min_value=50,
            max_value=500,
            value=300,
            step=50,
        )

    verified_ids = verified_draw_ids_from_details()
    use_verified_only = False
    if verified_ids:
        use_verified_only = st.checkbox(
            "只回測官方已通過校驗的期數",
            value=True,
        )
    else:
        st.caption("尚未有官方校驗明細；此處先使用本機歷史資料回測。")

    try:
        result = backtest_star_selection(
            history,
            stars=selected_stars,
            evaluation_draws=evaluation_draws,
            verified_draw_ids=verified_ids if use_verified_only else None,
        )
    except Exception as exc:
        st.info(f"目前無法產生回測：{exc}")
        return

    summary = result["summary"]
    metrics = st.columns(4)
    metrics[0].metric("回測期數", summary["checked_count"])
    metrics[1].metric("平均命中", f"{summary['mean_hits']:.2f}")
    metrics[2].metric("至少中 1 號", f"{summary['hit_rate']:.2%}")
    metrics[3].metric("零命中", f"{summary['zero_hit_rate']:.2%}")

    lift = summary.get("lift_vs_random")
    lift_label = f"{lift:.2f}x" if lift is not None else "－"
    st.caption(
        f"同星數隨機平均命中約 {summary['random_mean_hits']:.2f}；"
        f"模型相對隨機平均命中：{lift_label}。"
    )

    details = result["details"]
    st.dataframe(
        backtest_table(details.tail(50)),
        hide_index=True,
        use_container_width=True,
    )


def show_star_selection(history: pd.DataFrame) -> None:
    st.markdown("#### 建議選號區")
    star_options = list(range(STAR_MAX, STAR_MIN - 1, -1))
    selected_stars = st.selectbox(
        "選擇星數",
        options=star_options,
        format_func=lambda stars: STAR_LABELS[stars],
        key="star_selection_count",
    )
    selection = build_star_selection(history, selected_stars)
    st.caption(
        f"80 選 {selected_stars}｜推定下一期時間：{selection['next_draw_label']}"
    )
    st.markdown(
        chip_list(selection["selected_numbers"], "#6366f1"),
        unsafe_allow_html=True,
    )
    st.caption(f"模型：{selection['model_note']}")
    st.dataframe(
        star_selection_table(selection["selected_details"]),
        hide_index=True,
        use_container_width=True,
    )


def show_forecast() -> None:
    try:
        history = load_history(DATA_DIR / "bingo_history.csv")
        forecast = build_forecast(history)
    except Exception:
        st.info("預告區需要先有 bingo_history.csv。請先按「抓取並分析」。")
        return

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

    st.divider()
    show_star_selection(history)


def show_summary(summary: dict[str, Any]) -> None:
    metrics = st.columns(4)
    metrics[0].metric("期數", summary["draw_count"])
    with metrics[1]:
        st.caption("日期")
        st.markdown(f"**{summary['date_start']}**  \n**{summary['date_end']}**")
    metrics[2].metric("相鄰重複平均", f"{summary.get('mean_overlap') or 0:.2f}")
    metrics[3].metric("中位 gap", f"{summary.get('median_gap') or 0:.1f}")

    with st.expander("預告區", expanded=True):
        show_forecast()

    with st.expander("過往開獎號碼解析", expanded=False):
        show_history_draws()

    with st.expander("官方資料校驗", expanded=False):
        show_official_verification()

    with st.expander("預測回測", expanded=False):
        show_prediction_backtest()

    with st.expander("熱號 / 冷號", expanded=False):
        hot, cold = st.columns(2)
        with hot:
            st.markdown("#### 熱號")
            st.dataframe(
                number_table(summary["hot_numbers"]),
                hide_index=True,
                use_container_width=True,
            )
        with cold:
            st.markdown("#### 冷號")
            st.dataframe(
                number_table(summary["cold_numbers"]),
                hide_index=True,
                use_container_width=True,
            )

    with st.expander("圖表", expanded=False):
        for left_index in range(0, len(CHART_FILENAMES), 2):
            chart_columns = st.columns(2)
            for column, filename in zip(
                chart_columns,
                CHART_FILENAMES[left_index : left_index + 2],
            ):
                chart_path = OUTPUT_DIR / filename
                if chart_path.exists():
                    column.image(str(chart_path), caption=CHART_TITLES[filename])

    with st.expander("FFT 高能量週期", expanded=False):
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
    html {
        scroll-behavior: smooth;
    }
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
    div[data-testid="stExpander"] {
        border-radius: 8px;
        margin-bottom: 0.65rem;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 1rem;
        font-weight: 800;
    }
    .back-to-top {
        align-items: center;
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 999px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
        color: #ffffff !important;
        display: inline-flex;
        font-size: 0.78rem;
        font-weight: 800;
        height: 2.75rem;
        justify-content: center;
        letter-spacing: 0;
        position: fixed;
        right: 18px;
        text-decoration: none !important;
        top: 33vh;
        width: 2.75rem;
        z-index: 9999;
    }
    .back-to-top:hover {
        background: #1f2937;
        color: #ffffff !important;
        text-decoration: none !important;
    }
    @media (max-width: 640px) {
        .back-to-top {
            height: 2.5rem;
            right: 12px;
            top: 34vh;
            width: 2.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div id="page-top"></div><a class="back-to-top" href="#page-top">TOP</a>',
    unsafe_allow_html=True,
)
st.title("賓果賓果時間分析")
st.caption("台灣 BINGO BINGO 歷史資料探索。圖表是診斷工具，不是未來開獎保證。")

choose_range = st.checkbox("指定日期範圍", key="choose_date_range")

with st.form("refresh"):
    st.subheader("抓取與分析")
    request_delay = st.number_input(
        "每日頁面抓取延遲（秒）",
        min_value=0.5,
        value=1.0,
        step=0.1,
    )
    if choose_range:
        today = taipei_today()
        start_column, end_column = st.columns(2)
        start_date = start_column.date_input(
            "起日",
            value=today - timedelta(days=7),
            max_value=today,
        )
        end_date = end_column.date_input("迄日", value=today, max_value=today)
        days = None
    else:
        days = int(st.number_input("來源頁面最近天數", min_value=1, value=30, step=1))
        start_date = None
        end_date = None
    submitted = st.form_submit_button(
        "抓取並分析",
        use_container_width=True,
    )

if submitted:
    remaining_cooldown = cooldown_remaining_seconds(ANALYZE_COOLDOWN_KEY)
    if remaining_cooldown > 0:
        st.session_state["refresh_notice"] = None
        st.session_state["refresh_warnings"] = []
        st.session_state["refresh_error"] = f"請再等 {remaining_cooldown} 秒後再抓取。"
        st.rerun()
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
        st.session_state[ANALYZE_COOLDOWN_KEY] = time.time() + ANALYZE_COOLDOWN_SECONDS
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

inject_submit_countdown("抓取並分析", cooldown_remaining_seconds(ANALYZE_COOLDOWN_KEY))
