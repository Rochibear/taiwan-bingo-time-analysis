from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

CHART_FILENAMES = [
    "number_frequency.png",
    "overlap_distribution.png",
    "gap_distribution.png",
    "hourly_heatmap.png",
    "weekday_heatmap.png",
    "autocorrelation.png",
    "fft_periodogram.png",
]
NUMBERS = list(range(1, 81))


class AnalysisError(RuntimeError):
    """Raised when stored history cannot be analyzed."""


def parse_numbers(raw: object) -> list[int]:
    if isinstance(raw, list):
        values = [int(value) for value in raw]
    elif isinstance(raw, str):
        values = [int(value) for value in raw.split(";") if value.strip()]
    else:
        raise AnalysisError(f"unsupported numbers value {raw!r}")

    if len(values) != 20 or len(set(values)) != 20:
        raise AnalysisError(f"expected 20 unique numbers, got {values!r}")
    if min(values) < 1 or max(values) > 80:
        raise AnalysisError(f"numbers must stay within 1..80, got {values!r}")
    return values


def load_history(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise AnalysisError(f"history CSV does not exist: {csv_path}")

    history = pd.read_csv(csv_path, dtype={"draw_id": "string"})
    expected = {
        "draw_id",
        "date",
        "time",
        "numbers",
        "super_number",
        "big_small",
        "odd_even",
    }
    missing = expected.difference(history.columns)
    if missing:
        raise AnalysisError(f"history CSV missed columns: {sorted(missing)}")

    history = history.copy()
    history["numbers"] = history["numbers"].map(parse_numbers)
    history["datetime"] = pd.to_datetime(
        history["date"].astype(str) + " " + history["time"].astype(str),
        errors="raise",
    )
    history["super_number"] = history["super_number"].astype(int)
    history = history.sort_values(["datetime", "draw_id"]).drop_duplicates("draw_id")
    history = history.reset_index(drop=True)
    if history.empty:
        raise AnalysisError("history CSV has no draw rows")
    return history


def build_appearance_matrix(history: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=history.index, columns=NUMBERS, dtype=np.int8)
    for row_index, numbers in history["numbers"].items():
        matrix.loc[row_index, numbers] = 1
    return matrix


def overlap_counts(matrix: pd.DataFrame) -> np.ndarray:
    if len(matrix) < 2:
        return np.array([], dtype=int)
    return np.logical_and(matrix.iloc[:-1].to_numpy(), matrix.iloc[1:].to_numpy()).sum(
        axis=1
    )


def gap_values(matrix: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    all_gaps: list[int] = []
    rows: list[dict[str, float | int]] = []

    for number in NUMBERS:
        positions = np.flatnonzero(matrix[number].to_numpy())
        gaps = np.diff(positions) - 1
        all_gaps.extend(gaps.astype(int).tolist())
        rows.append(
            {
                "number": number,
                "appearances": int(len(positions)),
                "gap_count": int(len(gaps)),
                "mean_gap": float(np.mean(gaps)) if len(gaps) else np.nan,
                "median_gap": float(np.median(gaps)) if len(gaps) else np.nan,
                "max_gap": int(np.max(gaps)) if len(gaps) else np.nan,
            }
        )

    return np.asarray(all_gaps, dtype=int), pd.DataFrame(rows)


def rate_deviation_by_bucket(
    history: pd.DataFrame,
    matrix: pd.DataFrame,
    bucket: pd.Series,
    bucket_index: list[int],
) -> pd.DataFrame:
    grouped = matrix.groupby(bucket).mean().reindex(bucket_index)
    return grouped.subtract(matrix.mean(axis=0), axis=1).T


def mean_autocorrelation(matrix: pd.DataFrame, max_lag: int = 200) -> pd.DataFrame:
    if len(matrix) < 2:
        return pd.DataFrame({"lag": [], "mean_autocorrelation": []})

    centered = matrix.to_numpy(dtype=float) - matrix.mean(axis=0).to_numpy(dtype=float)
    denominator = np.square(centered).sum(axis=0)
    usable = denominator > 0
    final_lag = min(max_lag, len(matrix) - 1)
    rows: list[dict[str, float | int]] = []

    for lag in range(1, final_lag + 1):
        numerator = (centered[:-lag] * centered[lag:]).sum(axis=0)
        values = numerator[usable] / denominator[usable]
        rows.append(
            {
                "lag": lag,
                "mean_autocorrelation": float(np.nanmean(values)),
            }
        )
    return pd.DataFrame(rows)


def mean_fft_periodogram(matrix: pd.DataFrame) -> pd.DataFrame:
    if len(matrix) < 4:
        return pd.DataFrame({"period_draws": [], "mean_power": []})

    centered = matrix.to_numpy(dtype=float) - matrix.mean(axis=0).to_numpy(dtype=float)
    spectrum = np.fft.rfft(centered, axis=0)
    frequencies = np.fft.rfftfreq(len(matrix), d=1.0)
    mean_power = np.square(np.abs(spectrum)).mean(axis=1) / len(matrix)
    usable = frequencies > 0
    result = pd.DataFrame(
        {
            "period_draws": 1 / frequencies[usable],
            "mean_power": mean_power[usable],
        }
    )
    return result.sort_values("period_draws").reset_index(drop=True)


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d7d2", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def _save_empty_plot(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_number_frequency(path: Path, frequencies: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(14, 5.2))
    colors = ["#007b83" if value >= frequencies.mean() else "#c44d58" for value in frequencies]
    ax.bar(frequencies.index, frequencies.values, color=colors, width=0.82)
    ax.axhline(frequencies.mean(), color="#222222", linewidth=1.1, linestyle="--")
    ax.set_title("BINGO BINGO Number Frequency")
    ax.set_xlabel("Number")
    ax.set_ylabel("Appearances")
    ax.set_xticks(np.arange(1, 81, 4))
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_overlap_distribution(path: Path, overlaps: np.ndarray) -> None:
    if len(overlaps) == 0:
        _save_empty_plot(path, "Adjacent Draw Overlap", "Need at least two draws.")
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bins = np.arange(overlaps.min() - 0.5, overlaps.max() + 1.5, 1)
    ax.hist(overlaps, bins=bins, color="#007b83", edgecolor="#f7f5ef")
    ax.axvline(overlaps.mean(), color="#c44d58", linewidth=2)
    ax.axvline(5, color="#222222", linewidth=1.1, linestyle="--")
    ax.set_title("Repeated Balls Between Adjacent Draws")
    ax.set_xlabel("Overlap count")
    ax.set_ylabel("Adjacent draw pairs")
    ax.set_xticks(np.arange(overlaps.min(), overlaps.max() + 1))
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_gap_distribution(path: Path, gaps: np.ndarray) -> None:
    if len(gaps) == 0:
        _save_empty_plot(path, "Gap Distribution", "Need repeated appearances.")
        return

    capped_max = max(12, int(np.quantile(gaps, 0.995)))
    clipped = np.clip(gaps, 0, capped_max)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bins = np.arange(-0.5, capped_max + 1.5, 1)
    ax.hist(clipped, bins=bins, color="#e0a526", edgecolor="#f7f5ef")
    ax.set_title("Draw Gaps Before Each Number Reappears")
    ax.set_xlabel(f"Intervening draws, values above {capped_max} capped")
    ax.set_ylabel("Reappearance events")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_rate_heatmap(
    path: Path,
    data: pd.DataFrame,
    title: str,
    x_labels: list[str],
    x_title: str,
) -> None:
    array = data.to_numpy(dtype=float)
    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        _save_empty_plot(path, title, "No usable time buckets.")
        return

    bound = max(float(np.nanmax(np.abs(finite_values))), 0.01)
    fig_height = 12 if len(data) > 20 else 7
    fig, ax = plt.subplots(figsize=(12, fig_height))
    masked = np.ma.masked_invalid(array)
    image = ax.imshow(
        masked,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
    )
    ax.set_title(title)
    ax.set_xlabel(x_title)
    ax.set_ylabel("Number")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(0, 80, 4))
    ax.set_yticklabels([str(number) for number in NUMBERS[::4]])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    colorbar.set_label("Appearance rate deviation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_autocorrelation(path: Path, autocorrelation: pd.DataFrame) -> None:
    if autocorrelation.empty:
        _save_empty_plot(path, "Mean Autocorrelation", "Need at least two draws.")
        return

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(
        autocorrelation["lag"],
        autocorrelation["mean_autocorrelation"],
        color="#007b83",
        linewidth=1.8,
    )
    ax.axhline(0, color="#222222", linewidth=1)
    ax.set_title("Mean Number Appearance Autocorrelation")
    ax.set_xlabel("Lag in draws")
    ax.set_ylabel("Mean autocorrelation")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_fft_periodogram(path: Path, periodogram: pd.DataFrame) -> None:
    if periodogram.empty:
        _save_empty_plot(path, "FFT Periodogram", "Need at least four draws.")
        return

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(
        periodogram["period_draws"],
        periodogram["mean_power"],
        color="#c44d58",
        linewidth=1.6,
    )
    ax.set_xscale("log")
    ax.set_title("Mean FFT Power by Period")
    ax.set_xlabel("Period in draws, log scale")
    ax.set_ylabel("Mean power")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_history(csv_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history = load_history(csv_path)
    matrix = build_appearance_matrix(history)
    frequencies = matrix.sum(axis=0).astype(int)
    frequency_table = pd.DataFrame(
        {
            "number": frequencies.index,
            "appearances": frequencies.values,
            "appearance_rate": (frequencies / len(history)).values,
        }
    )

    overlaps = overlap_counts(matrix)
    gaps, gap_table = gap_values(matrix)
    hourly = rate_deviation_by_bucket(
        history,
        matrix,
        history["datetime"].dt.hour,
        list(range(24)),
    )
    weekday = rate_deviation_by_bucket(
        history,
        matrix,
        history["datetime"].dt.weekday,
        list(range(7)),
    )
    autocorrelation = mean_autocorrelation(matrix)
    periodogram = mean_fft_periodogram(matrix)

    save_number_frequency(output_dir / "number_frequency.png", frequencies)
    save_overlap_distribution(output_dir / "overlap_distribution.png", overlaps)
    save_gap_distribution(output_dir / "gap_distribution.png", gaps)
    save_rate_heatmap(
        output_dir / "hourly_heatmap.png",
        hourly,
        "Hourly Number Bias",
        [f"{hour:02d}" for hour in range(24)],
        "Hour",
    )
    save_rate_heatmap(
        output_dir / "weekday_heatmap.png",
        weekday,
        "Weekday Number Bias",
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Weekday",
    )
    save_autocorrelation(output_dir / "autocorrelation.png", autocorrelation)
    save_fft_periodogram(output_dir / "fft_periodogram.png", periodogram)

    frequency_table.to_csv(output_dir / "number_frequency.csv", index=False)
    gap_table.to_csv(output_dir / "gap_by_number.csv", index=False)
    autocorrelation.to_csv(output_dir / "autocorrelation.csv", index=False)
    periodogram.to_csv(output_dir / "fft_periodogram.csv", index=False)

    hot = frequency_table.sort_values(
        ["appearances", "number"], ascending=[False, True]
    ).head(10)
    cold = frequency_table.sort_values(
        ["appearances", "number"], ascending=[True, True]
    ).head(10)
    dominant_periods = periodogram.sort_values("mean_power", ascending=False).head(8)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "draw_count": int(len(history)),
        "date_start": history["date"].iloc[0],
        "date_end": history["date"].iloc[-1],
        "draw_start": str(history["draw_id"].iloc[0]),
        "draw_end": str(history["draw_id"].iloc[-1]),
        "mean_overlap": float(np.mean(overlaps)) if len(overlaps) else None,
        "median_gap": float(np.median(gaps)) if len(gaps) else None,
        "hot_numbers": hot.to_dict(orient="records"),
        "cold_numbers": cold.to_dict(orient="records"),
        "dominant_periods": dominant_periods.to_dict(orient="records"),
        "charts": CHART_FILENAMES,
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary

