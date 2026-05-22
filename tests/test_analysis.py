import pandas as pd

from bingo_analysis.analysis import build_appearance_matrix, gap_values, overlap_counts


def test_overlap_and_gap_values_follow_draw_order() -> None:
    history = pd.DataFrame(
        {
            "numbers": [
                list(range(1, 21)),
                list(range(11, 31)),
                list(range(1, 11)) + list(range(31, 41)),
            ]
        }
    )
    matrix = build_appearance_matrix(history)
    gaps, gap_table = gap_values(matrix)

    assert overlap_counts(matrix).tolist() == [10, 0]
    assert set(gaps.tolist()) == {0, 1}
    assert gap_table.loc[gap_table["number"] == 1, "gap_count"].item() == 1
