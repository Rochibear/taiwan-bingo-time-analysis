import zipfile

import pandas as pd

from bingo_analysis.official import (
    compare_with_official,
    is_official_download_host,
    parse_official_bingo_zip,
)


def write_official_zip(path):
    header = [
        "遊戲名稱",
        "期別",
        "開獎日期",
        "銷售總額",
        "銷售注數",
        "總獎金",
        *[f"獎號{index}" for index in range(1, 21)],
        "超級獎號",
        "猜大小",
        "猜單雙",
    ]
    row = [
        "賓果賓果",
        "115000001",
        "2026/01/01",
        "0",
        "0",
        "0",
        *[str(number) for number in range(1, 21)],
        "05",
        "－",
        "單",
    ]
    content = ",".join(header) + "\n" + ",".join(row) + "\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("賓果賓果_2026.csv", content.encode("utf-8-sig"))


def local_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "draw_id": "115000001",
                "date": "2026-01-01",
                "time": "07:05",
                "numbers": list(range(1, 21)),
                "super_number": 5,
                "big_small": "－",
                "odd_even": "單",
            },
            {
                "draw_id": "115000002",
                "date": "2026-01-01",
                "time": "07:10",
                "numbers": list(range(2, 22)),
                "super_number": 6,
                "big_small": "－",
                "odd_even": "－",
            },
        ]
    )


def test_parse_official_bingo_zip_reads_bingo_rows(tmp_path) -> None:
    zip_path = tmp_path / "official.zip"
    write_official_zip(zip_path)

    official = parse_official_bingo_zip(zip_path)

    assert official.loc[0, "draw_id"] == "115000001"
    assert official.loc[0, "date"] == "2026-01-01"
    assert official.loc[0, "numbers"] == list(range(1, 21))
    assert official.loc[0, "numbers_key"] == "01;02;03;04;05;06;07;08;09;10;11;12;13;14;15;16;17;18;19;20"


def test_compare_with_official_marks_verified_and_pending(tmp_path) -> None:
    zip_path = tmp_path / "official.zip"
    write_official_zip(zip_path)
    official = parse_official_bingo_zip(zip_path)

    details = compare_with_official(local_history(), official)

    assert details["status"].tolist() == ["verified", "pending_official"]


def test_is_official_download_host_only_allows_taiwan_lottery_domains() -> None:
    assert is_official_download_host("https://api.taiwanlottery.com/example")
    assert is_official_download_host("https://cdn.taiwanlottery.com.tw/example")
    assert not is_official_download_host("https://example.com/taiwanlottery.com")
