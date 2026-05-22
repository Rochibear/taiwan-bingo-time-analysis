from datetime import date

from bingo_analysis.scraper import parse_history_page


HTML = """
<table id="ltotable">
  <tr><td>2026/5/21 BINGO BINGO 賓果賓果開獎號碼</td></tr>
  <tr><td>
    <span>【期別: 115028421】</span><br>
    08, 12, 14, 19, 21, 24, 30, 40, 42, 43,
    52, 53, <span>58</span>, 59, 66, 69, 70, 72, 74, 76<br>
    超級獎號:<span>58</span> _ 猜大小:<span>－</span> _
    猜單雙:<span>雙</span> <span>(07:05)</span>
  </td></tr>
</table>
"""


def test_parse_history_page_extracts_draw_fields() -> None:
    records = parse_history_page(HTML, requested_date=date(2026, 5, 21))

    assert records == [
        {
            "draw_id": "115028421",
            "date": "2026-05-21",
            "time": "07:05",
            "numbers": [
                8,
                12,
                14,
                19,
                21,
                24,
                30,
                40,
                42,
                43,
                52,
                53,
                58,
                59,
                66,
                69,
                70,
                72,
                74,
                76,
            ],
            "super_number": 58,
            "big_small": "－",
            "odd_even": "雙",
        }
    ]

