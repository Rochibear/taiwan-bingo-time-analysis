from datetime import date

import pandas as pd
import pytest

from bingo_analysis.crosscheck import (
    CrosscheckError,
    compare_with_backup,
    local_history_for_dates,
    parse_twlottery_bingo_page,
)


HTML = """
<html>
  <body>
    <h2>2026/05/24 BINGO BINGO 賓果賓果開獎號碼</h2>
    <div>
      115029182 期
      <ul>
        <li>06</li><li>11</li><li>12</li><li>24</li><li>28</li>
        <li>32</li><li>33</li><li>37</li><li>40</li><li>44</li>
        <li>45</li><li>48</li><li>50</li><li>53</li><li>60</li>
        <li>61</li><li>71</li><li>75</li><li>76</li><li>80</li>
      </ul>
      超級獎號 ：71
      大小：－
      單雙：－
    </div>
    <div>
      115029181 期
      <ul>
        <li>08</li><li>10</li><li>13</li><li>14</li><li>18</li>
        <li>19</li><li>23</li><li>42</li><li>43</li><li>44</li>
        <li>47</li><li>48</li><li>52</li><li>53</li><li>58</li>
        <li>65</li><li>68</li><li>69</li><li>71</li><li>73</li>
      </ul>
      超級獎號 ：10
      大小：大
      單雙：－
    </div>
  </body>
</html>
"""


def test_parse_twlottery_bingo_page_reads_draws() -> None:
    frame = parse_twlottery_bingo_page(HTML, date(2026, 5, 24))

    assert frame["draw_id"].tolist() == ["115029182", "115029181"]
    assert frame.loc[0, "date"] == "2026-05-24"
    assert frame.loc[0, "numbers"] == [
        6,
        11,
        12,
        24,
        28,
        32,
        33,
        37,
        40,
        44,
        45,
        48,
        50,
        53,
        60,
        61,
        71,
        75,
        76,
        80,
    ]
    assert frame.loc[0, "super_number"] == 71
    assert frame.loc[1, "big_small"] == "大"


def test_parse_twlottery_bingo_page_rejects_wrong_date() -> None:
    with pytest.raises(CrosscheckError):
        parse_twlottery_bingo_page(HTML, date(2026, 5, 23))


def test_compare_with_backup_marks_verified_mismatch_and_missing() -> None:
    history = pd.DataFrame(
        [
            {
                "draw_id": "115029182",
                "date": "2026-05-24",
                "time": "19:45",
                "numbers": [6, 11, 12, 24, 28, 32, 33, 37, 40, 44, 45, 48, 50, 53, 60, 61, 71, 75, 76, 80],
                "super_number": 71,
                "big_small": "－",
                "odd_even": "－",
            },
            {
                "draw_id": "115029181",
                "date": "2026-05-24",
                "time": "19:40",
                "numbers": [8, 10, 13, 14, 18, 19, 23, 42, 43, 44, 47, 48, 52, 53, 58, 65, 68, 69, 71, 73],
                "super_number": 10,
                "big_small": "－",
                "odd_even": "－",
            },
            {
                "draw_id": "115029180",
                "date": "2026-05-24",
                "time": "19:35",
                "numbers": list(range(1, 21)),
                "super_number": 1,
                "big_small": "－",
                "odd_even": "－",
            },
        ]
    )
    local = local_history_for_dates(history, [date(2026, 5, 24)])
    backup = parse_twlottery_bingo_page(HTML, date(2026, 5, 24))

    details = compare_with_backup(local, backup)

    assert details["status"].tolist() == ["missing_backup", "mismatch", "verified"]
    assert details.loc[1, "mismatch_fields"] == "big_small"
