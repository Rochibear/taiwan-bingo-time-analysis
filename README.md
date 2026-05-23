# Taiwan BINGO BINGO Time Analysis

This Python project scrapes daily BINGO BINGO history pages from Pilio with
`requests` and `BeautifulSoup`, saves `bingo_history.csv`, and builds an
exploratory time-pattern dashboard for desktop and mobile browsers.

The source page currently exposes a date form using the `indate` query
parameter. The scraper keeps requests serial, retries transient failures, and
sleeps between daily pages.

## Setup

```powershell
cd "C:\Users\USER\Documents\New project\taiwan-bingo-time-analysis"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Scrape and analyze

Scrape the dates shown by the source page and generate every chart:

```powershell
bingo-time run --days 30
```

Use an explicit date range when the source site still serves those dates:

```powershell
bingo-time run --start-date 2026-05-01 --end-date 2026-05-21
```

The default outputs are:

- `bingo_history.csv`
- `output/number_frequency.png`
- `output/overlap_distribution.png`
- `output/gap_distribution.png`
- `output/hourly_heatmap.png`
- `output/weekday_heatmap.png`
- `output/autocorrelation.png`
- `output/fft_periodogram.png`
- `output/analysis_summary.json`

The CSV columns are `draw_id`, `date`, `time`, `numbers`, `super_number`,
`big_small`, and `odd_even`. The `numbers` column stores the 20 balls as
semicolon-separated two-digit values.

## Web app

```powershell
python app.py
```

The Flask dashboard opens on `http://127.0.0.1:5000`. To browse it from a phone
on the same network, use the computer's LAN address with port `5000` after the
local firewall permits that connection.

## Free cloud URL with Streamlit Community Cloud

For a free personal cloud URL, deploy `streamlit_app.py` on Streamlit Community
Cloud:

1. Sign in to Streamlit Community Cloud with GitHub.
2. Choose **Create app**.
3. Select this GitHub repository and the `main` branch.
4. Set the app entrypoint to `streamlit_app.py`.
5. Deploy and open the generated `.streamlit.app` URL.
6. On the first cloud visit, press **抓取並分析** once to build the runtime CSV
   and charts.

The Streamlit app uses the same scraper, pandas analysis, PNG outputs, retry
logic, and request delay as the Flask dashboard.

The app also includes a **預告區**:

- 下一期候選號碼：uses global frequency, recent frequency, and same-hour bias to produce
  one exploratory 20-number candidate set for the next draw time.
- 預測連號：shows consecutive-number pairs found in the candidate set and ranked
  historical consecutive-pair candidates.
- The forecast area is explicitly marked as an entertainment/statistical
  diagnostic, not a prediction guarantee or betting recommendation.

## Optional Flask cloud host

This repository also includes a `Procfile` for Flask hosts that accept a
Gunicorn start command:

```text
web: gunicorn --bind :$PORT --workers 1 --threads 4 --timeout 600 app:app
```

Useful Flask host environment variables:

- `FLASK_SECRET_KEY`: a long random secret for Flask sessions.
- `BINGO_DATA_DIR`: optional runtime folder for `bingo_history.csv` and
  generated charts.
- `PORT`: the cloud host port for the Gunicorn command in `Procfile`.

The generated CSV and charts are runtime data. A free cloud app can start with
an empty runtime filesystem after a redeploy or restart, so both dashboards are
designed to rebuild them from the source page.

## Analysis notes

- Number frequency counts appearances of each ball from 1 to 80.
- Hot and cold numbers are the top and bottom ten frequencies in the selected
  dataset.
- Overlap is the count of balls repeated from one draw to the next draw.
- Gap is the number of intervening draws before the same number appears again.
- Hourly and weekday heatmaps show per-number appearance-rate deviation from
  that number's overall rate.
- Autocorrelation and FFT operate on the per-draw binary appearance series for
  each number, then average the results across 80 numbers.
- Forecast candidates are deterministic for the same dataset and next draw time,
  so the page does not reshuffle numbers on every reload.

These plots are exploratory diagnostics. A visible peak or hot number in a
finite sample is not evidence that future lottery draws are predictable.
