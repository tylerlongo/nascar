# NASCAR Prediction Dashboard

A local NASCAR Cup Series prediction dashboard that scrapes race data, builds track-type-specific models, and shows predicted finish distributions for the last race, next race, and selected historical races.

## What it does

- Scrapes completed race results from DriverAverages.
- Uses Racing-Reference qualifying results when available.
- Falls back to Racing-Reference entry lists when qualifying is not posted yet.
- Builds driver history features with no future lookahead.
- Trains separate models for:
  - Speedway
  - Superspeedway
  - Road course
- Predicts expected finish, median finish, win %, top 3, top 5, top 10, top 15, top 20, and top 25 probabilities.
- Provides a browser dashboard for filtering, ranking, and comparing drivers.

## Files

```text
app.py          Flask server for the dashboard and API routes
getdata.py      Scrapes race data and builds training/testing datasets
predict.py      Trains models and writes prediction JSON files
dashboard.html  Front-end dashboard
requirements.txt Python dependencies
```

Generated data/prediction files may include:

```text
raw_races_cache.json
training.csv
testing_last_race.json
testing_next_race.json
predictions_last_race.json
predictions_next_race.json
predictions_historical_index.json
predictions_historical_YYYY_RR.json
feature_cache.json
```

## Setup

Install Python 3.10+ if you do not already have it.

Then install the dependencies:

```bash
pip install -r requirements.txt
```

If Racing-Reference scraping needs Playwright, install the browser package too:

```bash
playwright install chromium
```

## Running the dashboard

Start the Flask app:

```bash
python3 app.py
```

Then open this in your browser:

```text
http://localhost:5000
```

## First-time data build

On the first run, click **Refresh data** in the dashboard.

This will:

1. Build or update the raw race cache.
2. Create training/testing datasets.
3. Generate predictions for the last completed race and the next race.

The first refresh can take longer because it may need to scrape many historical races. Later refreshes should be much faster because the cache is reused.

## Dashboard tabs

### Last Race

Shows predictions for the most recently completed race. This uses the actual known track type and actual field/start information from the race data.

### Next Race

Shows predictions for the upcoming race. If qualifying is available, starting positions are used. If qualifying is not available, the app tries the entry list and uses fallback starting-position estimates.

### Historical

Lets you pick a past race and generate a no-lookahead prediction for that race. The model only uses data from races before the selected historical race.

## Ranking and filters

You can rank the field by:

- Expected finish
- Median finish
- Win %

You can also change odds format and filter by manufacturer.

Selecting drivers opens the comparison panel:

- 1 driver: shows driver markets and finish distribution.
- 2 drivers: shows head-to-head comparison.
- 3+ drivers: shows the chance each driver finishes best among the selected group.

## Track types

The app uses three track types:

```text
s   = Speedway
ss  = Superspeedway
rc  = Road course
```

For historical and known next-race predictions, only the known track type is predicted. If the next race track type is unknown, the app falls back to predicting all three track buckets.

## Notes

- This is a local app. It is not meant to be deployed publicly without additional cleanup.
- Racing-Reference may show a Cloudflare challenge. If that happens, solve it in the browser window that opens.
- Predictions are model estimates, not betting advice.
- If results look stale, click **Refresh data**.

## Common commands

Run the full data/prediction pipeline manually:

```bash
python3 getdata.py
python3 predict.py
```

Generate a historical prediction manually:

```bash
python3 predict.py --historical 2024 10
```

Start the dashboard:

```bash
python3 app.py
```
