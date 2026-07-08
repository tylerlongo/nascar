import requests
from bs4 import BeautifulSoup
import numpy as np
import json
import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import argparse

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEASONS = range(2010, datetime.now().year + 1)
RACES_PER_SEASON = 36
MAX_WORKERS      = 12
USE_RR_ENTRY_LIST = True
RR_HEADLESS       = False  # headless=True is more likely to hit Cloudflare

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

TRACK_TYPES = {
    "Daytona":                "ss",
    "Talladega":              "ss",
    "Atlanta":                "ss",
    "Sonoma":                 "rc",
    "Watkins Glen":           "rc",
    "Charlotte Roval":        "rc",
    "COTA":                   "rc",
    "Road America":           "rc",
    "Indy Road Course":       "rc",
    "Chicago Street":         "rc",
    "Daytona Road Course":    "rc",
    "Richmond":               "s",
    "Phoenix":                "s",
    "Martinsville":           "s",
    "Charlotte":              "s",
    "Texas":                  "s",
    "Kansas":                 "s",
    "Dover":                  "s",
    "Bristol":                "s",
    "New Hampshire":          "s",
    "Pocono":                 "s",
    "Las Vegas":              "s",
    "Michigan":               "s",
    "Darlington":             "s",
    "California (Auto Club)": "s",
    "Homestead":              "s",
    "Indianapolis":           "s",
    "Nashville":              "s",
    "Bristol Dirt":           "s",
    "Gateway (WWT)":          "s",
    "Kentucky":               "s",
    "Chicagoland":            "s",
    "Iowa":                   "s",
    "Mexico City":            "rc",
    "San Diego":              "rc",
}

def get_track_type(track_name, season):
    if track_name == "Atlanta" and season < 2022:
        return "s"
    return TRACK_TYPES.get(track_name)

METRICS = ["finish", "mid_pos", "closer_pos", "high_pos", "low_pos", "avg_pos"]
PCTS    = [10, 25, 50, 75]

# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------
session = requests.Session()
session.headers.update(HEADERS)

def fetch_race(season, race_num):
    """Fetch and parse one race page. Returns (season, race_num, track_name, drivers) or None."""
    sked_id = f"{season}{race_num:03d}"
    url = f"https://www.driveraverages.com/nascar/race.php?sked_id={sked_id}"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            return None
    except Exception as e:
        print(f"  Request error {sked_id}: {e}")
        return None

    soup = BeautifulSoup(resp.content, "html.parser")
    tables = soup.find_all("table")
    if len(tables) < 6:
        return None

    # Identify track from page title or headings
    track_name = None
    for tag in soup.find_all(["title", "h1", "h2", "h3", "b", "strong"]):
        text = tag.get_text(strip=True)
        for t in TRACK_TYPES:
            if t.lower() in text.lower():
                track_name = t
                break
        if track_name:
            break
    if track_name is None:
        print(f"  WARNING: no track identified for {sked_id}")
        return None

    # --- Main results table (tables[2]) ---
    # Columns: Finish, Start, #, Driver, Make, Pts, Laps, Led, Status, Team, ...
    results = {}
    main_header_found = False
    for row in tables[2].find_all("tr"):
        cells = row.find_all(["th", "td"])
        texts = [c.get_text(strip=True) for c in cells]
        if not texts:
            continue
        if not main_header_found:
            if texts[0] == "Finish":
                main_header_found = True
            continue
        try:
            finish       = int(texts[0])
            start        = int(texts[1])          # <-- NEW: capture start position
            car_num      = texts[2].lstrip("#")
            driver       = texts[3]
            manufacturer = normalize_make(texts[4]) if len(texts) > 4 else ""
            team         = clean_cell(texts[9]) if len(texts) > 9 else ""
        except (ValueError, IndexError):
            continue
        results[driver] = {
            "car_num":      car_num,
            "finish":       finish,
            "start":        start,                # <-- NEW
            "manufacturer": manufacturer,
            "team":         team,
        }

    # --- Loop data table (tables[5]) ---
    loop_table = tables[5]
    col_map = {}
    header_found = False
    for row in loop_table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        texts = [c.get_text(strip=True) for c in cells]
        if not texts:
            continue

        if not header_found:
            if texts[0] == "Finish" and len(texts) > 5:
                for i, t in enumerate(texts):
                    tl = t.lower().replace(" ", "")
                    if tl == "midpos":    col_map["mid_pos"]    = i
                    if tl == "closerpos": col_map["closer_pos"] = i
                    if tl == "highpos":   col_map["high_pos"]   = i
                    if tl == "lowpos":    col_map["low_pos"]    = i
                    if tl == "avgpos":    col_map["avg_pos"]    = i
                header_found = True
            continue

        driver = texts[1] if len(texts) > 1 else ""
        if driver not in results:
            continue
        try:
            for key, idx in col_map.items():
                results[driver][key] = float(texts[idx])
        except (ValueError, IndexError):
            continue

    # Keep only drivers with complete loop data
    complete = {
        d: v for d, v in results.items()
        if all(k in v for k in ["mid_pos","closer_pos","high_pos","low_pos","avg_pos"])
    }
    return (season, race_num, track_name, complete)

def scrape_all():
    jobs = [(s, r) for s in SEASONS for r in range(1, RACES_PER_SEASON + 1)]
    raw  = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_race, s, r): (s, r) for s, r in jobs}
        done = 0
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result is not None:
                s, r, track, drivers = result
                raw[(s, r)] = (track, drivers)
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)} races scraped...")
    return raw


# ---------------------------------------------------------------------------
# Racing-Reference entry list + qualifying scraping
# ---------------------------------------------------------------------------
def norm_driver_name(name):
    """Loose key for matching DriverAverages names to Racing-Reference names."""
    name = (name or "").lower()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\.?\b", "", name)
    name = re.sub(r"[^a-z0-9]+", "", name)
    return name

def clean_cell(text):
    return re.sub(r"\s+", " ", text or "").strip()

def normalize_make(make):
    m = clean_cell(make)
    ml = m.lower()
    if "chev" in ml: return "Chevrolet"
    if "ford" in ml: return "Ford"
    if "toyota" in ml: return "Toyota"
    return m

# How long to wait for the user to solve a Cloudflare challenge (seconds).
RR_CF_TIMEOUT = 60

def _rr_page_html_and_text(url):
    """
    Open a Racing-Reference page in Playwright and return (html, lines).

    Cloudflare handling: if the page shows a challenge, print a message and
    wait up to RR_CF_TIMEOUT seconds for the user to click the checkbox.
    The browser stays open until the real page content loads or we time out —
    it never auto-closes while a challenge is visible.
    """
    if sync_playwright is None:
        return "", []
    browser = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=RR_HEADLESS)
            page = browser.new_page(user_agent=HEADERS["User-Agent"])
            page.goto(url, wait_until="domcontentloaded", timeout=60000)

            # Detect Cloudflare challenge by title or known CF text.
            # Poll until either the challenge clears or we time out.
            cf_wait_s = 0
            cf_poll   = 2   # seconds between checks
            while cf_wait_s < RR_CF_TIMEOUT:
                title = page.title().lower()
                body  = page.locator("body").inner_text(timeout=5000)
                is_cf = (
                    "just a moment" in title
                    or "checking your browser" in title
                    or "cf-browser-verification" in (page.content().lower())
                    or ("enable javascript" in body.lower() and "cloudflare" in body.lower())
                )
                if not is_cf:
                    break
                if cf_wait_s == 0:
                    print(f"  Cloudflare challenge detected — solve the checkbox in the browser window ({RR_CF_TIMEOUT}s to complete)...")
                page.wait_for_timeout(cf_poll * 1000)
                cf_wait_s += cf_poll
            else:
                print(f"  Cloudflare challenge not solved within {RR_CF_TIMEOUT}s — giving up on {url}")
                browser.close()
                return "", []

            # Small settle wait after challenge clears / on normal load
            page.wait_for_timeout(1000)

            html  = page.content()
            text  = page.locator("body").inner_text(timeout=15000)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            browser.close()
            return html, lines
    except Exception as e:
        print(f"  Playwright error fetching {url}: {e}")
        try:
            if browser:
                browser.close()
        except Exception:
            pass
        return "", []


def _rr_page_text(url):
    """Compatibility wrapper for older code."""
    _, lines = _rr_page_html_and_text(url)
    return lines


def _split_rr_line(line):
    """Racing-Reference inner_text sometimes uses tabs, sometimes only spaces."""
    tab_parts = [p.strip() for p in line.split("\t") if p.strip()]
    if len(tab_parts) > 1:
        return tab_parts
    return line.split()


def _find_make_token(parts):
    for i, p in enumerate(parts):
        if normalize_make(p) in {"Chevrolet", "Ford", "Toyota"}:
            return i
    return None


def _table_rows_from_html(html, required_headers):
    """
    Return (headers, rows) for the first HTML table containing the required headers.
    rows are already text-cleaned cell lists.
    """
    if not html:
        return [], []

    soup = BeautifulSoup(html, "html.parser")
    required = [h.lower() for h in required_headers]

    for table in soup.find_all("table"):
        trs = table.find_all("tr")
        if not trs:
            continue

        header_cells = []
        header_tr_index = None
        for idx, tr in enumerate(trs):
            cells = [clean_cell(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
            lower = [c.lower() for c in cells]
            if cells and all(any(req in c for c in lower) for req in required):
                header_cells = cells
                header_tr_index = idx
                break

        if header_tr_index is None:
            continue

        rows = []
        for tr in trs[header_tr_index + 1:]:
            cells = [clean_cell(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        return header_cells, rows

    return [], []


def scrape_rr_qualifying(season, race_num):
    """
    Scrape Racing-Reference qualifying results.
    Returns dict: {car_num_str -> start_position_int}.
    Returns {} if not yet posted/unavailable/blocked.

    Table layout: RANK | DRIVER | NBR | CAR | TIME | SPEED
    We want: qual[NBR] = RANK  (e.g. qual["97"] = 1)
    """
    url = f"https://www.racing-reference.info/qual-results/{season}-{race_num:02d}/W"
    print(f"Checking Racing-Reference qualifying: {url}")

    html, lines = _rr_page_html_and_text(url)
    if not lines:
        print("  Could not fetch qualifying page.")
        return {}

    text_upper = " ".join(lines).upper()
    if "QUALIFYING RESULTS" not in text_upper and not ("RANK" in text_upper and "DRIVER" in text_upper):
        print("  Qualifying results not yet posted.")
        return {}

    def _looks_valid(q):
        """True if result looks like {car_num -> rank}, not {time_digits -> car_num}."""
        if not q:
            return False
        good = 0
        for car, rank in q.items():
            try:
                car_i = int(car)
                rank_i = int(rank)
            except (TypeError, ValueError):
                continue
            # NASCAR car numbers are normally 0-99, with rare 3-digit oddities.
            # If keys look like 214788, we accidentally parsed lap times as car numbers.
            if 0 <= car_i <= 999 and 1 <= rank_i <= 80:
                good += 1
        if good < max(1, len(q) * 0.8):
            return False
        matched = sum(1 for car, rank in q.items() if car == str(rank))
        return matched < len(q) * 0.5

    def _add_from_parts(parts, qual):
        """
        Parse one visible row/cell run.
        Works for both:
          ['1','Shane Van Gisbergen','97','Chevrolet','2:14.788','90.809']
        and looser space-split rows where the driver name spans tokens.
        """
        if not parts:
            return
        if not str(parts[0]).isdigit():
            return
        try:
            rank = int(re.sub(r"\D", "", str(parts[0])))
        except ValueError:
            return
        if not (1 <= rank <= 80):
            return

        make_i = _find_make_token(parts)
        if make_i is None or make_i < 1:
            return

        car = re.sub(r"\D", "", str(parts[make_i - 1]))
        if not car:
            return
        try:
            car_i = int(car)
        except ValueError:
            return
        if not (0 <= car_i <= 999):
            return

        qual[str(car_i)] = rank

    # ── Path 1: robust HTML parsing ──────────────────────────────────────────
    # Racing-Reference sometimes gives BeautifulSoup a weird flattened row where
    # the "headers" list contains the whole table. Do NOT trust header indices.
    # Instead, scan visible cell tokens and identify rows by: rank ... nbr make.
    if html:
        soup = BeautifulSoup(html, "html.parser")
        for table in soup.find_all("table"):
            table_text = table.get_text(" ", strip=True).upper()
            if "RANK" not in table_text or "DRIVER" not in table_text or "NBR" not in table_text:
                continue

            qual = {}

            # First try normal tr-by-tr rows.
            for tr in table.find_all("tr"):
                cells = [clean_cell(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
                cells = [c for c in cells if c]

                # Normal table row.
                _add_from_parts(cells, qual)

                # Weird flattened table row: scan the cells for repeated row starts.
                for i, cell in enumerate(cells):
                    if not cell.isdigit():
                        continue
                    window = cells[i:i + 8]
                    _add_from_parts(window, qual)

            if qual and _looks_valid(qual):
                print(f"  Qualifying (HTML): {len(qual)} cars. Sample: {dict(list(qual.items())[:3])}")
                return qual
            elif qual:
                print(f"  HTML parse looked suspicious: {dict(list(qual.items())[:3])} — falling back to text.")

    # ── Path 2: plain-text fallback ──────────────────────────────────────────
    # Playwright inner_text usually gives rows like:
    #   "1\tShane Van Gisbergen\t97\tChevrolet\t2:14.788\t90.809"
    header_idx = None
    for i, line in enumerate(lines):
        lu = line.upper()
        if "RANK" in lu and "DRIVER" in lu and "NBR" in lu:
            header_idx = i
            break

    if header_idx is None:
        print("  Could not find qualifying table header in text.")
        return {}

    qual = {}
    for line in lines[header_idx + 1:]:
        lu = line.upper()
        if lu.startswith("OP:") or lu.startswith("PC:") or lu.startswith("FAILED TO QUALIFY"):
            break

        parts = _split_rr_line(line)
        _add_from_parts(parts, qual)

    if qual and _looks_valid(qual):
        print(f"  Qualifying (text): {len(qual)} cars. Sample: {dict(list(qual.items())[:3])}")
        return qual

    if qual:
        print(f"  Text parse looked suspicious: {dict(list(qual.items())[:3])}")
    else:
        print("  Could not parse qualifying from text either.")
    return {}

def scrape_rr_entry_list(season, race_num):
    """
    Scrape Racing-Reference preliminary entry list.
    Returns list of {car_num, driver_name, manufacturer, team}.
    """
    url = f"https://www.racing-reference.info/entrylist/{season}-{race_num:02d}/W"
    print(f"Checking Racing-Reference entry list: {url}")

    html, lines = _rr_page_html_and_text(url)
    if not lines:
        print("  Could not fetch entry list page.")
        return []

    text_upper = " ".join(lines).upper()
    if "ENTRY LIST" not in text_upper and "DRIVER" not in text_upper:
        print("  Entry list not posted yet.")
        return []

    # Find the actual entry-list header in the visible page text.
    header_idx = None
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        upper_parts = [p.upper() for p in parts]

        if (
            "NBR" in upper_parts
            and "DRIVER" in upper_parts
            and "OWNER" in upper_parts
            and "CAR" in upper_parts
        ):
            header_idx = i
            break

    if header_idx is None:
        print("  Could not find entry list table header.")
        return []

    entries = []
    seen = set()

    for line in lines[header_idx + 1:]:
        # Stop once the table is over.
        if line.startswith("Vehicles withdrawn"):
            break

        # IMPORTANT:
        # Racing-Reference preserves this table cleanly with tabs.
        # Do NOT use .split(), because names/owners/sponsors contain spaces.
        parts = [p.strip() for p in line.split("\t") if p.strip()]

        # Expected:
        # row_index, NBR, DRIVER, OWNER, CAR, CREW CHIEF, SPONSOR
        #
        # Example:
        # 37, 91, Kevin Magnussen, Trackhouse Racing, Chevrolet, Phil Surgen, Qualcomm
        if len(parts) < 5:
            continue

        if not parts[0].isdigit():
            continue

        # Most RR entry-list rows include a leading row index.
        #
        # IMPORTANT: rows with a blank sponsor cell may only have 6 non-empty
        # parts after we strip empty cells:
        #   row_index, NBR, DRIVER, OWNER, CAR, CREW CHIEF
        # Example: 22, 35, Riley Herbst, 23XI Racing, Toyota, Davin Restivo
        # So do NOT require len(parts) >= 7 here. Requiring 7 made blank-sponsor
        # rows fall into the no-row-index fallback, where the row index was
        # mistaken for the car number and the row got skipped.
        if len(parts) >= 5 and parts[0].isdigit() and parts[1].isdigit():
            car_num = re.sub(r"\D", "", parts[1])
            driver = clean_cell(parts[2])
            team = clean_cell(parts[3])
            make = normalize_make(parts[4])
        else:
            # Fallback if no row index:
            # NBR, DRIVER, OWNER, CAR, CREW CHIEF, SPONSOR
            car_num = re.sub(r"\D", "", parts[0])
            driver = clean_cell(parts[1])
            team = clean_cell(parts[2]) if len(parts) > 2 else ""
            make = normalize_make(parts[3]) if len(parts) > 3 else ""

        if not car_num or not driver or driver.isdigit():
            continue

        if make not in {"Chevrolet", "Ford", "Toyota"}:
            continue

        if car_num in seen:
            continue

        seen.add(car_num)
        entries.append({
            "car_num": car_num,
            "driver_name": driver,
            "manufacturer": make,
            "team": team,
        })

    if entries:
        print(f"  Found {len(entries)} entry-list drivers.")
    else:
        print("  Could not parse any entry-list rows.")

    return entries



def scrape_driveraverages_schedule(season):
    """
    Scrape DriverAverages year page schedule sidebar.
    Returns dict: race_num -> track_name.

    Example:
      {17: "San Diego", 18: "Sonoma"}
    """
    url = f"https://www.driveraverages.com/nascar/year.php?yr_id={season}"
    print(f"Checking DriverAverages schedule: {url}")

    try:
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            print(f"  Could not fetch schedule page: status {resp.status_code}")
            return {}
    except Exception as e:
        print(f"  Request error fetching schedule: {e}")
        return {}

    soup = BeautifulSoup(resp.content, "html.parser")
    text = soup.get_text("\n", strip=True)
    lines = [clean_cell(x) for x in text.splitlines() if clean_cell(x)]

    schedule = {}
    in_schedule = False

    for line in lines:
        if line == f"{season} Cup Series Schedule":
            in_schedule = True
            continue

        if in_schedule and line in {"Averages by Track", "Averages by NASCAR Driver"}:
            break

        if not in_schedule:
            continue

        # Example lines:
        #   Feb 15 - Daytona
        #   Feb 22 - Atlanta (EchoPark)
        m = re.match(r"^[A-Z][a-z]{2}\s+\d{1,2}\s+-\s+(.+)$", line)
        if not m:
            continue

        track = clean_cell(m.group(1))
        # Atlanta (EchoPark) -> Atlanta
        track = re.sub(r"\s+\([^)]*\)$", "", track).strip()
        schedule[len(schedule) + 1] = track

    if schedule:
        print(f"  Found {len(schedule)} scheduled races.")
    else:
        print("  Could not parse DriverAverages schedule.")

    return schedule


def fallback_start_pos(history, track_type=None):
    """
    If qualifying/start is unknown, use the median start from the driver's
    last 10 races of this track type. Fallback to last 10 overall, then 20.0.
    """
    if track_type is not None:
        typed_starts = [
            r["start"] for r in history
            if r.get("track_type") == track_type and r.get("start") is not None
        ][-10:]
        if typed_starts:
            return float(np.percentile(typed_starts, 50))

    overall_starts = [r["start"] for r in history if r.get("start") is not None][-10:]
    if overall_starts:
        return float(np.percentile(overall_starts, 50))

    return 20.0

# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------
def safe_pcts(arr, metric):
    """Percentiles with neutral fallback for empty arrays."""
    if len(arr) == 0:
        return [20.0, 20.0, 20.0, 20.0]
    return [float(np.percentile(arr, p)) for p in PCTS]


def build_features(history, start_pos=None):
    """
    history: list of race dicts (chronological, excluding current race)
    start_pos: actual starting position for this race (int/float), or None if unknown.

    Returns flat list of 73 floats, or None if < 10 races of history.

    Feature layout:
      Overall windows 10/20/36 x 6 metrics x 4 pcts      = 72
      Starting position for this race (nan=20.0 fallback) =  1
                                                          ----
                                                            73

    The start position is always the last feature. For next-race rows,
    unknown starts should already be replaced by fallback_start_pos().
    """
    if len(history) < 10:
        return None

    feats = []

    # Overall windows
    for window in [10, 20, 36]:
        w = history[-window:]
        for metric in METRICS:
            vals = [r[metric] for r in w if r.get(metric) is not None]
            feats.extend(safe_pcts(vals, metric))

    # Track-type specific windows (last 36 of each type)
    for tt in ["ss", "rc", "s"]:
        typed = [r for r in history if r["track_type"] == tt][-36:]
        for metric in METRICS:
            vals = [r[metric] for r in typed if r.get(metric) is not None]
            feats.extend(safe_pcts(vals, metric))

    # Starting position (73rd feature)
    feats.append(float(start_pos) if start_pos is not None else float("nan"))

    return feats  # length 73


# ---------------------------------------------------------------------------
# Build datasets
# ---------------------------------------------------------------------------

FEATURE_CACHE_VERSION = 1
FEATURE_CACHE_PATH = "feature_cache.json"


def _race_key_str(key):
    return f"{int(key[0])}-{int(key[1]):02d}"


def _race_key_from_str(s):
    year_s, race_s = str(s).split("-", 1)
    return (int(year_s), int(race_s))


def _row_to_json(row):
    feats, finish, track_type = row
    return [feats, finish, track_type]


def _row_from_json(row):
    feats, finish, track_type = row
    return (list(feats), finish, track_type)


def _copy_history(history):
    """Cheap JSON-safe deep copy of driver history."""
    return {driver: [dict(r) for r in races] for driver, races in history.items()}


def _drivers_by_season(raw):
    out = {}
    for (season, race_num), (track_name, drivers) in raw.items():
        for driver in drivers:
            out.setdefault(season, set()).add(driver)
    return out


def _history_row(stats, track_type, season):
    return {
        "finish":     stats["finish"],
        "start":      stats.get("start"),
        "mid_pos":    stats["mid_pos"],
        "closer_pos": stats["closer_pos"],
        "high_pos":   stats["high_pos"],
        "low_pos":    stats["low_pos"],
        "avg_pos":    stats["avg_pos"],
        "track_type": track_type,
        "season":     season,
    }


def _build_training_rows_for_one_race(raw, key, driver_history, drivers_by_season, norm_to_driver, car_to_driver):
    """
    Build training rows for exactly one completed race using driver_history
    BEFORE the race, then mutate driver_history so it includes the race.
    """
    season, race_num = key
    track_name, drivers = raw[key]
    track_type = get_track_type(track_name, season)
    if track_type is None:
        return []

    rows = []
    prev_season_drivers = drivers_by_season.get(season - 1, set())

    for driver, stats in drivers.items():
        history = driver_history.get(driver, [])

        if driver in prev_season_drivers and season >= 2011:
            feats = build_features(history, start_pos=stats.get("start"))
            if feats is not None:
                rows.append((feats, stats["finish"], track_type))

        driver_history.setdefault(driver, []).append(_history_row(stats, track_type, season))
        norm_to_driver[norm_driver_name(driver)] = driver

        car_num_for_lookup = str(stats.get("car_num", "")).strip()
        if car_num_for_lookup:
            car_to_driver[car_num_for_lookup] = driver

    return rows


def _new_empty_feature_cache():
    return {
        "version": FEATURE_CACHE_VERSION,
        "race_keys": [],
        "training_by_race": {},
        "driver_history_after_latest": {},
        "driver_history_before_latest": {},
        "norm_to_driver": {},
        "car_to_driver": {},
    }


def _load_feature_cache(raw_keys):
    try:
        with open(FEATURE_CACHE_PATH) as f:
            cache = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"  Feature cache unreadable ({e}) — rebuilding once.", flush=True)
        return None

    if cache.get("version") != FEATURE_CACHE_VERSION:
        print("  Feature cache version changed — rebuilding once.", flush=True)
        return None

    cached_keys = cache.get("race_keys", [])
    wanted_keys = [_race_key_str(k) for k in raw_keys]

    # Safe fast path: the cache must be an exact prefix of current raw races.
    # If anything was deleted/reordered, rebuild from raw cache.
    if wanted_keys[:len(cached_keys)] != cached_keys:
        print("  Feature cache does not match raw race order — rebuilding once.", flush=True)
        return None

    return cache


def _save_feature_cache(cache):
    with open(FEATURE_CACHE_PATH, "w") as f:
        json.dump(cache, f)
    print(f"Saved feature cache to {FEATURE_CACHE_PATH}.", flush=True)


def _prepare_feature_cache(raw):
    """
    Reuse feature_cache.json when possible. If new races were added to
    raw_races_cache.json, append only those race rows instead of rebuilding
    thousands of historical feature rows from scratch.
    """
    race_keys = sorted(raw.keys())
    if not race_keys:
        return _new_empty_feature_cache()

    drivers_by_season = _drivers_by_season(raw)
    wanted_key_strings = [_race_key_str(k) for k in race_keys]

    cache = _load_feature_cache(race_keys)
    if cache is None:
        print("  Building feature cache from raw races...", flush=True)
        cache = _new_empty_feature_cache()
        driver_history = {}
        norm_to_driver = {}
        car_to_driver = {}
        training_by_race = {}
        latest_key = race_keys[-1]

        for i, key in enumerate(race_keys, start=1):
            if key == latest_key:
                cache["driver_history_before_latest"] = _copy_history(driver_history)

            rows = _build_training_rows_for_one_race(
                raw, key, driver_history, drivers_by_season, norm_to_driver, car_to_driver
            )
            training_by_race[_race_key_str(key)] = [_row_to_json(r) for r in rows]

            if i % 50 == 0 or i == len(race_keys):
                print(f"    Feature rows built through {i}/{len(race_keys)} races...", flush=True)

        cache["race_keys"] = wanted_key_strings
        cache["training_by_race"] = training_by_race
        cache["driver_history_after_latest"] = driver_history
        cache["norm_to_driver"] = norm_to_driver
        cache["car_to_driver"] = car_to_driver
        _save_feature_cache(cache)
        return cache

    cached_key_strings = cache.get("race_keys", [])
    new_keys = race_keys[len(cached_key_strings):]

    if not new_keys:
        print(f"  Reusing feature cache through {_race_key_str(race_keys[-1])}.", flush=True)
        return cache

    print(f"  Feature cache is {len(cached_key_strings)} races behind; appending {len(new_keys)} new race(s)...", flush=True)

    driver_history = cache.get("driver_history_after_latest", {})
    norm_to_driver = cache.get("norm_to_driver", {})
    car_to_driver = cache.get("car_to_driver", {})
    training_by_race = cache.get("training_by_race", {})
    latest_key = race_keys[-1]

    for key in new_keys:
        if key == latest_key:
            cache["driver_history_before_latest"] = _copy_history(driver_history)

        rows = _build_training_rows_for_one_race(
            raw, key, driver_history, drivers_by_season, norm_to_driver, car_to_driver
        )
        training_by_race[_race_key_str(key)] = [_row_to_json(r) for r in rows]
        print(f"    Added feature rows for {_race_key_str(key)} ({len(rows)} rows).", flush=True)

    cache["race_keys"] = wanted_key_strings
    cache["training_by_race"] = training_by_race
    cache["driver_history_after_latest"] = driver_history
    cache["norm_to_driver"] = norm_to_driver
    cache["car_to_driver"] = car_to_driver
    _save_feature_cache(cache)
    return cache


def _rows_for_keys(cache, keys):
    training_by_race = cache.get("training_by_race", {})
    rows = []
    for key in keys:
        for raw_row in training_by_race.get(_race_key_str(key), []):
            rows.append(_row_from_json(raw_row))
    return rows


def build_datasets(raw):
    print('Building training/testing datasets...', flush=True)
    race_keys = sorted(raw.keys())
    if not race_keys:
        return [], [], {}, {}

    latest_key = race_keys[-1]
    latest_season, latest_race_num = latest_key
    latest_track_name, latest_race_drivers = raw[latest_key]

    cache = _prepare_feature_cache(raw)

    training = _rows_for_keys(cache, race_keys)
    last_race_training = _rows_for_keys(cache, race_keys[:-1])

    driver_history = cache.get("driver_history_after_latest", {})
    dh_excl = cache.get("driver_history_before_latest", {})
    norm_to_driver = cache.get("norm_to_driver", {})
    car_to_driver = cache.get("car_to_driver", {})

    # -----------------------------------------------------------------------
    # "Last race" testing set
    # -----------------------------------------------------------------------
    last_race_testing = {"_meta": {
        "mode": "last_race",
        "race": {
            "season":     latest_season,
            "race_num":   latest_race_num,
            "track_name": latest_track_name,
            "track_type": get_track_type(latest_track_name, latest_season),
        },
        "qualifying_available": True,
    }}

    skipped_last = []
    for driver, stats in latest_race_drivers.items():
        car_num = str(stats.get("car_num", "")).strip()
        history = dh_excl.get(driver, [])
        feats = build_features(history, start_pos=stats.get("start"))
        if feats is None:
            skipped_last.append(driver)
            continue
        last_race_testing[car_num] = {
            "driver_name":         driver,
            "history_driver_name": driver,
            "manufacturer":        stats.get("manufacturer", ""),
            "team":                stats.get("team", ""),
            "features":            feats,
            "track_type":          get_track_type(latest_track_name, latest_season),
        }

    if skipped_last:
        print(f"  Last-race: skipped {len(skipped_last)} drivers with <10 races history.")

    # -----------------------------------------------------------------------
    # "Next race" testing set
    # -----------------------------------------------------------------------
    target_season   = latest_season
    target_race_num = latest_race_num + 1

    print(f'Fetching schedule for {target_season}...', flush=True)
    schedule = scrape_driveraverages_schedule(target_season)
    target_track_name = schedule.get(target_race_num)
    target_track_type = get_track_type(target_track_name, target_season) if target_track_name else None

    if target_track_name:
        print(f"Next race appears to be: {target_track_name} ({target_track_type or 'unknown type'})")
    else:
        print("WARNING: could not determine next race track from DriverAverages schedule.")

    rr_entries = []
    qual_positions = {}
    entry_list_available = False
    qualifying_available = False
    qualifying_url = f"https://www.racing-reference.info/qual-results/{target_season}-{target_race_num:02d}/W"
    entry_list_url = f"https://www.racing-reference.info/entrylist/{target_season}-{target_race_num:02d}/W"

    if USE_RR_ENTRY_LIST and target_race_num <= RACES_PER_SEASON:
        print(f'Fetching qualifying for {target_season} race {target_race_num}...', flush=True)
        qual_positions = scrape_rr_qualifying(target_season, target_race_num)
        qualifying_available = bool(qual_positions)
        if qualifying_available:
            print(f"  Qualifying available: {len(qual_positions)} cars with start positions.")
        else:
            print(f"  Qualifying not available — trying entry list.")

        if not qualifying_available:
            print(f'Fetching entry list for {target_season} race {target_race_num}...', flush=True)
            rr_entries = scrape_rr_entry_list(target_season, target_race_num)
            entry_list_available = bool(rr_entries)

    if rr_entries:
        field_source   = f"Racing-Reference entry list {target_season}-{target_race_num}"
        projected_field = rr_entries
    elif qual_positions:
        field_source   = f"Racing-Reference qualifying {target_season}-{target_race_num}"
        qual_entries_by_car = {}
        try:
            el = scrape_rr_entry_list(target_season, target_race_num)
            if el:
                qual_entries_by_car = {str(e["car_num"]): e for e in el}
                print(f"  Supplemented qualifying with entry list for {len(el)} drivers.")
        except Exception as _e:
            print(f"  Could not supplement qualifying with entry list: {_e}")
        latest_by_car = {
            str(stats.get("car_num", "")).strip(): (driver, stats)
            for driver, stats in latest_race_drivers.items()
        }
        projected_field = []
        for car_num in sorted(qual_positions, key=lambda c: qual_positions[c]):
            if car_num in qual_entries_by_car:
                e = qual_entries_by_car[car_num]
                projected_field.append({
                    "car_num":      car_num,
                    "driver_name":  e["driver_name"],
                    "manufacturer": e["manufacturer"],
                    "team":         e["team"],
                })
            else:
                old_driver, old_stats = latest_by_car.get(car_num, (f"Car #{car_num}", {}))
                projected_field.append({
                    "car_num":      car_num,
                    "driver_name":  old_driver,
                    "manufacturer": old_stats.get("manufacturer", ""),
                    "team":         old_stats.get("team", ""),
                })
    else:
        field_source   = f"DriverAverages fallback (last race drivers) {latest_season}-{latest_race_num}"
        projected_field = [
            {
                "car_num":      stats["car_num"],
                "driver_name":  driver,
                "manufacturer": stats.get("manufacturer", ""),
                "team":         stats.get("team", ""),
            }
            for driver, stats in latest_race_drivers.items()
        ]

    next_race_testing = {
        "_meta": {
            "mode": "next_race",
            "field_source":          field_source,
            "entry_list_available":  entry_list_available,
            "qualifying_available":  qualifying_available,
            "entry_list_url":        entry_list_url if entry_list_available else None,
            "qualifying_url":        qualifying_url if qualifying_available else None,
            "latest_completed_race": {
                "season":     latest_season,
                "race_num":   latest_race_num,
                "track_name": latest_track_name,
            },
            "target_race": {
                "season":     target_season,
                "race_num":   target_race_num,
                "track_name": target_track_name,
                "track_type": target_track_type,
            },
        }
    }

    skipped_next = []
    for entry_info in projected_field:
        rr_driver = entry_info["driver_name"]
        car_num   = str(entry_info["car_num"]).strip()
        rr_norm   = norm_driver_name(rr_driver)

        if car_num in car_to_driver:
            exact_driver = car_to_driver[car_num]
            rr_is_name = not rr_driver.isdigit() and len(rr_driver) > 3
            rr_prefix = rr_norm[:3]
            da_prefix = norm_driver_name(exact_driver)[:3]
            if rr_is_name and rr_prefix and da_prefix and rr_prefix != da_prefix:
                print(
                    f"  WARNING: car #{car_num} name mismatch — "
                    f"RR={rr_driver!r} vs DriverAverages={exact_driver!r}"
                )
        elif rr_driver in driver_history:
            exact_driver = rr_driver
        elif rr_norm in norm_to_driver:
            exact_driver = norm_to_driver[rr_norm]
        else:
            exact_driver = rr_driver

        history = driver_history.get(exact_driver, [])

        start_pos = qual_positions.get(car_num, None)
        if start_pos is None:
            start_pos = fallback_start_pos(history, track_type=target_track_type)

        feats = build_features(history, start_pos=start_pos)
        if feats is None:
            skipped_next.append(rr_driver)
            continue

        next_race_testing[car_num] = {
            "driver_name":         rr_driver,
            "history_driver_name": exact_driver,
            "manufacturer":        entry_info.get("manufacturer", ""),
            "team":                entry_info.get("team", ""),
            "features":            feats,
        }

    if skipped_next:
        print(f"  Next-race: skipped {len(skipped_next)} drivers with <10 races history: "
              + ", ".join(skipped_next[:8]))
        if len(skipped_next) > 8:
            print("  ...")

    return training, last_race_training, last_race_testing, next_race_testing

def build_datasets_for_race(raw, target_season, target_race_num):
    """
    Build a no-lookahead historical backtest dataset for one completed race.

    Training rows are ONLY races strictly before (target_season, target_race_num).
    Testing rows are the actual field from that target race, with actual start
    positions, actual finish positions, and features built from pre-race history.
    """
    target_key = (int(target_season), int(target_race_num))
    race_keys = sorted(raw.keys())
    if target_key not in raw:
        raise ValueError(f"Race {target_season}-{target_race_num} was not found in scraped raw data.")

    target_track_name, target_drivers = raw[target_key]
    target_track_type = get_track_type(target_track_name, int(target_season))
    if target_track_type is None:
        raise ValueError(f"Unknown track type for {target_track_name!r} in {target_season}-{target_race_num}.")

    # Pre-compute which drivers appeared in each season, but only using races
    # before the target race so the same-season current race cannot leak in.
    drivers_by_season = {}
    for key in race_keys:
        if key >= target_key:
            continue
        season, race_num = key
        track_name, drivers = raw[key]
        for driver in drivers:
            drivers_by_season.setdefault(season, set()).add(driver)

    driver_history = {}
    training = []

    for key in race_keys:
        if key >= target_key:
            break

        season, race_num = key
        track_name, drivers = raw[key]
        track_type = get_track_type(track_name, season)
        if track_type is None:
            continue

        prev_season_drivers = drivers_by_season.get(season - 1, set())

        for driver, stats in drivers.items():
            history = driver_history.get(driver, [])

            if driver in prev_season_drivers and season >= 2011:
                feats = build_features(history, start_pos=stats.get("start"))
                if feats is not None:
                    training.append((feats, stats["finish"], track_type))

            driver_history.setdefault(driver, []).append({
                "finish":     stats["finish"],
                "start":      stats.get("start"),
                "mid_pos":    stats["mid_pos"],
                "closer_pos": stats["closer_pos"],
                "high_pos":   stats["high_pos"],
                "low_pos":    stats["low_pos"],
                "avg_pos":    stats["avg_pos"],
                "track_type": track_type,
                "season":     season,
            })

    testing = {
        "_meta": {
            "mode": "historical",
            "race": {
                "season":     int(target_season),
                "race_num":   int(target_race_num),
                "track_name": target_track_name,
                "track_type": target_track_type,
            },
            "qualifying_available": True,
            "training_cutoff": {
                "before_season": int(target_season),
                "before_race_num": int(target_race_num),
            },
        }
    }

    skipped = []
    for driver, stats in target_drivers.items():
        car_num = str(stats.get("car_num", "")).strip()
        if not car_num:
            continue
        history = driver_history.get(driver, [])
        feats = build_features(history, start_pos=stats.get("start"))
        if feats is None:
            skipped.append(driver)
            continue

        testing[car_num] = {
            "driver_name":         driver,
            "history_driver_name": driver,
            "manufacturer":        stats.get("manufacturer", ""),
            "team":                stats.get("team", ""),
            "features":            feats,
            "track_type":          target_track_type,
            "actual_finish":       stats.get("finish"),
        }

    if skipped:
        print(f"  Historical {target_season}-{target_race_num}: skipped {len(skipped)} drivers with <10 races history.")

    return training, testing



def load_raw_cache(path="raw_races_cache.json"):
    """Load raw_races_cache.json -> raw dict with tuple keys. Returns {} if missing."""
    import os
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception as e:
        print(f"  Cache load error: {e}")
        return {}
    races = payload.get("races", payload)
    raw = {}
    for key, value in races.items():
        try:
            year_s, race_s = str(key).split("-", 1)
            raw[(int(year_s), int(race_s))] = (value[0], value[1])
        except Exception:
            continue
    return raw


def scrape_incremental(cache_path="raw_races_cache.json"):
    """
    Load cache, then walk forward one race at a time from the last cached
    entry. Stop as soon as a slot comes back empty (race hasn't happened yet).

    First run (no cache): falls back to full scrape.
    Typical run: 1-2 fetches maximum.
    """
    raw = load_raw_cache(cache_path)

    if not raw:
        print("  No cache — full scrape (one-time, ~30-60 s)...")
        raw = scrape_all()
        save_raw_cache(raw, cache_path)
        return raw

    last_season, last_race = max(raw.keys())
    print(f"  Cache: {len(raw)} completed races. Last: {last_season} race {last_race}.", flush=True)

    new_count = 0
    season, race_num = last_season, last_race + 1

    while True:
        if race_num > RACES_PER_SEASON:
            season += 1
            race_num = 1

        print(f"  Checking {season} race {race_num}...", flush=True)
        result = fetch_race(season, race_num)

        if result is None:
            print(f"  {season} race {race_num} not found — no more new races.", flush=True)
            break

        _, _, track_name, drivers = result
        raw[(season, race_num)] = (track_name, drivers)
        print(f"  Found: {season} race {race_num} at {track_name} ({len(drivers)} drivers).", flush=True)
        new_count += 1
        race_num += 1

    if new_count:
        print(f"  {new_count} new race(s) added. Saving cache.")
        save_raw_cache(raw, cache_path)
    else:
        print("  No new completed races.")

    return raw


def save_raw_cache(raw, path="raw_races_cache.json"):
    """Save scraped raw races so historical predictions can slice cached data without scraping."""
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "races": {
            f"{int(k[0])}-{int(k[1]):02d}": [track, drivers]
            for k, (track, drivers) in sorted(raw.items())
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Saved raw race cache to {path}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def write_training_csv(path, rows):
    feature_count = len(rows[0][0]) if rows else 145
    feature_cols  = [f"x{i}" for i in range(feature_count)]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(feature_cols + ["finish", "track_type"])
        for feats, finish, track_type in rows:
            writer.writerow(feats + [finish, track_type])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical",
        nargs=2,
        metavar=("YEAR", "RACE_NUM"),
        type=int,
        help="Build a no-lookahead dataset for one completed historical race.",
    )
    args = parser.parse_args()

    print("Checking for new completed races (incremental)...")
    raw = scrape_incremental()
    print(f"Cache now has {len(raw)} completed races.")

    if args.historical:
        year, race_num = args.historical
        print(f"Building historical dataset for {year} race {race_num}...")
        training_historical, testing_historical = build_datasets_for_race(raw, year, race_num)
        print(f"{len(training_historical)} historical training rows.")
        write_training_csv("training_historical.csv", training_historical)
        with open("testing_historical.json", "w") as f:
            json.dump(testing_historical, f, indent=2)
        print("Done. Wrote training_historical.csv and testing_historical.json.")
    else:
        print("Building datasets...")
        training, last_race_training, last_race_testing, next_race_testing = build_datasets(raw)

        print(f"{len(training)} next-race training rows.")
        print(f"{len(last_race_training)} last-race training rows.")

        write_training_csv("training.csv", training)
        write_training_csv("training_last_race.csv", last_race_training)

        with open("testing_last_race.json", "w") as f:
            json.dump(last_race_testing, f, indent=2)

        with open("testing_next_race.json", "w") as f:
            json.dump(next_race_testing, f, indent=2)

        print("Done. Wrote training.csv, training_last_race.csv, testing_last_race.json, testing_next_race.json.")