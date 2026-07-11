"""
NASCAR Prediction Model — Flask server

Run:  python3 app.py
Open: http://localhost:5000
"""

from __future__ import annotations
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

APP_DIR        = Path(__file__).resolve().parent
RAW_CACHE_FILE = APP_DIR / "raw_races_cache.json"
HISTORICAL_DIR = APP_DIR / "predictions_historical"

app = Flask(__name__, static_folder=None)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_raw_cache() -> dict:
    try:
        from getdata import load_raw_cache
        return load_raw_cache(str(RAW_CACHE_FILE))
    except Exception:
        return {}

def _get_track_type(track_name: str, season: int) -> str:
    try:
        from getdata import get_track_type
        return get_track_type(track_name, season) or "s"
    except Exception:
        return "s"

def _historical_relpath(season: int, race_num: int) -> Path:
    return Path("predictions_historical") / str(int(season)) / f"predictions_historical_{int(season)}_{int(race_num):02d}.json"

def _historical_abspath(season: int, race_num: int) -> Path:
    return APP_DIR / _historical_relpath(season, race_num)

def _migrate_legacy_historical_file(season: int, race_num: int) -> Path:
    destination = _historical_abspath(season, race_num)
    legacy = APP_DIR / f"predictions_historical_{int(season)}_{int(race_num):02d}.json"
    if not destination.exists() and legacy.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        legacy.replace(destination)
        print(f"Moved {legacy.name} -> {destination.relative_to(APP_DIR)}", flush=True)
    return destination

def _build_historical_index() -> list[dict[str, Any]]:
    raw = _load_raw_cache()
    out = []
    for (season, race_num), (track_name, _) in sorted(raw.items()):
        # Historical predictions are available from 2010 onward.
        # Raw data starts in 2005, and every historical prediction trains on
        # all available prior races from 2005 onward.
        if season < 2010:
            continue
        prediction_path = _migrate_legacy_historical_file(season, race_num)
        relpath = _historical_relpath(season, race_num).as_posix()
        out.append({
            "season":     season,
            "race_num":   race_num,
            "track_name": track_name,
            "track_type": _get_track_type(track_name, season),
            "file":       relpath,
            "predicted":  prediction_path.exists(),
        })
    return out

_COMPARISON_METHODS = [
    ("Tyler", "expected_finish", None),
    ("Start Pos", "starting_position", None),
    ("Last 10 Finish", "last10_median_finish", "baseline_last10_p50_finish"),
    ("Last 20 Finish", "last20_median_finish", None),
    ("Last 36 Finish", "last36_median_finish", None),
    ("Last 10 Avg Pos", "last10_median_avg_pos", None),
    ("Last 20 Avg Pos", "last20_median_avg_pos", None),
    ("Last 36 Avg Pos", "last36_median_avg_pos", None),
]


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _comparison_value(driver: dict[str, Any], key: str) -> float | None:
    if key in {"expected_finish", "median_finish", "baseline_last10_p50_finish", "starting_position"}:
        return _finite_number(driver.get(key))
    return _finite_number((driver.get("baselines") or {}).get(key))


def _score_prediction_file(payload: dict[str, Any]) -> dict[str, float]:
    tracks = payload.get("tracks") or {}
    if not tracks:
        return {}
    drivers_by_car = next(iter(tracks.values()), {}) or {}
    drivers = [
        d for d in drivers_by_car.values()
        if _finite_number(d.get("actual_finish")) is not None
    ]
    if len(drivers) < 5:
        return {}

    scores: dict[str, float] = {}

    # Expected MAE for a completely random ordering of this race's field.
    # This is deterministic for field size n: (n² - 1) / (3n).
    random_n = len(drivers)
    if random_n >= 2:
        scores["Random"] = (random_n * random_n - 1) / (3 * random_n)

    for name, key, fallback in _COMPARISON_METHODS:
        valued = []
        for driver in drivers:
            value = _comparison_value(driver, key)
            if value is None and fallback:
                value = _comparison_value(driver, fallback)
            if value is not None:
                valued.append((driver, value))
        if len(valued) < 5:
            continue

        valued.sort(key=lambda item: (
            item[1],
            _comparison_value(item[0], "expected_finish") or float("inf"),
            int(item[0].get("car")) if str(item[0].get("car", "")).isdigit() else 9999,
            str(item[0].get("car", "")),
        ))
        ranks = {str(driver.get("car")): i + 1 for i, (driver, _) in enumerate(valued)}
        error = sum(
            abs(float(driver["actual_finish"]) - ranks[str(driver.get("car"))])
            for driver, _ in valued
        ) / len(valued)
        scores[name] = error
    return scores


def _season_comparison(season: int) -> dict[str, Any]:
    files_by_race: dict[int, Path] = {}
    season_dir = HISTORICAL_DIR / str(int(season))
    if season_dir.exists():
        for path in season_dir.glob(f"predictions_historical_{int(season)}_*.json"):
            match = path.stem.rsplit("_", 1)[-1]
            if match.isdigit():
                files_by_race[int(match)] = path

    last_race_path = APP_DIR / "predictions_last_race.json"
    if last_race_path.exists():
        try:
            payload = json.loads(last_race_path.read_text())
            race = ((payload.get("meta") or {}).get("mode_meta") or {}).get("race") or {}
            if int(race.get("season", -1)) == int(season):
                files_by_race.setdefault(int(race.get("race_num")), last_race_path)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for race_num, path in sorted(files_by_race.items()):
        try:
            scores = _score_prediction_file(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
        for name, score in scores.items():
            totals[name] = totals.get(name, 0.0) + score
            counts[name] = counts.get(name, 0) + 1

    return {
        "season": int(season),
        "averages": {name: totals[name] / counts[name] for name in totals},
        "race_counts": counts,
    }


def _run(cmd: list[str], timeout: int = 600) -> dict[str, Any]:
    """Run a subprocess, streaming its output live to the terminal."""
    import time
    try:
        proc = subprocess.Popen(
            cmd, cwd=APP_DIR, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        stdout_lines = []
        deadline = time.monotonic() + timeout
        for line in proc.stdout:
            print(line, end="", flush=True)   # live terminal output
            stdout_lines.append(line)
            if time.monotonic() > deadline:
                proc.kill()
                return {"ok": False, "stdout": "".join(stdout_lines),
                        "stderr": "", "code": -1, "error": f"Timed out after {timeout}s."}
        proc.wait()
        stdout = "".join(stdout_lines)
        return {"ok": proc.returncode == 0, "stdout": stdout,
                "stderr": "", "code": proc.returncode}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": "", "code": -1, "error": str(e)}

def _cache_age_seconds() -> float | None:
    """Return seconds since cache was last written, or None if unknown."""
    if not RAW_CACHE_FILE.exists():
        return None
    try:
        payload  = json.loads(RAW_CACHE_FILE.read_text())
        updated  = payload.get("updated_at", "")
        if not updated:
            return None
        ts = datetime.fromisoformat(updated)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(tz=timezone.utc) - ts).total_seconds()
    except Exception:
        return None


# ── static files ──────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return send_from_directory(APP_DIR, "dashboard.html")

@app.get("/dashboard.html")
def dashboard():
    return send_from_directory(APP_DIR, "dashboard.html")

@app.get("/<path:filename>")
def static_files(filename: str):
    safe = Path(filename)
    if safe.is_absolute() or ".." in safe.parts:
        return jsonify({"ok": False, "error": "Bad path."}), 400
    return send_from_directory(APP_DIR, filename)


# ── API ───────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    age    = _cache_age_seconds()
    count  = 0
    updated_at = None
    if RAW_CACHE_FILE.exists():
        try:
            payload    = json.loads(RAW_CACHE_FILE.read_text())
            updated_at = payload.get("updated_at")
            count      = len(payload.get("races", {}))
        except Exception:
            pass
    return jsonify({
        "cache_exists":    RAW_CACHE_FILE.exists(),
        "cache_age_s":     age,
        "updated_at":      updated_at,
        "race_count":      count,
        "index":           _build_historical_index(),
    })

@app.get("/api/historical_index")
def api_historical_index():
    return jsonify(_build_historical_index())

@app.get("/api/season_comparison/<int:season>")
def api_season_comparison(season: int):
    return jsonify(_season_comparison(season))

@app.post("/api/scrape")
def api_scrape():
    """
    Incremental scrape of completed races + re-fetch qualifying/entry list
    + rebuild last/next-race predictions.

    After the first run, getdata.py only checks ~10-20 recent race slots
    instead of all 500+, so this is fast.
    """
    # Run getdata.py — it now uses scrape_incremental() internally
    r1 = _run([sys.executable, "getdata.py"], timeout=600)
    if not r1["ok"]:
        return jsonify({
            "ok":     False,
            "error":  r1.get("error", f"getdata.py failed (code {r1['code']})"),
            "stdout": r1["stdout"],
            "stderr": r1["stderr"],
        }), 500

    # Rebuild predictions
    r2 = _run([sys.executable, "predict.py"], timeout=300)
    return jsonify({
        "ok":            r2["ok"],
        "scrape_stdout": r1["stdout"],
        "scrape_stderr": r1["stderr"],
        "pred_stdout":   r2["stdout"],
        "pred_stderr":   r2["stderr"],
        "error":         None if r2["ok"] else f"predict.py failed (code {r2['code']})",
        "index":         _build_historical_index(),
    }), (200 if r2["ok"] else 500)

@app.post("/api/run_historical")
def api_run_historical():
    """
    Build all missing no-lookahead predictions for the selected season.
    The requested race is returned after the seasonal batch finishes.
    """
    payload = request.get_json(silent=True) or {}
    try:
        year = int(payload["year"])
        race = int(payload["race"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "Provide {year, race} as integers."}), 400

    if not RAW_CACHE_FILE.exists():
        return jsonify({
            "ok":    False,
            "error": "raw_races_cache.json not found. Click ↻ Refresh data first.",
        }), 400

    index = _build_historical_index()
    entry = next((e for e in index if e["season"] == year and e["race_num"] == race), None)
    if entry is None:
        return jsonify({
            "ok":    False,
            "error": f"Race {year} #{race} not found in cache. Try ↻ Refresh data first.",
        }), 404

    out_path = _migrate_legacy_historical_file(year, race)
    out_file = _historical_relpath(year, race).as_posix()

    # Already predicted — return immediately
    if out_path.exists():
        return jsonify({
            "ok":     True,
            "file":   out_file,
            "cached": True,
            "index":  _build_historical_index(),
        })

    r = _run([sys.executable, "predict.py", "--historical", str(year), str(race)],
             timeout=3600)
    if not r["ok"]:
        return jsonify({
            "ok":     False,
            "error":  r.get("error", f"predict.py failed (code {r['code']})"),
            "stdout": r["stdout"],
            "stderr": r["stderr"],
        }), 500

    return jsonify({
        "ok":     True,
        "file":   out_file,
        "cached": False,
        "stdout": r["stdout"],
        "index":  _build_historical_index(),
    })


if __name__ == "__main__":
    print("NASCAR dashboard → http://localhost:5000", flush=True)
    if not RAW_CACHE_FILE.exists():
        print(
            "\n  NOTE: raw_races_cache.json not found.\n"
            "  Open the dashboard and click '↻ Refresh data' to build it,\n"
            "  or run: python3 getdata.py\n",
            flush=True,
        )
    app.run(host="127.0.0.1", port=5000, debug=True)