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
SERIES_CONFIG = {"cup": "Cup Series", "oreilly": "O'Reilly Series"}

def _series_key(value: str | None) -> str:
    key = str(value or "cup").lower()
    return key if key in SERIES_CONFIG else "cup"

def _series_suffix(series: str) -> str:
    return "" if series == "cup" else f"_{series}"

def _raw_cache_file(series: str) -> Path:
    return APP_DIR / f"raw_races_cache{_series_suffix(series)}.json"

def _historical_dir(series: str) -> Path:
    return APP_DIR / "predictions_historical" / series

app = Flask(__name__, static_folder=None)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_raw_cache(series: str = "cup") -> dict:
    try:
        from getdata import load_raw_cache
        return load_raw_cache(str(_raw_cache_file(series)))
    except Exception:
        return {}

def _get_track_type(track_name: str, season: int) -> str:
    try:
        from getdata import get_track_type
        return get_track_type(track_name, season) or "s"
    except Exception:
        return "s"

def _historical_relpath(season: int, race_num: int, series: str = "cup") -> Path:
    return Path("predictions_historical") / series / str(int(season)) / f"predictions_historical_{int(season)}_{int(race_num):02d}.json"

def _historical_abspath(season: int, race_num: int, series: str = "cup") -> Path:
    return APP_DIR / _historical_relpath(season, race_num, series)

def _migrate_legacy_historical_file(season: int, race_num: int, series: str = "cup") -> Path:
    destination = _historical_abspath(season, race_num, series)
    legacy_candidates = []
    if series == "cup":
        legacy_candidates.extend([
            APP_DIR / "predictions_historical" / str(int(season)) / f"predictions_historical_{int(season)}_{int(race_num):02d}.json",
            APP_DIR / f"predictions_historical_{int(season)}_{int(race_num):02d}.json",
        ])
    else:
        legacy_candidates.append(
            APP_DIR / f"predictions_historical_{series}" / str(int(season)) / f"predictions_historical_{int(season)}_{int(race_num):02d}.json"
        )
    if not destination.exists():
        for legacy in legacy_candidates:
            if legacy.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                legacy.replace(destination)
                print(f"Moved {legacy.relative_to(APP_DIR)} -> {destination.relative_to(APP_DIR)}", flush=True)
                break
    return destination

def _build_historical_index(series: str = "cup") -> list[dict[str, Any]]:
    raw = _load_raw_cache(series)
    out = []
    for (season, race_num), (track_name, _) in sorted(raw.items()):
        # Historical predictions are available from 2010 onward.
        # Raw data starts in 2005, and every historical prediction trains on
        # all available prior races from 2005 onward.
        if season < 2010:
            continue
        prediction_path = _migrate_legacy_historical_file(season, race_num, series)
        relpath = _historical_relpath(season, race_num, series).as_posix()
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
    ("Tyler", "prediction", None),
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
    if key == "prediction":
        stored = _finite_number(driver.get("prediction"))
        if stored is not None:
            return stored
        pmf = driver.get("pmf") or []
        before = 0.0
        for index, raw_probability in enumerate(pmf):
            probability = _finite_number(raw_probability) or 0.0
            after = before + probability
            if after >= 0.5:
                if probability <= 0:
                    return float(index + 1)
                return max(1.0, index + (0.5 - before) / probability)
            before = after
        return _finite_number(driver.get("median_finish"))
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


def _season_comparison(season: int, series: str = "cup") -> dict[str, Any]:
    files_by_race: dict[int, Path] = {}
    season_dir = _historical_dir(series) / str(int(season))
    if season_dir.exists():
        for path in season_dir.glob(f"predictions_historical_{int(season)}_*.json"):
            match = path.stem.rsplit("_", 1)[-1]
            if match.isdigit():
                files_by_race[int(match)] = path


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

def _cache_age_seconds(series: str = "cup") -> float | None:
    """Return seconds since cache was last written, or None if unknown."""
    cache_file = _raw_cache_file(series)
    if not cache_file.exists():
        return None
    try:
        payload  = json.loads(cache_file.read_text())
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
    series = _series_key(request.args.get("series"))
    cache_file = _raw_cache_file(series)
    age    = _cache_age_seconds(series)
    count  = 0
    updated_at = None
    if cache_file.exists():
        try:
            payload    = json.loads(cache_file.read_text())
            updated_at = payload.get("updated_at")
            count      = len(payload.get("races", {}))
        except Exception:
            pass
    return jsonify({
        "series":          series,
        "cache_exists":    cache_file.exists(),
        "cache_age_s":     age,
        "updated_at":      updated_at,
        "race_count":      count,
        "index":           _build_historical_index(series),
    })

@app.get("/api/historical_index")
def api_historical_index():
    series = _series_key(request.args.get("series"))
    return jsonify(_build_historical_index(series))

@app.get("/api/season_comparison/<int:season>")
def api_season_comparison(season: int):
    series = _series_key(request.args.get("series"))
    return jsonify(_season_comparison(season, series))

@app.post("/api/scrape")
def api_scrape():
    """
    Incremental scrape of completed races + re-fetch qualifying/entry list
    + rebuild last/next-race predictions.

    After the first run, getdata.py only checks ~10-20 recent race slots
    instead of all 500+, so this is fast.
    """
    series = _series_key(request.args.get("series"))

    # Combined-series training needs both raw caches current. Refresh both
    # caches without building datasets, then build the selected series once.
    scrape_results = []
    for scrape_series in (series, "oreilly" if series == "cup" else "cup"):
        result = _run(
            [sys.executable, "getdata.py", "--series", scrape_series, "--scrape-only"],
            timeout=1800,
        )
        scrape_results.append((scrape_series, result))
        if not result["ok"]:
            return jsonify({
                "ok": False,
                "error": result.get("error", f"getdata.py failed for {scrape_series} (code {result['code']})"),
                "stdout": result["stdout"],
                "stderr": result["stderr"],
            }), 500

    dataset_result = _run([sys.executable, "getdata.py", "--series", series], timeout=1800)
    if not dataset_result["ok"]:
        return jsonify({
            "ok": False,
            "error": dataset_result.get("error", f"dataset build failed for {series} (code {dataset_result['code']})"),
            "stdout": dataset_result["stdout"],
            "stderr": dataset_result["stderr"],
        }), 500

    scrape_results.append((f"{series}-dataset", dataset_result))
    combined_scrape_stdout = "".join(
        f"\n=== {name.upper()} DATA ===\n{result['stdout']}"
        for name, result in scrape_results
    )
    combined_scrape_stderr = "".join(result["stderr"] for _, result in scrape_results)

    # Rebuild predictions for the series currently selected in the dashboard.
    r2 = _run([sys.executable, "predict.py", "--series", series], timeout=300)
    return jsonify({
        "ok":            r2["ok"],
        "scrape_stdout": combined_scrape_stdout,
        "scrape_stderr": combined_scrape_stderr,
        "pred_stdout":   r2["stdout"],
        "pred_stderr":   r2["stderr"],
        "error":         None if r2["ok"] else f"predict.py failed (code {r2['code']})",
        "index":         _build_historical_index(series),
    }), (200 if r2["ok"] else 500)

@app.post("/api/run_historical")
def api_run_historical():
    """
    Build all missing no-lookahead predictions for the selected season.
    The requested race is returned after the seasonal batch finishes.
    """
    payload = request.get_json(silent=True) or {}
    series = _series_key(payload.get("series") or request.args.get("series"))
    try:
        year = int(payload["year"])
        race = int(payload["race"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False, "error": "Provide {year, race} as integers."}), 400

    cache_file = _raw_cache_file(series)
    if not cache_file.exists():
        return jsonify({
            "ok":    False,
            "error": f"{cache_file.name} not found. Click ↻ Refresh data first.",
        }), 400

    index = _build_historical_index(series)
    entry = next((e for e in index if e["season"] == year and e["race_num"] == race), None)
    if entry is None:
        return jsonify({
            "ok":    False,
            "error": f"Race {year} #{race} not found in cache. Try ↻ Refresh data first.",
        }), 404

    out_path = _migrate_legacy_historical_file(year, race, series)
    out_file = _historical_relpath(year, race, series).as_posix()

    # Already predicted — return immediately
    if out_path.exists():
        return jsonify({
            "ok":     True,
            "file":   out_file,
            "cached": True,
            "index":  _build_historical_index(series),
        })

    r = _run([sys.executable, "predict.py", "--series", series, "--historical", str(year), str(race)],
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
        "index":  _build_historical_index(series),
    })


if __name__ == "__main__":
    print("NASCAR dashboard → http://localhost:5000", flush=True)
    cache_file = _raw_cache_file("cup")
    if not cache_file.exists():
        print(
            "\n  NOTE: raw_races_cache.json not found.\n"
            "  Open the dashboard and click '↻ Refresh data' to build it,\n"
            "  or run: python3 getdata.py\n",
            flush=True,
        )
    app.run(host="127.0.0.1", port=5000, debug=True)