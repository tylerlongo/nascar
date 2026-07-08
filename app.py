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

def _build_historical_index() -> list[dict[str, Any]]:
    raw = _load_raw_cache()
    out = []
    for (season, race_num), (track_name, _) in sorted(raw.items()):
        if season < 2022:
            continue
        fname = f"predictions_historical_{season}_{race_num:02d}.json"
        out.append({
            "season":     season,
            "race_num":   race_num,
            "track_name": track_name,
            "track_type": _get_track_type(track_name, season),
            "file":       fname,
            "predicted":  (APP_DIR / fname).exists(),
        })
    return out

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
    Build a no-lookahead prediction for a specific past race.
    Reads only from raw_races_cache.json — no scraping.
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

    out_file = f"predictions_historical_{year}_{race:02d}.json"

    # Already predicted — return immediately
    if (APP_DIR / out_file).exists():
        return jsonify({
            "ok":     True,
            "file":   out_file,
            "cached": True,
            "index":  _build_historical_index(),
        })

    r = _run([sys.executable, "predict.py", "--historical", str(year), str(race)],
             timeout=600)
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