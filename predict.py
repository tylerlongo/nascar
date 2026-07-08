import json
import csv
import warnings
import argparse
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=RuntimeWarning)

TRACKS = ["s", "ss", "rc"]
TRACK_LABELS = {"s": "Speedway", "ss": "Superspeedway", "rc": "Road Course"}

MAX_FINISH = 40

MODEL_PARAMS = dict(
    solver="saga",
    penalty="l2",
    C=0.02,
    max_iter=5000,
    tol=1e-4,
    random_state=1,
    n_jobs=-1,
)


def clean_feature_array(x):
    return np.nan_to_num(x, nan=20.0, posinf=40.0, neginf=1.0)


def load_training(path):
    rows = []
    track_types = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if not row:
                continue
            try:
                feats      = [float(v) for v in row[:-2]]
                finish     = float(row[-2])
                track_type = row[-1]
            except (ValueError, IndexError):
                continue

            rows.append(feats + [finish])
            track_types.append(track_type)

    if not rows:
        raise ValueError(f"No usable training rows found in {path}")

    data = np.array(rows, dtype=float)
    X    = clean_feature_array(data[:, :-1])
    y    = data[:, -1].astype(int)
    y    = np.clip(y, 1, MAX_FINISH)
    return X, y, np.array(track_types)


def get_track_masks(track_types):
    return {
        "ss": track_types == "ss",
        "rc": track_types == "rc",
        "s":  track_types == "s",
    }


def normalize_track_list(track_types_to_train=None):
    """Return a clean list of track types to train. None means all tracks."""
    if track_types_to_train is None:
        return list(TRACKS)

    if isinstance(track_types_to_train, str):
        track_types_to_train = [track_types_to_train]

    out = []
    for tt in track_types_to_train:
        if tt not in TRACKS:
            raise ValueError(f"Unknown track type to train: {tt!r}")
        if tt not in out:
            out.append(tt)

    if not out:
        return list(TRACKS)
    return out


def get_required_track_types(mode_meta):
    """
    Decide which model buckets are actually needed for this prediction job.

    Historical / last-race jobs have a known race.track_type, so only that
    model is needed. Next-race jobs also usually know target_race.track_type;
    if they do not, keep the old fallback and train all three.
    """
    mode = mode_meta.get("mode", "next_race")

    if mode in {"last_race", "historical"}:
        track_type = mode_meta.get("race", {}).get("track_type")
        if track_type in TRACKS:
            return [track_type]
        return list(TRACKS)

    target_track_type = mode_meta.get("target_race", {}).get("track_type")
    if target_track_type in TRACKS:
        return [target_track_type]

    return list(TRACKS)


def train_models(X, y, track_types, track_types_to_train=None):
    masks  = get_track_masks(track_types)
    models = {}

    for tt in normalize_track_list(track_types_to_train):
        mask = masks[tt]
        print(f"  Training {TRACK_LABELS[tt]}: {mask.sum()} rows", flush=True)

        if mask.sum() == 0:
            raise ValueError(f"No training rows for track type {tt}")

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(**MODEL_PARAMS)
        )
        model.fit(X[mask], y[mask])
        models[tt] = model

    skipped = [tt for tt in TRACKS if tt not in models]
    if skipped:
        pretty = ", ".join(TRACK_LABELS[tt] for tt in skipped)
        print(f"  Skipped unused model(s): {pretty}", flush=True)

    return models


def predict_pmf(model, x):
    probs   = model.predict_proba(x.reshape(1, -1))[0]
    classes = model.named_steps["logisticregression"].classes_

    pmf = np.zeros(MAX_FINISH)
    for cls, p in zip(classes, probs):
        if 1 <= cls <= MAX_FINISH:
            pmf[cls - 1] = p

    total = pmf.sum()
    if total > 0:
        pmf /= total

    cdf = np.cumsum(pmf)
    return pmf, cdf


def summarize_driver(car, driver_info, track_type, pmf, cdf):
    positions = np.arange(1, MAX_FINISH + 1)

    expected    = float(np.sum(positions * pmf))
    median      = float(np.searchsorted(cdf, 0.5) + 1)
    driver_name = driver_info.get("driver_name", f"#{car}")

    raw_start = None
    if driver_info.get("features"):
        raw_start = driver_info["features"][-1]

    starting_position = None
    try:
        if raw_start is not None and np.isfinite(float(raw_start)):
            starting_position = int(round(float(raw_start)))
    except (TypeError, ValueError):
        starting_position = None

    return {
        "car":                 car,
        "name":                driver_name,
        "driver_name":         driver_name,
        "history_driver_name": driver_info.get("history_driver_name", driver_name),
        "manufacturer":        driver_info.get("manufacturer", ""),
        "team":                driver_info.get("team", ""),
        "track_type":          track_type,
        "track_label":         TRACK_LABELS[track_type],

        "starting_position": starting_position,
        "actual_finish": driver_info.get("actual_finish"),

        "expected_finish": expected,
        "median_finish":   median,

        "win":   float(cdf[0]),
        "top3":  float(cdf[2]),
        "top5":  float(cdf[4]),
        "top10": float(cdf[9]),
        "top15": float(cdf[14]),
        "top20": float(cdf[19]),
        "top25": float(cdf[24]),

        "pmf": pmf.tolist(),
        "cdf": cdf.tolist(),
    }


def car_sort_key(z):
    z = str(z)
    return (0, int(z)) if z.isdigit() else (1, z)


def predict_field(models, testing, mode_meta):
    """
    Run predictions for all drivers in a testing dict.

    For last_race mode: each driver has a single "features" vector and a
    known "track_type", so we only run the matching model.

    For next_race mode: if getdata.py knows the target track type, we only
    run that matching model. If it does not, we fall back to all three buckets.
    """
    mode = mode_meta.get("mode", "next_race")
    cars = [k for k in testing.keys() if not str(k).startswith("_")]

    output = {
        "meta": {
            "model":      "regularized multinomial logistic regression",
            "max_finish": MAX_FINISH,
            "tracks":     TRACK_LABELS,
            "params":     MODEL_PARAMS,
            "mode_meta":  mode_meta,
        },
        "tracks": {},
    }

    if mode in {"last_race", "historical"}:
        # Single known track type — populate just that track bucket
        track_type = mode_meta.get("race", {}).get("track_type")
        if track_type not in TRACKS:
            raise ValueError(f"Unknown track_type in {mode} meta: {track_type!r}")

        output["tracks"][track_type] = {}

        for car in sorted(cars, key=car_sort_key):
            driver_info = testing[car]
            x = clean_feature_array(np.array(driver_info["features"], dtype=float))
            pmf, cdf = predict_pmf(models[track_type], x)
            output["tracks"][track_type][car] = summarize_driver(car, driver_info, track_type, pmf, cdf)

        print(f"  Predicted {len(cars)} drivers for {TRACK_LABELS[track_type]}.", flush=True)

    else:
        # next_race: if getdata.py determined the actual next track type,
        # predict only that bucket. This makes the dashboard open on the
        # correct track automatically instead of defaulting to Speedway.
        target_track_type = mode_meta.get("target_race", {}).get("track_type")

        if target_track_type in TRACKS:
            track_types_to_predict = [target_track_type]
            print(
                f"  Next race track type known: {TRACK_LABELS[target_track_type]}",
                flush=True,
            )
        else:
            # Fallback for older testing_next_race.json files that do not yet
            # include target_race.track_type. Keep the old behavior.
            track_types_to_predict = TRACKS
            print(
                "  Next race track type unknown; predicting all track buckets.",
                flush=True,
            )

        for tt in track_types_to_predict:
            print(f"  Predicting {TRACK_LABELS[tt]}...", flush=True)
            output["tracks"][tt] = {}

            for car in sorted(cars, key=car_sort_key):
                driver_info = testing[car]
                x = clean_feature_array(np.array(driver_info["features"], dtype=float))
                pmf, cdf = predict_pmf(models[tt], x)
                output["tracks"][tt][car] = summarize_driver(car, driver_info, tt, pmf, cdf)

    return output


def write_predictions(input_training, input_testing, output_path):
    with open(input_testing) as f:
        testing = json.load(f)

    meta = testing.get("_meta", {})
    required_tracks = get_required_track_types(meta)

    X, y, tt = load_training(input_training)
    models = train_models(X, y, tt, required_tracks)

    preds = predict_field(models, testing, meta)

    with open(output_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {output_path}", flush=True)
    return preds


def update_historical_index(preds, filename):
    mm = preds.get("meta", {}).get("mode_meta", {})
    race = mm.get("race", {})
    if not race.get("season") or not race.get("race_num"):
        return

    index_path = Path("predictions_historical_index.json")
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text())
        except json.JSONDecodeError:
            index = []
    else:
        index = []

    entry = {
        "season": race.get("season"),
        "race_num": race.get("race_num"),
        "track_name": race.get("track_name"),
        "track_type": race.get("track_type"),
        "file": filename,
    }

    index = [e for e in index if not (e.get("season") == entry["season"] and e.get("race_num") == entry["race_num"])]
    index.append(entry)
    index.sort(key=lambda e: (int(e.get("season", 0)), int(e.get("race_num", 0))))
    index_path.write_text(json.dumps(index, indent=2))
    print(f"Updated {index_path}", flush=True)



def rows_to_arrays(rows):
    if not rows:
        raise ValueError("No usable training rows were built.")
    X = []
    y = []
    track_types = []
    for feats, finish, track_type in rows:
        X.append(feats)
        y.append(finish)
        track_types.append(track_type)
    X = clean_feature_array(np.array(X, dtype=float))
    y = np.clip(np.array(y, dtype=int), 1, MAX_FINISH)
    return X, y, np.array(track_types)


def write_predictions_from_rows(training_rows, testing, output_path):
    meta = testing.get("_meta", {})
    required_tracks = get_required_track_types(meta)

    X, y, tt = rows_to_arrays(training_rows)
    models = train_models(X, y, tt, required_tracks)

    preds = predict_field(models, testing, meta)
    with open(output_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {output_path}", flush=True)
    return preds


def load_raw_cache(path="raw_races_cache.json"):
    """Load raw race cache. Delegates to getdata.load_raw_cache for consistent parsing."""
    from getdata import load_raw_cache as _gd_load
    raw = _gd_load(path)
    if not raw:
        cache_path = Path(path)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python3 getdata.py` once to build the cache."
            )
        raise ValueError(f"{path} exists but contained no usable race data.")
    return raw


def run_historical_from_cache(year, race_num):
    print(f"=== Historical predictions: {year} race {race_num} ===", flush=True)
    print("Reading raw_races_cache.json only. No scraping. No getdata.", flush=True)
    from getdata import build_datasets_for_race

    raw = load_raw_cache()
    training_rows, testing = build_datasets_for_race(raw, int(year), int(race_num))
    print(f"Built {len(training_rows)} no-lookahead training rows.", flush=True)

    output = f"predictions_historical_{int(year)}_{int(race_num):02d}.json"
    preds = write_predictions_from_rows(training_rows, testing, output)
    update_historical_index(preds, output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical",
        nargs=2,
        metavar=("YEAR", "RACE_NUM"),
        type=int,
        help="Build and predict one historical race from raw_races_cache.json only. Does not scrape.",
    )
    args = parser.parse_args()

    if args.historical:
        year, race_num = args.historical
        run_historical_from_cache(year, race_num)
        return

    print("=== Last-race predictions ===", flush=True)
    write_predictions("training_last_race.csv", "testing_last_race.json", "predictions_last_race.json")

    print("\n=== Next-race predictions ===", flush=True)
    write_predictions("training.csv", "testing_next_race.json", "predictions_next_race.json")


if __name__ == "__main__":
    main()