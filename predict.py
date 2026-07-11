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

# Fallback only. Normal prediction jobs use the actual target field size
# (number of cars in the race being predicted).
DEFAULT_FIELD_SIZE = 40


def _cars_in_testing(testing):
    return [k for k in testing.keys() if not str(k).startswith("_")]


def get_target_field_size(testing):
    """Return the number of possible finishing positions for this prediction job.

    Prefer explicit metadata when present, otherwise use the number of target
    cars. For completed historical/last-race jobs, also consider the max actual
    finish as a safety check in case a testing file is missing a car row.
    """
    meta = testing.get("_meta", {}) if isinstance(testing, dict) else {}

    for key in ("field_size", "target_field_size", "num_cars", "entry_count"):
        try:
            value = int(meta.get(key))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass

    cars = _cars_in_testing(testing)
    field_size = len(cars)

    max_actual = 0
    for car in cars:
        try:
            actual = testing[car].get("actual_finish")
            if actual is not None:
                max_actual = max(max_actual, int(actual))
        except (AttributeError, TypeError, ValueError):
            pass

    field_size = max(field_size, max_actual)
    return field_size if field_size > 0 else DEFAULT_FIELD_SIZE

MODEL_PARAMS = dict(
    # lbfgs is substantially faster than saga for dense, L2-regularized
    # multinomial logistic regression like this dataset.
    solver="lbfgs",
    penalty="l2",
    C=0.01,
    max_iter=1000,
    tol=1e-3,
    random_state=1,
)


def clean_feature_array(x):
    return np.nan_to_num(x, nan=20.0, posinf=float(DEFAULT_FIELD_SIZE), neginf=1.0)


def load_training(path, target_field_size=None):
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
    field_size = int(target_field_size or DEFAULT_FIELD_SIZE)
    y    = np.clip(y, 1, field_size)
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


def predict_pmf(model, x, field_size):
    probs   = model.predict_proba(x.reshape(1, -1))[0]
    classes = model.named_steps["logisticregression"].classes_

    field_size = int(field_size or DEFAULT_FIELD_SIZE)
    pmf = np.zeros(field_size)
    for cls, p in zip(classes, probs):
        if 1 <= cls <= field_size:
            pmf[cls - 1] = p

    total = pmf.sum()
    if total > 0:
        pmf /= total

    cdf = np.cumsum(pmf)
    return pmf, cdf



def balance_field_pmfs(pmf_matrix, max_iter=500, tol=1e-10):
    """Sinkhorn-balance a square driver-by-position PMF matrix."""
    matrix = np.asarray(pmf_matrix, dtype=float).copy()
    if matrix.ndim != 2:
        raise ValueError("pmf_matrix must be two-dimensional")

    n_drivers, n_positions = matrix.shape
    if n_drivers != n_positions:
        # A coherent one-driver-per-position matrix requires a square field.
        # Leave the model PMFs alone rather than applying an invalid balance.
        matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1e-15)
        return matrix

    matrix = np.maximum(matrix, 1e-15)
    for _ in range(max_iter):
        matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1e-15)
        matrix /= np.maximum(matrix.sum(axis=0, keepdims=True), 1e-15)

        row_error = np.max(np.abs(matrix.sum(axis=1) - 1.0))
        col_error = np.max(np.abs(matrix.sum(axis=0) - 1.0))
        if max(row_error, col_error) < tol:
            break

    matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1e-15)
    return matrix


def predict_pmfs_batch(model, feature_matrix, field_size):
    """Predict every driver's PMF in one model call, then field-balance it."""
    probs = model.predict_proba(feature_matrix)
    classes = model.named_steps["logisticregression"].classes_

    pmfs = np.zeros((len(feature_matrix), int(field_size)), dtype=float)
    for class_index, finish_class in enumerate(classes):
        finish_class = int(finish_class)
        if 1 <= finish_class <= field_size:
            pmfs[:, finish_class - 1] = probs[:, class_index]

    pmfs /= np.maximum(pmfs.sum(axis=1, keepdims=True), 1e-15)
    return balance_field_pmfs(pmfs)

def summarize_driver(car, driver_info, track_type, pmf, cdf, field_size):
    field_size = int(field_size or len(pmf) or DEFAULT_FIELD_SIZE)
    positions = np.arange(1, field_size + 1)

    expected    = float(np.sum(positions * pmf))
    median      = float(np.searchsorted(cdf, 0.5) + 1)
    driver_name = driver_info.get("driver_name", f"#{car}")

    features = driver_info.get("features") or []

    def feature_value(index):
        """Return a clean numeric feature value, or None if unavailable."""
        try:
            if index < 0 or index >= len(features):
                return None
            value = float(features[index])
            if np.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
        return None

    # Keep these names synchronized with getdata.METRICS. Each metric has
    # performance-oriented p10/p25/p50 slots, and each overall window is
    # one complete metric block. Historical starting position is now included
    # just like finish and average running position; the final feature remains
    # the current race's actual/fallback starting position.
    metric_order = [
        "finish", "start", "mid_pos", "closer_pos", "high_pos", "low_pos",
        "avg_pos", "pct_laps_completed", "pct_fastest_laps",
        "pct_laps_top15", "pct_laps_led",
    ]
    percentiles_per_metric = 3
    metric_block_size = len(metric_order) * percentiles_per_metric

    def overall_p50(window_index, metric):
        return feature_value(window_index * metric_block_size + metric_order.index(metric) * percentiles_per_metric + 2)

    baselines = {
        "last10_median_finish": overall_p50(0, "finish"),
        "last20_median_finish": overall_p50(1, "finish"),
        "last36_median_finish": overall_p50(2, "finish"),
        "last10_median_start": overall_p50(0, "start"),
        "last20_median_start": overall_p50(1, "start"),
        "last36_median_start": overall_p50(2, "start"),
        "last10_median_avg_pos": overall_p50(0, "avg_pos"),
        "last20_median_avg_pos": overall_p50(1, "avg_pos"),
        "last36_median_avg_pos": overall_p50(2, "avg_pos"),
        "last10_median_laps_completed": overall_p50(0, "pct_laps_completed"),
        "last20_median_laps_completed": overall_p50(1, "pct_laps_completed"),
        "last36_median_laps_completed": overall_p50(2, "pct_laps_completed"),
        "last10_median_fastest_laps": overall_p50(0, "pct_fastest_laps"),
        "last20_median_fastest_laps": overall_p50(1, "pct_fastest_laps"),
        "last36_median_fastest_laps": overall_p50(2, "pct_fastest_laps"),
        "last10_median_laps_top15": overall_p50(0, "pct_laps_top15"),
        "last20_median_laps_top15": overall_p50(1, "pct_laps_top15"),
        "last36_median_laps_top15": overall_p50(2, "pct_laps_top15"),
        "last10_median_laps_led": overall_p50(0, "pct_laps_led"),
        "last20_median_laps_led": overall_p50(1, "pct_laps_led"),
        "last36_median_laps_led": overall_p50(2, "pct_laps_led"),
    }

    raw_start = features[-1] if features else None

    starting_position = None
    try:
        if raw_start is not None and np.isfinite(float(raw_start)):
            starting_position = int(round(float(raw_start)))
    except (TypeError, ValueError):
        starting_position = None

    baselines["starting_position"] = float(starting_position) if starting_position is not None else None
    baseline_last10_p50_finish = baselines["last10_median_finish"]

    history_count = driver_info.get("history_count")
    try:
        history_count = int(history_count)
    except (TypeError, ValueError):
        history_count = None

    def clean_int_field(key):
        try:
            value = driver_info.get(key)
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    model_history_count = clean_int_field("model_history_count")
    team_history_fill_count = clean_int_field("team_history_fill_count")
    neutral_history_fill_count = clean_int_field("neutral_history_fill_count")

    limited_history = bool(driver_info.get("limited_history"))
    if history_count is not None:
        limited_history = history_count < 10

    def cdf_at(cdf_values, finish_pos):
        """Probability of finishing at or better than finish_pos.

        If the requested threshold is larger than the field size, it is a
        guaranteed event. Example: top25 in a 20-car race is 100%.
        """
        if len(cdf_values) == 0:
            return 0.0
        idx = min(max(int(finish_pos), 1), field_size) - 1
        return float(cdf_values[idx])

    return {
        "car":                 car,
        "name":                driver_name,
        "driver_name":         driver_name,
        "history_driver_name": driver_info.get("history_driver_name", driver_name),
        "manufacturer":        driver_info.get("manufacturer", ""),
        "team":                driver_info.get("team", ""),
        "canonical_team":      driver_info.get("canonical_team", ""),
        "track_type":          track_type,
        "track_label":         TRACK_LABELS[track_type],

        "starting_position": starting_position,
        "actual_finish": driver_info.get("actual_finish"),
        "history_count": history_count,
        "model_history_count": model_history_count,
        "team_history_fill_count": team_history_fill_count,
        "neutral_history_fill_count": neutral_history_fill_count,
        "limited_history": limited_history,
        "baseline_last10_p50_finish": baseline_last10_p50_finish,
        "baselines": baselines,

        "expected_finish": expected,
        "median_finish":   median,

        "field_size": field_size,

        "win":   cdf_at(cdf, 1),
        "top3":  cdf_at(cdf, 3),
        "top5":  cdf_at(cdf, 5),
        "top10": cdf_at(cdf, 10),
        "top15": cdf_at(cdf, 15),
        "top20": cdf_at(cdf, 20),
        "top25": cdf_at(cdf, 25),

        "pmf": pmf.tolist(),
        "cdf": cdf.tolist(),
    }


def car_sort_key(z):
    z = str(z)
    return (0, int(z)) if z.isdigit() else (1, z)


def predict_field(models, testing, mode_meta):
    """Run batched, field-balanced predictions for all target drivers."""
    mode = mode_meta.get("mode", "next_race")
    cars = sorted(_cars_in_testing(testing), key=car_sort_key)
    field_size = get_target_field_size(testing)

    # A complete target field should have one possible position per driver.
    # Prefer the actual target rows if stale metadata disagrees.
    if cars and field_size != len(cars):
        print(
            f"  WARNING: metadata field size {field_size} != {len(cars)} target cars; "
            f"using {len(cars)} for coherent field probabilities.",
            flush=True,
        )
        field_size = len(cars)

    output = {
        "meta": {
            "model": "fast regularized multinomial logistic regression + Sinkhorn",
            "field_size": field_size,
            "max_finish": field_size,
            "tracks": TRACK_LABELS,
            "params": MODEL_PARAMS,
            "mode_meta": mode_meta,
        },
        "tracks": {},
    }

    if mode in {"last_race", "historical"}:
        track_types_to_predict = [mode_meta.get("race", {}).get("track_type")]
    else:
        target_track_type = mode_meta.get("target_race", {}).get("track_type")
        track_types_to_predict = [target_track_type] if target_track_type in TRACKS else list(TRACKS)

    for tt in track_types_to_predict:
        if tt not in TRACKS:
            raise ValueError(f"Unknown track type: {tt!r}")

        print(f"  Predicting {TRACK_LABELS[tt]} in one batch...", flush=True)
        features = clean_feature_array(np.asarray([testing[car]["features"] for car in cars], dtype=float))
        pmfs = predict_pmfs_batch(models[tt], features, field_size)

        output["tracks"][tt] = {}
        for car, pmf in zip(cars, pmfs):
            cdf = np.cumsum(pmf)
            output["tracks"][tt][car] = summarize_driver(
                car, testing[car], tt, pmf, cdf, field_size
            )

        win_total = sum(d["win"] for d in output["tracks"][tt].values())
        print(
            f"  Predicted {len(cars)} drivers for {TRACK_LABELS[tt]}; "
            f"field win total={win_total:.6f}.",
            flush=True,
        )

    return output

def write_predictions(input_training, input_testing, output_path):
    with open(input_testing) as f:
        testing = json.load(f)

    meta = testing.get("_meta", {})
    required_tracks = get_required_track_types(meta)

    field_size = get_target_field_size(testing)
    X, y, tt = load_training(input_training, field_size)
    models = train_models(X, y, tt, required_tracks)

    preds = predict_field(models, testing, meta)

    with open(output_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {output_path}", flush=True)
    return preds


HISTORICAL_DIR = Path("predictions_historical")

def historical_output_path(year, race_num):
    return HISTORICAL_DIR / str(int(year)) / f"predictions_historical_{int(year)}_{int(race_num):02d}.json"

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
        "file": Path(filename).as_posix(),
    }

    index = [e for e in index if not (e.get("season") == entry["season"] and e.get("race_num") == entry["race_num"])]
    index.append(entry)
    index.sort(key=lambda e: (int(e.get("season", 0)), int(e.get("race_num", 0))))
    index_path.write_text(json.dumps(index, indent=2))
    print(f"Updated {index_path}", flush=True)



def rows_to_arrays(rows, target_field_size=None):
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
    field_size = int(target_field_size or DEFAULT_FIELD_SIZE)
    y = np.clip(np.array(y, dtype=int), 1, field_size)
    return X, y, np.array(track_types)


def write_predictions_from_rows(training_rows, testing, output_path):
    meta = testing.get("_meta", {})
    required_tracks = get_required_track_types(meta)

    field_size = get_target_field_size(testing)
    X, y, tt = rows_to_arrays(training_rows, field_size)
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
    """Generate every missing historical prediction in the selected season."""
    year = int(year)
    race_num = int(race_num)
    print(
        f"=== Historical season batch: {year} "
        f"(requested race {race_num}) ===",
        flush=True,
    )
    print(
        "Reading raw_races_cache.json and building the season in one "
        "chronological no-lookahead pass.",
        flush=True,
    )
    from getdata import build_datasets_for_season, fetch_race, save_raw_cache

    raw = load_raw_cache()
    target_key = (year, race_num)
    if target_key not in raw:
        raise ValueError(f"Race {year}-{race_num} was not found in raw cache.")

    # Preserve the old safety repair for the race the user explicitly selected.
    cached = raw.get(target_key)
    cached_count = (
        len(cached[1])
        if cached and len(cached) > 1 and isinstance(cached[1], dict)
        else 0
    )
    if 0 < cached_count < 36:
        print(
            f"  Cached requested race has only {cached_count} drivers; "
            "re-fetching it once with the loop-data fallback parser.",
            flush=True,
        )
        repaired = fetch_race(year, race_num)
        if repaired is not None:
            _, _, track_name, drivers = repaired
            if len(drivers) > cached_count:
                raw[target_key] = (track_name, drivers)
                save_raw_cache(raw)
                print(
                    f"  Repaired requested race to {len(drivers)} drivers "
                    "and saved raw cache.",
                    flush=True,
                )

    season_jobs = build_datasets_for_season(raw, year)
    requested_output = historical_output_path(year, race_num)
    generated = 0
    skipped = 0

    for current_race, training_rows, testing in season_jobs:
        output = historical_output_path(year, current_race)
        if output.exists():
            skipped += 1
            print(f"  Race {current_race:02d}: already cached; skipping.", flush=True)
            continue

        print(
            f"\n--- {year} race {current_race:02d}: "
            f"{len(training_rows)} no-lookahead training rows ---",
            flush=True,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        preds = write_predictions_from_rows(training_rows, testing, output)
        update_historical_index(preds, output)
        generated += 1

    if not requested_output.exists():
        raise RuntimeError(
            f"Season batch completed but did not create {requested_output}."
        )

    print(
        f"\nSeason {year} complete: generated {generated}, "
        f"reused {skipped} cached prediction file(s).",
        flush=True,
    )
    return str(requested_output.as_posix())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--historical",
        nargs=2,
        metavar=("YEAR", "RACE_NUM"),
        type=int,
        help="Build all missing historical predictions for the selected season in one pass.",
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