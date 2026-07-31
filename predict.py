import json
import csv
import warnings
import argparse
import hashlib
import os
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=RuntimeWarning)

TRACKS = ["s", "ss", "rc"]
TRACK_LABELS = {"s": "Speedway", "ss": "Superspeedway", "rc": "Road Course"}

SERIES_CONFIG = {"cup": "Cup Series", "oreilly": "O'Reilly Series", "truck": "Truck Series"}
SERIES = "cup"
SERIES_SUFFIX = ""
HISTORICAL_DIR = Path("predictions_historical") / "cup"

# Historical prediction files are invalidated automatically whenever the
# source code that determines their contents changes.
HISTORICAL_CACHE_SOURCE_FILES = ("predict.py", "getdata.py", "career_totals_pre_2005.json")


def historical_cache_signature():
    """Fingerprint all code that determines historical prediction contents.

    When launched by app.py, use the parent process's exact signature so the
    generated file and the Flask cache checker cannot disagree.
    """
    inherited = os.environ.get("HISTORICAL_CACHE_SIGNATURE")
    if inherited:
        return inherited

    digest = hashlib.sha256()
    base = Path(__file__).resolve().parent
    for filename in HISTORICAL_CACHE_SOURCE_FILES:
        path = base / filename
        digest.update(f"{filename}\0".encode())
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def historical_cache_metadata():
    return {
        "signature": historical_cache_signature(),
        "source_files": list(HISTORICAL_CACHE_SOURCE_FILES),
    }


def historical_cache_is_current(path):
    """Return True only when an existing JSON matches current model/data code."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        stored = payload.get("meta", {}).get("historical_cache", {})
        return stored.get("signature") == historical_cache_signature()
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def remove_stale_historical_cache(path):
    """Delete an obsolete historical JSON and report whether it was removed."""
    path = Path(path)
    if path.exists() and not historical_cache_is_current(path):
        path.unlink()
        print(f"  Removed stale historical cache: {path}", flush=True)
        return True
    return False

def configure_series(series):
    global SERIES, SERIES_SUFFIX, HISTORICAL_DIR
    series = str(series or "cup").lower()
    if series not in SERIES_CONFIG:
        raise ValueError(f"Unknown series: {series}")
    SERIES = series
    SERIES_SUFFIX = "" if series == "cup" else f"_{series}"
    HISTORICAL_DIR = Path("predictions_historical") / SERIES

def series_filename(stem, extension):
    return f"{stem}{SERIES_SUFFIX}.{extension}"

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

        # n_samples is far larger than n_features here (roughly 59k x 404).
        # covariance_eigh preserves the 99% explained-variance rule while
        # avoiding the very large sample-side matrices created by full SVD.
        # It is available in scikit-learn 1.5+.
        version_parts = []
        for part in str(sklearn.__version__).split(".")[:2]:
            digits = "".join(ch for ch in part if ch.isdigit())
            version_parts.append(int(digits or 0))
        sklearn_version = tuple((version_parts + [0, 0])[:2])
        pca_solver = "covariance_eigh" if sklearn_version >= (1, 5) else "full"
        print(f"    PCA solver: {pca_solver} (scikit-learn {sklearn.__version__})", flush=True)

        model = make_pipeline(
            StandardScaler(),
            PCA(n_components=0.99, svd_solver=pca_solver),
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

    expected = float(np.sum(positions * pmf))

    # Integer median for display, plus a linearly interpolated median for
    # precise ordering within the finishing-position bucket where the CDF
    # crosses 50%. Example: if F(15)=0.49 and F(16)=0.505, prediction=15.667.
    median_index = int(np.searchsorted(cdf, 0.5))
    median = float(median_index + 1)
    cdf_before = float(cdf[median_index - 1]) if median_index > 0 else 0.0
    bucket_probability = float(pmf[median_index]) if median_index < len(pmf) else 0.0
    if bucket_probability > 0:
        fraction_into_bucket = (0.5 - cdf_before) / bucket_probability
        prediction = float(max(1.0, median_index + fraction_into_bucket))
    else:
        prediction = median

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

    # Overall blocks are ordered last 5, last 10, last 20, last 36.
    baselines = {
        "last5_median_finish": overall_p50(0, "finish"),
        "last10_median_finish": overall_p50(1, "finish"),
        "last20_median_finish": overall_p50(2, "finish"),
        "last36_median_finish": overall_p50(3, "finish"),
        "last5_median_start": overall_p50(0, "start"),
        "last10_median_start": overall_p50(1, "start"),
        "last20_median_start": overall_p50(2, "start"),
        "last36_median_start": overall_p50(3, "start"),
        "last5_median_avg_pos": overall_p50(0, "avg_pos"),
        "last10_median_avg_pos": overall_p50(1, "avg_pos"),
        "last20_median_avg_pos": overall_p50(2, "avg_pos"),
        "last36_median_avg_pos": overall_p50(3, "avg_pos"),
        "last5_median_laps_completed": overall_p50(0, "pct_laps_completed"),
        "last10_median_laps_completed": overall_p50(1, "pct_laps_completed"),
        "last20_median_laps_completed": overall_p50(2, "pct_laps_completed"),
        "last36_median_laps_completed": overall_p50(3, "pct_laps_completed"),
        "last5_median_fastest_laps": overall_p50(0, "pct_fastest_laps"),
        "last10_median_fastest_laps": overall_p50(1, "pct_fastest_laps"),
        "last20_median_fastest_laps": overall_p50(2, "pct_fastest_laps"),
        "last36_median_fastest_laps": overall_p50(3, "pct_fastest_laps"),
        "last5_median_laps_top15": overall_p50(0, "pct_laps_top15"),
        "last10_median_laps_top15": overall_p50(1, "pct_laps_top15"),
        "last20_median_laps_top15": overall_p50(2, "pct_laps_top15"),
        "last36_median_laps_top15": overall_p50(3, "pct_laps_top15"),
        "last5_median_laps_led": overall_p50(0, "pct_laps_led"),
        "last10_median_laps_led": overall_p50(1, "pct_laps_led"),
        "last20_median_laps_led": overall_p50(2, "pct_laps_led"),
        "last36_median_laps_led": overall_p50(3, "pct_laps_led"),
    }

    raw_start = features[204] if len(features) > 204 else None

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
        "prediction":      prediction,

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




def build_pca_summary(model, raw_features, cars, top_components=8, max_top_drivers=20):
    """Return per-driver raw expected-finish contributions for each PCA component.

    For each retained component, predict every driver normally, then predict
    them again with only that component set to its neutral PCA value of zero.
    A positive contribution means the driver's actual component value improves
    their raw model expected finish:

        contribution = neutral_component_EV - actual_EV

    These values are calculated before Sinkhorn balancing so they isolate the
    direct model effect of the component on that driver rather than field-wide
    probability redistribution.
    """
    scaler = model.named_steps["standardscaler"]
    pca = model.named_steps["pca"]
    clf = model.named_steps["logisticregression"]

    standardized = scaler.transform(raw_features)
    scores = pca.transform(standardized)
    classes = np.asarray(clf.classes_, dtype=float)

    normal_probs = clf.predict_proba(scores)
    normal_expected = normal_probs @ classes

    components = []
    for index in range(scores.shape[1]):
        neutral_scores = scores.copy()
        neutral_scores[:, index] = 0.0
        neutral_probs = clf.predict_proba(neutral_scores)
        neutral_expected = neutral_probs @ classes

        # Positive = this driver's real value on the component lowers EV,
        # meaning it helps the driver's predicted finishing position.
        contributions = neutral_expected - normal_expected
        positive_order = [
            int(j) for j in np.argsort(-contributions)
            if contributions[j] > 1e-9
        ][:min(max_top_drivers, len(cars))]

        loading_order = np.argsort(-np.abs(pca.components_[index]))[:3]
        top_loadings = [
            {"feature": f"x{int(j)}", "loading": float(pca.components_[index, j])}
            for j in loading_order
        ]

        positive_values = contributions[contributions > 1e-9]
        mean_abs_help = float(np.mean(np.abs(contributions))) if len(contributions) else 0.0
        components.append({
            "component": int(index + 1),
            "explained_variance": float(pca.explained_variance_ratio_[index]),
            "top_loadings": top_loadings,
            "max_driver_help": float(np.max(positive_values)) if len(positive_values) else 0.0,
            "mean_positive_help": float(np.mean(positive_values)) if len(positive_values) else 0.0,
            "mean_abs_help": mean_abs_help,
            "drivers": [
                {
                    "car": str(cars[j]),
                    "ev_help": float(contributions[j]),
                    "pc_score": float(scores[j, index]),
                }
                for j in positive_order
            ],
        })

    # Rank components by their typical field-wide influence, not by a single
    # outlier driver. Absolute values count both helpful and harmful movement.
    components.sort(key=lambda item: item["mean_abs_help"], reverse=True)
    return {
        "observed_variables": int(raw_features.shape[1]),
        "retained_components": int(pca.n_components_),
        "retained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "contribution_basis": "raw model EV before Sinkhorn; component neutralized to zero",
        "components": components[:min(top_components, len(components))],
    }


def build_similar_historical_profiles(model, target_features, training_X, training_y,
                                      training_track_types, training_rows,
                                      track_type):
    """Return the single closest same-series observation from every season.

    Distances are measured directly in PCA score space, so components with
    greater explained variance contribute more to similarity. Candidates are
    restricted to the active series and target track type, then reduced to the
    nearest observation within each season. Results are displayed newest first
    and can extend all the way back to 2005 when data are available.
    """
    scaler = model.named_steps["standardscaler"]
    pca = model.named_steps["pca"]

    valid_indices = []
    for index, row in enumerate(training_rows):
        metadata = dict(row[3] or {}) if len(row) > 3 else {}
        if training_track_types[index] != track_type:
            continue
        if str(metadata.get("series", SERIES)).lower() != SERIES:
            continue
        try:
            season = int(metadata.get("season"))
        except (TypeError, ValueError):
            continue
        if season < 2005:
            continue
        valid_indices.append(index)

    candidate_indices = np.asarray(valid_indices, dtype=int)
    if candidate_indices.size == 0:
        return [[] for _ in range(len(target_features))]

    hist_scores = pca.transform(scaler.transform(training_X[candidate_indices]))
    target_scores = pca.transform(scaler.transform(target_features))
    out = []
    for target in target_scores:
        distances = np.sqrt(np.mean((hist_scores - target) ** 2, axis=1))
        best_by_season = {}
        for local_index, distance_value in enumerate(distances):
            source_index = int(candidate_indices[local_index])
            row = training_rows[source_index]
            metadata = dict(row[3] or {}) if len(row) > 3 else {}
            season = int(metadata["season"])
            distance = float(distance_value)
            current = best_by_season.get(season)
            if current is None or distance < current[0]:
                best_by_season[season] = (distance, source_index, metadata)

        matches = []
        for season in sorted(best_by_season, reverse=True):
            distance, source_index, metadata = best_by_season[season]
            matches.append({
                "car": metadata.get("car", ""),
                "driver": metadata.get("driver", ""),
                "season": season,
                "race_num": metadata.get("race_num"),
                "track_name": metadata.get("track_name", ""),
                "series": SERIES,
                "similarity": 100.0 / (1.0 + distance),
                "distance": distance,
                "finish": int(training_y[source_index]),
            })
        out.append(matches)
    return out

def predict_field(models, testing, mode_meta, training_X=None, training_y=None, training_track_types=None, training_rows=None):
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
            "model": "standardized PCA (95% variance) + regularized multinomial logistic regression + Sinkhorn",
            "field_size": field_size,
            "max_finish": field_size,
            "tracks": TRACK_LABELS,
            "params": MODEL_PARAMS,
            "mode_meta": mode_meta,
            "historical_cache": historical_cache_metadata(),
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

        output.setdefault("pca", {})[tt] = build_pca_summary(models[tt], features, cars)
        similar_profiles = build_similar_historical_profiles(
            models[tt], features, training_X, training_y,
            training_track_types, training_rows, tt
        ) if training_rows is not None else [[] for _ in cars]
        output["tracks"][tt] = {}
        for car, pmf, matches in zip(cars, pmfs, similar_profiles):
            cdf = np.cumsum(pmf)
            output["tracks"][tt][car] = summarize_driver(
                car, testing[car], tt, pmf, cdf, field_size
            )
            output["tracks"][tt][car]["similar_historical_profiles"] = matches

        win_total = sum(d["win"] for d in output["tracks"][tt].values())
        print(
            f"  Predicted {len(cars)} drivers for {TRACK_LABELS[tt]}; "
            f"field win total={win_total:.6f}.",
            flush=True,
        )

    return output

def _json_safe(value):
    """Recursively convert NumPy values and non-finite floats to valid JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _write_json(payload, output_path):
    safe_payload = _json_safe(payload)
    with open(output_path, "w") as f:
        json.dump(safe_payload, f, indent=2, allow_nan=False)
    return safe_payload


def write_predictions(input_training, input_testing, output_path):
    with open(input_testing) as f:
        testing = json.load(f)

    meta = testing.get("_meta", {})
    required_tracks = get_required_track_types(meta)

    field_size = get_target_field_size(testing)
    training_rows_path = Path(series_filename("training_rows", "json"))
    training_rows = None
    if training_rows_path.exists():
        try:
            raw_rows = json.loads(training_rows_path.read_text())
            training_rows = [
                (list(row[0]), row[1], row[2], dict(row[3] or {}))
                for row in raw_rows
                if isinstance(row, list) and len(row) >= 4
            ]
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"  Could not load {training_rows_path}: {exc}; similarity profiles disabled.", flush=True)
            training_rows = None

    if training_rows:
        X, y, tt = rows_to_arrays(training_rows, field_size)
    else:
        X, y, tt = load_training(input_training, field_size)
    models = train_models(X, y, tt, required_tracks)

    preds = predict_field(models, testing, meta, X, y, tt, training_rows)

    preds = _write_json(preds, output_path)
    print(f"Wrote {output_path}", flush=True)
    return preds


def historical_output_path(year, race_num):
    return HISTORICAL_DIR / str(int(year)) / f"predictions_historical_{int(year)}_{int(race_num):02d}.json"

def update_historical_index(preds, filename):
    mm = preds.get("meta", {}).get("mode_meta", {})
    race = mm.get("race", {})
    if not race.get("season") or not race.get("race_num"):
        return

    index_path = Path(series_filename("predictions_historical_index", "json"))
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
    for row in rows:
        feats, finish, track_type = row[:3]
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

    preds = predict_field(models, testing, meta, X, y, tt, training_rows)
    preds = _write_json(preds, output_path)
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
    from getdata import build_datasets_for_season, fetch_race, save_raw_cache, configure_series as configure_getdata
    configure_getdata(SERIES)

    raw = load_raw_cache(series_filename("raw_races_cache", "json"))
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
                save_raw_cache(raw, series_filename("raw_races_cache", "json"))
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
        remove_stale_historical_cache(output)
        if historical_cache_is_current(output):
            skipped += 1
            print(f"  Race {current_race:02d}: current cache; skipping.", flush=True)
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


def build_latest_historical_prediction():
    """Create/reuse the newest completed race prediction in the historical store."""
    from getdata import build_datasets_for_race, configure_series as configure_getdata
    configure_getdata(SERIES)
    raw = load_raw_cache(series_filename("raw_races_cache", "json"))
    if not raw:
        raise ValueError("No completed races are available.")
    year, race_num = max(raw)
    output = historical_output_path(year, race_num)
    remove_stale_historical_cache(output)
    if historical_cache_is_current(output):
        print(f"Latest completed race cache is current: {output}", flush=True)
        return str(output.as_posix())
    training_rows, testing = build_datasets_for_race(raw, year, race_num)
    output.parent.mkdir(parents=True, exist_ok=True)
    preds = write_predictions_from_rows(training_rows, testing, output)
    update_historical_index(preds, output)
    return str(output.as_posix())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", choices=sorted(SERIES_CONFIG), default="cup")
    parser.add_argument(
        "--historical",
        nargs=2,
        metavar=("YEAR", "RACE_NUM"),
        type=int,
        help="Build all missing historical predictions for the selected season in one pass.",
    )
    args = parser.parse_args()
    configure_series(args.series)

    if args.historical:
        year, race_num = args.historical
        run_historical_from_cache(year, race_num)
        return

    print("=== Latest completed race (historical cache) ===", flush=True)
    build_latest_historical_prediction()

    print("\n=== Next-race predictions ===", flush=True)
    write_predictions(series_filename("training", "csv"), series_filename("testing", "json"), series_filename("predictions", "json"))


if __name__ == "__main__":
    main()
