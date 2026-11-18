# PathSnap — Team Work Distribution (Deliverable 1, due before lunch)

> **For humans and their AI assistants.** This file is the single source of truth. Before you, or an LLM you use, write any code:
> 1. `git pull origin main` and **read the actual files in `src/pathsnap/`**. Never assume a function exists unless it's in this doc **and** in the repo.
> 2. Use **only** the file paths, column names, function signatures and output formats written here. Don't rename, add or drop columns in shared files.
> 3. Need something that isn't here? Ask the owner (see §2) in the group chat. Don't invent it.

Repo: https://github.com/Chunnu-Munnu/ML_HACKATHON · Python **3.11** · Seed **42** everywhere.

---

## 0. The problem in one paragraph

Each trip is a noisy GPS track (≈1 point per second) plus that trip's road network. Our pipeline:
1. For every GPS point, collect the nearby road segments (**arcs**) as **candidates**.
2. A **classifier** scores each (GPS point, candidate arc) pair: "is this the arc the car was on?"
3. An **HMM + Viterbi** decoder uses those probabilities plus road connectivity to choose one arc per point, so the result is a connected route.
4. A **FastAPI** server (Laptop 1) runs this for a chosen trip and scores it against the hidden reference route. A **Streamlit UI** (Laptop 2) displays the result.

**Deliverable 1** is the public GitHub repo containing all of this code plus every result file. The PDF report is later, but it will be built only from files in `results/`, so save everything.

---

## 1. Facts about the data (verified by scanning all 90 trips; do not assume otherwise)

**Folder layout** (local copy, gitignored; see §3):
```
data/train/<id>/<id>.track|.nodes|.arcs|.route        80 trips
data/student_test/<id>/<id>.track|.nodes|.arcs|.route 10 trips: 00000000 00000007 00000009 00000040 00000052 00000056 00000065 00000076 00000077 00000091
```
There is no evaluator data on our side. The missing IDs (00000015, 17, 23, 30, 49, 59, 67, 73, 87, 94) are presumably the evaluator's trips. **Never look for them or use them.**

**File formats** (no headers, whitespace-separated):
| File | Columns | Notes |
|---|---|---|
| `.track` | lon, lat, t (seconds) | **CRLF** line endings, tab-separated. t is strictly increasing but doesn't always start at 1. Trips 02, 71 and 97 have non-integer t. |
| `.nodes` | lon, lat | Row index = node ID |
| `.arcs` | u, v (node IDs) | Row index = arc ID; arcs are **directed** u→v |
| `.route` | arc_id | Ground-truth arcs in driving order. **No timestamps.** Row i of the route is NOT GPS point i. |

**Key facts:**
- **Every trip has its own road network:** 299 to 155,003 nodes and 649 to 307,392 arcs. Nothing is shared between trips.
- **Trips are spread over the whole world** (lon −100…137, lat −35…62). Distances are computed in metres after projecting **each trip to its own UTM zone**. This is already implemented in `geo.Projector`. **Never compute distances in degrees.**
- **About 91% of arcs have a reverse twin** (u→v and v→u, identical geometry), so only heading can tell the two directions apart.
- **Network oddities:**
  - Duplicate arcs (the same u→v twice, up to 322 in one trip). `RoadNetwork` collapses them to a **canonical ID** = the smallest arc ID with that (u, v). **All labels and predictions use canonical IDs.**
  - Self-loops and zero-length arcs are never candidates.
- **Large time gaps:** 153,064 s in trip 90, 52,810 s in trip 78, and 37 gaps over 30 s in trip 81. The median trip has 99% of steps at exactly 1 s.
- **GPS jumps:** trip 32 has 891 steps faster than 50 m/s. Trip 54 has 402 duplicate points.
- **Typical GPS-to-route distance:** median ≈ 4.5 m, p95 ≈ 13 m, p99 ≈ 21 m. Arc length median ≈ 25 m.
- **Bad reference labels:** trips **00000054** and **00000033** are 120–140 m from their own route (median), so the GPS doesn't line up with the map. They are **excluded from model fitting** using a train-only rule: median alignment distance > 50 m. Trip 02's end is 549 m from its route end.
- **Route breaks:** 11 routes have breaks where vᵢ ≠ uᵢ₊₁, including student-test trip 00000076.

---

## 2. Who does what (8 models: 2 per person, all from different families)

| Member | Main responsibility | Model 1 (family) | Model 2 (family) |
|---|---|---|---|
| **M1 (Amogh)** | Data core, labels, features, `predict_trip` pipeline, README, merges | Logistic Regression (linear) | Gaussian Naive Bayes (probabilistic) |
| **M2** | EDA notebook, Folium maps, FastAPI, Streamlit UI | KNN, distance-weighted (instance-based) | Decision Tree (tree) |
| **M3** | Experiment harness, metrics, model comparison, Bayesian HPO (Optuna) | LightGBM (boosting) | Random Forest (bagging ensemble) |
| **M4** | HMM/Viterbi, route metrics, HMM evaluation, decoder tuning | MLP (neural network) | SVM: LinearSVC + calibration (kernel) |

M3 also runs the **B0 baseline (nearest arc, no ML)**.

**File ownership.** Edit only your own files; that's how we avoid merge conflicts.
| Owner | Files |
|---|---|
| M1 | `src/pathsnap/{config,io,geo,network,candidates,labeling,features,pipeline}.py`, `scripts/build_dataset.py`, `scripts/export_artifacts.py`, `scripts/models/train_logreg.py`, `scripts/models/train_gnb.py`, `README.md`, `requirements.txt`, `.gitignore`, `pyproject.toml` |
| M2 | `notebooks/01_eda.ipynb`, `src/pathsnap/viz.py`, `scripts/make_maps.py`, `api/main.py`, `ui/app.py`, `scripts/models/train_knn.py`, `scripts/models/train_dtree.py`, `results/eda/*`, `results/maps/*` |
| M3 | `src/pathsnap/{metrics,experiment}.py`, `scripts/models/baseline_nearest.py`, `scripts/models/train_lgbm.py`, `scripts/models/train_rf.py`, `scripts/compare_models.py`, `scripts/tune_optuna.py`, `results/models/model_comparison.csv`, `results/optuna/*` |
| M4 | `src/pathsnap/{hmm,route_metrics}.py`, `scripts/eval_hmm.py`, `scripts/tune_decoder.py`, `scripts/models/train_mlp.py`, `scripts/models/train_svm.py`, `results/hmm/*`, `artifacts/hmm_config.json` |

Each model's folder `results/models/<name>/` belongs to whoever owns that model.

---

## 3. Setup (everyone, first 15 minutes)

```bash
git clone https://github.com/Chunnu-Munnu/ML_HACKATHON.git
cd ML_HACKATHON
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

- **`pip install -e .`** makes `from pathsnap.io import load_trip` work from any folder.
- **Data:** copy the provided `train` and `student_test` folders into `data/`, keeping the layout from §1. `data/` is gitignored, so never commit it.
- **Features:** once M1 pushes `features.py` (announced in the group), run `python scripts/build_dataset.py`. It creates `results/features/` locally (gitignored, ~a few minutes).
- **Branches:** `m1-core`, `m2-eda-api-ui`, `m3-models-hpo`, `m4-hmm`. Run `git pull origin main` often, open a PR or merge into `main` at each milestone (§10), and make small commits with clear messages.
- **New package needed?** Tell M1, who adds it to `requirements.txt`.

---

## 4. Code that already exists (on `main`). Use it; don't rewrite it.

```python
from pathsnap import config
config.ROOT, config.DATA_DIR, config.TRAIN_DIR, config.STUDENT_TEST_DIR, config.EVAL_DIR   # pathlib.Path
config.RESULTS_DIR, config.FEATURES_DIR, config.ARTIFACTS_DIR
config.SEED = 42; config.CAND_RADIUS_M = 50.0; config.CAND_K = 8
config.JUMP_SPEED_MS = 70.0; config.GAP_S = 5.0; config.STATIONARY_MS = 1.0; config.BAD_LABEL_MEDIAN_M = 50.0
# EVAL_DIR comes from the env var EVAL_DIR, default = data/student_test (dev stand-in)

from pathsnap.io import load_trip, load_route, list_trips, Trip
trip = load_trip("data/train/00000001", with_route=True)   # with_route=False on any prediction path
trip.trip_id   # "00000001"
trip.split     # "train" | "student_test" | folder name
trip.track     # DataFrame columns: lon, lat, t, dt_prev, is_dup, is_jump, is_gap   (one row per GPS point, never dropped)
trip.nodes     # DataFrame: lon, lat
trip.arcs      # DataFrame: u, v (int64)
trip.route     # np.ndarray[int64] of raw arc ids, or None
load_route(trip_dir) -> np.ndarray      # server-side scoring only
list_trips(split_dir) -> list[Path]     # sorted trip folders

from pathsnap.geo import Projector, point_segment, bearing, angle_diff, smoothed_heading
# Projector(lon0, lat0) / Projector.for_track(lon, lat): .epsg, .fwd(lon,lat)->(x,y) metres, .inv(x,y)->(lon,lat)
# point_segment(px,py,ax,ay,bx,by) -> (dist, t_clamped, t_raw, qx, qy)     broadcastable numpy
# bearing(dx,dy) -> degrees [0,360) (0 = north);  angle_diff(a,b) -> [0,180]

from pathsnap.network import RoadNetwork
net = RoadNetwork(trip)                 # ~1 s for the biggest trips
net.x, net.y, net.t                     # projected GPS points (metres), timestamps
net.nx, net.ny                          # projected nodes
net.u, net.v, net.ux, net.uy, net.vx, net.vy, net.length, net.bearing   # per raw arc id
net.canon                               # canonical id per arc (duplicates collapsed)
net.rev_of                              # canonical id of the reverse twin, or -1
net.out_deg, net.in_deg                 # per node
net.cand_ids                            # arc ids allowed as candidates
net.route_dist(a, ta, b, tb, cutoff=2000.0) -> float   # metres driving from fraction ta on arc a to tb on arc b; inf if unreachable within cutoff
net.node_to_node(src, dst, cutoff) -> float
net.connected(a, b) -> bool             # a == b or v[a] == u[b]
net.arc_lonlat(arc_ids) -> [[(lat,lon),(lat,lon)], ...]   # for folium
net.proj.inv(x, y) -> (lon, lat)

from pathsnap.candidates import get_candidates
cands = get_candidates(net, r=50.0, k=8)   # DataFrame: point_idx, arc_id, dist, t_proj, t_raw, qx, qy
# sorted by point_idx then dist; points with no arc within r have NO rows

from pathsnap.labeling import align_route
labels, align_dist, pos = align_route(net, trip.route)   # per GPS point: canonical label arc, distance (m), route index
```

---

## 5. Files M1 will push next (contract; M1 implements exactly this)

### 5.1 `results/features/` (created locally by `python scripts/build_dataset.py`)

| File | One row per | Columns |
|---|---|---|
| `train.parquet` | (GPS point, candidate arc) in the 80 train trips | `trip_id` (str), `point_idx` (int), `arc_id` (int, canonical), `label` (0/1), `align_dist` (float m), **all of `FEATURES`** |
| `student_test.parquet` | same, for the 10 student-test trips | same |
| `points_train.parquet` | GPS point | `trip_id`, `point_idx`, `t`, `lon`, `lat`, `label_arc` (canonical), `align_dist`, `n_cands`, `label_in_cands` (bool) |
| `points_student_test.parquet` | GPS point | same |
| `excluded_train_trips.json` | — | `{"rule": "...", "trips": ["00000033","00000054", ...]}`. Rows of these trips are **still in** `train.parquet`; the harness drops them before fitting. |

Also written: `results/eda/label_alignment.csv` (alignment stats per trip, for M2's EDA) and `results/eda/candidate_miss_rate.csv` (miss rate for r ∈ {20, 30, 50, 75} and K ∈ {4, 8, 12}, on train).

`label = 1` iff `arc_id == label_arc` of that point. A point is a **candidate miss** if none of its rows has `label = 1`, including points with no rows at all.

### 5.2 `FEATURES` (`from pathsnap.features import FEATURES`, a list of column names)

```
# geometry
dist, dist_rank, dist_minus_min, dist_ratio_min, dist_z, t_proj, off_end_m, dist_to_endpoint, arc_len, log_arc_len,
# direction
head_diff_s, head_diff_l, head_cos, head_rank, has_reverse, beats_reverse, turn_rate,
# kinematics
speed, speed_smooth, accel, dt_prev, dt_next, is_gap, is_jump, is_stationary,
# temporal context
ctx_dist_3, ctx_dist_7, ctx_nearest_frac_3, prev_dist, next_dist, in_prev_cands, in_next_cands, conn_prev, conn_prev_nearest,
# network / candidate set
out_deg_v, in_deg_u, n_cands, n_cands_10m, score_rank
```

| Feature | Meaning |
|---|---|
| `dist` | Perpendicular distance from the point to the arc, in metres |
| `dist_rank` | Rank of `dist` among this point's candidates (0 = nearest) |
| `dist_minus_min` | `dist` minus the nearest candidate's `dist` |
| `dist_ratio_min` | `dist / (min dist + 1)` |
| `dist_z` | `dist` standardised within the point's candidates |
| `t_proj` | Where the projection lands along the arc, 0–1 |
| `off_end_m` | Metres the unclamped projection falls past the arc's ends (0 if it lands on the arc) |
| `dist_to_endpoint` | Distance to the nearer of the arc's two end nodes |
| `arc_len`, `log_arc_len` | Arc length and its log |
| `head_diff_s`, `head_diff_l` | Angle between the GPS heading (smoothed over i±2 and i±5 respectively) and the arc bearing, 0–180 |
| `head_cos` | cos of `head_diff_s` |
| `head_rank` | Rank of `head_diff_s` among the point's candidates |
| `has_reverse` | 1 if the arc has a reverse twin |
| `beats_reverse` | 1 if the heading fits this arc better than its twin |
| `turn_rate` | Change in heading per second |
| `speed`, `speed_smooth`, `accel` | m/s, smoothed m/s, m/s² |
| `dt_prev`, `dt_next` | Seconds to the previous and next point |
| `is_gap`, `is_jump`, `is_stationary` | 0/1 flags |
| `ctx_dist_3`, `ctx_dist_7` | Mean distance of points i±3 and i±7 to the same arc |
| `ctx_nearest_frac_3` | Share of points i±3 whose nearest candidate is this arc |
| `prev_dist`, `next_dist` | Distance of points i−1 and i+1 to this arc |
| `in_prev_cands`, `in_next_cands` | 0/1: the arc is also a candidate of the previous / next point |
| `conn_prev` | 0/1: equal to, or directly connected from, any candidate of point i−1 |
| `conn_prev_nearest` | The same check, against point i−1's nearest candidate only |
| `out_deg_v`, `in_deg_u` | Node degrees at the arc's ends |
| `n_cands` | Number of candidates the point has |
| `n_cands_10m` | Number of candidates within 10 m |
| `score_rank` | Rank by `dist + 0.5·head_diff_s` |

**Never use as features:** `trip_id`, `point_idx`, `lon`, `lat`, `t`, `arc_id`, `label`, `align_dist`.

### 5.3 `src/pathsnap/features.py` and `src/pathsnap/pipeline.py`
```python
build_trip_features(trip: Trip, net: RoadNetwork | None = None, r=config.CAND_RADIUS_M, k=config.CAND_K) -> (rows_df, points_df)
# rows_df: trip_id, point_idx, arc_id, *FEATURES, plus label & align_dist only if trip.route is not None
predict_trip(trip: Trip, model, hmm_cfg: dict) -> dict  # see §8 (M1 builds; M2's API calls it)
```

---

## 6. M3: experiment harness and metrics. **Push by T+1:00; everyone's models depend on it.**

### 6.1 `src/pathsnap/metrics.py`
```python
def point_top1(rows: pd.DataFrame, proba: np.ndarray, points: pd.DataFrame) -> float
    # rows has trip_id, point_idx, arc_id, label. For each (trip_id, point_idx) pick the arc with the max proba.
    # Correct if that row's label == 1. Denominator = ALL points in `points` (points without candidates count as wrong).
def point_top1_covered(rows, proba) -> float          # same, but only over points that have >= 1 candidate
def candidate_miss_rate(points: pd.DataFrame) -> float # mean(~points.label_in_cands)
def row_metrics(y_true, proba, thr=0.5) -> dict       # accuracy, precision, recall, f1 (positive class),
                                                      # macro_f1 (average="macro"), log_loss (labels=[0,1], proba clipped 1e-7)
def confusion(y_true, proba, thr=0.5) -> np.ndarray   # 2x2, sklearn.metrics.confusion_matrix(labels=[0,1])
```

### 6.2 `src/pathsnap/experiment.py`
```python
def load_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]   # ("train"|"student_test") -> (rows, points) from results/features/
def run_experiment(name: str, family: str, estimator, config: dict,
                   feature_cols: list[str] = FEATURES, subsample_points: float | None = None,
                   seed: int = 42, save_model: bool = True) -> dict
```

**What `run_experiment` must do, in this order:**
1. Load train + student-test. **Drop rows of trips listed in `excluded_train_trips.json` from train only.** Keep the full student-test split.
2. If `subsample_points` is set (e.g. 0.2), sample that fraction of **(trip_id, point_idx) groups** from train with `random_state=seed`, keeping all candidate rows of each sampled point. Record the fraction in `config.json`.
3. **CV:** `GroupKFold(n_splits=5)` with `groups=trip_id` on the (filtered) train rows. For each fold: `clone(estimator).fit(X_tr, y_tr)`, `predict_proba(X_va)[:, 1]`. Record row_metrics + point_top1 (using the train points of the validation trips) + fit time.
4. **Refit** on all filtered train rows (timed). Score student-test: row_metrics, point_top1, point_top1_covered, confusion matrix, and **inference time** (total seconds over the 10 student-test trips / 10 = seconds per trip). Also report the overfitting check: train-set macro F1 and Top-1 next to the CV values.
5. **Write** `results/models/<name>/`:
   - `config.json`: name, family, `estimator.get_params()` (stringify non-JSON values), feature_cols, subsample_points, seed, excluded trips
   - `cv_folds.csv`: fold, n_train_rows, n_val_rows, macro_f1, f1, precision, recall, accuracy, log_loss, top1, fit_s
   - `metrics.json`: keys `cv_macro_f1_mean`, `cv_macro_f1_std`, `cv_top1_mean`, `cv_top1_std`, `cv_log_loss_mean`, `train_macro_f1`, `train_top1`, `st_accuracy`, `st_precision`, `st_recall`, `st_f1`, `st_macro_f1`, `st_log_loss`, `st_top1`, `st_top1_covered`, `st_miss_rate`, `fit_s`, `infer_s_per_trip`
   - `confusion_matrix.png`: student-test, threshold 0.5, with a title naming the model
   - `student_test_proba.parquet`: columns `trip_id, point_idx, arc_id, proba` (**M4 needs this for the HMM**)
   - `model.joblib` if under 50 MB (otherwise skip it and note that in `config.json`)
   - `loss_curve.png` **only if** the fitted estimator (or its last pipeline step) has `loss_curve_`. Never fake curves for other models.
6. **Append or replace** one row (keyed by `name`) in `results/models/model_comparison.csv` with columns: `name, family, key_config, cv_macro_f1_mean, cv_macro_f1_std, cv_top1_mean, st_top1, st_macro_f1, st_log_loss, st_miss_rate, fit_s, infer_s_per_trip, route_continuity`. Leave `route_continuity` empty; M4 fills it via `eval_hmm.py`.
7. Return the metrics dict and print a one-line summary.

**Preprocessing:** any scaler lives **inside** an sklearn `Pipeline`, so it's fitted on the training folds only. Missing values: `features.py` fills NaNs (e.g. heading when stationary), but the harness should still do `X = X.fillna(0)` to be safe.

### 6.3 How each model script looks (identical pattern for all 8)
```python
# scripts/models/train_<name>.py
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathsnap.experiment import run_experiment

est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
run_experiment("logreg", "Linear", est, {"C": 1.0, "class_weight": "balanced"})
```
If you try more than one configuration, give each a separate `name` (e.g. `knn_k15`, `knn_k31`). Keep every run, because every model and configuration you try has to be documented.

**Suggested starting configurations** (record whatever you actually use):
| Name | Estimator |
|---|---|
| logreg | `StandardScaler` + `LogisticRegression(max_iter=2000, class_weight="balanced")` |
| gnb | `GaussianNB()` (scaling not needed; note the independence assumption) |
| knn | `StandardScaler` + `KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)`, `subsample_points=0.2` (it's slow) |
| dtree | `DecisionTreeClassifier(max_depth=12, min_samples_leaf=50, class_weight="balanced", random_state=42)` |
| lgbm | `LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)` |
| rf | `RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, class_weight="balanced_subsample", random_state=42)` |
| mlp | `StandardScaler` + `MLPClassifier(hidden_layer_sizes=(64,32), early_stopping=True, max_iter=200, random_state=42)`; save `loss_curve_` and `validation_scores_` |
| svm | `StandardScaler` + `CalibratedClassifierCV(LinearSVC(C=1.0, dual=False), cv=3)`, `subsample_points=0.3` |

**B0 baseline** (`scripts/models/baseline_nearest.py`, M3): proba = 1 for the `dist_rank == 0` row and 0 for the others. Use rank-based scores in [0, 1] for log loss, or report N/A with the reason. No fitting. Same output files.

### 6.4 `scripts/compare_models.py` (M3)
- Reads every `results/models/*/metrics.json`.
- **Plots** to `results/models/`:
  - `cv_macro_f1.png` (bar chart with std error bars)
  - `st_top1_macro_f1.png` (grouped bars)
  - `log_loss.png`
  - `inference_time.png`
  - `top1_before_after_hmm.png` (reads `results/hmm/hmm_comparison.csv` once M4 has written it)
- Writes `selection.md`. **Selection rule:** highest student-test Top-1 after HMM; tiebreaker lower student-test log loss; also consider CV stability (std) and inference time. **Never select on training accuracy.**

### 6.5 `scripts/tune_optuna.py` (M3, Bayesian HPO on the selected model; probably LightGBM)
- `optuna.create_study(direction="maximize", sampler=TPESampler(seed=42), pruner=MedianPruner(), storage="sqlite:///results/optuna/study.db", study_name="pathsnap_<model>")`.
- **Objective:** mean GroupKFold(5) CV **point Top-1** on filtered train only. Report each fold with `trial.report` so pruning works.
- **Search space** (LightGBM):
  - `learning_rate` loguniform 0.01–0.3
  - `num_leaves` int 15–255
  - `max_depth` int 3–12
  - `min_child_samples` int 5–200
  - `reg_lambda` loguniform 1e-3–10
  - `colsample_bytree` 0.5–1.0
  - `subsample` 0.5–1.0
  - `n_estimators` int 100–800
- 40–60 trials, or `timeout=1800`.
- **Outputs:** `results/optuna/trials.csv` (`study.trials_dataframe()`), `best_params.json`, `history.png`, `param_importance.png`. Refit with the best parameters via `run_experiment("lgbm_tuned", ...)`. Write `tuned_vs_untuned.csv` comparing CV macro F1, ST Top-1, ST macro F1, log loss and inference time. Copy the tuned model to `artifacts/model.joblib`.
- **Student-test is used only after tuning, for the final comparison. Never inside the objective.**

---

## 7. M4: HMM / Viterbi (`src/pathsnap/hmm.py`, `route_metrics.py`)

### 7.1 Signature
```python
DEFAULT_HMM_CFG = {"lam": 1.0, "beta": 10.0, "sigma": 8.0, "v_max": 60.0, "gap_s": 30.0,
                   "backward_tol_m": 10.0, "route_cutoff_m": 3000.0, "min_logp": -50.0}

def viterbi(cands: pd.DataFrame, net: RoadNetwork, cfg: dict = DEFAULT_HMM_CFG,
            emission: str = "proba") -> tuple[np.ndarray, dict]
# cands columns: point_idx, arc_id, dist, t_proj, proba    (proba required when emission == "proba")
# emission == "proba": log e = lam * log(clip(proba, 1e-6, 1))
# emission == "dist" : log e = -0.5 * (dist / sigma)**2       (baseline B1)
# returns: arcs (np.ndarray[int64], length = len(net.x), one canonical arc per GPS point, -1 only if the
#          whole trip has no candidates), info = {"n_breaks", "n_no_cand", "n_gap_splits", "no_cand_mask"}
```

### 7.2 Algorithm
- **States** at step i are the candidate rows of point i. Points with no candidates are skipped in the chain.
- **Transition** between consecutive *decoded* points i → j:
  - `d_gps = hypot(net.x[j]-net.x[i], net.y[j]-net.y[i])`
  - `d_route = net.route_dist(a, ta, b, tb, cfg["route_cutoff_m"])`
  - If a == b and tb < ta with `(ta - tb) * net.length[a] <= backward_tol_m`, set `d_route = 0` (GPS jitter).
  - `log T = -abs(d_route - d_gps) / beta`
  - log T = −∞ if `d_route` is inf **or** `d_route > v_max * dt + 50`.
- **Recurrence:** `δ_j(b) = max_a [δ_i(a) + log T(a,b)] + log e_j(b)`, with backpointers, all in log space. K ≤ 8 states per step, so loop over states with numpy.
- **Break:** if every transition into step j is −∞, end the chain at i (backtrack it), start a new chain at j with `δ = log e_j`, and `n_breaks += 1`.
- **Gap:** if `t[j] - t[i] > gap_s`, start a new chain without a transition and `n_gap_splits += 1`.
- **No-candidate points** get the decoded arc of the nearest decoded point in time (ties → previous). They're marked in `no_cand_mask`.
- **Speed:** cache `net.route_dist` per (v(a), u(b)) through `net.node_dists` (already cached inside `RoadNetwork`). A trip with 20k points must decode in under ~60 s.

### 7.3 `src/pathsnap/route_metrics.py`
```python
def continuity(arcs: np.ndarray, net: RoadNetwork) -> float
    # over consecutive points with arcs >= 0: share where net.connected(prev, cur) is True
def point_accuracy(pred_arcs, label_arcs) -> float        # mean(pred == label)
def route_prf(pred_arcs, route_canon) -> dict             # precision/recall/f1 of the SET of arcs vs set(net.canon[route])
def compress(arcs) -> np.ndarray                          # consecutive duplicates removed -> predicted route sequence
```

### 7.4 `scripts/eval_hmm.py`
- **Compare 4 variants on the 10 student-test trips:**
  - (B0) nearest arc
  - (B1) nearest + HMM (`emission="dist"`)
  - (C) classifier only (argmax proba per point)
  - (C+H) classifier + HMM (`emission="proba"`)
- **Run** C and C+H for **every** model that has `results/models/<name>/student_test_proba.parquet`.
- **Metrics:** point accuracy vs `label_arc` (from `points_student_test.parquet`; points without candidates count as wrong), continuity, route P/R/F1, n_breaks, decode time.
- **Outputs:**
  - `results/hmm/hmm_comparison.csv` with columns `model, variant, trip_id, top1, continuity, route_precision, route_recall, route_f1, n_breaks, decode_s`, plus rows with `trip_id = "ALL"` (point-weighted mean).
  - `results/hmm/hmm_gain.png`
  - Fill the `route_continuity` column of `results/models/model_comparison.csv` with each model's C+H continuity.
  - Maps (via M2's `viz.trip_map`): `results/maps/hmm_helped_<trip>.html` ×2 and `hmm_hurt_<trip>.html` ×1.

### 7.5 `scripts/tune_decoder.py`
- Grid or Optuna over `lam ∈ [0.3, 2]`, `beta ∈ [2, 50]`, `v_max ∈ [40, 80]`, `gap_s ∈ [10, 120]`.
- Objective: C+H point Top-1 on **train trips' out-of-fold probabilities** or the student-test probabilities (allowed: the decoder may be tuned on train + student-test). **Never evaluator data.**
- **Outputs:** `results/hmm/decoder_search.csv`, `artifacts/hmm_config.json`.

---

## 8. M2: EDA, maps, API, UI

### 8.1 `notebooks/01_eda.ipynb` (partition = **train** unless stated; write it in every section title)
1. **Inventory table:** trip, split, files present, file sizes, n_points, n_nodes, n_arcs, n_route.
2. **Data quality:** dtypes, NaN counts, duplicate points (`is_dup`), jumps (`is_jump`), non-integer timestamps.
3. **Time:** Δt histogram (log y-axis), share of 1 s steps per trip, a table of the 10 largest gaps.
4. **Geography:** a Folium world map of trip start points, plus a lon/lat extent table.
5. **Network:** n_nodes and n_arcs distribution; weakly and strongly connected components (networkx) for ~10 trips (the biggest ones are slow, so say which you used); degree histogram; reverse-twin share; duplicate arcs; self-loops; isolated nodes (there are 0).
6. **Routes:** length (arcs and km), breaks (vᵢ ≠ uᵢ₊₁).
7. **GPS-to-route distance:** histogram of `align_dist` from `results/features/points_train.parquet`, plus the suspect trips (54, 33, 02) and why they're excluded.
8. **Speed** distribution and outliers.
9. **Candidate miss rate vs radius/K** from `results/eda/candidate_miss_rate.csv` (justifies r = 50, K = 8).
10. **At least one Folium map** of one train trip: raw GPS + nearby network + reference route.

Save figures as `results/eda/<name>.png` and tables as `.csv`. Write `results/eda/data_quality.md`: a bullet list of issues → decisions.

### 8.2 `src/pathsnap/viz.py`
```python
def trip_map(trip: Trip, net: RoadNetwork | None = None, pred_arcs: np.ndarray | None = None,
             show_route: bool = True, network_buffer_m: float = 150.0) -> folium.Map
```
- **Layers**, each a `folium.FeatureGroup` with a name:
  - "GPS observations": blue `CircleMarker` radius 2, thin out to ≤ 3000 points
  - "Road network (nearby)": grey lines, only arcs within `network_buffer_m` of the track, using `net.query_radius`, capped at 5000 arcs
  - "Reference route": green, width 5, from `net.canon[trip.route]`, only if `show_route and trip.route is not None`
  - "Predicted route": red, width 3, from `pred_arcs`
  - Start/end markers
- `folium.LayerControl()`, an HTML legend, and `fit_bounds` on the track.
- Use `net.arc_lonlat(ids)` for coordinates.

`scripts/make_maps.py`: write `results/maps/train_<id>_raw.html` (1 train trip) and `results/maps/st_<id>_pred.html` (2 student-test trips, reference + prediction).

### 8.3 `api/main.py` (FastAPI)
- At startup, load `artifacts/model.joblib` and `artifacts/hmm_config.json` once.
- Set up `logging` to the console **and** `logs/api.log` with the format `%(asctime)s %(levelname)s %(message)s`.
- Use `CORSMiddleware(allow_origins=["*"])`.

**`GET /health`**:
```json
{"status": "ok", "model": "lgbm_tuned", "eval_dir": "..."}
```

**`GET /trips`**:
```json
{"trips": ["00000000", "..."]}
```
These are the folder names in `config.EVAL_DIR`, sorted.

**`POST /predict`**, body `{"trip_id": "00000000"}`:
1. Validate that `trip_id` is in `/trips`; otherwise return 404.
2. `trip = load_trip(EVAL_DIR / trip_id, with_route=False)` and `out = predict_trip(trip, model, hmm_cfg)`.
3. **Separately**, `route = load_route(EVAL_DIR / trip_id)`. Compute `top1` (point accuracy vs the labels from `align_route` on the server) and `continuity`. The route is never put in the response.
4. Respond with:
   ```json
   {"trip_id": "...", "n_points": 0,
    "gps": [[lat, lon], ...],
    "pred_route": [[[lat, lon], [lat, lon]], ...],
    "point_arcs": [...],
    "metrics": {"top1_accuracy": 0.0, "continuity": 0.0, "n_breaks": 0, "route_f1": 0.0},
    "timings": {"load_s": 0, "features_s": 0, "model_s": 0, "hmm_s": 0, "total_s": 0}}
   ```
   `gps` may be thinned to ≤ 3000 points for transfer; say so in the logs.
5. Log one line per request: `PREDICT trip=<id> n_points=<n> top1=<x> continuity=<y> total_s=<z>`.

**Run with:** `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Until M1's `predict_trip` lands, use a stub that returns the nearest candidate arc, keeping the same response format.

### 8.4 `ui/app.py` (Streamlit + streamlit-folium)
- **Sidebar:** text input "API URL", defaulting to `os.environ.get("PATHSNAP_API", "http://localhost:8000")`, and a health check indicator.
- `requests.get(f"{api}/trips")` fills a selectbox. A "Reconstruct route" button POSTs to `/predict`.
- **Display:**
  - A Folium map built from the response only: blue GPS points, red predicted route, legend, LayerControl.
  - `st.metric` cards: Top-1 accuracy, continuity, n_points, total latency.
  - The timings table.
- **The UI does no computation and never reads local data files.**
- Run with `streamlit run ui/app.py`.

---

## 9. M1: remaining tasks (for reference)
1. `features.py` + `scripts/build_dataset.py` → everything in §5. **Announce in the group when it's pushed.**
2. `train_logreg.py`, `train_gnb.py` via the harness.
3. `pipeline.predict_trip(trip, model, hmm_cfg)`:
   - Steps: build features with `with_route=False` → `model.predict_proba` → `hmm.viterbi` → `point_arcs`.
   - Then derive `route_arcs = compress(point_arcs)` and `route_coords = net.arc_lonlat(route_arcs)`.
   - Return `{"point_arcs", "route_arcs", "route_coords", "gps", "info", "timings"}`.
   - It **must not read `trip.route`.**
4. `scripts/export_artifacts.py`: `artifacts/feature_list.json`, `metadata.json` (date, seed, library versions, CV scores).
5. README finalisation and merges.

---

## 10. Timeline (T = when you start; freeze 30 min before lunch)

| Time | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| T+0:15 | Skeleton + core modules ✅ | Setup, start EDA | Setup, start harness | Setup, Viterbi on toy data |
| T+1:00 | **features + parquet pushed** | EDA sections 1–6 | **metrics + harness pushed** | HMM on a real trip (B1, `emission="dist"`) |
| T+1:45 | logreg, gnb | knn, dtree, viz.py, API stub | B0, lgbm, rf | mlp, svm |
| T+2:30 | pipeline.py | UI + API wired to `predict_trip` | compare_models, start Optuna | eval_hmm on all models |
| T+3:00 | export, README | maps, EDA finished | Optuna finished, tuned model → artifacts | tune_decoder → hmm_config.json |
| T+3:30 | **Freeze**: everyone merges, then a fresh clone → README steps → API + UI on all 10 student-test trips | | | |

**If you're behind,** keep the path *features → 1 model → HMM → API → UI* working first. Extra configurations come last.

---

## 11. Hard rules (breaking them costs marks; LLMs must follow them too)
1. **Never read, list or guess evaluator data.** Only `api/main.py` uses `config.EVAL_DIR`.
2. **All choices are made on train (CV) + student-test only:** features, models, hyperparameters, r/K, decoder parameters.
3. **`predict_trip` and anything it calls must never read `trip.route`.** Load trips with `with_route=False` on the prediction path. The UI never receives the reference route.
4. **No hardcoded predictions, trip-specific `if` statements, or copied expected answers.**
5. **No lon/lat, trip_id or point_idx as model features.** No fitting a scaler on student-test.
6. **Seed 42.** Never fabricate metrics or plots. Every number in the report must come from a file in `results/`. No fake loss curves for models that don't produce them.
7. **No secrets in the repo.** Don't commit `data/` or `results/features/`.
8. **Every member must be able to explain their 2 models** (why they suit the problem, hyperparameters, loss, results, over- or underfitting) **and their code in the viva.** Read and understand AI-generated code before committing it.

---

## 12. Paste-ready prompt for your LLM

> I'm member **M_** of a 4-person team on the PathSnap map-matching hackathon. The repo is https://github.com/Chunnu-Munnu/ML_HACKATHON and the full spec is `docs/TEAM_TASKS.md` (pasted below). Implement **only my tasks from §2 and my sections (§6 / §7 / §8)**.
> - Use exactly the file paths, function signatures, column names and output formats in the doc.
> - Import existing code from `pathsnap.*` (§4) instead of rewriting it.
> - If something I need isn't defined in the doc, stop and tell me instead of inventing it.
> - Follow every rule in §11.
> - Python 3.11, seed 42, numpy/pandas/scikit-learn/lightgbm/optuna/fastapi/streamlit/folium only (see `requirements.txt`).
>
> [paste docs/TEAM_TASKS.md here]

## 13. Pre-lunch checklist (M1 ticks these at freeze)
- [ ] All 4 branches merged into `main`; repo public
- [ ] `results/models/`: 8 models + B0, each with config/metrics/cv_folds/confusion matrix; `model_comparison.csv`; comparison plots; MLP loss curve; `selection.md`
- [ ] `results/optuna/`: trials.csv, best_params.json, history/importance plots, tuned_vs_untuned.csv
- [ ] `results/hmm/hmm_comparison.csv` + decoder_search.csv; `artifacts/{model.joblib, hmm_config.json, feature_list.json, metadata.json}`
- [ ] EDA notebook + `results/eda/*` + Folium maps in `results/maps/`
- [ ] API: `/trips` lists 10 trips, `/predict` works on all 10, `logs/api.log` written; UI displays map + metrics through the API
- [ ] README reproduces everything from a fresh clone
