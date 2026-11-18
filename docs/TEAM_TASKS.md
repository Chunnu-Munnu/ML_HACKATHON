# PathSnap — Team Task Split (Deliverable 1, before lunch)

**Scope before lunch = Deliverable 1:** the public GitHub repo with *all* code and results. The PDF report comes later, but save every CSV, plot and log to `results/` now, because the report will be built from them.

**Model rule: 8 models in total, 2 per member, each from a different algorithm family.** Every model runs through the **same shared experiment harness** (`src/pathsnap/experiment.py`), so all of them use the same data, folds, metrics and output format.

| Member | Main work | Their 2 models |
|---|---|---|
| **M1 (Amogh)** | Data core, labels, **features**, prediction pipeline, README, repo | **Logistic Regression** (linear) · **Gaussian Naive Bayes** (probabilistic) |
| **M2** | EDA, Folium maps, FastAPI, Streamlit UI | **KNN, distance-weighted** (instance-based) · **Decision Tree** (tree) |
| **M3** | Experiment harness, metrics, comparison plots, **Bayesian HPO** | **LightGBM** (boosting) · **Random Forest** (bagging ensemble) |
| **M4** | HMM / Viterbi, route metrics, decoder tuning | **MLP** (neural net) · **SVM**, linear + calibrated, or RBF on a subsample (kernel) |

Baseline **B0 = nearest arc** (no training; M3 adds it to the comparison). Optional extras if anyone has time: Extra Trees, XGBoost, AdaBoost, SGD.
Final model = the best by **student-test Top-1 after HMM**, with log loss as the tiebreaker. M3 then runs Optuna on it.

---

## 0. Verified data facts (from scanning all 90 trips)

- **Layout:** `data/{train,student_test}/<id>/<id>.{track,nodes,arcs,route}`. 80 train + 10 student-test trips. **No evaluator data is given to us**; the missing IDs 15, 17, 23, 30, 49, 59, 67, 73, 87, 94 are presumably the evaluator's.
- **Every trip has its own road network:** 299–155k nodes and 649–307k arcs, so a spatial index (shapely STRtree) is needed.
- **Trips are global** (lon −100…137, lat −35…62), so each trip gets its own UTM projection.
- **Track files:** CRLF line endings, tab-separated. Timestamps are increasing but don't always start at 1. Trips 02, 71 and 97 have non-integer timestamps.
- **Gaps:** up to 153,064 s (trip 90) and 52,810 s (trip 78). The median trip has 99% of steps at 1 s.
- **About 91% of arcs have a reverse twin** with identical geometry, so heading features are required. There are also duplicate arcs (up to 322 in one trip), self-loops and zero-length arcs.
- **Routes:** all arc IDs are valid. 11 routes have breaks (vᵢ ≠ uᵢ₊₁), including student-test trip 76.
- **GPS-to-route distance:** median ≈ 4.5 m, p99 ≈ 21 m. Arc length median ≈ 25 m.
- **Suspect labels:** trips 54 and 33 are 120–140 m from their route at the median. Trip 02 ends 549 m from the end of its route.

---

## 1. Setup (first 15 min)

```bash
git clone https://github.com/Chunnu-Munnu/ML_HACKATHON.git
cd ML_HACKATHON
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

- Copy the data into `data/train/` and `data/student_test/` (gitignored).
- Then run `python scripts/build_dataset.py` (M1), which writes `results/features/*.parquet` (gitignored, so everyone regenerates it locally).
- **Git:**
  - Work on your own branch: `m1-core`, `m2-eda-api-ui`, `m3-models-hpo`, `m4-hmm`.
  - Only edit the files you own (listed below).
  - Pull `main` often and merge at every milestone.
  - Small commits, because the repo history is checked.
  - Need a new package? Ask M1 to add it to `requirements.txt`.

---

## 2. Who does what

### M1 — Data core, features, pipeline, repo (Amogh)
Files: `src/pathsnap/{config,io,geo,network,candidates,labeling,features,pipeline}.py`, `scripts/build_dataset.py`, `scripts/export_artifacts.py`, `scripts/models/train_logreg.py`, `scripts/models/train_gnb.py`, `README.md`, `requirements.txt`, `.gitignore`

1. Repo skeleton and requirements (**done**).
2. Core modules:
   - `io.py`: loading and cleaning.
   - `geo.py`: per-trip UTM projection and point-to-segment geometry.
   - `network.py`: STRtree, CSR graph, canonical and reverse arcs, `route_dist`.
   - `candidates.py`: radius + top-K search.
3. `labeling.py`: monotone DP alignment of route arcs to GPS points, giving the label and alignment distance for each point.
4. `features.py`: the **full feature set in §3**. `build_dataset.py` writes the parquets plus a candidate-miss-rate sweep.
5. Models: **Logistic Regression** and **Gaussian NB**, run through the harness.
6. `pipeline.py`: `predict_trip(trip, model, hmm_cfg)`, which never reads the route. Then `export_artifacts.py`, README, final merge.

### M2 — EDA, maps, API, UI
Files: `notebooks/01_eda.ipynb`, `src/pathsnap/viz.py`, `scripts/make_maps.py`, `api/main.py`, `ui/app.py`, `scripts/models/train_knn.py`, `scripts/models/train_dtree.py`

1. **EDA notebook** on the train partition (state the partition in every section). It must cover:
   - Inventory: trips, files, sizes, observation counts.
   - Data quality: dtypes, NaNs, duplicate points.
   - Time: the Δt histogram and a table of the largest gaps.
   - Geography: a **world map of trip locations**.
   - Network: size, connected components (networkx), degree distribution, reverse-twin share, duplicate arcs and self-loops.
   - Routes: lengths and breaks.
   - Distances: GPS-to-route distance (M1's `align_dist` column).
   - Speed outliers and the suspect trips.
   - Figures go to `results/eda/`. Also write `results/eda/data_quality.md`.
2. **`viz.trip_map(track, net=None, route=None, pred=None)`** (Folium)
   - Layers: raw GPS (blue), nearby network only (grey), reference route (green), prediction (red), start/end markers, LayerControl, legend.
   - `make_maps.py` saves HTML maps for at least 2 student-test trips to `results/maps/`.
3. **Models:** **KNN (distance-weighted)** and **Decision Tree** through the harness. For KNN, subsample training points if it's slow and record that you did. Try 2–3 values of k and max_depth, and log every run.
4. **`api/main.py`** (FastAPI)
   - **Endpoints:** `GET /health`, `GET /trips` (folders in `EVAL_DIR`, which defaults to `data/student_test`), and `POST /predict {"trip_id"}`.
   - **`/predict`** loads the trip **without its route** and calls `pipeline.predict_trip`. It then compares the result against the route **on the server only** and returns GPS coordinates, the predicted route coordinates, arc IDs, metrics (Top-1, continuity) and timings.
   - **Never send the reference route to the UI.**
   - Log every request to the console and `logs/api.log`.
   - Use CORS and run with `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
   - Build it against a dummy predictor first, then wire in the real pipeline.
5. **`ui/app.py`** (Streamlit + streamlit-folium)
   - The API URL is set in the sidebar, defaulting to the `PATHSNAP_API` env var.
   - A trip dropdown filled from `/trips` and a Reconstruct button.
   - Show the map and metric cards. Every value comes from the API.

### M3 — Harness, metrics, comparison, Bayesian HPO
Files: `src/pathsnap/{metrics,experiment}.py`, `scripts/compare_models.py`, `scripts/tune_optuna.py`, `scripts/models/train_lgbm.py`, `scripts/models/train_rf.py`, `scripts/models/baseline_nearest.py`

1. **Push the harness by T+1:00. Everyone depends on it.**
   - `run_experiment(name, family, estimator, config, feature_cols=FEATURES, subsample_points=None, seed=42)`
   - **Steps:** load the parquets → `GroupKFold(5)` by `trip_id` on train → fit and score each fold → refit on all of train → score student-test → save everything.
   - **Metrics:** row-level accuracy, precision, recall, macro F1, log loss, confusion matrix; **point-level Top-1** (argmax candidate per point); **candidate miss rate**; training time; inference time per trip.
   - **Outputs:** `results/models/<name>/{metrics.json, cv_folds.csv, config.json, confusion_matrix.png, student_test_proba.parquet}`. Also append a row to `results/models/model_comparison.csv`, save `model.joblib` if under 50 MB, and save `loss_curve.png` *only* if the estimator has `loss_curve_`.
   - Put the scaler inside an sklearn `Pipeline`. Seed 42.
   - **`student_test_proba.parquet`** holds (trip_id, point_idx, arc_id, proba). M4 feeds it into the HMM to get route continuity for every model.
2. **Models:** **LightGBM** and **Random Forest**, plus the **B0 nearest-arc** baseline.
3. **`compare_models.py`:**
   - Builds the consolidated table (report §3.1 columns: family, key config, CV macro F1, Top-1, route continuity, student-test macro F1, log loss, inference time).
   - **Plots:** CV macro F1 with error bars, student-test Top-1 + macro F1, log loss, Top-1 before and after HMM.
   - Writes `results/models/selection.md` with the selection rule and the winner.
4. **`tune_optuna.py`:**
   - Optuna TPE on the selected model. Objective: mean GroupKFold CV Top-1 on **train only**.
   - Use a justified search space and 40–60 trials.
   - **Outputs:** `results/optuna/{study.db, trials.csv, best_params.json, history.png, param_importance.png, tuned_vs_untuned.csv}` and `artifacts/model.joblib`.

### M4 — HMM / Viterbi, route metrics, decoder tuning
Files: `src/pathsnap/{hmm,route_metrics}.py`, `scripts/eval_hmm.py`, `scripts/tune_decoder.py`, `scripts/models/train_mlp.py`, `scripts/models/train_svm.py`

1. **`hmm.viterbi(cands_with_proba, net, x, y, t, cfg)`**: one arc per point. Start on synthetic data right away; use M1's `RoadNetwork` once it's pushed.
   - **States:** the candidates of each point.
   - **Emission:** `λ·log p_clf`. The B1 baseline uses a Gaussian on distance instead.
   - **Transition:** `−|d_route − d_gps|/β` with `net.route_dist`. It's −∞ when the arc is unreachable or the implied speed exceeds v_max.
   - **Edge cases:**
     - A point with no candidates takes its neighbour's arc and gets flagged.
     - If every transition is −∞, it's a **break**: restart the chain and count it.
     - A large Δt splits the chain or scales β.
2. **`route_metrics.py`:** continuity (the share of consecutive arcs that are the same or connected), route arc precision/recall, break count.
3. **Models:** **MLP**, which saves its loss curve (the only model with training curves), and **SVM** (LinearSVC + `CalibratedClassifierCV`, or RBF on a point subsample; record which).
4. **`eval_hmm.py`:**
   - Compare **B0, B1 (nearest + HMM), classifier only and classifier + HMM** on every model's `student_test_proba.parquet`.
   - Measure point Top-1, continuity and route P/R, per trip and overall.
   - **Outputs:** `results/hmm/hmm_comparison.csv`, a plot, and maps where the HMM helped and where it hurt (using `viz.trip_map`).
5. **`tune_decoder.py`:** λ, β, v_max and the gap threshold, on **train + student-test only** → `artifacts/hmm_config.json`, `results/hmm/decoder_search.csv`.

---

## 3. Feature set (M1 builds, everyone uses `features.FEATURES`)

One row per (GPS point, candidate arc). Label = 1 if the candidate is the point's aligned route arc. Duplicate arcs are collapsed to a canonical ID.

| Group | Features |
|---|---|
| **Geometry** | `dist`, `dist_rank`, `dist_minus_min`, `dist_ratio_min`, `dist_z` (within the point's candidates), `t_proj`, `off_end_m` (how far the projection falls past the segment end), `dist_to_endpoint`, `arc_len`, `log_arc_len` |
| **Direction** | `head_diff_s` (smoothed GPS heading over i±2 vs arc bearing, 0–180), `head_diff_l` (i±5), `head_cos`, `head_rank`, `has_reverse`, `beats_reverse` (heading fits better than its reverse twin), `turn_rate` |
| **Kinematics** | `speed`, `speed_smooth`, `accel`, `dt_prev`, `dt_next`, `is_gap` (Δt > 5 s), `is_jump`, `is_stationary` |
| **Temporal context** | `ctx_dist_3`, `ctx_dist_7` (mean distance of points i±3 / i±7 to this arc), `ctx_nearest_frac_3` (share of the window whose nearest arc is this one), `prev_dist`, `next_dist`, `in_prev_cands`, `in_next_cands`, `conn_prev` (connected to one of the previous point's candidates), `conn_prev_nearest` |
| **Network / candidate set** | `out_deg_v`, `in_deg_u`, `n_cands`, `n_cands_10m`, `score_rank` (rank by distance + heading combined) |

There is deliberately **no raw lon/lat and no trip ID**, so models can't memorise locations.

---

## 4. Interface contract (don't change without telling everyone)

```python
# io.py
load_trip(trip_dir, with_route=True) -> Trip(trip_id, track[lon,lat,t,...], nodes[lon,lat], arcs[u,v], route|None)
# network.py
RoadNetwork(trip)  # .proj .x/.y(track) .ux .uy .vx .vy .length .bearing .canon .rev_of .graph .route_dist(a,ta,b,tb,cutoff)
# candidates.py
get_candidates(net, r, k) -> DataFrame[point_idx, arc_id, dist, t_proj, t_raw, qx, qy]
# features.py
FEATURES: list[str];  build_trip_features(trip, net=None, r, k) -> DataFrame[trip_id, point_idx, arc_id, *FEATURES, (label, align_dist)]
# experiment.py (M3)
run_experiment(name, family, estimator, config, ...) -> dict
# hmm.py (M4)
viterbi(cands_with_proba, net, cfg) -> (arc per point ndarray, info dict)
# pipeline.py (M1)
predict_trip(trip, model, hmm_cfg) -> dict(point_arcs, route_arcs, route_coords, timings)
```

---

## 5. Timeline (T = start; freeze 30 min before lunch)

| Time | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| T+0:15 | Skeleton pushed ✅ | env, data, start EDA | env, start harness | env, start Viterbi (synthetic) |
| T+1:00 | io/geo/network/candidates | EDA | **harness + metrics pushed** | HMM on the real network |
| T+1:30 | **features parquet** | viz.py, API stub | LGBM, RF, B0 | MLP, SVM |
| T+2:15 | LogReg, GNB, pipeline.py | KNN, DTree, UI | compare_models | eval_hmm on every model |
| T+3:00 | export, README | API wired to the pipeline, maps | **Optuna** + tuned model | tune_decoder, hmm_config |
| T+3:30 | **Freeze**: fresh clone → README → API + UI on all 10 student-test trips → push | | | |

If you're behind, keep the end-to-end path working first: features → one model → HMM → API → UI.

---

## 6. Rules that cost marks if broken
- **Never use evaluator data while developing.** Only the API reads `EVAL_DIR`, and only on demo day.
- **Make every choice (features, models, hyperparameters, r/K, decoder parameters) using train + student-test only.**
- **`predict_trip` never sees the route, and the UI never receives it.**
- **No hardcoded outputs.** Seed 42 everywhere. Every reported number must come from a file in `results/`.
- **Everyone must be able to explain their 2 models and their code in the viva.** Understand anything an AI tool generated before you commit it.

## 7. Pre-lunch checklist
- [ ] `main` has all 4 branches merged; repo public
- [ ] 8 models + B0 in `results/models/model_comparison.csv`, plots, confusion matrices, MLP loss curve, selection.md
- [ ] `results/optuna/` complete; `artifacts/model.joblib` + `hmm_config.json` + `feature_list.json`
- [ ] `results/hmm/hmm_comparison.csv` (classifier only vs + HMM)
- [ ] EDA notebook + `results/eda/` + Folium maps in `results/maps/`
- [ ] API + UI work on all 10 student-test trips; `logs/api.log` written
- [ ] README reproduces everything from a fresh clone
