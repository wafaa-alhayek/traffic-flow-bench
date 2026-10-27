# TrafficFlowBench pipeline (2026 IEEE Big Data Cup)

Automated pipeline for the Kaggle competition
[2026 IEEE Big Data Traffic Flow Bench](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench).
One command builds all task outputs, merges them into the upload file, and writes a report.

The competition scores four tasks on ten synthetic freeway corridors at five-minute resolution:

```
S_total = 0.35 * S_state (Task 1) + 0.30 * S_queue (Task 2) + 0.15 * S_physics (Task 3) + 0.20 * S_ODME (Task 4)
```

| Task | What it asks | What this pipeline does | Local check |
|---|---|---|---|
| 1 State | Fill masked speed and flow cells | Tries 8 methods per corridor on train (time interpolation, profile plus decaying residual, each with or without a neighbour-link correction). Keeps the best for speed and for flow separately. | Exact Task 1 score on train (matches the official scorer) |
| 2 Queue | Which links are queued at T+5..T+30 | LightGBM per condition (onset, ongoing), trained on windows generated from every train day. Causal features only (at or before T). Onset windows are reweighted to match the organizers' window selection. The cell-selection rule is picked by day-grouped CV. | CV IoU on the organizers' own train windows, vs the persistence baseline |
| 3 Physics | Is the Task 1 answer physically consistent | Nothing separate: it is scored on the Task 1 rows | Leaderboard only |
| 4 ODME | Path flows from link counts | Non-negative least squares on each split's own released counts, anchored to its weak prior (lambda 0.05) | S_link per corridor |

## Run it

```bash
pip install -r requirements.txt
./run.sh                                   # data already unpacked under data/kaggle_public
./run.sh --download                        # fetch from Kaggle first (needs a Kaggle API token)
./run.sh --data-url gs://bucket/file.zip   # or fetch the zip from Cloud Storage or an https link
./run.sh --panel D12_I5_N                  # one corridor, for a quick run
```

For `--download`, accept the competition rules on Kaggle and put your API token in `~/.kaggle/kaggle.json`.
The data is about 13 GB unpacked. The dense caches go to `cache/`, or to the folder set in `TFB_CACHE`.

### Running inside Kaggle instead (no download)

`kaggle/traffic_flow_pipeline.ipynb` bundles the whole pipeline into one notebook. On Kaggle, go to Create, New Notebook, File, Import Notebook.
Add the competition as input data and click Run All. The upload file is written to `/kaggle/working/outputs/submission.csv`.
If you change the code, rebuild the notebook with `python kaggle/build_notebook.py`.

## Outputs (in `outputs/`)

- `submission.csv` and `submission.csv.zip`: the upload file, in the six columns the leaderboard reads
- `state_submission.csv`, `queue_submission.csv`, `odme_submission.csv`: one file per task
- `report.json`: every Task 1 candidate's train RMSE, the Task 2 CV IoU by rule against persistence, top features, and Task 4 S_link by lambda
- `task1_methods.png`, `task2_iou.png`: diagnostics

## Layout

- `src/pipeline.py`: the orchestrator (data, then Task 1, Task 2, Task 4, merge, report)
- `src/task1_state.py`, `src/task2_queue.py`, `src/task4_odme.py`: one module per task
- `src/common.py`: the release reader, dense `[day, slot, link]` caches, network parameters, and the Task 1 scorer
- `official/`: the organizers' MIT-licensed scorers, baselines and merge script, from
  [jacky850/trafficflowbench-public](https://github.com/jacky850/trafficflowbench-public)
- `tests/make_fixture.py`: writes a tiny fake release in the Kaggle layout, used to smoke-test the code end to end

## Things worth knowing

- Validation (the public leaderboard) and private (the final ranking) are different months, generated independently. Tune on train and treat the public leaderboard as a transfer check.
- Tasks 2 and 3 cannot be scored locally. The official `score_task1.py` finds targets by looking for blank cells. On a test fixture, that also counted the Task 2 horizon blanks as targets. The scorer here takes targets from the template, as the leaderboard does, and matches the official score exactly when both use the template cells.
- The final submission deadline is listed as 2026-11-06. Winners must reproduce their file from code, so the pipeline is seeded (`--seed`).
