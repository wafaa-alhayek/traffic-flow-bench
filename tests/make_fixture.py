"""Write a tiny fake release in the Kaggle layout, for smoke-testing the pipeline without the real data.

The numbers are synthetic toy traffic and say nothing about leaderboard scores.

    python tests/make_fixture.py /tmp/tfb_fixture
    python src/pipeline.py --release-root /tmp/tfb_fixture/kaggle_public --output-dir /tmp/tfb_fixture/out
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PANELS = ["D12_I5_N", "D12_I405_N"]
L = 14
SPLITS = {"train": ("2030-06-01", 28), "validation": ("2031-03-01", 7), "private": ("2031-04-01", 7)}
RATE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}


def traffic(rng, days):
    t = np.arange(288)[None, :, None] / 12.0
    x = np.arange(L)[None, None, :]
    vf = 105.0
    v = np.full((days, 288, L), vf) + rng.normal(0, 2, (days, 288, L))
    for d in range(days):
        if rng.random() < 0.8:
            start, b, depth = rng.uniform(6.5, 8.5), rng.integers(6, L), rng.uniform(25, 55)
            front = b - (t[0] - start) * 3.0
            jam = ((t[0] >= start) & (t[0] <= start + 2.5) & (x[0] <= b) & (x[0] >= front))
            v[d][jam] -= depth
    v = np.clip(v, 8, 120)
    k = np.where(v > 63, 2000 * 4 / vf * (1 - (v - 63) / 120), 4 * 120 * (1 - v / 110))
    q = np.clip(v * np.clip(k, 5, None) + rng.normal(0, 80, v.shape), 50, None)
    return v.astype(np.float32), q.astype(np.float32)


def main(root: Path):
    rng = np.random.default_rng(0)
    rel = root / "kaggle_public"
    (rel / "config").mkdir(parents=True, exist_ok=True)
    (rel / "config" / "corridors.json").write_text(json.dumps(
        {"panels": [{"corridor_id": p, "family_id": p.rsplit("_", 1)[0]} for p in PANELS]}))
    key_rows = []
    for panel in PANELS:
        pdir = rel / "corridors" / panel
        net = pdir / "network"
        net.mkdir(parents=True, exist_ok=True)
        links = [f"{panel}_L{i:02d}" for i in range(L)]
        pd.DataFrame({"link_id": links, "length_km": 0.8, "lanes": 4, "free_speed_kmh": 105.0,
                      "capacity_vph": 8000.0}).to_csv(net / "links.csv", index=False)
        pd.DataFrame({"link_id": links, "station_id": [f"S{i}" for i in range(L)], "free_speed_kmh": 105.0,
                      "capacity_vph": 8000.0, "lanes": 4, "k_jam": 480.0}).to_csv(net / "fd_parameters.csv", index=False)
        pd.DataFrame({"link_id": links, "order_index": range(L),
                      "incoming_link_ids": [""] + links[:-1], "outgoing_link_ids": links[1:] + [""],
                      "on_ramp_link_ids": "", "off_ramp_link_ids": ""}).to_csv(net / "lwr_mainline_topology.csv", index=False)
        paths = pd.DataFrame({"path_id": [f"P{i}" for i in range(20)],
                              "origin_zone": rng.integers(0, 5, 20), "destination_zone": rng.integers(5, 10, 20)})
        paths.to_csv(net / "path_set.csv", index=False)
        inc = [(f"P{i}", links[j]) for i in range(20) for j in range(rng.integers(0, 6), rng.integers(8, L))]
        pd.DataFrame(inc, columns=["path_id", "link_id"]).to_csv(net / "path_link_incidence.csv", index=False)
        A = pd.crosstab(pd.DataFrame(inc, columns=["p", "l"]).l, pd.DataFrame(inc, columns=["p", "l"]).p)
        for split, (start, days) in SPLITS.items():
            dates = pd.date_range(start, periods=days, freq="D").strftime("%Y-%m-%d").tolist()
            v, q = traffic(rng, days)
            pct = np.where(rng.random(v.shape) < 0.85, 100, rng.integers(0, 100, v.shape))
            regimes = [list(RATE)[i % 3] for i in range(days)]
            mask = np.zeros(v.shape, bool)
            for d, r in enumerate(regimes):
                mask[d] = rng.random((288, L)) < RATE[r]
            elig = pct >= 75
            target = mask & elig
            # Task 2 windows from the truth
            windows = []
            vcut = 63.0
            for d in range(days):
                qd = v[d] <= vcut
                for s0 in range(84, 120, 3):
                    if not qd[s0 + 1:s0 + 7].any():
                        continue
                    cond = "queue_onset" if not qd[s0 - 12:s0 + 1].any() else "queue_ongoing"
                    if sum(w[3] == cond for w in windows) < 5 and all(w[0] != d for w in windows if w[3] == cond):
                        windows.append((d, s0, f"{panel}_{split}_w{len(windows)}", cond))
            vm, qm = v.copy(), q.copy()
            vm[target], qm[target] = np.nan, np.nan
            for d, s0, _, _ in windows:
                vm[d, s0 + 1:s0 + 19], qm[d, s0 + 1:s0 + 19] = np.nan, np.nan
                target[d, s0 + 1:s0 + 19] = False
            ts = np.array([f"{dt}T{s // 12:02d}:{(s % 12) * 5:02d}:00Z" for dt in dates for s in range(288)])
            for layer, vv, qq in (("mainline_states", v, q), ("mainline_states_masked", vm, qm)):
                if layer == "mainline_states" and split != "train":
                    continue
                for d, dt in enumerate(dates):
                    sub = "year_month=" + dt[:7] if layer == "mainline_states" else f"mask_regime={regimes[d]}"
                    out = pdir / split / layer / sub
                    out.mkdir(parents=True, exist_ok=True)
                    frame = pd.DataFrame({
                        "date": dt, "timestamp": np.repeat(ts[d * 288:(d + 1) * 288], L),
                        "station_id": np.tile([f"S{i}" for i in range(L)], 288), "link_id": np.tile(links, 288),
                        "milepost": np.tile(np.arange(L, dtype=float), 288),
                        "speed_kmh": vv[d].reshape(-1), "flow_vph": qq[d].reshape(-1), "occupancy": 5.0,
                        "pct_observed": pct[d].reshape(-1), "is_score_eligible": elig[d].reshape(-1),
                        "mask_regime": regimes[d]})
                    frame.to_parquet(out / f"{dt}.parquet", index=False)
            (pdir / split / "ramp_states").mkdir(parents=True, exist_ok=True)
            d_, s_, l_ = np.nonzero(target)
            t1 = pd.DataFrame({"panel": panel, "timestamp": ts[d_ * 288 + s_], "station_id": [f"S{i}" for i in l_],
                               "link_id": np.asarray(links)[l_], "mask_regime": np.asarray(regimes)[d_],
                               "speed_kmh": 0.0, "flow_vph": 0.0})
            (rel / "task1" / panel / split).mkdir(parents=True, exist_ok=True)
            t1.to_csv(rel / "task1" / panel / split / "sample_submission_state.csv", index=False)
            if split != "train":
                key_rows.append(t1.assign(task="state")[["task", "panel", "timestamp", "station_id", "link_id", "mask_regime"]])
            if panel != "D12_I405_N":
                t2 = rel / "task2" / panel / split
                t2.mkdir(parents=True, exist_ok=True)
                wi = pd.DataFrame([{"window_id": w, "panel": panel, "family_id": panel.rsplit("_", 1)[0], "split": split,
                                    "condition": c, "date": dates[d], "forecast_origin": ts[d * 288 + s0]}
                                   for d, s0, w, c in windows])
                wi.to_csv(t2 / "window_index.csv", index=False)
                hist, tpl = [], []
                for d, s0, w, c in windows:
                    for s in range(s0 - 12, s0):
                        hist.append(pd.DataFrame({"window_id": w, "timestamp": ts[d * 288 + s], "link_id": links,
                                                  "speed_kmh": v[d, s], "flow_vph": q[d, s], "is_score_eligible": elig[d, s]}))
                    for s in range(s0 + 1, s0 + 7):
                        tpl.append(pd.DataFrame({"window_id": w, "timestamp": ts[d * 288 + s], "link_id": links, "queue_pred": 0}))
                pd.concat(hist).to_parquet(t2 / "window_history.parquet", index=False)
                tpl = pd.concat(tpl)
                tpl.to_csv(t2 / "sample_submission_queue.csv", index=False)
                if split != "train":
                    key_rows.append(tpl.assign(task="queue")[["task", "window_id", "timestamp", "link_id"]])
            t4 = rel / "task4" / panel / split
            t4.mkdir(parents=True, exist_ok=True)
            f_true = rng.uniform(100, 600, 20)
            counts = A.to_numpy() @ f_true[[list(A.columns).index(f"P{i}") for i in range(20) if f"P{i}" in A.columns]] \
                if A.shape[1] == 20 else A.to_numpy() @ rng.uniform(100, 600, A.shape[1])
            pd.DataFrame({"link_id": A.index, "count": counts}).to_csv(t4 / "synthetic_link_counts.csv", index=False)
            tok = f"{split.upper()}-PM"
            pd.DataFrame({"path_id": paths.path_id, "departure_time": tok,
                          "path_flow": f_true * rng.uniform(0.6, 1.4, 20)}).to_csv(t4 / "synthetic_weak_prior.csv", index=False)
            p4 = paths.assign(panel=panel, departure_time=tok, path_flow=0.0)
            p4.to_csv(t4 / "sample_submission_path_flow.csv", index=False)
            if split != "train":
                key_rows.append(p4.assign(task="odme")[["task", "panel", "departure_time", "path_id"]])
    key = pd.concat(key_rows, ignore_index=True)
    key.insert(0, "submission_id", np.arange(len(key)))
    key.to_csv(rel / "submission_key.csv", index=False)
    print(f"fixture written to {rel} ({len(key):,} submission rows)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tfb_fixture"))
