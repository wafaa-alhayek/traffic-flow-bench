"""Shared helpers: release paths, dense per-panel caches, network parameters, local scoring.

A cache holds one panel and one split as arrays shaped [day, slot(288), link]:
  speed, flow, occ   values as released in the masked layer (NaN where blanked)
  elig               is_score_eligible
  target             Task 1 target cells, read from the published template
  regime             per-day mask regime index (0, 1, 2 = R1, R2, R3)
  true_speed/flow    train only: the unmasked layer, which is the Task 1 truth
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RELEASE = ROOT / "data" / "kaggle_public"
CACHE = Path(os.environ.get("TFB_CACHE", ROOT / "cache"))
OUTPUTS = ROOT / "outputs"
SPLITS = ("train", "validation", "private")
REGIMES = ("R1", "R2", "R3")
SLOTS = 288

# Task 1 scoring constants, from official/src/task1/baseline_task1_historical_mean.py
SPEED_NORM, FLOW_NORM, SPEED_W, FLOW_W = 25.0, 600.0, 0.54, 0.46


def panels(release: Path) -> list[str]:
    manifest = release / "config" / "corridors.json"
    if not manifest.exists():
        manifest = ROOT / "official" / "config" / "corridors.json"
    found = {p.name for p in (release / "corridors").iterdir() if p.is_dir()}
    listed = [p["corridor_id"] for p in json.loads(manifest.read_text())["panels"]]
    return [p for p in listed if p in found] + sorted(found - set(listed))


def slot_of(ts: pd.Series) -> np.ndarray:
    """5-minute slot of a canonical text timestamp, YYYY-MM-DDTHH:MM:SSZ."""
    s = ts.astype(str)
    return (s.str.slice(11, 13).astype(int) * 12 + s.str.slice(14, 16).astype(int) // 5).to_numpy()


def read_parquet_dir(root: Path, columns: list[str]) -> pd.DataFrame:
    files = sorted(root.glob("**/*.parquet"))
    if not files:
        return pd.DataFrame(columns=columns)
    parts = []
    for f in files:
        have = set(pq.read_schema(f).names)
        parts.append(pq.read_table(f, columns=[c for c in columns if c in have]).to_pandas())
    return pd.concat(parts, ignore_index=True)


def link_order(panel_dir: Path, frame: pd.DataFrame) -> list[str]:
    """Mainline links in corridor order: topology order_index, else milepost, else id."""
    seen = set(frame.link_id.astype(str))
    topo = panel_dir / "network" / "lwr_mainline_topology.csv"
    if topo.exists():
        t = pd.read_csv(topo, dtype=str)
        if "order_index" in t.columns:
            t["order_index"] = pd.to_numeric(t.order_index, errors="coerce")
            order = [x for x in t.sort_values("order_index").link_id.drop_duplicates() if x in seen]
            return order + sorted(seen - set(order))
    if "milepost" in frame.columns:
        m = frame[["link_id", "milepost"]].drop_duplicates("link_id").copy()
        m["link_id"] = m.link_id.astype(str)
        m["milepost"] = pd.to_numeric(m.milepost, errors="coerce")
        return m.sort_values(["milepost", "link_id"]).link_id.tolist()
    return sorted(seen)


MAIN_COLS = ["date", "timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "occupancy",
             "pct_observed", "is_score_eligible", "mask_regime", "milepost"]


def build_cache(release: Path, panel: str, split: str, links: list[str] | None = None) -> dict:
    """Dense arrays for one panel/split, written to cache/<panel>_<split>.npz."""
    pdir = release / "corridors" / panel
    m = read_parquet_dir(pdir / split / "mainline_states_masked", MAIN_COLS)
    if m.empty:
        raise FileNotFoundError(f"{panel}/{split}: no mainline_states_masked partitions under {pdir / split}")
    m["link_id"] = m.link_id.astype(str)
    m["station_id"] = m.station_id.astype(str)
    if "date" not in m.columns:
        m["date"] = m.timestamp.astype(str).str.slice(0, 10)
    if links is None:
        links = link_order(pdir, m)
    dates = sorted(m.date.astype(str).unique())
    lmap = {x: i for i, x in enumerate(links)}
    dmap = {x: i for i, x in enumerate(dates)}
    m = m[m.link_id.isin(lmap)]
    d, s, l = m.date.astype(str).map(dmap).to_numpy(), slot_of(m.timestamp), m.link_id.map(lmap).to_numpy()

    def dense(values, fill=np.nan, dtype=np.float32):
        a = np.full((len(dates), SLOTS, len(links)), fill, dtype=dtype)
        a[d, s, l] = values
        return a

    out = {
        "speed": dense(pd.to_numeric(m.speed_kmh, errors="coerce").to_numpy(np.float32)),
        "flow": dense(pd.to_numeric(m.flow_vph, errors="coerce").to_numpy(np.float32)),
        "elig": dense(m.is_score_eligible.fillna(False).astype(bool).to_numpy(), False, bool),
    }
    if "occupancy" in m.columns:
        out["occ"] = dense(pd.to_numeric(m.occupancy, errors="coerce").to_numpy(np.float32))
    reg = m.groupby(m.date.astype(str)).mask_regime.first().reindex(dates).astype(str)
    out["regime"] = np.array([REGIMES.index(r) if r in REGIMES else 0 for r in reg], dtype=np.int8)
    st = m.drop_duplicates("link_id").set_index("link_id").station_id.reindex(links).fillna("").astype(str)

    tpl_path = release / "task1" / panel / split / "sample_submission_state.csv"
    target = np.zeros(out["speed"].shape, dtype=bool)
    if tpl_path.exists():
        tpl = pd.read_csv(tpl_path, usecols=["timestamp", "link_id"], dtype=str)
        tpl = tpl[tpl.link_id.isin(lmap) & tpl.timestamp.str.slice(0, 10).isin(dmap)]
        target[tpl.timestamp.str.slice(0, 10).map(dmap).to_numpy(), slot_of(tpl.timestamp),
               tpl.link_id.map(lmap).to_numpy()] = True
    out["target"] = target

    if split == "train":
        f = read_parquet_dir(pdir / split / "mainline_states", ["date", "timestamp", "link_id", "speed_kmh", "flow_vph"])
        f["link_id"] = f.link_id.astype(str)
        if "date" not in f.columns:
            f["date"] = f.timestamp.astype(str).str.slice(0, 10)
        f = f[f.link_id.isin(lmap) & f.date.astype(str).isin(dmap)]
        fd_, fs, fl = f.date.astype(str).map(dmap).to_numpy(), slot_of(f.timestamp), f.link_id.map(lmap).to_numpy()
        for col, name in (("speed_kmh", "true_speed"), ("flow_vph", "true_flow")):
            a = np.full(out["speed"].shape, np.nan, dtype=np.float32)
            a[fd_, fs, fl] = pd.to_numeric(f[col], errors="coerce").to_numpy(np.float32)
            out[name] = a

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"{panel}_{split}.npz", dates=np.array(dates, dtype=str), links=np.array(links, dtype=str),
             stations=st.to_numpy().astype(str), **out)
    return load_cache(panel, split)


def load_cache(panel: str, split: str) -> dict:
    z = np.load(CACHE / f"{panel}_{split}.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def cached(release: Path, panel: str, split: str, rebuild: bool = False) -> dict:
    path = CACHE / f"{panel}_{split}.npz"
    if path.exists() and not rebuild:
        return load_cache(panel, split)
    links = None
    if split != "train":
        links = list(cached(release, panel, "train", rebuild=False)["links"])
    return build_cache(release, panel, split, links)


def network(release: Path, panel: str, links) -> pd.DataFrame:
    """Per-link length, lanes, free speed, capacity, jam density and queue cutoff, in cache link order.

    fd_parameters.csv wins over links.csv where both carry a value, as in the official Task 3 scorer.
    """
    ndir = release / "corridors" / panel / "network"
    links = [str(x) for x in links]
    net = pd.DataFrame(index=pd.Index(links, name="link_id"))
    if (ndir / "links.csv").exists():
        lk = pd.read_csv(ndir / "links.csv", dtype={"link_id": str}).drop_duplicates("link_id").set_index("link_id")
        for c in ("length_km", "lanes", "free_speed_kmh", "capacity_vph"):
            if c in lk.columns:
                net[c] = pd.to_numeric(lk[c], errors="coerce").reindex(links)
    if (ndir / "fd_parameters.csv").exists():
        fd = pd.read_csv(ndir / "fd_parameters.csv", dtype={"link_id": str, "station_id": str})
        fd = fd.groupby("link_id").mean(numeric_only=True).reindex(links)
        for c in ("length_km", "lanes", "free_speed_kmh", "capacity_vph", "k_jam", "v_cut"):
            if c in fd.columns:
                net[c] = fd[c].combine_first(net[c]) if c in net.columns else fd[c]
    defaults = {"length_km": 0.5, "lanes": 4.0, "free_speed_kmh": 105.0}
    for c, v in defaults.items():
        net[c] = pd.to_numeric(net.get(c), errors="coerce").fillna(v) if c in net.columns else v
    net.loc[(net.lanes < 1) | (net.lanes > 16), "lanes"] = 4.0
    if "capacity_vph" not in net.columns:
        net["capacity_vph"] = np.nan
    net["capacity_vph"] = net.capacity_vph.fillna(2000.0 * net.lanes)
    if "k_jam" not in net.columns:
        net["k_jam"] = np.nan
    net["k_jam"] = net.k_jam.fillna(110.0 * net.lanes)
    if "v_cut" not in net.columns:
        net["v_cut"] = np.nan
    # The Task 2 queue label: speed at or below 0.60 x free-flow speed.
    net["v_cut"] = net.v_cut.fillna(0.60 * net.free_speed_kmh)
    return net


def weekday_of(dates) -> np.ndarray:
    return pd.to_datetime(pd.Series(list(dates))).dt.dayofweek.to_numpy()


def score_state(pv: np.ndarray, pq_: np.ndarray, z: dict, lanes: np.ndarray, days=None) -> dict:
    """Task 1 score on a train cache, same formula as official/src/task1/score_task1.py."""
    days = np.arange(len(z["dates"])) if days is None else np.asarray(days)
    tv, tq = z["true_speed"], z["true_flow"]
    out, parts = {}, []
    for r, name in enumerate(REGIMES):
        dd = days[z["regime"][days] == r]
        if len(dd) == 0:
            continue
        m = z["target"][dd] & z["elig"][dd] & np.isfinite(tv[dd]) & np.isfinite(tq[dd])
        ev = np.nan_to_num(pv[dd], nan=0.0)[m] - tv[dd][m]
        eq = ((np.nan_to_num(pq_[dd], nan=0.0) - tq[dd]) / lanes)[m]
        rv, rq = float(np.sqrt(np.mean(ev ** 2))), float(np.sqrt(np.mean(eq ** 2)))
        s = SPEED_W * max(0.0, 1 - rv / SPEED_NORM) + FLOW_W * max(0.0, 1 - rq / FLOW_NORM)
        out[name] = {"rmse_speed_kmh": rv, "rmse_flow_vph_per_lane": rq, "S_state": s, "n": int(m.sum())}
        parts.append(s)
    out["S_state"] = float(np.mean(parts)) if parts else 0.0
    return out
