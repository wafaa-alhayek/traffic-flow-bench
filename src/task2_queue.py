"""Task 2: predict which links are queued at T+5 .. T+30 from one hour of history.

A link is queued when its speed is at or below v_cut (0.60 x free-flow speed). The leaderboard scores
space-time IoU per window, averaged over windows, with queue_onset and queue_ongoing weighted equally.

The leaderboard scores every link in the horizon (window_index future_coverage is 1.0, labels come from the
complete simulated state), including links whose detector is missing. So the history is filled across links
before the features are built, with the observed share kept as a feature, and training labels on missing cells
are filled the same way from neighbouring links.

Training windows are generated from every train day, where the unmasked layer gives the labels, using the
same conditions the organizers select on. Onset windows are reweighted so that the step at which the queue
first appears matches the shipped train windows. Features only use timestamps at or before T.

One LightGBM model per condition is evaluated with day-grouped cross-validation, on generated windows and on
the organizers' own train windows, against the persistence baseline. The cell selection rule (a fixed
probability threshold, or the set that maximizes expected IoU) is chosen by the same cross-validation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import SLOTS, weekday_of

HIST = 12          # history steps T-60 .. T-5, plus the cell at T
H = 6              # horizon steps T+5 .. T+30
CONDITIONS = ("queue_onset", "queue_ongoing")
NBR = (-5, -3, -2, -1, 1, 2, 3, 5)
FEATURES = (
    [f"r_lag{k}" for k in range(HIST + 1)]
    + ["r_last", "r_min", "r_mean3", "r_slope3", "q_lane_last", "q_lane_slope3"]
    + [f"nbr{k:+d}_r_last" for k in NBR] + [f"nbr{k:+d}_r_min" for k in (-2, -1, 1, 2)]
    + ["obs_frac", "obs_last"]
    + ["dist_q_down", "dist_q_up", "n_queued_now", "frac_queued_hist",
       "freq_now", "freq_h", "freq_h_max_nbr",
       "h", "tod", "weekend", "pos", "v_cut", "lanes"]
)


# ---------------------------------------------------------------- features

def _ffill(a):
    idx = np.where(np.isfinite(a), np.arange(a.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = a[idx, np.arange(a.shape[1])[None, :]]
    return out


def _shift(x, k):
    """x[..., l+k], NaN outside the corridor."""
    out = np.full_like(x, np.nan)
    if k > 0:
        out[..., :-k] = x[..., k:]
    elif k < 0:
        out[..., -k:] = x[..., :k]
    return out


def fill_links(a):
    """[T, L] with each row's NaNs filled by linear interpolation along the corridor (ends held flat)."""
    out = np.array(a, dtype=float, copy=True)
    idx = np.arange(out.shape[1])
    for t in range(out.shape[0]):
        ok = np.isfinite(out[t])
        if 2 <= ok.sum() < len(ok):
            out[t] = np.interp(idx, idx[ok], out[t, ok])
    return out


def _dist(flag, direction):
    L = flag.shape[0]
    out = np.full(L, 99.0)
    rng = range(L - 2, -1, -1) if direction > 0 else range(1, L)
    for i in rng:
        j = i + direction
        out[i] = 1.0 if flag[j] else min(out[j] + 1, 99.0)
    return out


def window_features(hv, hq, vcut, lanes, freq_we, s0, weekend):
    """Feature matrix [H * L, F] for one window.

    hv, hq: [HIST + 1, L] speed and flow at T-60 .. T (row HIST is T), NaN where unavailable.
    freq_we: [SLOTS, L] train queue frequency for this window's day type.
    """
    L = hv.shape[1]
    obs = np.isfinite(hv)
    obs_frac, obs_last = obs[:-1].mean(axis=0), obs[-1] | obs[-2]
    hv, hq = fill_links(hv), fill_links(hq)
    r = hv / vcut[None, :]
    rf = _ffill(r)
    r_last = rf[-1]
    with np.errstate(all="ignore"):
        r_min = np.nanmin(r, axis=0)
        r_mean3 = np.nanmean(r[-3:], axis=0)
        ql = hq / lanes[None, :]
        qf = _ffill(ql)
    r_slope3 = rf[-1] - rf[-4]
    queued_now = np.nan_to_num(r_last, nan=9.0) <= 1.0
    per_link = [r[::-1].T, r_last, r_min, r_mean3, r_slope3, qf[-1], qf[-1] - qf[-4]]
    per_link += [_shift(r_last, k) for k in NBR] + [_shift(r_min, k) for k in (-2, -1, 1, 2)]
    per_link += [obs_frac, obs_last.astype(float)]
    per_link += [_dist(queued_now, +1), _dist(queued_now, -1),
                 np.full(L, queued_now.sum(), float),
                 np.full(L, np.nanmean(np.nan_to_num(r, nan=9.0) <= 1.0), float),
                 freq_we[min(s0, SLOTS - 1)]]
    base = np.column_stack([p if p.ndim == 2 else p[:, None] for p in per_link])
    rows = []
    for h in range(1, H + 1):
        s = min(s0 + h, SLOTS - 1)
        fh = freq_we[s]
        fnb = np.nanmax(np.stack([_shift(fh, k) for k in (-2, -1, 1, 2)]), axis=0)
        extra = np.column_stack([fh, fnb, np.full(L, h), np.full(L, s0), np.full(L, weekend),
                                 np.arange(L) / max(L - 1, 1), vcut, lanes])
        rows.append(np.hstack([base, extra]))
    return np.vstack(rows).astype(np.float32)


# ---------------------------------------------------------------- queue frequency from train

def queue_frequency(z_train, vcut):
    tv = z_train["true_speed"]
    ok = np.isfinite(tv) & z_train["elig"]
    q = ok & (tv <= vcut[None, None, :])
    we = (weekday_of(z_train["dates"]) >= 5).astype(int)
    S = np.stack([q[we == k].sum(0) for k in (0, 1)]).astype(np.float64)
    N = np.stack([ok[we == k].sum(0) for k in (0, 1)]).astype(np.float64)
    return q, ok, we, S, N


def freq_for(S, N, we_k, own_q=None, own_ok=None):
    s, n = S[we_k].copy(), N[we_k].copy()
    if own_q is not None:
        s -= own_q
        n -= own_ok
    return s / np.maximum(n, 1)


# ---------------------------------------------------------------- windows

def _labels_and_conditions(tv, el, vcut):
    """Per (slot, link) queue truth for one day, with NaN where not scoreable."""
    lab = np.where(np.isfinite(tv) & el, (tv <= vcut[None, :]).astype(float), np.nan)
    return lab


def full_labels(tv, vcut):
    """Queue truth for every link: observed speeds where present (eligible or not), filled along the corridor
    where the detector is missing. NaN only for slots with fewer than two observed links."""
    f = fill_links(tv)
    return np.where(np.isfinite(f), (f <= vcut[None, :]).astype(float), np.nan)


def generate_windows(z_train, vcut, rng, max_per_condition):
    """Origins on train days that satisfy the organizers' condition definitions."""
    tv = z_train["true_speed"]
    el = z_train["elig"]
    out = {c: [] for c in CONDITIONS}
    for d in range(len(z_train["dates"])):
        lab = _labels_and_conditions(tv[d], el[d], vcut)
        q = np.nan_to_num(lab) > 0
        cover = np.isfinite(lab)
        for s0 in range(HIST, SLOTS - H):
            hist, fut = q[s0 - HIST:s0 + 1], q[s0 + 1:s0 + 1 + H]
            if not fut.any() or cover[s0 - HIST:s0].mean() < 0.7:
                continue
            if not hist.any():
                first = int(np.argmax(fut.any(axis=1))) + 1
                out["queue_onset"].append((d, s0, first))
            elif q[s0].any() or q[s0 - 1].any():
                pers = np.repeat((q[s0] | q[s0 - 1])[None, :], H, 0)
                m = cover[s0 + 1:s0 + 1 + H]
                inter, union = (pers & fut & m).sum(), ((pers | fut) & m).sum()
                if union and inter / union <= 0.9:
                    out["queue_ongoing"].append((d, s0, 0))
    for c in CONDITIONS:
        w = out[c]
        if len(w) > max_per_condition:
            w = [w[i] for i in sorted(rng.choice(len(w), max_per_condition, replace=False))]
        out[c] = w
    return out


def read_task2(release: Path, panel: str, split: str):
    base = release / "task2" / panel / split
    if not (base / "window_index.csv").exists():
        return None
    wi = pd.read_csv(base / "window_index.csv", dtype=str)
    origin_col = next(c for c in ("forecast_origin", "origin", "t0", "forecast_origin_utc") if c in wi.columns)
    wi["forecast_origin"] = wi[origin_col]
    if "date" not in wi.columns:
        wi["date"] = wi.forecast_origin.str.slice(0, 10)
    hist = pd.read_parquet(base / "window_history.parquet")
    hist["window_id"] = hist.window_id.astype(str)
    hist["link_id"] = hist.link_id.astype(str)
    tpl = pd.read_csv(base / "sample_submission_queue.csv", dtype=str) \
        if (base / "sample_submission_queue.csv").exists() else None
    return wi, hist, tpl


def released_window_arrays(w, hist_w, z, links_index):
    """History speed/flow [HIST+1, L] for a published window: T-60 .. T-5 from window_history, T from the
    masked layer of the same split (released data stamped at T)."""
    L = len(links_index)
    hv = np.full((HIST + 1, L), np.nan)
    hq = np.full((HIST + 1, L), np.nan)
    origin = pd.Timestamp(w.forecast_origin)
    ts = pd.to_datetime(hist_w.timestamp, utc=True)
    step = ((ts - origin) / pd.Timedelta(minutes=5)).round().astype(int).to_numpy() + HIST
    li = hist_w.link_id.map(links_index)
    ok = (step >= 0) & (step < HIST) & li.notna().to_numpy()
    li = li.fillna(0).astype(int).to_numpy()
    hv[step[ok], li[ok]] = pd.to_numeric(hist_w.speed_kmh, errors="coerce").to_numpy()[ok]
    if "flow_vph" in hist_w.columns:
        hq[step[ok], li[ok]] = pd.to_numeric(hist_w.flow_vph, errors="coerce").to_numpy()[ok]
    s0 = origin.hour * 12 + origin.minute // 5
    dmap = {x: i for i, x in enumerate(z["dates"].astype(str))}
    d = dmap.get(str(w.date))
    if d is not None:
        hv[HIST] = z["speed"][d, s0]
        hq[HIST] = z["flow"][d, s0]
        if "flow_vph" not in hist_w.columns:
            lo = max(s0 - HIST, 0)
            hq[HIST - (s0 - lo):HIST] = z["flow"][d, lo:s0]
    return hv, hq, s0, d


def persistence(hv, vcut):
    """The official baseline: repeat the queue state at the last history step through the horizon."""
    last = _ffill(hv)[HIST - 1]
    return np.repeat((np.nan_to_num(last, nan=9e9) <= vcut)[None, :], H, 0).astype(float)


# ---------------------------------------------------------------- selection rules

def select_cells(p, rule):
    """Binary prediction for one window from probabilities p [H*L]."""
    if rule[0] == "threshold":
        pred = p >= rule[1]
        if not pred.any():
            pred = p >= 0.5 * p.max()
        return pred
    order = np.argsort(-p)
    cs = np.cumsum(p[order])
    k = np.arange(1, len(p) + 1)
    exp_iou = cs / (k + p.sum() - cs)
    best = int(np.argmax(exp_iou)) + 1
    pred = np.zeros(len(p), bool)
    pred[order[:best]] = True
    return pred


def iou(pred, y):
    m = np.isfinite(y)
    yt = y[m] > 0
    pr = pred[m]
    union = (pr | yt).sum()
    return 1.0 if union == 0 else float((pr & yt).sum() / union)


RULES = [("threshold", t) for t in (0.2, 0.3, 0.4, 0.5, 0.6)] + [("expected_iou", None)]


# ---------------------------------------------------------------- tables, training, prediction

def panel_tables(release: Path, panel: str, z_tr: dict, scored: dict, net: pd.DataFrame, rng, max_windows: int):
    """Rows for one panel: generated train windows, shipped train windows, and the scored splits' windows."""
    vcut, lanes = net.v_cut.to_numpy(float), net.lanes.to_numpy(float)
    lmap = {x: i for i, x in enumerate(z_tr["links"].astype(str))}
    L = len(lmap)
    q, ok, we, S, N = queue_frequency(z_tr, vcut)
    wd = weekday_of(z_tr["dates"])
    rows = []

    def add(kind, cond, split, date, wid, X, y, first, meta=None):
        rows.append(dict(kind=kind, condition=cond, split=split, panel=panel, date=date, window_id=wid,
                         X=X, y=y, first=first, meta=meta))

    gen = generate_windows(z_tr, vcut, rng, max_windows)
    for cond, ws in gen.items():
        for d, s0, first in ws:
            hv = z_tr["true_speed"][d, s0 - HIST:s0 + 1].astype(float).copy()
            hq = z_tr["true_flow"][d, s0 - HIST:s0 + 1].astype(float).copy()
            hv[HIST], hq[HIST] = z_tr["speed"][d, s0], z_tr["flow"][d, s0]
            freq = freq_for(S, N, we[d], q[d], ok[d])
            X = window_features(hv, hq, vcut, lanes, freq, s0, float(wd[d] >= 5))
            lab = full_labels(z_tr["true_speed"][d, s0 + 1:s0 + 1 + H], vcut)
            add("generated", cond, "train", str(z_tr["dates"][d]), f"gen_{panel}_{d}_{s0}", X, lab.reshape(-1), first)

    for split, z in [("train", z_tr)] + list(scored.items()):
        t2 = read_task2(release, panel, split)
        if t2 is None:
            continue
        wi, hist, tpl = t2
        by_window = dict(tuple(hist.groupby("window_id")))
        for w in wi.itertuples(index=False):
            hw = by_window.get(str(w.window_id))
            if hw is None:
                continue
            hv, hq, s0, d = released_window_arrays(w, hw, z, lmap)
            if split == "train" and d is not None:
                freq = freq_for(S, N, we[d], q[d], ok[d])
                lab = _labels_and_conditions(z_tr["true_speed"][d], z_tr["elig"][d], vcut)
                lab = np.full((H, L), np.nan) if s0 + H >= SLOTS else lab[s0 + 1:s0 + 1 + H]
                y = np.full((H, L), np.nan) if s0 + H >= SLOTS else \
                    full_labels(z_tr["true_speed"][d, s0 + 1:s0 + 1 + H], vcut)
            else:
                weekend = pd.Timestamp(w.date).dayofweek >= 5
                freq = freq_for(S, N, int(weekend))
                lab = y = np.full((H, L), np.nan)
            weekend = float(pd.Timestamp(w.date).dayofweek >= 5)
            X = window_features(hv, hq, vcut, lanes, freq, s0, weekend)
            y = y.reshape(-1)
            first = int(np.argmax(np.nan_to_num(lab).any(axis=1))) + 1 if np.nan_to_num(lab).any() else 0
            add("shipped", str(w.condition), split, str(w.date), str(w.window_id), X, y, first,
                meta=dict(forecast_origin=str(w.forecast_origin), persistence=persistence(hv, vcut).reshape(-1),
                          tpl=None if tpl is None else tpl[tpl.window_id == str(w.window_id)], lmap=lmap))
    return rows


def _fit(X, y, w, seed):
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=0.03, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  seed=seed, deterministic=True, verbose=-1, num_threads=0)
    return lgb.train(params, lgb.Dataset(X, y, weight=w, feature_name=list(FEATURES), free_raw_data=True),
                     num_boost_round=600)


def _training_set(rows, weights):
    X = np.vstack([r["X"] for r in rows])
    y = np.concatenate([r["y"] for r in rows])
    w = np.concatenate([np.full(len(r["y"]), weights[id(r)]) for r in rows])
    keep = np.isfinite(y)
    return X[keep], y[keep], w[keep]


def onset_weights(rows):
    """Reweight generated onset windows so the first-queue step matches the shipped train windows."""
    w = {id(r): 1.0 for r in rows}
    gen = [r for r in rows if r["kind"] == "generated" and r["condition"] == "queue_onset"]
    ship = [r for r in rows if r["kind"] == "shipped" and r["split"] == "train"
            and r["condition"] == "queue_onset" and r["first"] > 0]
    if not gen or len(ship) < 5:
        return w, None
    pg = np.bincount([r["first"] for r in gen], minlength=H + 1)[1:] + 0.5
    ps = np.bincount([r["first"] for r in ship], minlength=H + 1)[1:] + 0.5
    ratio = (ps / ps.sum()) / (pg / pg.sum())
    for r in gen:
        w[id(r)] = float(ratio[r["first"] - 1])
    return w, {"shipped_first_step_counts": (ps - 0.5).astype(int).tolist(),
               "generated_first_step_counts": (pg - 0.5).astype(int).tolist()}


def onset_at_last_step(rows):
    """True when the organizers' train onset windows put the first queue at T+30.

    Their selector takes the first origin with a queue in the horizon, so the queue first shows at the last
    step. Training then keeps only generated windows built the same way, and predictions are restricted to
    that step.
    """
    ship = [r for r in rows if r["kind"] == "shipped" and r["split"] == "train"
            and r["condition"] == "queue_onset" and r["first"] > 0]
    return len(ship) >= 5 and float(np.mean([r["first"] == H for r in ship])) >= 0.8


def _mask_last(p):
    q = p.copy()
    q[:len(p) - len(p) // H] = 0.0
    return q


def run(rows, folds: int, seed: int, log=print):
    """Cross-validate, pick the selection rule, refit on everything, predict the scored windows."""
    report, preds = {}, {}
    for cond in CONDITIONS:
        rc = [r for r in rows if r["condition"] == cond]
        train_rows = [r for r in rc if r["split"] == "train" and np.isfinite(r["y"]).any()]
        if not train_rows:
            log(f"  {cond}: no train windows, falling back to persistence")
            continue
        last_only = cond == "queue_onset" and onset_at_last_step(train_rows)
        if last_only:
            train_rows = [r for r in train_rows if r["kind"] == "shipped" or r["first"] == H]
            weights, wrep = {id(r): 1.0 for r in train_rows}, {"last_step_only": True}
        elif cond == "queue_onset":
            weights, wrep = onset_weights(train_rows)
        else:
            weights, wrep = {id(r): 1.0 for r in train_rows}, None
        post = _mask_last if last_only else (lambda p: p)
        dates = sorted({r["date"] for r in train_rows})
        rs = np.random.default_rng(seed)
        fold_of = dict(zip(dates, rs.integers(0, folds, len(dates))))
        cv = {rule: {"generated": [], "shipped": []} for rule in RULES}
        pers = {"generated": [], "shipped": []}
        for f in range(folds):
            tr = [r for r in train_rows if fold_of[r["date"]] != f]
            te = [r for r in train_rows if fold_of[r["date"]] == f]
            if not tr or not te:
                continue
            model = _fit(*_training_set(tr, weights), seed)
            for r in te:
                p = post(model.predict(r["X"]))
                for rule in RULES:
                    cv[rule][r["kind"]].append(iou(select_cells(p, rule), r["y"]))
                if r["kind"] == "shipped":
                    pers["shipped"].append(iou(r["meta"]["persistence"] > 0, r["y"]))
            log(f"  {cond}: fold {f + 1}/{folds} done")

        def mean(v):
            return float(np.mean(v)) if v else float("nan")
        table = {f"{rule[0]}{'' if rule[1] is None else '@' + str(rule[1])}":
                 {k: mean(v) for k, v in cv[rule].items()} for rule in RULES}
        key = "shipped" if len(cv[RULES[0]]["shipped"]) >= 10 else "generated"
        best = max(RULES, key=lambda rl: np.nanmean([mean(cv[rl][key]), mean(cv[rl]["generated"])]))
        report[cond] = {"cv_iou_by_rule": table, "chosen_rule": list(best),
                        "persistence_iou_on_shipped_train_windows": mean(pers["shipped"]),
                        "n_generated_windows": sum(r["kind"] == "generated" for r in train_rows),
                        "n_shipped_train_windows": sum(r["kind"] == "shipped" for r in train_rows),
                        "onset_reweighting": wrep}
        log(f"  {cond}: chosen rule {best}, CV IoU shipped {mean(cv[best]['shipped']):.4f} "
            f"generated {mean(cv[best]['generated']):.4f}, persistence {mean(pers['shipped']):.4f}")
        model = _fit(*_training_set(train_rows, weights), seed)
        imp = dict(zip(FEATURES, model.feature_importance("gain").round(1).tolist()))
        report[cond]["top_features"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:12])
        for r in rc:
            if r["split"] != "train":
                preds[r["window_id"]] = select_cells(post(model.predict(r["X"])), best)
    return report, preds


def submission_rows(rows, preds, split):
    """Fill each scored window's template; persistence wherever no model prediction exists."""
    out = []
    for r in rows:
        if r["kind"] != "shipped" or r["split"] != split or r["meta"]["tpl"] is None:
            continue
        t = r["meta"]["tpl"][["window_id", "timestamp", "link_id"]].copy()
        p = preds.get(r["window_id"], r["meta"]["persistence"] > 0)
        L = len(r["meta"]["lmap"])
        origin = pd.Timestamp(r["meta"]["forecast_origin"])
        h = ((pd.to_datetime(t.timestamp, utc=True) - origin) / pd.Timedelta(minutes=5)).round().astype(int).to_numpy()
        li = t.link_id.map(r["meta"]["lmap"])
        ok = (h >= 1) & (h <= H) & li.notna().to_numpy()
        idx = (np.clip(h, 1, H) - 1) * L + li.fillna(0).astype(int).to_numpy()
        t["queue_pred"] = np.where(ok, p[idx], 0).astype(int)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["window_id", "timestamp", "link_id", "queue_pred"])
