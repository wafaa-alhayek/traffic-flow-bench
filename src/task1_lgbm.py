"""Task 1, v2: a gradient-boosted correction on top of the time interpolation.

For every target cell the model predicts the gap between the truth and the linear time interpolation, from
features that are all available at a masked cell of any split:

  the interpolation itself, the gaps to the nearest released cells, the weekday x slot profile,
  the last and next released values on the same link,
  neighbouring links' same-instant deviations from their profile (l-3..l+3) and from their own
  leave-one-out interpolation (l-2..l+2), neighbours' occupancy against its profile,
  neighbours' deviations one slot before and after, slot of day, link, regime and lanes.

Speed is modelled in km/h and flow per lane, the units the scorer uses. One model per channel and corridor,
trained on train targets only, where the truth is released. A day-grouped holdout (half the train days)
measures the gain against the v1 method before the final model is refitted on every train day.
"""
from __future__ import annotations

import numpy as np

from common import score_state, weekday_of
import task1_state as t1

PARAMS = dict(objective="regression", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4)
ROUNDS = 400
MAX_ROWS = 1_500_000


def _shift(a: np.ndarray, k: int = 0, dt: int = 0) -> np.ndarray:
    """a [T, L] moved so that out[t, l] = a[t + dt, l + k], NaN outside."""
    out = np.full_like(a, np.nan)
    T, L = a.shape
    t0, t1_ = max(0, -dt), min(T, T - dt)
    l0, l1 = max(0, -k), min(L, L - k)
    if t1_ > t0 and l1 > l0:
        out[t0:t1_, l0:l1] = a[t0 + dt:t1_ + dt, l0 + k:l1 + k]
    return out


def _prev_next(x: np.ndarray):
    T, L = x.shape
    idx = np.arange(T)[:, None]
    ok = np.isfinite(x)
    prev = np.maximum.accumulate(np.where(ok, idx, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, idx, 2 * T)[::-1], axis=0)[::-1]
    cols = np.arange(L)[None, :]
    p = np.where(prev >= 0, x[np.clip(prev, 0, T - 1), cols], np.nan)
    n = np.where(nxt < T, x[np.clip(nxt, 0, T - 1), cols], np.nan)
    return p, n


def features(z: dict, P_v, P_q, P_o, lanes: np.ndarray, rows: np.ndarray):
    """Feature matrix for the flattened cells `rows` of cache z, plus the interpolation base (iv, iq)."""
    D, S, L = z["speed"].shape
    wd = weekday_of(z["dates"])
    xv, xq = t1._flat(z["speed"]), t1._flat(z["flow"])
    Pv, Pq = t1._flat(P_v[wd]), t1._flat(P_q[wd])
    cols, names = [], []

    def add(name, a):
        names.append(name)
        cols.append(np.asarray(a, dtype=np.float32).ravel()[rows])

    iv, gp, gn = t1.time_interp(xv)
    iq, _, _ = t1.time_interp(xq)
    iv = np.where(np.isfinite(iv), iv, Pv)
    iq = np.where(np.isfinite(iq), iq, Pq)
    base_v, base_q = iv.ravel()[rows].copy(), iq.ravel()[rows].copy()
    add("iv", iv); add("iq", iq); add("Pv", Pv); add("Pq", Pq)
    add("gp", np.minimum(gp, 300)); add("gn", np.minimum(gn, 300))
    add("riv", iv - Pv); add("riq", iq / np.maximum(Pq, 1) - 1)
    del iv, iq, gp, gn
    for nm, x in (("v", xv), ("q", xq)):
        p, n = _prev_next(x)
        add(f"p{nm}", p); add(f"x{nm}", n)
    rv, rq = xv - Pv, xq / np.maximum(Pq, 1) - 1
    for k in (-3, -2, -1, 1, 2, 3):
        add(f"nv{k}", _shift(rv, k)); add(f"nq{k}", _shift(rq, k))
    for k in (-1, 1):
        for dt in (-1, 1):
            add(f"nv{k}t{dt}", _shift(rv, k, dt)); add(f"nq{k}t{dt}", _shift(rq, k, dt))
    del rv, rq, Pv, Pq
    lv, _, _ = t1.time_interp(xv, True)
    Rv = xv - lv
    del lv
    lq, _, _ = t1.time_interp(xq, True)
    Rq = xq / np.maximum(lq, 1) - 1
    del lq
    for k in (-2, -1, 1, 2):
        add(f"Rv{k}", _shift(Rv, k)); add(f"Rq{k}", _shift(Rq, k))
    del Rv, Rq
    if "occ" in z:
        ro = t1._flat(z["occ"]) / np.maximum(t1._flat(P_o[wd]), 1e-6) - 1
        for k in (-2, -1, 1, 2):
            add(f"ro{k}", _shift(ro, k))
        del ro
    T = D * S
    r_t, r_l = rows // L, rows % L
    names += ["slot", "link", "reg", "lanes"]
    cols += [(r_t % S).astype(np.float32), r_l.astype(np.float32),
             z["regime"][r_t // S].astype(np.float32), lanes[r_l].astype(np.float32)]
    return np.column_stack(cols), names, base_v, base_q


def _train(X, y, seed):
    import lightgbm as lgb
    return lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(X, y), num_boost_round=ROUNDS)


def _sample(idx, rng):
    return idx if len(idx) <= MAX_ROWS else rng.choice(idx, size=MAX_ROWS, replace=False)


def target_rows(z: dict, need_truth: bool) -> np.ndarray:
    m = z["target"]
    if need_truth:
        m = m & z["elig"] & np.isfinite(z["true_speed"]) & np.isfinite(z["true_flow"])
    return np.flatnonzero(m.reshape(-1))


def run(z_tr: dict, scored: dict, P_v, P_q, P_o, lanes, vmax, v1_pred, seed: int, log=print, physics=None):
    """Holdout check on half the train days, then fit on all train targets and fill every scored split.

    v1_pred: (pv, pq) of the v1 method on train, for the comparison. physics(pv, pq, days) -> dict, optional.
    Returns (report, {split: (pv, pq)}).
    """
    D, S, L = z_tr["speed"].shape
    rng = np.random.default_rng(seed)
    rows = target_rows(z_tr, True)
    X, names, bv, bq = features(z_tr, P_v, P_q, P_o, lanes, rows)
    tv, tq = z_tr["true_speed"].reshape(-1)[rows], z_tr["true_flow"].reshape(-1)[rows]
    ln = lanes[rows % L]
    yv, yq = tv - bv, (tq - bq) / ln
    del tv, tq

    fold_of_day = rng.permutation(D) % 2
    hold = fold_of_day[rows // L // S] == 0
    tr = _sample(np.flatnonzero(~hold), rng)
    te = np.flatnonzero(hold)
    pv_h = v1_pred[0].copy().reshape(-1)
    pq_h = v1_pred[1].copy().reshape(-1)
    pv_h[rows[te]] = np.clip(bv[te] + _train(X[tr], yv[tr], seed).predict(X[te]), 1.0, vmax[rows[te] % L])
    pq_h[rows[te]] = np.clip(bq[te] + _train(X[tr], yq[tr], seed).predict(X[te]) * ln[te], 0.0, None)
    pv_h, pq_h = pv_h.reshape(D, S, L), pq_h.reshape(D, S, L)
    days = np.flatnonzero(fold_of_day == 0)
    rep = {"holdout_days": int(len(days)), "features": names,
           "holdout_v1": score_state(v1_pred[0], v1_pred[1], z_tr, lanes, days),
           "holdout_v2": score_state(pv_h, pq_h, z_tr, lanes, days)}
    if physics is not None:
        rep["holdout_physics_v1"] = physics(v1_pred[0], v1_pred[1], days)
        rep["holdout_physics_v2"] = physics(pv_h, pq_h, days)
    log(f"  Task 1 v2 holdout ({len(days)} days): S_state {rep['holdout_v1']['S_state']:.4f} -> "
        f"{rep['holdout_v2']['S_state']:.4f}"
        + (f", Task 3 proxy {rep['holdout_physics_v1']['S_physics']:.4f} -> "
           f"{rep['holdout_physics_v2']['S_physics']:.4f}" if physics is not None else ""))
    del pv_h, pq_h

    allr = _sample(np.arange(len(rows)), rng)
    mv, mq = _train(X[allr], yv[allr], seed), _train(X[allr], yq[allr], seed)
    del X, yv, yq
    out = {}
    for s, z in scored.items():
        Ds = z["speed"].shape[0]
        r = target_rows(z, False)
        Xs, _, bvs, bqs = features(z, P_v, P_q, P_o, lanes, r)
        pv = np.full(Ds * S * L, np.nan, np.float32)
        pq = np.full(Ds * S * L, np.nan, np.float32)
        pv[r] = np.clip(bvs + mv.predict(Xs), 1.0, vmax[r % L])
        pq[r] = np.clip(bqs + mq.predict(Xs) * lanes[r % L], 0.0, None)
        out[s] = (pv.reshape(Ds, S, L), pq.reshape(Ds, S, L))
    return rep, out
