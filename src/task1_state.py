"""Task 1: reconstruct the masked mainline speed and flow cells.

Every candidate method is scored on the train split, where the truth is released, and the best one is chosen
per panel and per channel (speed, flow) before it is applied to validation and private.

Candidates:
  interp           linear interpolation in time on the same link, between the nearest released cells
  profile+resid    weekday x time-of-day profile plus the time-interpolated residual from it, which decays
                   back to the profile as the gap grows (tau = decay length in 5-minute slots)
  ...+nbr          adds a same-instant correction from neighbouring links' residuals, fitted by least squares
                   on released cells, where no labels are needed

Profiles are built from released (masked) cells only, so scoring on train never sees a target's truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import REGIMES, SLOTS, score_state, weekday_of

TAUS = (6.0, 24.0, np.inf)
NBR_OFFSETS = (-2, -1, 1, 2)


def time_interp(x: np.ndarray, loo: bool = False):
    """Linear interpolation along axis 0 of x [T, L] from the nearest finite cells before and after.

    With loo=True the cell itself is excluded, so a released cell gets the estimate it would have had as a
    target. Returns (estimate, gap to previous, gap to next); a gap is 10**6 where there is no neighbour.
    """
    T, L = x.shape
    ok = np.isfinite(x)
    idx = np.arange(T)[:, None]
    prev = np.maximum.accumulate(np.where(ok, idx, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, idx, 2 * T)[::-1], axis=0)[::-1]
    if loo:
        prev = np.vstack([np.full((1, L), -1), prev[:-1]])
        nxt = np.vstack([nxt[1:], np.full((1, L), 2 * T)])
    hp, hn = prev >= 0, nxt < T
    cols = np.arange(L)[None, :]
    xp = np.where(hp, x[np.clip(prev, 0, T - 1), cols], np.nan)
    xn = np.where(hn, x[np.clip(nxt, 0, T - 1), cols], np.nan)
    gp = np.where(hp, idx - prev, 10 ** 6).astype(np.float64)
    gn = np.where(hn, nxt - idx, 10 ** 6).astype(np.float64)
    w = np.where(hp & hn, gn / np.maximum(gp + gn, 1), np.where(hp, 1.0, 0.0))
    est = np.where(hp & hn, w * xp + (1 - w) * xn, np.where(hp, xp, xn))
    return est, gp, gn


def profile(arrays: list[np.ndarray], weekdays: list[np.ndarray]) -> np.ndarray:
    """Mean by (weekday, slot, link) over released cells of several splits; link mean where a slot is empty."""
    L = arrays[0].shape[2]
    s, n = np.zeros((7, SLOTS, L)), np.zeros((7, SLOTS, L))
    for a, wd in zip(arrays, weekdays):
        for k in range(7):
            sel = a[wd == k]
            s[k] += np.nansum(sel, axis=0)
            n[k] += np.isfinite(sel).sum(axis=0)
    link_mean = s.sum((0, 1)) / np.maximum(n.sum((0, 1)), 1)
    return np.where(n > 0, s / np.maximum(n, 1), link_mean[None, None, :])


def neighbour_feature(R: np.ndarray) -> np.ndarray:
    """Mean of R [T, L] over links l-2..l+2 (excluding l) at the same instant, NaN-aware."""
    num, den = np.zeros_like(R), np.zeros_like(R)
    for k in NBR_OFFSETS:
        sh = np.full_like(R, np.nan)
        if k > 0:
            sh[:, :-k] = R[:, k:]
        else:
            sh[:, -k:] = R[:, :k]
        w = 1.0 / abs(k)
        num += w * np.nan_to_num(sh)
        den += w * np.isfinite(sh)
    return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)


def candidate(x: np.ndarray, P: np.ndarray, tau: float | None, nbr: bool, relative: bool):
    """One candidate estimate for every cell of x [T, L] (flattened days), with profile P aligned to x."""
    def base(loo):
        if tau is None:
            est, gp, gn = time_interp(x, loo)
            return np.where(np.isfinite(est), est, P), gp, gn
        r = (x / np.maximum(P, 1e-6) - 1.0) if relative else (x - P)
        er, gp, gn = time_interp(r, loo)
        decay = np.exp(-np.minimum(gp, gn) / tau) if np.isfinite(tau) else 1.0
        er = np.nan_to_num(er) * decay
        return (P * (1 + er) if relative else P + er), gp, gn

    est, _, _ = base(False)
    if not nbr:
        return est
    loo, gp, gn = base(True)
    R = (x / np.maximum(loo, 1e-6) - 1.0) if relative else (x - loo)
    R = np.where(np.isfinite(R) & (np.maximum(gp, gn) <= 3), R, np.nan)
    lim = np.nanstd(R) * 5 if np.isfinite(R).any() else 0.0
    R = np.clip(R, -lim, lim)
    F = neighbour_feature(R)
    fit = np.isfinite(R) & (F != 0)
    if fit.sum() < 1000:
        return est
    beta = float((F[fit] * R[fit]).sum() / max((F[fit] ** 2).sum(), 1e-12))
    beta = float(np.clip(beta, 0.0, 1.0))
    return est * (1 + beta * F) if relative else est + beta * F


def configs():
    for tau in (None,) + TAUS:
        for nbr in (False, True):
            name = ("interp" if tau is None else f"profile+resid(tau={tau:g})") + ("+nbr" if nbr else "")
            yield name, tau, nbr


def _flat(a):
    return a.reshape(-1, a.shape[-1]).astype(np.float64)


def predict(z: dict, P_v: np.ndarray, P_q: np.ndarray, cfg_v, cfg_q, vmax: np.ndarray):
    D, S, L = z["speed"].shape
    wd = weekday_of(z["dates"])
    Pv, Pq = _flat(P_v[wd]), _flat(P_q[wd])
    pv = candidate(_flat(z["speed"]), Pv, cfg_v[0], cfg_v[1], relative=False)
    pq = candidate(_flat(z["flow"]), Pq, cfg_q[0], cfg_q[1], relative=True)
    pv = np.clip(pv, 1.0, vmax[None, :]).reshape(D, S, L)
    pq = np.clip(pq, 0.0, None).reshape(D, S, L)
    return pv.astype(np.float32), pq.astype(np.float32)


def select(z_train: dict, P_v, P_q, lanes: np.ndarray, vmax: np.ndarray, log=print) -> dict:
    """Score every candidate on train, separately for speed and flow, and keep the best of each."""
    D, S, L = z_train["speed"].shape
    wd = weekday_of(z_train["dates"])
    Pv, Pq = _flat(P_v[wd]), _flat(P_q[wd])
    xv, xq = _flat(z_train["speed"]), _flat(z_train["flow"])
    m = z_train["target"] & z_train["elig"] & np.isfinite(z_train["true_speed"]) & np.isfinite(z_train["true_flow"])
    m = m.reshape(D * S, L)
    tv, tq = _flat(z_train["true_speed"]), _flat(z_train["true_flow"])
    lane = np.broadcast_to(lanes[None, :], m.shape)
    results = []
    for name, tau, nbr in configs():
        ev = np.clip(candidate(xv, Pv, tau, nbr, False), 1.0, vmax[None, :])
        eq = np.clip(candidate(xq, Pq, tau, nbr, True), 0.0, None)
        rv = float(np.sqrt(np.mean((ev[m] - tv[m]) ** 2)))
        rq = float(np.sqrt(np.mean(((eq[m] - tq[m]) / lane[m]) ** 2)))
        results.append({"method": name, "tau": tau, "nbr": nbr, "rmse_speed": rv, "rmse_flow_per_lane": rq})
        log(f"    {name:32s} speed RMSE {rv:6.3f} km/h   flow RMSE {rq:7.2f} veh/h/lane")
    best_v = min(results, key=lambda r: r["rmse_speed"])
    best_q = min(results, key=lambda r: r["rmse_flow_per_lane"])
    pv, pq = predict(z_train, P_v, P_q, (best_v["tau"], best_v["nbr"]), (best_q["tau"], best_q["nbr"]), vmax)
    score = score_state(pv, pq, z_train, lanes)
    hist = score_state(np.asarray(P_v[wd], np.float32), np.asarray(P_q[wd], np.float32), z_train, lanes)
    return {"speed_method": best_v, "flow_method": best_q, "candidates": results,
            "train_score": score, "train_score_profile_only": hist}


def submission_rows(panel: str, template: pd.DataFrame, z: dict, pv: np.ndarray, pq: np.ndarray) -> pd.DataFrame:
    """Fill the published template rows, keeping its keys verbatim."""
    from common import slot_of
    dmap = {x: i for i, x in enumerate(z["dates"].astype(str))}
    lmap = {x: i for i, x in enumerate(z["links"].astype(str))}
    t = template.copy()
    d = t.timestamp.str.slice(0, 10).map(dmap)
    l = t.link_id.map(lmap)
    ok = d.notna() & l.notna()
    d, l, s = d.fillna(0).astype(int).to_numpy(), l.fillna(0).astype(int).to_numpy(), slot_of(t.timestamp)
    t["panel"] = panel
    if "mask_regime" not in t.columns or t.mask_regime.isna().any():
        t["mask_regime"] = np.asarray(REGIMES)[z["regime"][d]]
    t["speed_kmh"] = np.where(ok, np.round(pv[d, s, l], 3), np.nan)
    t["flow_vph"] = np.where(ok, np.round(pq[d, s, l], 2), np.nan)
    return t[["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]]
