"""Local Task 3 proxy on the train split.

The leaderboard's conservation term uses organizer boundary flows "projected onto the released
observations", so the released truth satisfies N(t+dt) - N(t) = rhs exactly. On train the truth is released,
so rhs = dN_true and a submission's S_LWR is approximately 1 - sum|dN_hat - dN_true| / sum|dN_true| over
within-day transitions where both cells are eligible. Non-target cells keep their truth, as in the scorer.
S_FD uses the official per-lane triangular diagram. Both are averaged over the three regimes.

It ignores the scorer's link-observability and ramp-validity filters, so read it as a relative measure.
"""
from __future__ import annotations

import numpy as np

from common import REGIMES


def _state(pv, pq, z, days):
    tv, tq = z["true_speed"][days], z["true_flow"][days]
    tgt = z["target"][days]
    ok = z["elig"][days] & np.isfinite(tv) & np.isfinite(tq)
    hv = np.where(tgt, pv[days], tv)
    hq = np.where(tgt, pq[days], tq)
    return tv, tq, hv, hq, ok


def s_lwr(pv, pq, z, length, days) -> float:
    tv, tq, hv, hq, ok = _state(pv, pq, z, days)
    Nt = tq / np.maximum(tv, 1.0) * length[None, None, :]
    Nh = hq / np.maximum(hv, 1.0) * length[None, None, :]
    m = ok[:, 1:] & ok[:, :-1] & np.isfinite(Nh[:, 1:]) & np.isfinite(Nh[:, :-1])
    rhs = (Nt[:, 1:] - Nt[:, :-1])[m]
    res = (Nh[:, 1:] - Nh[:, :-1])[m] - rhs
    return float(max(0.0, 1 - min(1.0, np.abs(res).sum() / max(np.abs(rhs).sum(), 1e-9))))


def s_fd(pv, pq, z, net, days) -> float:
    _, _, hv, hq, ok = _state(pv, pq, z, days)
    lanes = net.lanes.to_numpy(float)
    vf = net.free_speed_kmh.to_numpy(float)
    cap = net.capacity_vph.to_numpy(float)
    kc = cap / np.maximum(vf, 1)
    kj = np.maximum(net.k_jam.to_numpy(float), kc * 1.05)
    ql = hq / lanes
    kl = hq / np.maximum(hv, 1.0) / lanes
    qfd = np.where(kl <= kc / lanes, vf * kl,
                   (cap / lanes) / np.maximum((kj - kc) / lanes, 1e-9) * np.maximum(kj / lanes - kl, 0))
    ok = ok & np.isfinite(ql)
    return float(max(0.0, 1 - np.abs(ql - qfd)[ok].sum() / max(np.abs(ql)[ok].sum(), 1e-9)))


def score(pv, pq, z, net, days=None) -> dict:
    days = np.arange(len(z["dates"])) if days is None else np.asarray(days)
    length = net.length_km.to_numpy(float)
    out, lw, fd = {}, [], []
    for r, name in enumerate(REGIMES):
        dd = days[z["regime"][days] == r]
        if len(dd) == 0:
            continue
        a, b = s_lwr(pv, pq, z, length, dd), s_fd(pv, pq, z, net, dd)
        out[name] = {"S_LWR": a, "S_FD": b}
        lw.append(a)
        fd.append(b)
    out["S_LWR"], out["S_FD"] = float(np.mean(lw)), float(np.mean(fd))
    out["S_physics"] = (out["S_FD"] + 2 * out["S_LWR"]) / 3
    return out
