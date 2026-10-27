"""One-command TrafficFlowBench pipeline: data -> Task 1, 2, 4 -> Kaggle upload file + report.

    python src/pipeline.py                         # data already unpacked under data/kaggle_public
    python src/pipeline.py --download              # fetch from Kaggle first (needs a Kaggle API token)
    python src/pipeline.py --data-url gs://bucket/2026-ieee-big-data-traffic-flow-bench.zip
    python src/pipeline.py --panel D12_I5_N        # one corridor, for a quick run

Task 3 has no output of its own: it is scored on the Task 1 rows.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import task1_state as t1  # noqa: E402
import task2_queue as t2  # noqa: E402
import task4_odme as t4  # noqa: E402
from common import CACHE, DEFAULT_RELEASE, OUTPUTS, ROOT, cached, network, panels, weekday_of  # noqa: E402

SLUG = "2026-ieee-big-data-traffic-flow-bench"
SCORED = ("validation", "private")
T0 = time.time()


def log(msg=""):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def find_release(root: Path) -> Path | None:
    if not root.exists():
        return None
    for cand in [root, *sorted(p for p in root.glob("*/") if p.is_dir()), *sorted(p for p in root.glob("*/*/") if p.is_dir())]:
        if (cand / "corridors").is_dir() and (cand / "task1").is_dir():
            return cand
    return None


def acquire(args) -> Path:
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    archive = None
    if args.download:
        import os
        if not os.environ.get("KAGGLE_API_TOKEN"):
            alias = next((k for k in os.environ if k.lower() == "kaggle_competition"), None)
            if alias:
                os.environ["KAGGLE_API_TOKEN"] = os.environ[alias]
        log("Downloading from Kaggle (about 13 GB unpacked)")
        subprocess.run(["kaggle", "competitions", "download", "-c", SLUG, "-p", str(data)], check=True)
        archive = data / f"{SLUG}.zip"
    elif args.data_url:
        archive = data / Path(args.data_url.split("?")[0]).name
        log(f"Fetching {args.data_url}")
        if args.data_url.startswith("gs://"):
            subprocess.run(["gsutil", "cp", args.data_url, str(archive)], check=True)
        else:
            subprocess.run(["curl", "-fL", "-o", str(archive), args.data_url], check=True)
    if archive is not None and archive.exists():
        log(f"Unpacking {archive.name}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data)
        if not args.keep_archive:
            archive.unlink()
    rel = find_release(Path(args.release_root)) if args.release_root else find_release(data)
    if rel is None or not (Path(rel) / "corridors").is_dir():
        raise SystemExit(
            f"No release found under {data}. Unpack the Kaggle data so that data/kaggle_public/corridors exists, "
            "or pass --release-root, --download or --data-url.")
    return Path(rel).resolve()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-root", type=Path)
    ap.add_argument("--download", action="store_true", help="download the data with the Kaggle CLI")
    ap.add_argument("--data-url", help="gs:// or https:// URL of the competition zip")
    ap.add_argument("--keep-archive", action="store_true")
    ap.add_argument("--panel", action="append", help="limit to these corridors (repeatable)")
    ap.add_argument("--max-windows", type=int, default=600,
                    help="generated Task 2 training windows per corridor and condition")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--lam", type=float, default=0.05, help="Task 4 prior regularization")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=OUTPUTS)
    args = ap.parse_args()

    rel = acquire(args)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    all_panels = panels(rel)
    sel = [p for p in all_panels if not args.panel or p in args.panel]
    log(f"Release {rel}; corridors: {', '.join(sel)}")
    rng = np.random.default_rng(args.seed)
    report = {"release_root": str(rel), "panels": sel, "task1": {}, "task2": {}, "task4": {}}
    state_path, odme_parts, t2_rows = out / "state_submission.csv", [], []
    if state_path.exists():
        state_path.unlink()
    first_state = True

    for panel in sel:
        log(f"== {panel}")
        z_tr = cached(rel, panel, "train", args.rebuild_cache)
        scored = {}
        for s in SCORED:
            if (rel / "corridors" / panel / s / "mainline_states_masked").exists():
                scored[s] = cached(rel, panel, s, args.rebuild_cache)
        net = network(rel, panel, z_tr["links"])
        lanes, vmax = net.lanes.to_numpy(float), 1.3 * net.free_speed_kmh.to_numpy(float) + 10
        log(f"  {len(z_tr['links'])} links, train {len(z_tr['dates'])} days, "
            f"{int(z_tr['target'].sum()):,} train targets")

        # Task 1: profiles from released cells of every split, method chosen on train
        arrays_v = [z_tr["speed"]] + [z["speed"] for z in scored.values()]
        arrays_q = [z_tr["flow"]] + [z["flow"] for z in scored.values()]
        wds = [weekday_of(z_tr["dates"])] + [weekday_of(z["dates"]) for z in scored.values()]
        P_v, P_q = t1.profile(arrays_v, wds), t1.profile(arrays_q, wds)
        log("  Task 1 candidates on train:")
        sel1 = t1.select(z_tr, P_v, P_q, lanes, vmax, log=log)
        cfg_v = (sel1["speed_method"]["tau"], sel1["speed_method"]["nbr"])
        cfg_q = (sel1["flow_method"]["tau"], sel1["flow_method"]["nbr"])
        log(f"  Task 1 train S_state {sel1['train_score']['S_state']:.4f} "
            f"(profile only {sel1['train_score_profile_only']['S_state']:.4f}); "
            f"speed: {sel1['speed_method']['method']}, flow: {sel1['flow_method']['method']}")
        report["task1"][panel] = sel1
        for s, z in scored.items():
            tpl = pd.read_csv(rel / "task1" / panel / s / "sample_submission_state.csv", dtype=str)
            pv, pq = t1.predict(z, P_v, P_q, cfg_v, cfg_q, vmax)
            rows = t1.submission_rows(panel, tpl, z, pv, pq)
            rows.to_csv(state_path, mode="w" if first_state else "a", header=first_state, index=False)
            first_state = False
            log(f"  Task 1 {s}: {len(rows):,} rows")

        # Task 2 tables (trained jointly across corridors below)
        if (rel / "task2" / panel).exists():
            rows2 = t2.panel_tables(rel, panel, z_tr, scored, net, rng, args.max_windows)
            t2_rows += rows2
            log(f"  Task 2: {sum(r['kind'] == 'generated' for r in rows2)} generated windows, "
                f"{sum(r['kind'] == 'shipped' for r in rows2)} published windows")

        # Task 4
        for s in SCORED:
            sub, rep = t4.run_panel(rel, panel, s, args.lam)
            if sub is not None:
                odme_parts.append(sub)
                report["task4"][f"{panel}/{s}"] = rep
                log(f"  Task 4 {s}: S_link {rep['S_link']:.4f} (prior only {rep['S_link_prior_only']:.4f})")
        del z_tr, scored

    log("== Task 2 models")
    rep2, preds = t2.run(t2_rows, args.folds, args.seed, log=log)
    report["task2"] = rep2
    queue = pd.concat([t2.submission_rows(t2_rows, preds, s) for s in SCORED], ignore_index=True)
    queue.to_csv(out / "queue_submission.csv", index=False)
    log(f"  Task 2: {len(queue):,} rows, {int(queue.queue_pred.sum()):,} predicted queued")
    pd.concat(odme_parts, ignore_index=True).to_csv(out / "odme_submission.csv", index=False)

    key = rel / "submission_key.csv"
    if key.exists():
        log("== Merging into the upload file")
        subprocess.run([sys.executable, str(ROOT / "official" / "src" / "merge_submissions.py"),
                        "--state", str(state_path), "--queue", str(out / "queue_submission.csv"),
                        "--odme", str(out / "odme_submission.csv"), "--key", str(key),
                        "--output", str(out / "submission.csv")], check=True)
        with zipfile.ZipFile(out / "submission.csv.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out / "submission.csv", "submission.csv")
    else:
        log("No submission_key.csv in the release; per-task files written, merge skipped")

    report["summary"] = summarize(report)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    plots(report, out)
    log(f"Done. Outputs in {out}")
    print(json.dumps(report["summary"], indent=1, default=float))


def summarize(report):
    s1 = [v["train_score"]["S_state"] for v in report["task1"].values()]
    s1b = [v["train_score_profile_only"]["S_state"] for v in report["task1"].values()]
    q = {}
    for cond, r in report["task2"].items():
        rule = r["chosen_rule"]
        name = f"{rule[0]}{'' if rule[1] is None else '@' + str(rule[1])}"
        q[cond] = {"model_cv_iou_shipped": r["cv_iou_by_rule"][name]["shipped"],
                   "model_cv_iou_generated": r["cv_iou_by_rule"][name]["generated"],
                   "persistence_iou_shipped": r["persistence_iou_on_shipped_train_windows"]}
    sl = [v["S_link"] for v in report["task4"].values()]
    return {"task1_train_S_state_mean": float(np.mean(s1)) if s1 else None,
            "task1_train_S_state_profile_baseline": float(np.mean(s1b)) if s1b else None,
            "task2_cv": q,
            "task4_S_link_mean": float(np.mean(sl)) if sl else None}


def plots(report, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if report["task1"]:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for panel, v in report["task1"].items():
            c = pd.DataFrame(v["candidates"])
            ax[0].plot(c.method, c.rmse_speed, marker="o", label=panel)
            ax[1].plot(c.method, c.rmse_flow_per_lane, marker="o", label=panel)
        for a, t in zip(ax, ("Speed RMSE (km/h)", "Flow RMSE (veh/h/lane)")):
            a.set_title(t)
            a.tick_params(axis="x", rotation=60, labelsize=7)
        ax[1].legend(fontsize=7)
        fig.suptitle("Task 1 candidates on train target cells (lower is better)")
        fig.tight_layout()
        fig.savefig(out / "task1_methods.png", dpi=120)
        plt.close(fig)
    if report["task2"]:
        conds = list(report["task2"])
        model = [report["summary"]["task2_cv"][c]["model_cv_iou_shipped"] for c in conds]
        pers = [report["summary"]["task2_cv"][c]["persistence_iou_shipped"] for c in conds]
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(conds))
        ax.bar(x - 0.2, pers, 0.4, label="persistence baseline")
        ax.bar(x + 0.2, model, 0.4, label="LightGBM")
        ax.set_xticks(x, conds)
        ax.set_ylabel("IoU, organizers' train windows (CV)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "task2_iou.png", dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    main()
