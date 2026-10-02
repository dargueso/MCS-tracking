#!/usr/bin/env python
"""
Robustness of the PGW/present-day storm changes to the tracker configuration,
exp1 (ST1, the reference) to exp10, all Yang & Slingo on the clock-hour rain:

    exp1  reference         exp6   rain only, no cloud shield
    exp2  ST2 (bar raised)  exp7   loose thresholds
    exp3  ST3 (bar raised)  exp8   persistence (longer lifetimes)
    exp4  peak 30 mm/h      exp9   Gaussian smoothing, sigma 1
    exp5  peak 30, larger   exp10  linking with >= 30% overlap
                            exp11  intense rain only (peak 30, 1000 km2, no shield)

Same statistics and the same paired year-block bootstrap as
check_bt_robustness.py (whose load/ratios/bootstrap are reused): storms per
year and medians of area, duration, peak rain rate and rain volume, timesteps
clipped to WME.

Also, for the configurations tracked on the 0.1 deg grid (all ten), the
present-day model against IMERG + MERGIR: storms per year and medians, with
the model/observed ratio.

    python check_exp_robustness.py
    python check_exp_robustness.py --exps exp1 exp6 exp10 --nboot 2000

Writes check_exp_robustness.{txt,csv,png} beside the figures.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcs_config as cfg
import check_bt_robustness as cb

OBS_ROOT = "/scratch3/dargueso/obs-mcs-tracking/tracking"
DESC = {"exp1": "reference (ST1)", "exp2": "ST2: 1000 km2, shield 2000 km2", "exp3": "ST3: 1000 km2, shield 10000 km2, peak 10",
        "exp4": "15 mm/h, peak 30", "exp5": "15 mm/h, 1000 km2, peak 30", "exp6": "rain only, no cloud shield",
        "exp7": "loose: 3 mm/h, 250 km2, 2 h, peak 10", "exp8": "persistence: 6 h / 8 h / 8 h",
        "exp9": "smoothing, sigma 1 cell", "exp10": "linking, >= 30% overlap",
        "exp11": "intense rain only: peak 30, 1000 km2, no shield"}
METRICS = cb.METRICS
LABEL = {"count": "storms per year", "area": "max. area", "duration": "duration", "peak": "peak rain rate",
         "volume": "rain volume"}


def load_dir(base, years, months):
    """Per-storm characteristics from a tracking directory, the check_bt_robustness way."""
    lat0, lon0, lat1, lon1 = cfg.reg_coords["WME"]
    rows = []
    for year in years:
        for month in months:
            fin = f"{base}/MCS_{year}{month:02d}"
            if not os.path.exists(fin):
                continue
            for storm in pd.read_pickle(fin).values():
                track = storm["track"]
                inside = ((track[:, 0] > lat0) & (track[:, 0] < lat1) & (track[:, 1] > lon0) & (track[:, 1] < lon1))
                if not inside.any():
                    continue
                rows.append({"year": year, "area": np.nanmax(np.asarray(storm["size"])[inside]) / 1e6,
                             "duration": int(inside.sum()), "peak": np.nanmax(np.asarray(storm["max"])[inside]),
                             "volume": np.nansum(np.asarray(storm["volume"])[inside]) / 1e6})
    return pd.DataFrame(rows)


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--exps", nargs="+", default=[f"exp{i}" for i in range(1, 12)])
    par.add_argument("--syear", type=int, default=2011)
    par.add_argument("--eyear", type=int, default=2020)
    par.add_argument("--months", type=int, nargs="+", default=[8, 9, 10, 11])
    par.add_argument("--nboot", type=int, default=10000)
    par.add_argument("--stat", choices=["mean", "median"], default="mean",
                     help="statistic of the per-storm properties (the manuscript reports means)")
    args = par.parse_args()
    cb.STAT = args.stat
    suffix = "" if args.stat == "mean" else "_median"
    rng = np.random.default_rng(20261001)
    years = list(range(args.syear, args.eyear + 1))
    root = cfg.path_track_root
    fig_dir = f"{cfg.path_figures}/MCS-tracking"
    os.makedirs(fig_dir, exist_ok=True)

    lines = [f"Tracker-configuration robustness, Yang & Slingo, clock-hour rain, {args.syear}-{args.eyear}, "
             f"months {args.months}, WME (timesteps clipped to the box); {args.stat} of the per-storm properties", "",
             "Does the PGW/present change depend on how a storm is defined?", ""]
    rows = []
    res = {}
    for exp in args.exps:
        pres = load_dir(f"{root}/EPICC_2km_ERA5/ConvStormTracking_YS/{exp}", years, args.months)
        fut = load_dir(f"{root}/EPICC_2km_ERA5_CMIP6anom/ConvStormTracking_YS/{exp}", years, args.months)
        if pres.empty or fut.empty:
            lines.append(f"{exp}: no 2 km tracking found, skipped")
            continue
        r, p, f, = cb.ratios(pres, fut, years)
        ci = cb.bootstrap(pres, fut, years, args.nboot, rng)
        res[exp] = (r, p, f, ci, len(pres), len(fut))
        for k in METRICS:
            rows.append(dict(exp=exp, description=DESC.get(exp, ""), metric=k, pres=p[k], fut=f[k], ratio=r[k],
                             ci_lo=ci[k][0], ci_hi=ci[k][1], n_pres=len(pres), n_fut=len(fut)))

    lines += [f"{'exp':6s} {'definition':40s} {'n pres/fut':>12s}" + "".join(f"{LABEL[k]:>22s}" for k in METRICS),
              f"{'':6s} {'':40s} {'':>12s}" + "".join(f"{'fut/pres [95% CI]':>22s}" for _ in METRICS)]
    for exp, (r, p, f, ci, npres, nfut) in res.items():
        lines.append(f"{exp:6s} {DESC.get(exp, ''):40s} {npres:5d}/{nfut:5d} " +
                     "".join(f"{r[k]:6.2f} [{ci[k][0]:.2f},{ci[k][1]:.2f}]".rjust(22) for k in METRICS))
    ref = res.get("exp1")
    if ref:
        lines += ["", "Agreement with the reference (95% intervals overlap):"]
        for exp, (r, p, f, ci, _, _) in res.items():
            if exp == "exp1":
                continue
            ok = [not (ci[k][1] < ref[3][k][0] or ref[3][k][1] < ci[k][0]) for k in METRICS]
            lines.append(f"  {exp:6s} " + "  ".join(f"{LABEL[k]} {'yes' if o else 'NO'}" for k, o in zip(METRICS, ok)))
    lines += ["", f"Absolute values (per year for count, {args.stat} otherwise), present | PGW:", "",
              f"  {'exp':6s}" + "".join(f"{LABEL[k]:>24s}" for k in METRICS)]
    for exp, (r, p, f, ci, _, _) in res.items():
        lines.append(f"  {exp:6s}" + "".join(f"{p[k]:10.1f} | {f[k]:<10.1f}".rjust(24) for k in METRICS))

    # 0.1 deg: present-day model against the observations, same definitions
    lines += ["", f"0.1 deg grid, present day: EPICC (coarsened) / IMERG + MERGIR, same tracker and definition ({args.stat})", "",
              f"  {'exp':6s} {'n mod/obs':>12s}" + "".join(f"{LABEL[k]:>14s}" for k in METRICS)]
    for exp in args.exps:
        obs = load_dir(f"{OBS_ROOT}/obs/{exp}", years, args.months)
        mod = load_dir(f"{OBS_ROOT}/mod0.1_YS_pres/{exp}", years, args.months)
        if obs.empty or mod.empty:
            lines.append(f"  {exp:6s} no 0.1 deg tracking")
            continue
        r, o, m = cb.ratios(obs, mod, years)
        lines.append(f"  {exp:6s} {len(mod):5d}/{len(obs):5d} " + "".join(f"{r[k]:14.2f}" for k in METRICS))
        for k in METRICS:
            rows.append(dict(exp=exp, description=DESC.get(exp, ""), metric=f"obs01_{k}", pres=o[k], fut=m[k],
                             ratio=r[k], ci_lo=np.nan, ci_hi=np.nan, n_pres=len(obs), n_fut=len(mod)))
    lines += ["", "(0.1 deg: 'pres' column = observations, 'fut' column = model in the CSV; ratio = model/observed)"]

    text = "\n".join(lines)
    print(text)
    with open(f"{fig_dir}/check_exp_robustness{suffix}.txt", "w") as fh:
        fh.write(text + "\n")
    pd.DataFrame(rows).to_csv(f"{fig_dir}/check_exp_robustness{suffix}.csv", index=False, float_format="%.4g")

    # figure: fut/pres ratio with intervals, one panel per metric, exp1 first
    exps = list(res)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.0 * len(METRICS), 4.2), sharey=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, k in zip(axes, METRICS):
        for i, exp in enumerate(exps):
            r, _, _, ci, _, _ = res[exp]
            col = "#eb6834" if exp == "exp1" else ("#2a78d6" if int(exp[3:]) <= 5 else "#1baf7a")
            ax.errorbar(i, r[k], yerr=[[r[k] - ci[k][0]], [ci[k][1] - r[k]]], fmt="o", color=col, ms=5, capsize=3, lw=1.2)
        if ref:
            ax.axhspan(ref[3][k][0], ref[3][k][1], color="#eb6834", alpha=0.12, lw=0)
        ax.axhline(1, color="#52514e", lw=0.8)
        ax.set_xticks(range(len(exps))); ax.set_xticklabels([e.replace("exp", "") for e in exps], fontsize=8)
        ax.set_title(LABEL[k], loc="left", fontsize=9.5, fontweight="bold", color="#0b0b0b")
        ax.set_xlabel("experiment", fontsize=8, color="#0b0b0b")
        ax.grid(True, color="#dcdcd8", lw=0.5); ax.tick_params(labelsize=7.5, colors="#52514e")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("PGW / present", fontsize=8.5, color="#0b0b0b")
    fig.suptitle(f"Climate change signal of the storm statistics ({args.stat} of the per-storm properties) under the tracker definitions "
                 f"(orange: reference exp1 and its interval; blue: thresholds raised; green: definition changed) — "
                 f"WME, {args.syear}-{args.eyear}, months {'-'.join(map(str, args.months))}",
                 fontsize=9.5, fontweight="bold", color="#0b0b0b", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{fig_dir}/check_exp_robustness{suffix}.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nwrote {fig_dir}/check_exp_robustness{suffix}.{{txt,csv,png}}")


if __name__ == "__main__":
    main()
