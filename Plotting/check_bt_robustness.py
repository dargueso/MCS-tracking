#!/usr/bin/env python
"""
Robustness check: does the PGW/present-day change in storm statistics depend on
the OLR-to-brightness-temperature conversion?

Switching from Stefan-Boltzmann to Yang & Slingo roughly halves absolute storm
counts, so every absolute number in the analysis moves. The paper's claim is a
*relative* change between climates, so the question that matters is whether the
fut/pres ratio is the same under both conversions.

Note what is being compared: the ratio of interest is fut/pres WITHIN each
conversion, not YS/SB within each climate. The latter only restates that the
conversions differ.

    python check_bt_robustness.py
    python check_bt_robustness.py --months 8 9 10 11 --syear 2011 --eyear 2020

Writes check_bt_robustness_<exp>.txt beside the figures.
"""

import os
import argparse

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcs_config as cfg

# Both conversions tracked with the same code on the same clock-hour rain
# (2026-09-29 re-run), so SB and YS differ ONLY in the BT conversion. The first
# SB run on /scratch1 (ConvStormTracking/) used the 10-min-early rain and older
# code: do not mix it in.
ROOTS = {"SB": (cfg.path_track_root, "ConvStormTracking_SB"),
         "YS": (cfg.path_track_root, "ConvStormTracking_YS")}
RUNS = {"pres": "EPICC_2km_ERA5", "fut": "EPICC_2km_ERA5_CMIP6anom"}
DX = 2000.0
METRICS = ("count", "area", "duration", "peak", "volume")


def load(bt, period, syear, eyear, months, exp):
    """Per-storm characteristics for one conversion and one climate."""
    root, sub = ROOTS[bt]
    base = f"{root}/{RUNS[period]}/{sub}/{exp}"
    lat0, lon0, lat1, lon1 = cfg.reg_coords[cfg.region if hasattr(cfg, "region") else "WME"]
    rows = []
    for year in range(syear, eyear + 1):
        for month in months:
            fin = f"{base}/MCS_{year}{month:02d}"
            if not os.path.exists(fin):
                continue          # month with no qualifying storm; counts as zero
            for storm in pd.read_pickle(fin).values():
                track = storm["track"]
                inside = ((track[:, 0] >= lat0) & (track[:, 0] <= lat1)
                          & (track[:, 1] >= lon0) & (track[:, 1] <= lon1))
                if not inside.any():
                    continue
                # SB pickles predate the area-weighted 'volume' key, so fall back
                # to the uniform-cell equivalent. ~7% low, and applied to SB only,
                # so volume ratios below are the one metric to treat with care.
                vol = (np.nansum(storm["volume"]) if "volume" in storm
                       else np.nansum(storm["tot"]) * 1e-3 * DX ** 2)
                rows.append({"year": year,
                             "area": np.nanmax(storm["size"]) / 1e6,
                             "duration": len(storm["times"]),
                             "peak": np.nanmax(storm["max"]),
                             "volume": vol / 1e6})
    return pd.DataFrame(rows)


def summarise(frame, years):
    """Storm count per year, and the median of each property."""
    out = {"count": len(frame) / len(years)}
    for key in ("area", "duration", "peak", "volume"):
        out[key] = np.median(frame[key]) if len(frame) else np.nan
    return out


def ratios(pres, fut, years):
    p, f = summarise(pres, years), summarise(fut, years)
    return {k: f[k] / p[k] for k in METRICS}, p, f


def bootstrap(pres, fut, years, nboot, rng):
    """Paired year-block CI on the fut/pres ratio.

    The two experiments share a synoptic sequence, so a year drawn for one must
    be drawn for the other; resampling them independently would break the
    pairing the design relies on.
    """
    pg = {y: g for y, g in pres.groupby("year")}
    fg = {y: g for y, g in fut.groupby("year")}
    draws = {k: [] for k in METRICS}
    for _ in range(nboot):
        pick = rng.choice(years, size=len(years), replace=True)
        p = pd.concat([pg[y] for y in pick if y in pg]) if any(y in pg for y in pick) else pres.iloc[:0]
        f = pd.concat([fg[y] for y in pick if y in fg]) if any(y in fg for y in pick) else fut.iloc[:0]
        if not len(p) or not len(f):
            continue
        r, _, _ = ratios(p, f, pick)
        for k in METRICS:
            draws[k].append(r[k])
    return {k: np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)
            for k, v in draws.items()}


def main():
    par = argparse.ArgumentParser()
    par.add_argument("--exp", default="exp1")
    par.add_argument("--syear", type=int, default=2011)
    par.add_argument("--eyear", type=int, default=2020)
    par.add_argument("--months", type=int, nargs="+", default=[8, 9, 10, 11])
    par.add_argument("--nboot", type=int, default=10000)
    args = par.parse_args()
    rng = np.random.default_rng(20260929)
    years = list(range(args.syear, args.eyear + 1))

    lines = [f"Brightness-temperature robustness check, {args.exp}, "
             f"{args.syear}-{args.eyear}, months {args.months}", "",
             "Does the PGW/present change depend on the OLR-to-Tb conversion?", ""]

    res = {}
    for bt in ("SB", "YS"):
        pres, fut = (load(bt, p, args.syear, args.eyear, args.months, args.exp)
                     for p in ("pres", "fut"))
        r, p, f = ratios(pres, fut, years)
        ci = bootstrap(pres, fut, years, args.nboot, rng)
        res[bt] = (r, p, f, ci)
        lines.append(f"{bt}:  {len(pres)} present-day storms, {len(fut)} PGW storms")

    lines += ["", f"{'metric':10s}" + "".join(f"{b:>26s}" for b in ('SB', 'YS'))
              + f"{'agree?':>10s}",
              f"{'':10s}" + "".join(f"{'fut/pres [95% CI]':>26s}" for _ in range(2))]
    for k in METRICS:
        cells, los, his = [], [], []
        for bt in ("SB", "YS"):
            r, _, _, ci = res[bt]
            cells.append(f"{r[k]:6.2f} [{ci[k][0]:.2f},{ci[k][1]:.2f}]")
            los.append(ci[k][0]); his.append(ci[k][1])
        overlap = not (his[0] < los[1] or his[1] < los[0])
        lines.append(f"{k:10s}" + "".join(f"{c:>26s}" for c in cells)
                     + f"{'yes' if overlap else 'NO':>10s}")

    lines += ["", "Absolute values behind the ratios (per year for count, "
                  "median otherwise):", "",
              f"  {'':8s}" + "".join(f"{m:>12s}" for m in METRICS)]
    for bt in ("SB", "YS"):
        _, p, f, _ = res[bt]
        lines.append(f"  {bt} pres" + "".join(f"{p[m]:12.1f}" for m in METRICS))
        lines.append(f"  {bt} fut " + "".join(f"{f[m]:12.1f}" for m in METRICS))
    lines += ["",
              "'agree?' compares the two 95% intervals: overlapping means the",
              "climate signal is consistent between conversions, which is the",
              "claim this check exists to support.",
              "NOTE: SB pickles predate the area-weighted volume, so SB volumes use",
              "the uniform-cell equivalent (~7% low). Treat the volume row as",
              "indicative; the other four are like for like."]

    text = "\n".join(lines)
    print(text)
    out = f"{cfg.path_figures}/MCS-tracking/check_bt_robustness_{args.exp}.txt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write(text + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
