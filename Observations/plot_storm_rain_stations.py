#!/usr/bin/env python
"""
Rain at the gauges only while a tracked storm is overhead, against the model's
rain while one of ITS storms is overhead.

The hourly gauge evaluation mixes every rain system. This one keeps the storms
the study is about. Each side is conditioned on its own storms
(extract_storms_at_stations.py):

    gauges      hours when an observed storm (IMERG + MERGIR, 0.1 deg) covers the station
    EURADCLIM   the same hours, at the station's cell (second observation)
    EPICC 0.1   hours when a storm tracked on the model coarsened to 0.1 deg covers it:
                the like-for-like pair, same grid, tracker and criteria
    EPICC 2 km  hours when a storm of the native 2 km tracking covers it (sensitivity)

The rain itself is always at the point: gauge, or the model's / radar's 2 km
cell holding the gauge. Only hours valid at the gauge and in the model are used
(and at the radar for its series), so storms over missing data do not count on
either side.

What is compared, per region:
    - how often a station is under a storm (storm hours per station and year)
    - the rain it gets in those hours: mean, share of wet hours, quantiles, Q-Q
    - the storm share of all rain, and the storm rain per station and year
    - the diurnal cycle of storm hours and storm rain
    - co-occurrence: how often a model storm is overhead when an observed one is

**Selection, read before the numbers.** A storm mask is where the TRACKER's
rain is >= 5 mm/h. For the model that rain is the model's own, so its rain
under its mask is wet by construction; for the observations it is IMERG,
and the gauge under an IMERG storm is an independent measurement that IMERG's
position and intensity errors decorrelate. Gauge-under-IMERG against
model-under-model therefore favours a wetter model. The like-for-like
comparison is the tracker rain under each side's own mask (IMERG against the
coarsened model, both at 0.1 deg): the "tracker" series. The point series
still tell how much rain reaches a gauge when a storm is overhead, and the
storm-hour counts and their diurnal cycle are like for like.

Cold cloud alone (BT_objects) is no neutral alternative: the model has about a
third of the observed cold-cloud hours over the stations (reported below).

Uncertainty: paired year-block bootstrap.

Outputs, in {cfg.path_storm_figs}:
    storm_rain_qq_<tag>.png, storm_rain_diurnal_<tag>.png,
    storm_rain_<tag>_numbers.txt, storm_rain_qq_<tag>.csv

    python plot_storm_rain_stations.py            # ASON
    python plot_storm_rain_stations.py --all-months
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import station_config as cfg
import radar_utils as ru
import seasons
from plot_obs_model_comparison import INK, INK_MUTED, GRID
from plot_station_model import C_OBS, C_MOD, C_RAD, load, region_of

STORM_FILE = f"{cfg.path_eval}/STORMS_AT_STATIONS_{cfg.syear}-{cfg.eyear}.nc"
PROBS = np.array([0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99])
LABELLED = {0.5: "p50", 0.9: "p90", 0.99: "p99"}
EDGES = np.geomspace(0.1, 400.0, 1601)       # mm/h; below 0.1 mm counts as dry
MIN_BOOT_YEARS = 3
SERIES = {  # name: (rain key, storm key, colour, line style)
    "gauges": ("obs", "obs", C_OBS, "-"),
    "EURADCLIM": ("rad", "obs", C_RAD, "-"),
    "EPICC 0.1 storms": ("mod", "mod01", C_MOD, "-"),
    "EPICC 2 km storms": ("mod", "mod2km", C_MOD, "--"),
}
TRACKER = {  # the rain each tracker saw, under its own mask, 0.1 deg: like for like
    "IMERG (tracker)": ("imerg", "obs", C_OBS, ":"),
    "EPICC 0.1 (tracker)": ("mod01pr", "mod01", C_MOD, ":"),
}
ALLSERIES = {**SERIES, **TRACKER}


def load_all(months):
    d = load(months)
    with xr.open_dataset(STORM_FILE) as s:
        assert [str(c) for c in s.code.values] == d["codes"], "station order differs"
        t = pd.to_datetime(s.time.values)
        sel = np.isin(t.month, months)
        assert np.array_equal(t[sel], d["times"])
        storms = {k: s[k].values[:, sel] for k in ("obs", "mod01", "mod2km")}
        cloud = {k: s[f"{k}_btobj"].values[:, sel] for k in ("obs", "mod01", "mod2km")}
        imerg, mod01pr = s["obs_pr"].values[:, sel], s["mod01_pr"].values[:, sel]
    rain = {"obs": d["obs"], "mod": d["mod"]["cell"],
            "rad": d["rad"]["cell"] if d["rad"] is not None else None,
            "imerg": imerg, "mod01pr": mod01pr}
    valid = np.isfinite(d["obs"]) & np.isfinite(d["mod"]["cell"])
    vrad = valid & np.isfinite(rain["rad"]) if rain["rad"] is not None else None
    return dict(rain=rain, storms=storms, cloud=cloud, valid=valid, vrad=vrad, times=d["times"],
                lat=d["lat"], lon=d["lon"], codes=d["codes"])


def mask_for(d, name):
    rk, sk, _, _ = ALLSERIES[name]
    v = d["vrad"] if rk == "rad" else d["valid"]
    return d["rain"][rk], (d["storms"][sk] > 0) & v, v


###########################################################

def per_year(x, m, years, rm):
    out = {}
    for y in np.unique(years):
        s = years == y
        v = x[rm][:, s][m[rm][:, s]]
        out[y] = (np.histogram(v, EDGES)[0], v.size)
    return out


def quantiles(py, years):
    hist = sum(py[y][0] for y in years)
    n = sum(py[y][1] for y in years)
    if n == 0:
        return np.full(PROBS.size, np.nan)
    return ru.quantile_from_hist(hist, n - hist.sum(), PROBS, EDGES)


def ratio_ci(pa, pb, nboot, seed=0):
    years = sorted(set(pa) & set(pb))
    if nboot <= 0 or len(years) < MIN_BOOT_YEARS:
        return np.full((2, PROBS.size), np.nan)
    rng = np.random.default_rng(seed)
    r = np.empty((nboot, PROBS.size))
    for b in range(nboot):
        pick = rng.choice(years, size=len(years), replace=True)
        qa, qb = quantiles(pa, pick), quantiles(pb, pick)
        with np.errstate(invalid="ignore", divide="ignore"):
            r[b] = np.where(qa > 0, qb / qa, np.nan)
    return np.nanpercentile(r, [2.5, 97.5], axis=0)


def qq_table(d, regions, nboot, ref="gauges", names=tuple(SERIES)):
    years = d["times"].year.values
    rows = []
    for reg, rm in regions.items():
        x, m, _ = mask_for(d, ref)
        pg = per_year(x, m, years, rm)
        qg = quantiles(pg, list(pg))
        for name in names:
            x, m, _ = mask_for(d, name)
            if x is None:
                continue
            p = per_year(x, m, years, rm)
            q = quantiles(p, list(p))
            lo, hi = ratio_ci(pg, p, nboot if name != ref else 0)
            for pr, a, b, l, h in zip(PROBS, qg, q, lo, hi):
                rows.append(dict(region=reg, series=name, prob=pr, gauges=a, value=b,
                                 ratio=b / a if a > 0 else np.nan, ratio_lo=l, ratio_hi=h))
    return pd.DataFrame(rows)


def summary(d, regions):
    """Storm hours, storm-hour rain and storm share of rain, per region and series."""
    nyear = len(np.unique(d["times"].year))
    rows = []
    for reg, rm in regions.items():
        for name in ALLSERIES:
            x, m, v = mask_for(d, name)
            if x is None:
                continue
            xs, ms, vs = x[rm], m[rm], v[rm]
            # per station: storm hours per year of valid data
            nval = vs.sum(1) / (len(d["times"]) / nyear)          # valid "years" per station
            ok = nval > 0.5
            hours = ms.sum(1)[ok] / nval[ok]
            r = xs[ms]
            tot_storm = np.where(ms, xs, 0).sum(1)[ok] / nval[ok]
            tot_all = np.where(vs, np.nan_to_num(xs), 0).sum(1)[ok] / nval[ok]
            rows.append(dict(region=reg, series=name,
                             storm_hours_per_year=np.mean(hours),
                             mean_rain=r.mean() if r.size else np.nan,
                             wet_share=np.mean(r >= 0.1) if r.size else np.nan,
                             storm_rain_per_year=np.mean(tot_storm),
                             storm_share=np.sum(tot_storm) / np.sum(tot_all),
                             n=int(ms.sum())))
    return pd.DataFrame(rows)


def cooccurrence(d, regions):
    """P(model storm overhead | observed storm overhead), same hour and within +-3 h."""
    lines = ["\nCo-occurrence: share of observed-storm station-hours with a model storm overhead"
             " (same hour / within +-3 h), and the reverse"]
    obs = (d["storms"]["obs"] > 0) & d["valid"]
    for key in ("mod01", "mod2km"):
        mod = (d["storms"][key] > 0) & d["valid"]
        # +-3 h dilation along time (within the selected months; season edges barely matter)
        dil = np.zeros_like(mod)
        for s in range(-3, 4):
            dil |= np.roll(mod, s, axis=1)
        dilo = np.zeros_like(obs)
        for s in range(-3, 4):
            dilo |= np.roll(obs, s, axis=1)
        parts = []
        for reg, rm in regions.items():
            o, mm, dm, do = obs[rm], mod[rm], dil[rm], dilo[rm]
            parts.append(f"{reg} {np.mean(mm[o]):.2f}/{np.mean(dm[o]):.2f} "
                         f"(rev {np.mean(o[mm]):.2f}/{np.mean(do[mm]):.2f})")
        lines.append(f"  {key:6s} " + "  ".join(parts))
    return lines


def cold_cloud(d, regions):
    """Cold cloud only (BT_objects): hours and point rain, each side under its own cloud."""
    nyear = len(np.unique(d["times"].year))
    lines = ["\nCold cloud only (BT_objects, Tb <= 241 K for >= 5 h), each side under its own cloud:"
             " hours per station-year, mean rain (mm/h), share of all rain"]
    for reg, rm in regions.items():
        parts = []
        for lab, key, rk in (("gauges", "obs", "obs"), ("EPICC 0.1", "mod01", "mod"), ("EPICC 2 km", "mod2km", "mod")):
            v = d["valid"][rm]
            m = (d["cloud"][key][rm] > 0) & v
            x = d["rain"][rk][rm]
            years = v.sum() / (v.shape[1] / nyear)
            share = np.where(m, x, 0).sum() / np.nansum(np.where(v, x, 0))
            parts.append(f"{lab} {m.sum() / years:5.1f} h {np.nanmean(x[m]):.2f} {share:.0%}")
        lines.append(f"  {reg:4s} " + " | ".join(parts))
    return lines


###########################################################

def style(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.tick_params(labelsize=7, colors=INK_MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def figure_qq(q, qt, regions, tag):
    fig, axes = plt.subplots(2, len(regions), figsize=(3.3 * len(regions), 6.8), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for c, reg in enumerate(regions):
        ax = axes[1, c]
        g = qt[(qt.region == reg) & (qt.series == "EPICC 0.1 (tracker)")]
        ax.plot(g.gauges, g.value, color=C_MOD, marker="o", ms=3, lw=1.5, label="EPICC 0.1 vs IMERG")
        for _, r in g[g.prob.isin(list(LABELLED))].iterrows():
            ax.annotate(LABELLED[r.prob], (r.gauges, r.value), fontsize=6.5, color=INK_MUTED,
                        xytext=(4, -9), textcoords="offset points")
        vals = g[["gauges", "value"]].values
        vals = vals[np.isfinite(vals) & (vals > 0)]
        lo, hi = (vals.min() * 0.7, vals.max() * 1.3) if vals.size else (1, 100)
        ax.plot([lo, hi], [lo, hi], color=C_OBS, lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{reg} · like for like, 0.1°", loc="left", fontsize=9.5, color=INK, fontweight="bold")
        ax.set_xlabel("IMERG under observed storms (mm/h)", fontsize=7.5, color=INK)
        if c == 0:
            ax.set_ylabel("EPICC 0.1° under its storms (mm/h)", fontsize=7.5, color=INK)
        style(ax)
    for c, reg in enumerate(regions):
        ax = axes[0, c]
        g0 = q[(q.region == reg) & (q.series == "gauges")]
        for name, (_, _, col, ls) in SERIES.items():
            if name == "gauges":
                continue
            g = q[(q.region == reg) & (q.series == name)]
            if g.empty:
                continue
            ax.plot(g.gauges, g.value, color=col, ls=ls, marker="o", ms=3, lw=1.5, label=name)
            if name == "EPICC 0.1 storms":
                for _, r in g[g.prob.isin(list(LABELLED))].iterrows():
                    ax.annotate(LABELLED[r.prob], (r.gauges, r.value), fontsize=6.5, color=INK_MUTED,
                                xytext=(4, -9), textcoords="offset points")
        vals = q[q.region == reg][["gauges", "value"]].values
        vals = vals[np.isfinite(vals) & (vals > 0)]
        lo, hi = (vals.min() * 0.7, vals.max() * 1.3) if vals.size else (0.1, 100)
        ax.plot([lo, hi], [lo, hi], color=C_OBS, lw=1, label="1:1 (gauges)")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{reg} · at the gauge (selection favours EPICC)", loc="left", fontsize=8.5,
                     color=INK, fontweight="bold")
        ax.set_xlabel("gauges, observed-storm hours (mm/h)", fontsize=7.5, color=INK)
        if c == 0:
            ax.set_ylabel("in own storm hours (mm/h)", fontsize=7.5, color=INK)
        style(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"Hourly rain under storms only, {tag} (p25 … p99)", fontsize=11,
                 color=INK, fontweight="bold", x=0.01, ha="left", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{cfg.path_storm_figs}/storm_rain_qq_{tag}.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def figure_diurnal(d, regions, tag):
    hod = d["times"].hour.values
    nyear = len(np.unique(d["times"].year))
    fig, axes = plt.subplots(3, len(regions), figsize=(3.3 * len(regions), 7.6), sharex=True,
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    lines = ["\nDiurnal cycle of storm rain, ALL: afternoon (12-19 UTC) / night (00-09 UTC) ratio"
             " of storm hours and of storm rain"]
    for c, (reg, rm) in enumerate(regions.items()):
        for name, (rk, _, col, ls) in SERIES.items():
            x, m, v = mask_for(d, name)
            if x is None:
                continue
            xs, ms, vs = x[rm], m[rm], v[rm]
            freq, mean, amount = [], [], []
            for h in range(24):
                s = hod == h
                n_valid = vs[:, s].sum()
                n_storm = ms[:, s].sum()
                freq.append(100 * n_storm / max(n_valid, 1))
                mean.append(xs[:, s][ms[:, s]].mean() if n_storm else np.nan)
                amount.append(np.where(ms[:, s], xs[:, s], 0).sum() / max(n_valid, 1) * s.sum() / nyear)
            for r, y in enumerate((freq, mean, amount)):
                axes[r, c].plot(np.arange(24) + 0.5, y, color=col, ls=ls, lw=1.8, label=name)
            if reg == "ALL":
                a = np.array(amount); f = np.array(freq)
                aft, nig = slice(12, 19), slice(0, 9)
                lines.append(f"  {name:18s} storm hours {f[aft].mean() / f[nig].mean():.2f}   "
                             f"storm rain {a[aft].mean() / a[nig].mean():.2f}")
        axes[0, c].set_title(reg, loc="left", fontsize=9.5, color=INK, fontweight="bold")
        axes[-1, c].set_xlabel("hour (UTC)", fontsize=8, color=INK)
    axes[0, 0].set_ylabel("storm hours (% of hours)", fontsize=7.5, color=INK)
    axes[1, 0].set_ylabel("mean rain in storm hours (mm/h)\nselection favours EPICC", fontsize=7.5, color=INK)
    axes[2, 0].set_ylabel("storm rain (mm per station-year)\nselection favours EPICC", fontsize=7.5, color=INK)
    for ax in axes.ravel():
        ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24]); ax.set_ylim(bottom=0)
        style(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"Diurnal cycle of storm hours (like for like) and storm rain at the gauges, {tag}", fontsize=11,
                 color=INK, fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{cfg.path_storm_figs}/storm_rain_diurnal_{tag}.png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return lines


def main():
    par = argparse.ArgumentParser()
    seasons.add_args(par)
    par.add_argument("--nboot", type=int, default=1000)
    args = par.parse_args()
    logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    os.makedirs(cfg.path_storm_figs, exist_ok=True)
    months, tag = seasons.resolve(args)
    d = load_all(months)
    regions = region_of(d["lat"], d["lon"])

    lines = [f"Rain at the AEMET gauges under tracked storms only (MCS_objects, exp1), {tag}, "
             f"{cfg.syear}-{cfg.eyear}",
             "Each side conditioned on its own storms: gauges and EURADCLIM on observed storms "
             "(IMERG+MERGIR, 0.1 deg); EPICC on storms tracked on the model at 0.1 deg (like for like) "
             "or at 2 km (sensitivity). Rain at the point / 2 km cell.",
             "SELECTION: each mask is defined by its tracker's rain (IMERG / the model), so the model's rain "
             "under its own mask is wet by construction while the gauge under an IMERG storm is not. "
             "Like for like: the tracker series (IMERG vs EPICC, both 0.1 deg, each under its own mask)."]
    s = summary(d, regions)
    lines.append("\nStorm hours per station-year | mean rain in storm hours (mm/h) | wet share (>= 0.1 mm) |"
                 " storm rain (mm per station-year) | storm share of all rain | station-hours")
    for reg in regions:
        for _, r in s[s.region == reg].iterrows():
            lines.append(f"  {reg:4s} {r.series:18s} {r.storm_hours_per_year:6.1f} h | {r.mean_rain:5.2f} | "
                         f"{r.wet_share:4.0%} | {r.storm_rain_per_year:6.1f} mm | {r.storm_share:5.1%} | {r.n}")
    lines.append("\nLike for like at 0.1 deg, EPICC 0.1 (tracker) / IMERG (tracker):")
    for reg in regions:
        g = s[(s.region == reg)].set_index("series")
        lines.append(f"  {reg:4s} storm hours {g.loc['EPICC 0.1 (tracker)', 'storm_hours_per_year'] / g.loc['IMERG (tracker)', 'storm_hours_per_year']:.2f}"
                     f"   mean rain {g.loc['EPICC 0.1 (tracker)', 'mean_rain'] / g.loc['IMERG (tracker)', 'mean_rain']:.2f}"
                     f"   storm rain {g.loc['EPICC 0.1 (tracker)', 'storm_rain_per_year'] / g.loc['IMERG (tracker)', 'storm_rain_per_year']:.2f}")
    lines.append("\nRatios to the gauges (EPICC 0.1 storms / gauges; EPICC 2 km storms / gauges):")
    for reg in regions:
        g = s[(s.region == reg)].set_index("series")
        parts = []
        for col in ("storm_hours_per_year", "mean_rain", "storm_rain_per_year", "storm_share"):
            parts.append(f"{col.replace('_per_year', '').replace('_', ' ')} "
                         f"{g.loc['EPICC 0.1 storms', col] / g.loc['gauges', col]:.2f}/"
                         f"{g.loc['EPICC 2 km storms', col] / g.loc['gauges', col]:.2f}")
        lines.append(f"  {reg:4s} " + "   ".join(parts))

    logging.info("quantiles")
    q = qq_table(d, regions, args.nboot)
    q.to_csv(f"{cfg.path_storm_figs}/storm_rain_qq_{tag}.csv", index=False, float_format="%.4g")
    lines.append("\nQuantiles of hourly rain in storm hours, ratio to the gauges [95% year-block interval]:")
    for reg in regions:
        for name in SERIES:
            if name == "gauges":
                continue
            g = q[(q.region == reg) & (q.series == name)]
            if g.empty:
                continue
            parts = []
            for p, lab in LABELLED.items():
                r = g[np.isclose(g.prob, p)].iloc[0]
                parts.append(f"{lab} {r.ratio:.2f} [{r.ratio_lo:.2f}-{r.ratio_hi:.2f}]")
            lines.append(f"  {reg:4s} {name:18s} " + "  ".join(parts))
    qt = qq_table(d, regions, args.nboot, ref="IMERG (tracker)", names=("EPICC 0.1 (tracker)",))
    qt.to_csv(f"{cfg.path_storm_figs}/storm_rain_qq_{tag}_tracker.csv", index=False, float_format="%.4g")
    lines.append("\nLike for like, quantiles of the tracker rain under own storms, EPICC 0.1 / IMERG:")
    for reg in regions:
        g = qt[(qt.region == reg) & (qt.series == "EPICC 0.1 (tracker)")]
        parts = [f"{lab} {g[np.isclose(g.prob, p)].iloc[0].ratio:.2f} [{g[np.isclose(g.prob, p)].iloc[0].ratio_lo:.2f}-"
                 f"{g[np.isclose(g.prob, p)].iloc[0].ratio_hi:.2f}]" for p, lab in LABELLED.items()]
        lines.append(f"  {reg:4s} " + "  ".join(parts))
    lines += cooccurrence(d, regions)
    lines += cold_cloud(d, regions)
    figure_qq(q, qt, regions, tag)
    lines += figure_diurnal(d, regions, tag)
    text = "\n".join(lines)
    print(text)
    with open(f"{cfg.path_storm_figs}/storm_rain_{tag}_numbers.txt", "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
