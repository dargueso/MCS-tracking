#!/usr/bin/env python
"""
Brightness temperature: the model's OLR-derived Tb (Yang & Slingo and
Stefan-Boltzmann) against MERGIR, on the common 0.1 deg hourly grid the
tracker uses.

Why: over the gauges the model has about a third of the observed long-lived
cold-cloud hours (BT_objects), and cold cloud is half of what makes a tracked
storm. This shows whether that is the conversion (YS vs SB differ) or the
cloud field itself (both differ from MERGIR the same way), and when in the day.

Sample: hours and cells where MERGIR and both model fields are valid, so the
three distributions are over the same cell-hours.

Outputs, in {ocfg.path_figs}/tb:
    tb_eval_<tag>.png          Tb distribution, cold fractions by hour, per region
    tb_eval_<tag>_numbers.txt

    python plot_tb_evaluation.py             # ASON
    python plot_tb_evaluation.py --all-months
"""

import os
import argparse
import logging
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import obs_config as cfg
import seasons
import regions
from plot_obs_model_comparison import INK, INK_MUTED, GRID, COLORS

FILES = {"MERGIR": f"{cfg.path_track}/OBS_01H_TB_{{tag}}.nc",
         "EPICC YS": f"{cfg.path_modcoarse}/{cfg.model_runs['pres']}/MOD_01H_TB_YS_{{tag}}.nc",
         "EPICC SB": f"{cfg.path_modcoarse}/{cfg.model_runs['pres']}/MOD_01H_TB_SB_{{tag}}.nc"}
COL = {"MERGIR": COLORS["obs"], "EPICC YS": COLORS["YS"], "EPICC SB": COLORS["SB"]}
EDGES = np.arange(170.0, 331.0, 1.0)          # K
THRES = (241.0, 225.0)                        # tracker: BT object, MCS core
PATH_FIGS = f"{cfg.path_figs}/tb"
_R = {}


def do_month(tag):
    tb = {}
    for name, pat in FILES.items():
        f = pat.format(tag=tag)
        if not os.path.exists(f):
            return tag, None
        with xr.open_dataset(f) as d:
            tb[name] = d.TB.values
            if name == "MERGIR":
                hod = pd.to_datetime(d.time.values).hour.values
    valid = np.all([np.isfinite(v) for v in tb.values()], axis=0)
    out = {}
    for reg, rm in _R.items():
        sel = valid & rm[None]
        res = {}
        for name, v in tb.items():
            x = v[sel]
            res[name] = {"hist": np.histogram(x, EDGES)[0],
                         "n_hour": np.array([sel[hod == h].sum() for h in range(24)]),
                         **{f"cold{int(t)}_hour": np.array([(v[hod == h][sel[hod == h]] <= t).sum()
                                                            for h in range(24)]) for t in THRES},
                         "mean": x.sum(), "n": x.size}
        out[reg] = res
    return tag, out


def main():
    par = argparse.ArgumentParser()
    seasons.add_args(par)
    args = par.parse_args()
    logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
    os.makedirs(PATH_FIGS, exist_ok=True)
    months, tag = seasons.resolve(args)
    with xr.open_dataset(FILES["MERGIR"].format(tag=f"{cfg.syear}-01")) as d:
        lat, lon = d.lat.values, d.lon.values
    _R.update(regions.masks(lat, lon, box=[cfg.lat_min, cfg.lon_min, cfg.lat_max, cfg.lon_max]))
    tags = [f"{y}-{m:02d}" for y in range(cfg.syear, cfg.eyear + 1) for m in months]
    acc = {}
    with Pool(12) as pool:
        for t, res in pool.imap_unordered(do_month, tags):
            if res is None:
                logging.warning("%s: missing input", t)
                continue
            for reg, r in res.items():
                for name, v in r.items():
                    a = acc.setdefault(reg, {}).setdefault(name, {})
                    for k, val in v.items():
                        a[k] = a.get(k, 0) + val
    regs = list(acc)
    mid = 0.5 * (EDGES[1:] + EDGES[:-1])

    lines = [f"Brightness temperature, MERGIR vs EPICC (0.1 deg, hourly), {tag}, {cfg.syear}-{cfg.eyear}, "
             "joint valid cell-hours"]
    fig, axes = plt.subplots(3, len(regs), figsize=(3.2 * len(regs), 8.2), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for c, reg in enumerate(regs):
        a = acc[reg]
        n = a["MERGIR"]["n"]
        lines.append(f"\n{reg} ({n:,} cell-hours): mean Tb, fraction <= 241 K, <= 225 K, Tb p1 / p5 / p10")
        for name, v in a.items():
            cdf = np.cumsum(v["hist"]) / n
            q = [mid[np.searchsorted(cdf, p)] for p in (0.01, 0.05, 0.10)]
            f241, f225 = v["cold241_hour"].sum() / n, v["cold225_hour"].sum() / n
            lines.append(f"  {name:9s} {v['mean'] / n:6.1f} K   {100 * f241:5.2f}%  {100 * f225:5.2f}%   "
                         f"{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} K")
            axes[0, c].plot(mid, v["hist"] / n, color=COL[name], lw=1.8, label=name)
            for r, t in enumerate(THRES, 1):
                frac = 100 * v[f"cold{int(t)}_hour"] / np.maximum(v["n_hour"], 1)
                axes[r, c].plot(np.arange(24) + 0.5, frac, color=COL[name], lw=1.8, label=name)
        for t in THRES:
            fo = a["MERGIR"][f"cold{int(t)}_hour"].sum() / n
            lines.append(f"  <= {t:.0f} K, model/MERGIR: YS {a['EPICC YS'][f'cold{int(t)}_hour'].sum() / n / fo:.2f}"
                         f"  SB {a['EPICC SB'][f'cold{int(t)}_hour'].sum() / n / fo:.2f}")
        axes[0, c].set_yscale("log"); axes[0, c].set_xlim(180, 320)
        axes[0, c].set_title(f"{reg} · Tb distribution", loc="left", fontsize=9, color=INK, fontweight="bold")
        axes[0, c].set_xlabel("Tb (K)", fontsize=7.5, color=INK)
        for t in THRES:
            axes[0, c].axvline(t, color=GRID, lw=0.8)
        axes[1, c].set_title(f"{reg} · cold cloud, Tb ≤ 241 K", loc="left", fontsize=9, color=INK, fontweight="bold")
        axes[2, c].set_title(f"{reg} · cold core, Tb ≤ 225 K", loc="left", fontsize=9, color=INK, fontweight="bold")
        axes[2, c].set_xlabel("hour (UTC)", fontsize=7.5, color=INK)
        for r in (1, 2):
            axes[r, c].set_xlim(0, 24); axes[r, c].set_xticks([0, 6, 12, 18, 24]); axes[r, c].set_ylim(bottom=0)
        for ax in axes[:, c]:
            ax.grid(True, color=GRID, lw=0.5); ax.tick_params(labelsize=7, colors=INK_MUTED)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    axes[0, 0].set_ylabel("fraction per K", fontsize=7.5, color=INK)
    axes[1, 0].set_ylabel("% of cell-hours", fontsize=7.5, color=INK)
    axes[2, 0].set_ylabel("% of cell-hours", fontsize=7.5, color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"Brightness temperature, MERGIR vs EPICC OLR-derived Tb, {tag} {cfg.syear}-{cfg.eyear} (0.1°, hourly)",
                 fontsize=10.5, color=INK, fontweight="bold", x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{PATH_FIGS}/tb_eval_{tag}.png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    rows = []
    for reg in regs:
        for name, v in acc[reg].items():
            for h in range(24):
                rows.append(dict(region=reg, dataset=name, hour=h, n=int(v["n_hour"][h]),
                                 **{f"frac{int(t)}": v[f"cold{int(t)}_hour"][h] / max(v["n_hour"][h], 1)
                                    for t in THRES}))
    pd.DataFrame(rows).to_csv(f"{PATH_FIGS}/tb_eval_{tag}_diurnal.csv", index=False, float_format="%.5g")
    text = "\n".join(lines)
    print(text)
    with open(f"{PATH_FIGS}/tb_eval_{tag}_numbers.txt", "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
