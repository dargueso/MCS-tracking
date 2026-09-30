#!/usr/bin/env python
"""
Spatial comparison of EPICC 2 km against EURADCLIM, on the masked cells only.

Rows: mean rain, wet-hour frequency, an all-hour intensity quantile, and the
time of the diurnal maximum. Columns: EURADCLIM, EPICC, and their comparison
(model/radar ratio on a log2 scale; for the diurnal phase, the difference in
hours).

The phase is the time of maximum of the first (24 h) harmonic of the mean
diurnal cycle, not the arg-max hour: with ~8 seasons of data per cell the
arg-max jumps between neighbouring hours, while the harmonic uses every hour.
Cells whose diurnal amplitude is small relative to their mean have no
meaningful phase and are left blank (min_amplitude).

The intensity quantile of a single 2 km cell rests on few hours: at p99.9 over
eight ASON seasons, ~23 hours per cell. Read the 2 km quantile map for pattern,
not for cell values, or use --scale 5.

Outputs, in {cfg.path_rad_figs}:
    radar_model_maps_<months>_s<scale>.png
    radar_model_maps_<months>_s<scale>_numbers.txt

    python plot_radar_model_maps.py                  # ASON, 2 km
    python plot_radar_model_maps.py --scale 5 --q 0.999
    python plot_radar_model_maps.py --all-months     # or --season JJA, --months 6 7 8
"""

import argparse
import logging

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import radar_config as cfg
import regions
import radar_utils as ru
from plot_obs_model_comparison import INK, INK_MUTED, GRID
from plot_obs_model_maps_relative import DIV, SEQ
from plot_radar_model_qq import RAD, MOD
import seasons

MIN_AMPLITUDE = 0.3     # first-harmonic amplitude / mean, below which no phase


def harmonic_phase(dsum, dvalid):
    """Hour (UTC, 0-24) of maximum of the first diurnal harmonic, and rel. amplitude."""
    with np.errstate(invalid="ignore", divide="ignore"):
        cyc = dsum / dvalid                               # (24, y, x) mm/h
        mean = cyc.mean(0)
        # hour-start stamps: each value belongs to the middle of its hour
        ang = 2 * np.pi * (np.arange(24) + 0.5) / 24
        a = (cyc * np.cos(ang)[:, None, None]).mean(0) * 2
        b = (cyc * np.sin(ang)[:, None, None]).mean(0) * 2
        amp = np.hypot(a, b) / mean
    phase = (np.degrees(np.arctan2(b, a)) % 360) / 15.0
    return np.where(amp >= MIN_AMPLITUDE, phase, np.nan), amp


def fields(ds, q):
    with np.errstate(invalid="ignore", divide="ignore"):
        nvalid = ds.nvalid.values.astype(float)
        out = {"mean": ds.total.values / nvalid * 24.0,
               "freq": 100.0 * ds.nwet.values / nvalid,
               "quant": ru.cell_quantile(ds.hist.values, ds.nvalid.values, q,
                                         ru.hist_edges())}
    out["phase"], _ = harmonic_phase(ds.dsum.values, ds.dvalid.values)
    return out


def main():
    par = argparse.ArgumentParser()
    seasons.add_args(par)
    par.add_argument("--scale", type=int, default=1, choices=cfg.scales)
    par.add_argument("--q", type=float, default=0.999, help="all-hour quantile")
    args = par.parse_args()
    logging.basicConfig(format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                        level=logging.INFO)
    months, tag = seasons.resolve(args)
    years = range(cfg.syear, cfg.eyear + 1)

    with xr.open_dataset(cfg.mask_file) as m:
        mask = m[f"mask_s{args.scale}"].values.astype(bool)
    rad = ru.load_stats(RAD, args.scale, years, months)
    mod = ru.load_stats(MOD, args.scale, years, months)
    if rad is None or mod is None:
        raise RuntimeError("no statistics; run radar_model_stats.py first")
    lat, lon = rad.lat.values, rad.lon.values
    fr, fm = fields(rad, args.q), fields(mod, args.q)
    for f in (fr, fm):
        for k in f:
            f[k] = np.where(mask, f[k], np.nan)

    qlab = f"p{100 * args.q:g}"
    rows = [("mean", "mean rain (mm/day)", LogNorm(0.3, 10), "ratio"),
            ("freq", "wet hours, ≥0.1 mm/h (%)", LogNorm(1, 20), "ratio"),
            ("quant", f"all-hour {qlab} (mm/h)", LogNorm(1, 50), "ratio"),
            ("phase", "diurnal maximum (UTC)", None, "diff")]

    fig = plt.figure(figsize=(15, 3.1 * len(rows)))
    fig.patch.set_facecolor("#fcfcfb")
    lines = [f"EPICC vs EURADCLIM, {tag} {cfg.syear}-{cfg.eyear}, "
             f"{2 * args.scale} km, masked ({int(mask.sum())} cells)"]
    for r, (key, title, norm, how) in enumerate(rows):
        for c, (f, name) in enumerate(((fr, "EURADCLIM"), (fm, "EPICC"))):
            ax = fig.add_subplot(len(rows), 3, 3 * r + c + 1, projection=ccrs.PlateCarree())
            if key == "phase":
                mesh = draw(ax, lon, lat, f[key], cmap="twilight", vmin=0, vmax=24)
            else:
                # a log scale would leave dry (zero) cells blank, indistinguishable
                # from masked ones: pin them to the bottom of the scale instead
                mesh = draw(ax, lon, lat, np.maximum(f[key], norm.vmin), cmap=SEQ,
                            norm=norm)
            decorate(ax, f"{'abcdefghijkl'[3 * r + c]}  {name} · {title}")
            cb = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02,
                              extend="neither" if key == "phase" else "both")
            cb.ax.tick_params(labelsize=7, colors=INK_MUTED)
            if key == "phase":
                cb.set_ticks([0, 6, 12, 18, 24])

        ax = fig.add_subplot(len(rows), 3, 3 * r + 3, projection=ccrs.PlateCarree())
        ok = np.isfinite(fr[key]) & np.isfinite(fm[key])
        if how == "ratio":
            with np.errstate(invalid="ignore", divide="ignore"):
                cmpf = np.where(ok & (fr[key] > 0) & (fm[key] > 0),
                                np.log2(fm[key] / fr[key]), np.nan)
            mesh = draw(ax, lon, lat, cmpf, cmap=DIV.reversed(),
                        norm=TwoSlopeNorm(0, -2, 2))
            cb = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02, extend="both")
            cb.set_ticks([-2, -1, 0, 1, 2])
            cb.set_ticklabels(["¼", "½", "1", "2", "4"])
            decorate(ax, f"{'abcdefghijkl'[3 * r + 2]}  EPICC / EURADCLIM")
            if not ok.any():
                lines.append(f"\n{title}: no cell has enough hours "
                             f"(needs >= {1 / (1 - args.q):.0f} for {qlab}; "
                             "use more months or --scale 5)")
                cb.ax.tick_params(labelsize=7, colors=INK_MUTED)
                continue
            pos = ok & (fr[key] > 0) & (fm[key] > 0)
            corr = np.corrcoef(np.log(fr[key][pos]), np.log(fm[key][pos]))[0, 1] \
                if pos.sum() > 2 else np.nan
            lines.append(f"\n{title}: domain mean {np.nanmean(fr[key][ok]):.3g} vs "
                         f"{np.nanmean(fm[key][ok]):.3g}  (ratio "
                         f"{np.nanmean(fm[key][ok]) / np.nanmean(fr[key][ok]):.2f}), "
                         f"pattern r (log) {corr:.2f}")
            for reg, rmask in regions.masks(lat, lon, box=cfg.subregions["ALL"]).items():
                sub = ok & rmask
                if sub.sum():
                    lines.append(f"  {reg:4s} {np.mean(fr[key][sub]):8.3g} vs "
                                 f"{np.mean(fm[key][sub]):8.3g}  ratio "
                                 f"{np.mean(fm[key][sub]) / np.mean(fr[key][sub]):5.2f}")
        else:
            d = (fm[key] - fr[key] + 12) % 24 - 12            # wrapped to +-12 h
            mesh = draw(ax, lon, lat, np.where(ok, d, np.nan), cmap=DIV.reversed(),
                        norm=TwoSlopeNorm(0, -6, 6))
            cb = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02, extend="both")
            decorate(ax, f"{'abcdefghijkl'[3 * r + 2]}  EPICC − EURADCLIM (h)")
            lines.append(f"\n{title}: median phase difference "
                         f"{np.nanmedian(d[ok]):+.1f} h over {int(ok.sum())} cells "
                         f"with a clear cycle (amplitude >= {MIN_AMPLITUDE} of mean)")
        cb.ax.tick_params(labelsize=7, colors=INK_MUTED)

    fig.suptitle(f"EPICC 2 km vs EURADCLIM — {tag} {cfg.syear}-{cfg.eyear}, "
                 f"{2 * args.scale} km, quality-masked", fontsize=12, color=INK,
                 fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    base = f"{cfg.path_rad_figs}/radar_model_maps_{tag}_s{args.scale}"
    fig.savefig(f"{base}.png", dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    text = "\n".join(lines)
    print(text)
    with open(f"{base}_numbers.txt", "w") as fh:
        fh.write(text + "\n")
    logging.info("wrote %s.png", base)


def draw(ax, lon, lat, field, **kw):
    return ax.pcolormesh(lon, lat, np.ma.masked_invalid(field), shading="nearest",
                         transform=ccrs.PlateCarree(), rasterized=True, **kw)


def decorate(ax, title):
    # the masked area, not the whole box: EURADCLIM has no Italian radars, so
    # the east of the region is empty by construction
    ax.set_extent([cfg.lon_min, 10.0, 36.0, cfg.lat_max], ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.5, edgecolor="#4a4a48")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.3, edgecolor="#8a8a86")
    gl = ax.gridlines(draw_labels=True, lw=0.4, color=GRID, alpha=0.8)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": INK_MUTED}
    ax.set_title(title, loc="left", fontsize=9, color=INK, fontweight="bold")


if __name__ == "__main__":
    main()
