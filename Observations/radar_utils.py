#!/usr/bin/env python
"""
Shared pieces of the EURADCLIM evaluation: the two grids, the remapping
weights, reading EURADCLIM hours, and the per-cell statistics.

Geometry
    Model:     WRF Lambert conformal, 2 km, 749 x 1249, on the WRF sphere.
    EURADCLIM: ODIM "+proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000",
               2 km, 2200 rows x 1900 cols, upper-left CORNER at (0, 0), rows
               running south. Checked against CoordinatesHDF5ODIMWGS84.dat from
               EURADCLIM-tools: cell (r, c) has its centre at
               x = 2000 c + 1000, y = -(2000 r + 1000) to within 0.03 m.
"""

import os
import re
import glob
import logging
import zipfile

import numpy as np
import pandas as pd
import xarray as xr
import h5py
import pyproj
from scipy import sparse

import radar_config as cfg

EUR_PROJ = ("+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 "
            "+units=m +ellps=WGS84")
EUR_NROW, EUR_NCOL, EUR_DX = 2200, 1900, 2000.0

# End-of-hour stamp in EURADCLIM filenames, e.g. ..._201305311400.h5
_STAMP = re.compile(r"(\d{12})\.h5$")


###########################################################
# Model grid
###########################################################

def model_grid():
    """Model lat/lon, LANDMASK and HGT, cropped to the evaluation box.

    The crop is the smallest index rectangle holding every cell whose centre is
    in the box; `in_box` marks those cells, since the rectangle of a Lambert
    grid is not a lat/lon box.
    """
    with xr.open_dataset(cfg.geofile) as geo:
        lat = geo.XLAT_M[0].values
        lon = geo.XLONG_M[0].values
        land = geo.LANDMASK[0].values.astype(bool)
        hgt = geo.HGT_M[0].values
        attrs = dict(geo.attrs)
    in_box = ((lat >= cfg.lat_min) & (lat <= cfg.lat_max)
              & (lon >= cfg.lon_min) & (lon <= cfg.lon_max))
    rows, cols = np.where(in_box)
    ys = slice(rows.min(), rows.max() + 1)
    xs = slice(cols.min(), cols.max() + 1)
    return {"lat": lat[ys, xs], "lon": lon[ys, xs], "land": land[ys, xs],
            "hgt": hgt[ys, xs], "in_box": in_box[ys, xs], "ys": ys, "xs": xs,
            "attrs": attrs}


def model_projection(attrs):
    """pyproj for the WRF Lambert grid (WRF uses a sphere of radius 6370 km)."""
    return pyproj.Proj(proj="lcc", lat_1=attrs["TRUELAT1"], lat_2=attrs["TRUELAT2"],
                       lat_0=attrs["MOAD_CEN_LAT"], lon_0=attrs["STAND_LON"],
                       R=6370000.0)


###########################################################
# Remapping EURADCLIM -> model grid
###########################################################

def remap_weights(grid):
    """Sparse (model cells x radar window cells) area weights, and the window.

    Supersampling approximation to first-order conservative remapping: each
    model cell is split into nsub x nsub equal parts in its own projection, and
    each part is given to the radar cell containing it. Rows of the matrix sum
    to 1. Returns (W, (r0, r1, c0, c1)) with the radar window to read.
    """
    proj = model_projection(grid["attrs"])
    x, y = proj(grid["lon"], grid["lat"])
    dx = float(grid["attrs"]["DX"])
    # The grid must be regular in its own projection, or the sub-cell offsets
    # below are wrong. This checks both the projection and the reading of it.
    err = max(np.abs(np.diff(x, axis=1) - dx).max(), np.abs(np.diff(y, axis=0) - dx).max())
    if err > 5.0:
        raise RuntimeError(f"model grid not regular in its projection (max err {err:.1f} m)")

    n = cfg.nsub
    off = ((np.arange(n) + 0.5) / n - 0.5) * dx
    ox, oy = np.meshgrid(off, off)
    sx = (x[..., None] + ox.ravel()).ravel()
    sy = (y[..., None] + oy.ravel()).ravel()
    slon, slat = proj(sx, sy, inverse=True)
    ex, ey = pyproj.Proj(EUR_PROJ)(slon, slat)
    col = np.floor(ex / EUR_DX).astype(np.int64)
    row = np.floor(-ey / EUR_DX).astype(np.int64)
    ok = (row >= 0) & (row < EUR_NROW) & (col >= 0) & (col < EUR_NCOL)

    r0, r1 = row[ok].min(), row[ok].max() + 1
    c0, c1 = col[ok].min(), col[ok].max() + 1
    ncell = x.size
    target = np.repeat(np.arange(ncell), n * n)[ok]
    source = (row[ok] - r0) * (c1 - c0) + (col[ok] - c0)
    W = sparse.csr_matrix((np.full(target.size, 1.0 / (n * n)), (target, source)),
                          shape=(ncell, (r1 - r0) * (c1 - c0)))
    W.sum_duplicates()
    return W, (int(r0), int(r1), int(c0), int(c1))


def remap(W, fields, shape):
    """Apply the weights to a stack of radar windows (nt, nr, nc) -> (nt, ny, nx).

    NaN (nodata) is excluded and the weights renormalised, so a cell half
    covered by valid radar gets the mean of that half; below min_valid_frac of
    its area valid it is NaN.
    """
    flat = fields.reshape(fields.shape[0], -1).T          # (nsource, nt)
    valid = np.isfinite(flat)
    num = W @ np.where(valid, flat, 0.0)
    den = W @ valid.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den >= cfg.min_valid_frac, num / den, np.nan)
    return out.T.reshape(fields.shape[0], *shape).astype("float32")


###########################################################
# EURADCLIM files
###########################################################

def index_hours():
    """{hour START (Timestamp): (archive or None, path)} for everything in raw/.

    Works whether the platform delivers monthly zips or loose .h5 files, and
    across archive boundaries: the hour 23-00 UTC on 31 December is stamped
    00:00 on 1 January and may sit in either year's archive.
    """
    index = {}

    def add(name, archive):
        m = _STAMP.search(os.path.basename(name))
        if m and "HOURLY" in os.path.basename(name).upper():
            end = pd.Timestamp(pd.to_datetime(m.group(1), format="%Y%m%d%H%M"))
            index[end - pd.Timedelta(hours=1)] = (archive, name)

    for arch in sorted(glob.glob(f"{cfg.path_rad_raw}/**/*.zip", recursive=True)):
        try:
            with zipfile.ZipFile(arch) as zf:
                for name in zf.namelist():
                    add(name, arch)
        except zipfile.BadZipFile:
            logging.error("corrupt archive %s, ignored", arch)
    for fin in glob.glob(f"{cfg.path_rad_raw}/**/*.h5", recursive=True):
        add(fin, None)
    return index


def _str(attr):
    """ODIM string attributes come back as bytes or str depending on the writer."""
    return attr.decode() if isinstance(attr, bytes) else str(attr)


def read_hour(path, window):
    """One hourly accumulation (mm), window only, NaN where nodata.

    undetect is a valid 'no rain' and becomes 0; nodata (outside radar range or
    missing) becomes NaN. Returns (field, nodes) with the contributing radars.
    """
    r0, r1, c0, c1 = window
    with h5py.File(path, "r") as f:
        what = f["/dataset1/what"].attrs
        start = pd.to_datetime(_str(what["startdate"]) + _str(what["starttime"]),
                               format="%Y%m%d%H%M%S")
        end = pd.to_datetime(_str(what["enddate"]) + _str(what["endtime"]),
                             format="%Y%m%d%H%M%S")
        if abs((end - start) - pd.Timedelta(hours=1)) > pd.Timedelta(minutes=10):
            raise ValueError(f"{os.path.basename(path)} is not a 1-h accumulation "
                             f"({start} to {end})")
        raw = f["/dataset1/data1/data"][r0:r1, c0:c1].astype("float64")
        gain, offset = float(what["gain"]), float(what["offset"])
        nodata, undetect = float(what["nodata"]), float(what["undetect"])
        nodes = ""
        if "how" in f and "nodes" in f["how"].attrs:
            nodes = _str(f["how"].attrs["nodes"])
    data = offset + gain * raw
    data[raw == undetect] = 0.0
    data[(raw == nodata) | ~np.isfinite(raw)] = np.nan
    data[data < 0] = np.nan        # any other flag value
    return data, [n.split(":")[-1] for n in nodes.split(",") if n]


def extract(index, hours, where):
    """Local paths for the requested hours, extracting zip members to `where`."""
    paths, by_arch = {}, {}
    for h in hours:
        if h not in index:
            continue
        arch, name = index[h]
        if arch is None:
            paths[h] = name
        else:
            by_arch.setdefault(arch, []).append((h, name))
    for arch, items in by_arch.items():
        with zipfile.ZipFile(arch) as zf:
            for h, name in items:
                paths[h] = zf.extract(name, where)
    return paths


###########################################################
# Per-cell statistics
###########################################################

def hist_edges():
    return np.geomspace(cfg.hist_min, cfg.hist_max, cfg.hist_nbins + 1)


def aggregate(field, valid, k):
    """Block means over k x k cells of the valid cells; block valid if enough are.

    Used identically on the model and the radar with the RADAR validity, so both
    sides average exactly the same set of 2 km cells.
    """
    if k == 1:
        return np.where(valid, field, np.nan)
    nt, ny, nx = field.shape
    ny, nx = ny // k * k, nx // k * k
    f = np.where(valid, field, 0.0)[:, :ny, :nx].reshape(nt, ny // k, k, nx // k, k)
    v = valid[:, :ny, :nx].reshape(nt, ny // k, k, nx // k, k)
    n = v.sum(axis=(2, 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = f.sum(axis=(2, 4)) / n
    return np.where(n >= cfg.agg_min_valid * k * k, out, np.nan)


def aggregate_static(field, k, how="mean"):
    """Block-reduce a 2D field to scale k (lat/lon, land fraction, ...)."""
    if k == 1:
        return field
    ny, nx = field.shape[0] // k * k, field.shape[1] // k * k
    blk = field[:ny, :nx].reshape(ny // k, k, nx // k, k)
    return blk.mean(axis=(1, 3)) if how == "mean" else blk.max(axis=(1, 3))


class CellStats:
    """Running per-cell statistics of hourly rain for one dataset and scale.

    Everything a plot needs is kept in additive form (counts and sums), so
    months can be summed afterwards in any combination -- a season, a year for
    the bootstrap -- and the mask can be changed without recomputing anything.
    """

    def __init__(self, shape):
        ny, nx = shape
        self.shape = shape
        nb = cfg.hist_nbins
        self.nvalid = np.zeros(shape, np.int32)
        self.total = np.zeros(shape, np.float64)
        self.nwet = np.zeros(shape, np.int32)
        self.nexc = np.zeros((len(cfg.exceed_thres), ny, nx), np.int32)
        self.hist = np.zeros(nb * ny * nx, np.int64)
        self.dsum = np.zeros((24, ny, nx), np.float64)
        self.dvalid = np.zeros((24, ny, nx), np.int32)
        self.dwet = np.zeros((24, ny, nx), np.int32)
        self.maxval = np.zeros(shape, np.float32)
        self.edges = hist_edges()

    def add(self, field, hours):
        """field (nt, ny, nx) with NaN where invalid; hours (nt,) UTC hour of day."""
        valid = np.isfinite(field)
        vals = np.where(valid, field, 0.0)
        wet = vals >= cfg.wet_thres
        self.nvalid += valid.sum(0)
        self.total += vals.sum(0)
        self.nwet += wet.sum(0)
        for i, t in enumerate(cfg.exceed_thres):
            self.nexc[i] += (vals >= t).sum(0)
        np.maximum(self.maxval, vals.max(0), out=self.maxval)

        ncell = self.shape[0] * self.shape[1]
        cell = np.broadcast_to(np.arange(ncell).reshape(self.shape), vals.shape)[wet]
        b = np.clip(np.searchsorted(self.edges, vals[wet], side="right") - 1,
                    0, cfg.hist_nbins - 1)
        self.hist += np.bincount(b * ncell + cell, minlength=self.hist.size)

        for h in np.unique(hours):
            sel = hours == h
            self.dsum[h] += vals[sel].sum(0)
            self.dvalid[h] += valid[sel].sum(0)
            self.dwet[h] += wet[sel].sum(0)

    def to_dataset(self, lat, lon, attrs):
        nb = cfg.hist_nbins
        ny, nx = self.shape
        e = self.edges
        return xr.Dataset(
            {"nvalid": (("y", "x"), self.nvalid, {"long_name": "valid hours"}),
             "total": (("y", "x"), self.total.astype("float32"),
                       {"units": "mm", "long_name": "accumulated rain over valid hours"}),
             "nwet": (("y", "x"), self.nwet,
                      {"long_name": f"hours >= {cfg.wet_thres} mm/h"}),
             "nexc": (("thres", "y", "x"), self.nexc,
                      {"long_name": "hours >= threshold"}),
             "maxval": (("y", "x"), self.maxval, {"units": "mm h-1"}),
             "hist": (("bin", "y", "x"), self.hist.reshape(nb, ny, nx).astype(np.int32),
                      {"long_name": "wet-hour counts per intensity bin; "
                                    "dry hours = nvalid - sum(hist)"}),
             "dsum": (("hour", "y", "x"), self.dsum.astype("float32"),
                      {"units": "mm", "long_name": "rain by UTC hour of day (hour start)"}),
             "dvalid": (("hour", "y", "x"), self.dvalid),
             "dwet": (("hour", "y", "x"), self.dwet)},
            coords={"lat": (("y", "x"), lat.astype("float32")),
                    "lon": (("y", "x"), lon.astype("float32")),
                    "thres": ("thres", np.asarray(cfg.exceed_thres, "float32"),
                              {"units": "mm h-1"}),
                    "bin_lo": ("bin", e[:-1].astype("float32")),
                    "bin_hi": ("bin", e[1:].astype("float32")),
                    "hour": ("hour", np.arange(24))},
            attrs=attrs)


def stats_file(dataset, scale, year, month):
    return f"{cfg.path_rad_stats}/{dataset}/RADSTATS_{dataset}_s{scale}_{year}-{month:02d}.nc"


def _combine(a, b):
    """Add two statistics datasets: counts and sums add, the maximum is a max."""
    out = a + b.drop_vars(["lat", "lon"])
    out["maxval"] = np.maximum(a.maxval, b.maxval)
    return out.assign_coords(lat=a.lat, lon=a.lon)


def load_stats(dataset, scale, years, months, per_year=False):
    """Sum the monthly statistics over the requested years and months.

    With per_year, returns {year: Dataset} instead, for the year-block bootstrap.
    """
    out = {}
    for year in years:
        acc = None
        for month in months:
            fin = stats_file(dataset, scale, year, month)
            if not os.path.exists(fin):
                continue
            with xr.open_dataset(fin) as ds:
                ds = ds.load()
            acc = ds if acc is None else _combine(acc, ds)
        if acc is not None:
            out[year] = acc
    if per_year:
        return out
    if not out:
        return None
    total = None
    for ds in out.values():
        total = ds if total is None else _combine(total, ds)
    return total


def quantile_from_hist(hist, ndry, probs, edges):
    """Quantiles of a distribution given as dry count + log-binned wet counts.

    Interpolates log-linearly inside a bin. Quantiles falling in the dry part
    come back as 0.
    """
    counts = np.concatenate([[ndry], hist]).astype("float64")
    cdf = np.cumsum(counts) / counts.sum()
    out = np.zeros(len(probs))
    for i, p in enumerate(probs):
        j = int(np.searchsorted(cdf, p, side="left"))
        if j == 0:
            continue
        j = min(j, len(counts) - 1)
        lo_cdf = cdf[j - 1]
        frac = (p - lo_cdf) / max(cdf[j] - lo_cdf, 1e-300)
        lo, hi = np.log(edges[j - 1]), np.log(edges[j])
        out[i] = np.exp(lo + np.clip(frac, 0, 1) * (hi - lo))
    return out


def cell_quantile(hist, nvalid, p, edges):
    """Per-cell all-hour quantile p from (bin, y, x) histograms; NaN if too few hours."""
    ndry = nvalid - hist.sum(0)
    # number of hours above the quantile
    k = (1.0 - p) * nvalid
    # count from the top down
    above = np.cumsum(hist[::-1], axis=0)[::-1]          # hours in bin >= b
    out = np.zeros(nvalid.shape, "float32")
    nb = hist.shape[0]
    for b in range(nb):
        hi_cnt = above[b + 1] if b + 1 < nb else np.zeros_like(nvalid)
        inbin = (above[b] >= k) & (hi_cnt < k) & (hist[b] > 0)
        frac = np.where(inbin, (above[b] - k) / np.maximum(hist[b], 1), 0)
        val = np.exp(np.log(edges[b]) + frac * (np.log(edges[b + 1]) - np.log(edges[b])))
        out = np.where(inbin, val, out)
    out[ndry >= p * nvalid] = 0.0
    out[nvalid < 1.0 / (1.0 - p)] = np.nan
    return out
