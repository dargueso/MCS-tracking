#!/usr/bin/env python
"""
Evaluation regions: the Spanish autonomous communities on the Mediterranean
side, as polygons (Natural Earth admin-1, 10 m), plus ALL for the whole
evaluation box.

    CAT  Catalonia         VAL  Valencia (Comunitat Valenciana)
    BAL  Balearic Islands  MUR  Murcia
    AND  Andalusia

Polygons rather than lat/lon boxes, so Valencia, Murcia and Andalusia do not
overlap and nothing is counted twice. A station or grid cell belongs to the
region whose polygon holds it; sea cells belong to none. For tracked storms,
whose centres are often just offshore, use `masks(..., buffer_deg=0.5)`: the
polygons grown by ~50 km.

    import regions
    m = regions.masks(lat, lon)               # {"ALL": ..., "CAT": ..., ...}, bool like lat
    regions.outline(ax)                       # draw the boundaries on a cartopy axis
"""

import numpy as np
from matplotlib.path import Path
import cartopy.io.shapereader as shapereader
from shapely.ops import unary_union

ORDER = ["CAT", "VAL", "BAL", "MUR", "AND"]
NE_REGION = {"CAT": "Cataluña", "VAL": "Valenciana", "BAL": "Islas Baleares",
             "MUR": "Murcia", "AND": "Andalucía"}
LONG = {"ALL": "whole domain", "CAT": "Catalonia", "VAL": "Valencia", "BAL": "Balearic Islands",
        "MUR": "Murcia", "AND": "Andalusia"}
SHAPEFILE = shapereader.natural_earth(resolution="10m", category="cultural",
                                      name="admin_1_states_provinces_lakes")
_POLY = {}


def polygons(buffer_deg=0.0):
    """{region: shapely (Multi)Polygon}, the provinces of each community merged."""
    key = round(buffer_deg, 3)
    if key not in _POLY:
        parts = {r: [] for r in ORDER}
        for rec in shapereader.Reader(SHAPEFILE).records():
            a = rec.attributes
            if a.get("admin") != "Spain":
                continue
            for r, name in NE_REGION.items():
                if a.get("region") == name:
                    parts[r].append(rec.geometry)
        out = {}
        for r in ORDER:
            assert parts[r], f"no provinces found for {r}"
            g = unary_union(parts[r])
            out[r] = g.buffer(buffer_deg) if buffer_deg else g
        _POLY[key] = out
    return _POLY[key]


def _inside(geom, lat, lon):
    """Boolean array like lat: points inside a shapely (Multi)Polygon (holes ignored)."""
    pts = np.column_stack([np.ravel(lon), np.ravel(lat)])
    out = np.zeros(pts.shape[0], bool)
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for p in polys:
        x0, y0, x1, y1 = p.bounds
        cand = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
        if cand.any():
            out[cand] |= Path(np.asarray(p.exterior.coords)).contains_points(pts[cand])
    return out.reshape(np.shape(lat))


def masks(lat, lon, box=None, buffer_deg=0.0, names=ORDER):
    """{"ALL": in the box (everything if box is None), region: inside its polygon}."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if box is None:
        all_ = np.ones(lat.shape, bool)
    else:
        la0, lo0, la1, lo1 = box
        all_ = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    polys = polygons(buffer_deg)
    out = {"ALL": all_}
    for r in names:
        # a region is also cut to the box: Andalusia west of 5 W is outside the
        # evaluation domain of the radar and satellite comparisons
        out[r] = _inside(polys[r], lat, lon) & np.isfinite(lat) & np.isfinite(lon) & all_
    return out


def outline(ax, names=ORDER, lw=0.8, color="#4a4a48", **kw):
    """Draw the region boundaries on a cartopy GeoAxes."""
    import cartopy.crs as ccrs
    polys = polygons()
    ax.add_geometries([polys[r] for r in names], ccrs.PlateCarree(), facecolor="none",
                      edgecolor=color, linewidth=lw, zorder=3, **kw)


if __name__ == "__main__":
    import xarray as xr
    import station_config as cfg
    for label, f in (("AEMET/HyMEX hourly", cfg.file_01h), ("AEMET (Arnau) daily", cfg.file_arnau)):
        with xr.open_dataset(f) as d:
            m = masks(d.lat.values, d.lon.values, box=cfg.subregions["ALL"])
        print(f"{label}: " + ", ".join(f"{r} {int(v.sum())}" for r, v in m.items()))
