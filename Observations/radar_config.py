#!/usr/bin/env python
"""
Configuration for the radar (EURADCLIM) evaluation of the EPICC 2 km rainfall.

This is a separate branch from the IMERG/MERGIR one in obs_config.py: that one
compares *storms* at 0.1 deg; this one compares *rainfall* at 2 km, over land and
the coast, where gauge-adjusted radar is the best reference there is.

EURADCLIM (Overeem et al. 2023, ESSD 15, 1441) is 1-h accumulations on a 2 km
Lambert azimuthal equal-area grid (1900 x 2200), ODIM-HDF5, from the OPERA
composite merged with ECA&D gauges. Two facts matter throughout:

  * the filename time is the END of the accumulation hour. Everything here is
    re-stamped to the START of the hour, which is what the model (after
    flooring its HH:25 stamps) and the IMERG/MERGIR files use. The actual
    window is HH:05-HH+1:05 (ODIM start/end attributes); the 5-min offset
    from the model's HH:00-HH:50 is ignored.
  * each monthly zip holds the hours ENDING in that month, so the last hour of
    a month (stamped 00:00 on the 1st) is in the next month's zip. Process a
    month only once its successor has been downloaded.
  * Italian radars are not in the OPERA composite, so Sardinia, the Italian
    coast and most of the Tyrrhenian are nodata. The comparison is effectively
    Iberia, southern France, Corsica and the Balearics.

    python download_euradclim.py --list      # check the API sees the dataset
    python download_euradclim.py             # monthly zips, resumable
    python make_radar_input.py               # EURADCLIM onto the model grid
    python radar_model_stats.py              # per-cell statistics, both sides
    python make_radar_mask.py                # quality mask + diagnostic figure
    python plot_radar_model_qq.py            # Q-Q per subregion + diurnal cycle
    python plot_radar_model_maps.py          # spatial comparison
"""

import os
import sys

import obs_config as ocfg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mcs_config as mcfg

###########################################################
# Period and region
###########################################################

# EURADCLIM v2.0 covers 2013-2022; the EPICC evaluation period is 2011-2020.
syear, eyear = 2013, 2020
months = list(range(1, 13))       # processed; plots choose a subset (default ASON)

# Evaluation box: the same WME region as everything else.
lat_min, lat_max = ocfg.lat_min, ocfg.lat_max
lon_min, lon_max = ocfg.lon_min, ocfg.lon_max

# Subregions for the Q-Q and diurnal analysis, [lat0, lon0, lat1, lon1] as in
# mcs_config.reg_coords. "ALL" is the whole masked area. SFR (southern France:
# Languedoc, Cevennes, Provence) is defined here because it has no entry in
# mcs_config; it is where the Cevenol events and the best radar coverage are.
subregions = {
    "ALL": [lat_min, lon_min, lat_max, lon_max],
    "CAT": mcfg.reg_coords["CAT"],
    "LEV": mcfg.reg_coords["LEV"],
    "BAL": mcfg.reg_coords["BAL"],
    "SFR": [42.3, 2.0, 45.0, 7.8],
}

###########################################################
# Paths
###########################################################

path_obs = ocfg.path_obs
path_rad = f"{path_obs}/EURADCLIM"
path_rad_raw = f"{path_rad}/raw"          # archives as downloaded
path_rad_grid = f"{path_rad}/on_model_grid"   # RAD_01H_RAIN_YYYY-MM.nc
path_rad_stats = f"{path_rad}/stats"      # per-month per-cell statistics
path_rad_scratch = f"{ocfg.path_scratch}/euradclim"
path_rad_figs = path_rad

model_run = ocfg.model_runs["pres"]       # evaluation run: ERA5-driven present
# Clock-hour model rain (HH:00-HH+1:00), the definitive version on /scratch3.
# (The original, 10-min-early files on /scratch1 are not used: README, "Timing checks".)
path_model_rain = f"{ocfg.path_model}/{model_run}/RAIN"
geofile = mcfg.geofile_ref                # LANDMASK, HGT_M, projection

# OPERA radar database (EUMETNET), used only for the distance-to-radar mask.
# Downloaded by download_euradclim.py if it is not already here.
opera_db_url = ("https://www.eumetnet.eu/wp-content/themes/aeron-child/"
                "observations-programme/current-activities/opera/database/"
                "OPERA_Database/OPERA_RADARS_DB.json")
opera_db = f"{path_rad}/OPERA_RADARS_DB.json"

###########################################################
# Download (KNMI Open Data API)
###########################################################

# Dataset identifiers on the KNMI Data Platform. Check with --list: if the API
# answers "Not Found", look the name and version up on the dataset page
# (dataplatform.knmi.nl/dataset/rad-opera-hourly-rainfall-accumulation-euradclim-2-0,
# "Access" tab) and correct them here.
knmi_dataset = "RAD_OPERA_HOURLY_RAINFALL_ACCUMULATION_EURADCLIM"   # upper case: the
# lower-case slug of the web page gives 404. Files are monthly zips,
# RAD_OPERA_HOURLY_RAINFALL_ACCUMULATION_EURADCLIM_YYYYMM_0002.zip, 0.5-2.9 GB.
knmi_version = "2.0"
knmi_api = "https://api.dataplatform.knmi.nl/open-data/v1"

# API key: $KNMI_API_KEY, else ~/.knmi_api_key, else the public anonymous key.
# The anonymous key is shared by everyone and is rate-limited to 50 req/min for
# the whole world, so it is often exhausted; a personal key is free
# (developer.dataplatform.knmi.nl) and strongly recommended.
knmi_anonymous_key = ("eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjUzYTg1ZDBhMmQ5YzRk"
                      "YzJiYWNlNzQ4NTQ2Zjk4ODExIiwiaCI6Im11cm11cjEyOCJ9")   # expires 2027-08-01

###########################################################
# Regridding
###########################################################

# EURADCLIM is put on the MODEL grid, not the other way round: both are 2 km, the
# model is the thing being evaluated, and its grid carries LANDMASK and HGT.
# Conservative remapping is approximated by supersampling: each model cell is
# split into nsub x nsub points, each point takes the EURADCLIM cell it falls
# in, and the model cell gets the area-weighted mean. With cells of equal size
# a model cell overlaps at most 4 radar cells, so 5 x 5 gives the weights to 4%.
nsub = 5
# Minimum fraction of a model cell covered by valid radar data for it to be valid.
min_valid_frac = 0.5

###########################################################
# Statistics
###########################################################

wet_thres = 0.1          # mm/h, wet hour
exceed_thres = [1, 5, 10, 20, 50]   # mm/h, exceedance counts per cell

# Log-spaced intensity bins for the per-cell histograms, from which every
# quantile is derived. Everything below the first edge is "dry". 100 bins from
# 0.1 to 400 mm/h is ~8.6% per bin; quantiles are interpolated within a bin.
hist_min, hist_max, hist_nbins = 0.1, 400.0, 100

# Spatial scales, as block sizes in model cells: 1 = native 2 km, 5 = 10 km.
# Pixel-level extremes depend strongly on the effective resolution of each
# product; a statistic that holds at 10 km but not at 2 km is a resolution
# effect, not a model bias. A block is valid if >= agg_min_valid of its cells are.
scales = [1, 5]
agg_min_valid = 0.8

# Hours per chunk when computing statistics (memory control).
chunk_hours = 96

###########################################################
# Quality mask (make_radar_mask.py)
###########################################################

# 1. distance to the nearest radar that actually contributed. C- and S-band
#    radars are quantitatively useful to ~100-150 km; X-band gap-fillers less.
max_range_km = {"C": 150.0, "S": 150.0, "X": 60.0}
# a radar counts as contributing if it appears in at least this fraction of hours
radar_min_hours_frac = 0.2

# Old ODIM node codes found in the EURADCLIM files that are no longer in the
# current OPERA database. AEMET renamed its radars (the 2013 composite lists
# esbar, espma, ...; the database now has esgld, esllm, ...). Same sites.
node_alias = {
    "esbar": "esgld",   # Barcelona  -> Gelida
    "espma": "esllm",   # Palma      -> Llucmajor
    "esval": "escll",   # Valencia   -> Cullera
    "esmur": "esftn",   # Murcia     -> Fortuna
    "esalm": "esnjr",   # Almeria    -> Nijar
    "esmal": "esahr",   # Malaga     -> Alhaurin el Grande
    "eszar": "espdg",   # Zaragoza   -> Perdiguera
    "esmad": "estjv",   # Madrid     -> Torrejon de Velasco
    "essan": "essls",   # "Santander", in fact Aguion (Asturias), 43.4625 -6.3019
    "essev": "esclg",   # Sevilla    -> Castillo de las Guardas
    "esbad": "essft",   # Badajoz    -> Sierra de Fuentes (Caceres)
    "eslpa": "esatn",   # Las Palmas -> Artenara (Gran Canaria)
    "escor": "esccd",   # Coruna (not Cordoba: AEMET has no radar there) -> Cerceda
}
# With these, the 15 old AEMET codes map one-to-one onto the 15 pre-2025 Spanish
# sites in the OPERA database.
# (codes seen in a 2013 file; make_radar_mask.py logs any contributing node it
# cannot place, so an alias missing here shows up rather than going unnoticed)

# Contributing radars that are not in the OPERA database under any code, placed
# by hand as {code: {"location", "latitude", "longitude", "band"}}.
extra_radars = {}

# 2. land plus a coastal strip. Gauge adjustment only has gauges on land, so far
#    offshore the product is close to raw radar.
coast_buffer_km = 10.0

# 3. availability: fraction of hours with valid radar data.
min_availability = 0.9

# 4. climatological artefacts, relative to a smoothed version of the EURADCLIM
#    climatology (Gaussian, sigma in km). Shadows behind mountains show up as
#    anomalously low accumulation; clutter as anomalously frequent light rain.
artefact_sigma_km = 20.0
shadow_ratio = 0.5         # mean precip / smoothed mean precip below this -> out
clutter_ratio = 2.0        # wet-hour freq / smoothed wet-hour freq above this -> out
# Ratios are only trusted where the smoothed reference is not near zero: a dry
# neighbourhood makes the ratio noise, and would flag shadows that are not there.
artefact_min_mean = 0.2    # mm/day, smoothed mean below this -> shadow test skipped
artefact_min_freq = 0.005  # smoothed wet-hour fraction below this -> clutter test skipped
# Grow the flagged areas by this much, so their fringes go too.
artefact_dilate_km = 4.0

# 4b. radial artefacts (beam blockage sectors, interference spokes). Over
#     2013-2020 these are +-30-50% streaks along radials from the Spanish
#     radars, far too weak for the 0.5x / 2x tests above, and tightening those
#     would also remove real orographic gradients. So they are detected by
#     their SHAPE instead: around each radar, in polar bins, each bin is
#     compared with the median of the neighbouring azimuths at the same range.
#     A broad orographic feature is in the neighbours too and cancels; a narrow
#     radial streak does not. A bin is flagged when its ratio stays outside
#     [radial_low, radial_high] for radial_min_run consecutive range bins.
#     Each cell is assigned to its nearest contributing radar, as an
#     approximation of how the composite chooses.
# Settings chosen by comparing four variants on the 2013-2020 climatology:
# 2 deg / 0.75-1.33 removed 0.5% of the area and left obvious spokes;
# 1 deg / 0.85-1.18 removes 1.4% and takes out the clear ones (Murcia,
# Almeria, south of Valencia, part of Madrid). Weaker +-10-20% spokes remain
# across the Spanish composite: no setting removes them without large losses.
radial_az_bin_deg = 1.0
radial_range_bin_km = 10.0
radial_ref_halfwidth_deg = 10.0
radial_low, radial_high = 0.85, 1.18
radial_min_run = 3          # consecutive range bins, i.e. >= 30 km along the radial
radial_min_cells = 1        # cells needed in a polar bin (1-deg bins are sparse near the radar)

# 5. manual exclusions, [lat0, lon0, lat1, lon1] boxes, for anything the
#    diagnostic figure shows that the automatic criteria miss.
manual_exclude = []

mask_file = f"{path_rad}/RADAR_MASK.nc"
