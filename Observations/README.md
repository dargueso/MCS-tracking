# Observational evaluation branch

Everything needed to compare the EPICC storm tracking against satellite
observations: download IMERG and MERGIR, put both on a common grid, coarsen the
model onto that same grid, track storms in all of them with identical settings,
and compare the resulting statistics.

This exists because "we applied the same tracker to observations" is only
defensible if the tracker sees comparable data. Two things have to be made
comparable before the thresholds mean the same thing on both sides: the **grid**
(2 km vs 0.1°) and the **brightness temperature** (model OLR vs satellite window
BT). Both are handled here, and both change the answer.

---

## Quick start

```bash
conda activate MCStracking
cd /me01/dargueso/Scripts/MCS-tracking/Observations

# 1. download (two different hosts, can run together; ~41 h and ~16 h)
nohup python download_imerg.py  > imerg.out  2>&1 &
nohup python download_mergir.py > mergir.out 2>&1 &
python check_months.py                 # ALWAYS check before going on

# 2. put observations on the common hourly 0.1 deg grid   (~15 min)
python make_obs_tracking_input.py

# 3. coarsen the model onto that same grid                (~1.5 h per run)
python make_model_tracking_input.py pres
python make_model_tracking_input.py fut

# 4. track storms, identical tracker settings everywhere  (~15 min per dataset)
python track_storms.py --list
python track_storms.py obs mod0.1_YS_pres mod0.1_SB_pres

# 5. compare
python plot_obs_model_comparison.py    # distributions: KS + tail ratios
python plot_obs_model_maps.py          # where and when: density maps + diurnal
python plot_obs_model_maps_relative.py # relative pattern, all months and by season
python plot_storm_structure.py         # storm-relative rain structure (~90 s)
```

All settings live in `obs_config.py`: region, period, months, paths, worker
counts, BT methods, dataset registry.

---

## How the pieces fit

```
   IMERG 0.1 deg 30 min ----\
                             >--- OBS_01H_{RAIN,TB}      --\
   MERGIR 4 km 30 min ------/     (hourly, 0.1 deg)         \
                                                             >--- track_storms.py
   EPICC 2 km RAIN --------\                                /      -> storms
   EPICC 2 km OLR ---------/ >--- MOD_01H_{RAIN,TB_SB,TB_YS}
                                  (hourly, 0.1 deg, same grid)
```

| Script | Does |
|---|---|
| `obs_config.py` | every setting; no parameters live in the other scripts |
| `obs_download_utils.py` | Earthdata auth, retries, content validation, month bookkeeping |
| `download_imerg.py` | IMERG half-hourly precipitation for the region |
| `download_mergir.py` | MERGIR half-hourly brightness temperature for the region |
| `check_months.py` | **completeness audit**; `--delete` removes short months for redo |
| `make_obs_tracking_input.py` | hourly means, MERGIR block-averaged to the IMERG grid |
| `make_model_tracking_input.py` | EPICC 2 km coarsened to that grid, both BT conversions |
| `track_storms.py` | runs `MCStracking` on any dataset; writes provenance |
| `plot_obs_model_comparison.py` | distribution comparison: figure, KS, tail ratios |
| `plot_obs_model_maps.py` | spatial and diurnal comparison: absolute density maps |
| `plot_obs_model_maps_relative.py` | standardised density maps, all months and by season |
| `plot_storm_structure.py` | storm-relative rain-rate composites and radial profile |

---

## Where things land

Under `/scratch3/dargueso/obs-mcs-tracking` (on `/scratch3` because `/me01`,
which holds `/home/dargueso`, is near its 2 TB quota):

```
IMERG/          IMERG_30MIN_PR_YYYY-MM.nc        native 0.1 deg, 30 min
MERGIR/         MERGIR_30MIN_TB_YYYY-MM.nc       native 4 km, 30 min
ConvStormTracking/
                OBS_01H_RAIN_YYYY-MM.nc          hourly, 0.1 deg, 2D lat/lon
                OBS_01H_TB_YYYY-MM.nc
MODEL_0.1deg/<wrun>/
                MOD_01H_RAIN_YYYY-MM.nc          hourly, same grid
                MOD_01H_TB_SB_YYYY-MM.nc         Stefan-Boltzmann BT
                MOD_01H_TB_YS_YYYY-MM.nc         Yang & Slingo BT
tracking/<dataset>/<exp>/
                PR_YYYYMM, BT_YYYYMM, MCS_YYYYMM object pickles
                Storms_YYYY-MM.nc                 object masks + provenance
scratch/                                          transient, self-cleaning
```

---

## The two axes: dataset x exp

Storm tracking is indexed on two **independent** axes, and they are kept
separate deliberately.

- **`exp`** is the tracker threshold configuration — `exp1`..`exp5` in
  `mcs_config.py`. `exp1` is the reference (= ST1 in the manuscript).
- **`dataset`** is what was tracked — `obs`, `mod0.1_{SB,YS}_{pres,fut}`, or the
  native 2 km runs.

**Do not encode a data change as a new `exp` (e.g. "exp1b").** The `expN` values
are written into the global attributes of every tracked file, and that is how
provenance is recovered. Two runs differing only in the BT conversion would
carry byte-identical tracker attributes and be indistinguishable from the files
themselves; a label like "exp1b" hides in a name what should be visible in the
data. Name the dataset instead, and every `exp` label keeps meaning exactly one
thing.

`track_storms.py` enforces this: it **derives** the `exp` label by fingerprinting
the five threshold values actually in force against a table of the known
configurations, and refuses to run if they match none of them. Typing a label is
not possible. This is a direct response to a real incident — `mcs_config.py` was
once left on the exp5 thresholds while everything downstream assumed exp1.

The native 2 km tracking is not run from here; it keeps its existing home at
`{path_model}/<wrun>/ConvStormTracking/<exp>/`.

---

## Decisions that change the answer

### Grid: coarsen the model, don't compare across resolutions

`min_area_pr = 500 km²` is ~125 cells at 2 km but ~5 cells at 0.1°. Running
"identical settings" on data at two resolutions means the thresholds bite very
differently. The model is therefore coarsened onto the *observational* grid, read
straight out of an `OBS_01H_*` file so the two can never drift apart.

Note what this validates: the tracker and the storm statistics **at 0.1°**. It
does not by itself validate the 2 km storms, which resolve systems the 0.1°
comparison cannot see.

### Brightness temperature: SB vs YS

| | what it is | vs MERGIR (Sep 2014) |
|---|---|---|
| `SB` | `Tb = (OLR/σ)^0.25`, what `MCS_tracking_WRF.py` does and what produced exp1..exp5 | median **−24.6 K**, p95 **−29.2 K**; whole p1–p99 span only 230–273 K |
| `YS` | Ohring et al. (1984) as given in Yang & Slingo (2001) | median **+2.8 K**, p95 **+2.5 K** |

SB returns a broadband *effective emission* temperature, not a window BT, so it
is not the same physical quantity MERGIR reports. YS removes a 20–30 K bias
through the bulk of the distribution.

The catch: SB reproduces the observed fraction of cells below 241 K almost
exactly (4.08% vs 4.13%) while YS gives 1.31%. That SB agreement is
**compensating errors** — its distribution is so compressed that it happens to
sit near the observed cold tail. Under the physically correct conversion the
model has genuinely less cold cloud than MERGIR, partly real and partly the
coarsening (the model averages ~25 cells of 2 km per 0.1° box against ~7.5 for
MERGIR at 4 km).

Both are produced, so either can be tracked without redoing the coarsening, and
the choice should be judged on **storm** statistics rather than cell counts.

Tracked over the full record (exp1, WME, 2011-2020, all months):

| | storms | med area (km²) | med dur (h) | med peak (mm/h) | med vol (10⁶ m³) |
|---|---|---|---|---|---|
| IMERG + MERGIR | **1884** | **10154** | **12** | **26.5** | **473** |
| EPICC 0.1° YS | 1587 (−16%) | 6988 | 12 | 32.1 | 411 |
| EPICC 0.1° SB | 3182 (+69%) | 5527 | 11 | 28.7 | 284 |

Two-sample KS statistic against the observations:

| | area | duration | peak | volume |
|---|---|---|---|---|
| YS | **0.165** | 0.070 | 0.212 | **0.060** |
| SB | 0.266 | 0.077 | **0.097** | 0.163 |

and the tail summary — model/observed quantile ratio, 10,000 paired year-block
resamples, `*` = 95% interval excludes 1:

| | | p90 | p95 | p99 |
|---|---|---|---|---|
| **YS** | area | 0.62 * | 0.59 * | 0.58 * |
| | duration | 1.29 * | 1.40 * | 1.37 * |
| | peak | 1.26 * | 1.30 * | 1.37 * |
| | volume | 0.86 * | 0.93 | **1.03** |
| **SB** | area | 0.51 * | 0.49 * | 0.50 * |
| | duration | 1.08 * | 1.17 * | 1.20 * |
| | peak | 1.12 * | 1.21 * | 1.29 * |
| | volume | 0.57 * | 0.61 * | 0.68 * |

YS is closer on storm count and on the two cloud-influenced distributions (area,
volume). SB is closer only on peak rain rate — a precipitation quantity the BT
conversion does not touch directly, so it differs purely because a different set
of objects is identified. **YS is the recommended choice**, and the tail summary
strengthens that: at p95–p99 YS reproduces extreme rain volume (0.93–1.03, the
interval including 1), while SB underestimates it by a third (0.57–0.68, never
including 1). Rain volume is the paper's headline quantity, so that is the
comparison that matters most.

**Run both summaries, and do not read the KS alone.** They answer different
questions and disagree in an instructive way. Duration has the *lowest* KS D of
any variable (0.070, apparently the best match, and the medians are identical at
12 h) — yet the tail is 30–40% too long, robustly. KS has its power near the
median; the tail ratios are where the science sits.

Two things the comparison exposes, both worth reporting rather than tuning away:
the model makes somewhat smaller storms with higher peak rain rates than IMERG
(expected — IMERG is a smoothed retrieval), and it produces **too few winter
storms**: several Dec-Mar months contain no object meeting the MCS criteria at
all, against ~10 in the observations for the same month. That matters if the
study extends beyond ASO. A month with no qualifying storm writes no `MCS_`
pickle (the `Storms_*.nc` is still written, with `MCS_objects` all zero), which
is why the tracking month counts below 120 are expected, not a failure.

### Averaging Tb, not radiance

MERGIR supplies Tb and nothing else, so its coarse values are block means of Tb.
The model is treated identically: OLR is converted to Tb at 2 km and only then
averaged. Converting after averaging would make the two sides differ by the
nonlinearity of the T⁴ relation — the sort of difference this comparison exists
to detect, not to introduce.

---

## Adding new data

**Another period or region.** Change `syear`/`eyear`/`months` or the
`lat_min..lon_max` box in `obs_config.py` and re-run from step 1. The region has
a `margin` around it so storms are tracked before they enter the region, matching
how the model side is filtered. Completed months are skipped, so extending a
period only downloads what is missing.

**Another model run.** Add it to `model_runs` in `obs_config.py`; the dataset
registry picks it up automatically for both BT methods.

**Another threshold configuration.** Edit `mcs_config.py` to one of exp1..exp5
and re-run `track_storms.py`; output goes to a new `<dataset>/<exp>/` directory.
A configuration that is not one of the known five is refused by design — add it
to `KNOWN_EXPS` in `track_storms.py` with a name of its own first.

**Another satellite product.** Add its patterns to `datasets`; `track_storms.py`
needs only a `RAIN` file and a `TB` file on a common grid with identical time
axes and 2D `lat`/`lon`.

---

## Things that will bite you

- **Incomplete months are silent.** Months are skipped by *filename*, so a short
  month is never revisited. `check_months.py` reads both the recorded
  `granules`/`files` attribute and the actual number of time steps. Run it after
  every download session, before anything downstream.
- **A download that "looks fine" may not be.** GES DISC answers some requests
  with HTTP 200 and an HTML login page. `fetch()` checks for the netCDF/HDF5
  magic bytes and treats anything else as retryable, re-authenticating when it
  sees the login page. Before that check existed, ~12% of a MERGIR month was
  being dropped with nothing but a log line. Warnings like
  `session expired, re-authenticating` are now normal and harmless.
- **Don't raise `nworkers`.** Measured: 4 workers 0.84 s/granule, 8 workers
  1.97 s, 12 workers 1.79 s — more workers spend longer backing off from 503s
  than they gain. IMERG and MERGIR are on different hosts, but throttling appears
  to be per Earthdata *account*: MERGIR ran at 0.66 s/file alone and 2.43 s/file
  while IMERG was also running.
- **Editing a script does not affect a running job.** Python imports at startup.
  A job started before a fix keeps the old code — restart it.
- **Time conventions differ.** EPICC hourly files are stamped at `HH:25` (the
  centre of the accumulation window); observations are stamped at the hour start.
  `make_model_tracking_input.py` floors the model stamps to match. MERGIR stamps
  carry microsecond jitter, which also gets floored — and which is why hourly
  means are taken by reshape rather than `resample` (9 min/month → 39 s/month,
  bit-identical output).
- **NaN is a value to the tracker.** `track_storms.py` fills missing rain with 0
  and missing BT with 300 K, so a data gap can never create a storm.


---

## Comparison outputs, and which numbers live where

Everything lands in `{path_obs}` (i.e. `/scratch3/dargueso/obs-mcs-tracking`),
named on the experiment so an `exp2` run sits beside `exp1` rather than
overwriting it.

| File | Holds |
|---|---|
| `obs_model_comparison_<exp>.png` | 6 panels: seasonal cycle, interannual counts, CDFs of area/duration/peak/volume |
| `obs_model_comparison_<exp>_tail.csv` | **the tail statistics** — 24 rows: dataset, variable, quantile, ratio, 95% CI, and the absolute obs/model values |
| `obs_model_maps_<exp>.png` | 6 panels: 3 track-density maps, 2 difference maps, diurnal cycle |
| `obs_model_maps_<exp>_numbers.txt` | density, sub-region and diurnal numbers, as printed |
| `obs_model_maps_<exp>_relative.png` | standardised density: 3 fields + 2 differences |
| `obs_model_maps_<exp>_seasonal.png` | standardised density, 4 seasons x 3 datasets |
| `obs_model_maps_<exp>_relative_numbers.txt` | pattern correlation, seasonal shares |
| `storm_structure_<exp>.png` | 3 rain composites + radial profile |
| `storm_structure_<exp>_numbers.txt` | core intensity, e-folding radius, outer share |

Both scripts also print everything to stdout, so `python plot_… | tee` captures
it. The bootstrap is seeded, so the intervals reproduce exactly between runs.

### Three complementary comparisons, which disagree usefully

1. **Whole distribution (KS D).** One number per variable, but its power sits
   near the median. Duration scores best (D = 0.070, identical medians) while
   its p95 is 40% too long — do not read it alone.
2. **Tail (quantile ratios at p90/p95/p99, paired year-block bootstrap).** Where
   the science is. Under YS the extreme rain volume matches within sampling
   uncertainty (p95 0.93, p99 1.03, intervals including 1) while SB
   underestimates it by a third.
3. **Space and time (density maps, sub-regions, diurnal cycle).** A model can
   match every distribution and still put storms in the wrong place or start
   them at the wrong hour — neither of the first two would notice.

### Spatial and diurnal results (exp1, 2011-2020)

| | storms/yr | peak density | mean lat | mean lon | r vs obs |
|---|---|---|---|---|---|
| IMERG + MERGIR | 188.4 | 31.4 | 41.06 | 6.09 | — |
| EPICC 0.1° YS | 158.7 | 27.5 | 41.23 | 6.10 | 0.650 |
| EPICC 0.1° SB | 318.2 | 49.2 | 41.25 | 6.28 | 0.697 |

Density is storms yr⁻¹ per 10⁴ km², each storm counted once per 0.5° cell it
crosses. Note the pattern correlation is a *shape* measure and is insensitive to
the factor-of-two amplitude error in SB — another reason not to rank on one
statistic.

| | south <40.3N | north >40.5N | S/N |
|---|---|---|---|
| IMERG + MERGIR | 91.6 | 124.5 | 0.74 |
| EPICC 0.1° YS | 72.8 | 108.5 | 0.67 |
| EPICC 0.1° SB | 138.9 | 218.9 | 0.63 |

Diurnal cycle of storm initiation (UTC):

| | peak hr | % 09-15 | % 19-01 | 21 UTC |
|---|---|---|---|---|
| IMERG + MERGIR | 13 | 33.6 | **31.3** | **6.2** |
| EPICC 0.1° YS | 12 | 46.5 | 23.1 | 2.5 |
| EPICC 0.1° SB | 12 | 43.6 | 24.5 | 2.9 |

**The clearest model deficiency found so far.** The observations have a second,
nocturnal maximum near 21 UTC that neither model configuration produces (6.2% of
initiations against 2.5-2.9%), and the model over-concentrates initiation in the
early-afternoon window (46.5% against 33.6% in 09-15 UTC). Caveat worth keeping:
IMERG's passive-microwave sampling is not uniform through the day, so part of
the observed nocturnal signal could be retrieval-related rather than physical.


---

## Relative (standardised) density, and why it is the one to read

`plot_obs_model_maps_relative.py` divides each field by its own domain mean, so
1.0 is an average cell **for that dataset**. The absolute maps confound two
things — how many storms a dataset makes and where it puts them — and because SB
makes twice as many storms as observed, its absolute map is darker everywhere
and its pattern is unreadable. Standardising removes the count entirely.

On the colour ramp: the sequential panels use **viridis**, which is multi-hue but
*not* a rainbow — lightness still rises monotonically, so it survives greyscale
and colour-vision deficiency while resolving far more steps than a single-hue
blue. The difference panels stay on a two-hue diverging ramp with a neutral
midpoint, because there zero must read as absence of colour.

| | pattern r | mean abs diff | max excess | max deficit |
|---|---|---|---|---|
| EPICC 0.1° YS | 0.650 | 0.336 | +2.47 | −1.26 |
| EPICC 0.1° SB | 0.697 | 0.305 | +2.58 | −1.34 |

Seasonal share of each dataset's own storms:

| | DJF | MAM | JJA | SON |
|---|---|---|---|---|
| IMERG + MERGIR | **16.1** | 15.3 | 21.1 | **47.6** |
| EPICC 0.1° YS | 8.4 | 19.8 | 28.0 | 43.7 |
| EPICC 0.1° SB | 13.2 | 21.9 | 26.0 | 38.9 |

The seasonal maps show the model concentrating storms over orography (Pyrenees,
Alpine foothills, Gulf of Lion) in MAM and JJA far more sharply than the
observations do. Combined with the seasonal shares, the model under-produces
winter storms (8.4% of its total against 16.1% observed) and over-produces
summer ones.

### Which period the model reproduces best — and why the study uses ASON

Pattern correlation of the standardised density field, Yang & Slingo against
IMERG+MERGIR, 2011-2020:

| period | n obs | n model | pattern r |
|---|---|---|---|
| **ASON** (focus period) | 1082 | 857 | **0.611** |
| ASO | 806 | 656 | **0.443** |
| SON | 896 | 694 | 0.635 |
| JJA | 397 | 445 | 0.654 |
| MAM | 288 | 315 | 0.485 |
| DJF | 303 | 133 | 0.410 |
| whole year | 1884 | 1587 | 0.650 |

By single month: Aug 0.361, Sep **0.239**, Oct 0.496, Nov 0.411.

**The study focuses on ASON, and the model reproduces the observed spatial
pattern of storms better over ASON than over ASO** — 0.611 against 0.443.
November agrees comparatively well while September is the weakest month of the
year, so adding the fourth month dilutes rather than compounds the
disagreement. The storm-count ratio barely moves (0.79 for ASON, 0.81 for ASO).

Two caveats before quoting these anywhere. Pattern correlation **rises with
sample size** — a single month holds ~300 storms spread over 720 cells, so
month-level values are noise-limited. And it is a *shape* measure that ignores
the storm-count difference entirely. It ranks periods; it does not grade them.
On that ranking ASON is better than ASO but not uniquely best: SON, JJA and the
whole year all score marginally higher.

An earlier version of this file claimed the focus season was where the model
agreed best. That was read off the maps rather than computed, and it was wrong —
ASO is the worst of the multi-month groupings, which is part of the case for
ASON.

## Storm-relative rain structure

`plot_storm_structure.py` composites the rain field around every storm centre
(~27,000 time steps for the observations), giving three composite maps and a
radial profile. It answers a question the bulk statistics cannot: *is* the
model's smaller-storm/higher-peak signature a narrower core, a weaker shield, or
both?

| | centre | r=50 km | r=100 km | r=200 km | e-fold (km) | % rain >100 km |
|---|---|---|---|---|---|---|
| IMERG + MERGIR | 7.79 | 4.26 | 1.70 | 0.59 | **88** | 71.1 |
| EPICC 0.1° YS | 7.64 | 3.26 | 1.21 | 0.44 | **62** | 70.9 |
| EPICC 0.1° SB | 7.60 | 2.95 | 1.07 | 0.42 | **62** | 72.6 |

Rain rates in mm h⁻¹; `centre` is the composite peak, `r=X` the azimuthal mean
at that radius, `e-fold` the radius at which the profile falls to 1/e of the
innermost ring.

**The core is right; the shield is too narrow.** Composite peak intensity is
within 2% of observed (7.64 vs 7.79), but the profile decays much faster —
e-folding radius 62 km against 88 km, and at 100 km the model has 29% less rain.
This is the mechanism behind the ~40% area deficit in the tail statistics, and
it reconciles the apparent contradiction there (smaller storms yet higher peak
rain rates): the model's rain is more tightly concentrated, so a per-storm
maximum is higher while the storm covers less ground.

Note the *fraction* of composite rain beyond 100 km is nearly identical across
all three (71-73%), so this is a rescaling of the storm, not a change in the
core/shield partition.


---

## Storm wind speed (2 km analysis)

The storm wind property comes from `WSPD10`, not from `U10MET`/`V10MET`.

`WSPD10` is computed by the EPICC postprocessing (`compute_vars.compute_WSPD10`,
via `uvmet10_wspd_wdir`, earth-relative) and lives in
`{run}/WSPD10/UIB_01H_WSPD10_YYYY-MM.nc` alongside `RAIN/`, `OLR/`, `PW/` and
`CAPE2D/`. Both runs are complete, 120/120 months.

**Why not U and V.** The hourly files are hour-averages of the sub-hourly output.
Averaging the vector components and *then* taking the magnitude gives the speed
of the hourly-mean wind, which is always less than the hourly-mean speed, because
vector averaging cancels rotating and gusty flow. That is precisely what storms
have, so the bias is correlated with storm activity. `WSPD10` averages the scalar
speed and avoids it. Direction is never used by the analysis, so U/V also cost
twice the storage for no benefit.

**Do not aggregate `WDIR10` with `cdo hourmean`** if it is ever produced —
direction is circular, and averaging 350 deg and 10 deg gives 180 deg. Mean
direction has to come from averaging the components and taking atan2.

**What the storm property means.** The model writes one value per hour, so the
quantity is the *hourly-mean* wind speed, and the storm property is its maximum
over the footprint — not a gust. A true peak would need an `hourmax` aggregation
of sub-hourly speed, which the standard chain does not produce.

Wind numbers from this route are somewhat **higher** than the original AGU
figure, which used the U/V reconstruction. That is the correction, not a
regression.


---

## Applying Yang & Slingo to the 2 km production run

The evidence above is from the 0.1 deg evaluation, but the conclusion applies to
the native 2 km tracking, and that is now the production configuration.

**The decisive argument is not the evaluation.** The 241 K and 225 K
cloud-shield thresholds come from the satellite MCS literature and are defined
on *window-channel* brightness temperature. Stefan-Boltzmann does not produce a
window Tb; it produces a broadband effective emission temperature 20-30 K
colder. Applying satellite-calibrated thresholds to it is a category error
regardless of which conversion gives more agreeable numbers. The observational
comparison then supports the same choice independently.

Measured at 2 km native (`EPICC_2km_ERA5`, 2014-09, 225M points):

| | SB | YS |
|---|---|---|
| mean Tb | 262.9 | 289.9 |
| p50 | 264.5 | 292.4 |
| cells < 241 K | 2.787% | **0.780%** (x0.28) |
| cells < 225 K | 0.200% | **0.052%** (x0.26) |

and the effect on tracked storms, same month, same exp1 thresholds:

| | storms | med area (km²) | max area | med dur (h) | med peak (mm/h) |
|---|---|---|---|---|---|
| SB | 148 | 3711 | 38023 | 10 | 47.9 |
| **YS** | **89** | **5029** | 38023 | 11 | **58.2** |

YS/SB count ratio 0.60 at 2 km, against 0.50 at 0.1 deg. The surviving storms
are larger, longer, more intense and wetter: the stricter cold-cloud criterion
removes marginal systems whose shields only qualified because SB made everything
look ~22 K colder. Note the largest storm is **identical** in both (38,023 km²) —
the deep systems are unaffected, which is what the two conversions converging in
cold cloud predicts.

**Caveat to state in the Methods.** Yang & Slingo was fitted to coarse satellite
radiances (Nimbus-7, tens of km), so applying it per pixel at 2 km is an
extrapolation. It is standard practice for convection-permitting output
(PyFLEXTRKR does the same) and is in any case far closer to a window Tb than the
grey-body inversion, but it should be stated rather than passed over.

### How it is configured

In `mcs_config.py`:

```python
bt_method = "YS"          # or "SB"; written into the output netCDF attributes
exp_label = "exp1"        # the threshold set, and nothing else
write_nc  = True          # the Storms netCDF, ~2.2 GB per month
path_track_root = "/scratch3/dargueso/postprocessed/EPICC"
path_out = f"{path_track_root}/{wrun}/ConvStormTracking_{bt_method}/{exp_label}"
```

`olr_to_tb()` in `tracking_functions_optimized.py` is the single definition of
the conversion, shared by the tracker and by the coarsening script, so the two
cannot drift apart. `bt_method` is written into every tracked netCDF, so an SB
run and a YS run with identical thresholds are distinguishable **from the files
themselves** — the same provenance principle that caught `mcs_config` sitting on
the exp5 thresholds while everything downstream assumed exp1.

Output goes to `/scratch3` rather than beside the input: `/scratch1` is at 99%
with ~175 GB free and this run writes ~520 GB of netCDF. The directory layout
mirrors `/scratch1`, so the trees merge without rearranging.

**Robustness check worth doing.** The paper headline is a *relative* change
between climates, and ref. 30 argues trackers differ in absolute frequency while
agreeing on relative changes. Since the SB tracking already exists, compute the
PGW/present ratio under both conversions. If the climate signal is stable the
result is inoculated against exactly this objection; if it is not, that needs to
be known before submission.


---

## Radar evaluation of 2 km rainfall (EURADCLIM)

A second, independent branch: EPICC **rainfall** at its native 2 km against
EURADCLIM (Overeem et al. 2023, ESSD 15:1441), hourly, gauge-adjusted OPERA
radar on a 2 km grid. The satellite branch above validates storms at 0.1°; this
one validates the rain field at the model's own resolution, over land and the
coast. Settings in `radar_config.py`.

```bash
python download_euradclim.py --list   # FIRST: check the API sees the dataset
python download_euradclim.py          # monthly zips -> EURADCLIM/raw
python make_radar_input.py            # EURADCLIM onto the model grid  (~3 min/month/worker)
python radar_model_stats.py           # per-cell stats, both datasets  (~30 s/month/worker)
python make_radar_mask.py             # quality mask + radar_mask.png  -- LOOK AT IT
python plot_radar_model_qq.py         # Q-Q per subregion x scale, diurnal cycle (ASON)
python plot_radar_model_maps.py       # maps: mean, wet freq, p99.9, diurnal phase
python plot_radar_model_maps.py --scale 5
```

Output under `/scratch3/dargueso/obs-mcs-tracking/EURADCLIM/`.

**API key.** The KNMI API needs one: a personal key (free, at
developer.dataplatform.knmi.nl) is read from `$KNMI_API_KEY` or
`~/.knmi_api_key`. The anonymous key is shared by everyone and is rate-limited
most of the time. The dataset is `RAD_OPERA_HOURLY_RAINFALL_ACCUMULATION_EURADCLIM`
v2.0, **upper case**: the lower-case slug of the web page gives 404. It is
served as monthly zips of 0.5–2.9 GB (182 GB for 2013–2020).

### Decisions

- **Direction of the remapping: radar → model grid.** Both grids are 2 km. The
  model is the thing being evaluated, and its grid carries `LANDMASK` and
  `HGT_M`. The remap is conservative, approximated by supersampling each model
  cell 5×5 (`radar_utils.remap_weights`). The EURADCLIM georeferencing (LAEA,
  upper-left corner at 0,0) was checked against the official coordinate file.
- **Time.** EURADCLIM filenames carry the **end** of the hour; everything is
  re-stamped to the hour start, matching the floored model stamps.
- **Like with like.** The model is counted only in the cell-hours where the
  radar is valid. At 10 km both are averaged over the same valid 2 km cells.
- **Statistics are additive and unmasked** (counts, sums, 100-bin log
  histograms per cell). Masks and seasons are applied at plot time, so changing
  the mask never requires recomputing. Histogram quantiles were checked against
  exact `np.quantile`: within 1% above the 0.1 mm/h wet threshold.
- **Two scales, 2 km and 10 km.** A bias that disappears at 10 km comes from
  the effective resolution of the two products, not from the model physics.
- **Q-Q uses all hours** (p99 = exceeded 1% of the time), with wet-hour
  frequency reported alongside. The uncertainty comes from a paired year-block
  bootstrap, as in the tail analysis above.

### The mask (`make_radar_mask.py`)

A cell is kept if all of these hold: it is inside WME; it is within range of
a radar that actually contributed (150 km C/S-band, 60 km X-band, read from
each hour's ODIM `how/nodes` and located with the OPERA database); it is land
or within 10 km of the coast; it has ≥90% availability; it shows no
climatological shadow (mean < 0.5× its 20 km-smoothed surroundings); it shows
no clutter (wet-hour frequency > 2× surroundings); and it is outside every
`manual_exclude` box. The artefact tests compare EURADCLIM with itself, never
with the model, which would bias the evaluation towards agreement.

Things that bite:

- **The range test is not optional.** The Spanish composite reports valid
  zeros over a whole rectangle, far beyond any radar's useful range, so
  availability cannot see where Spanish coverage ends.
- **No Italian radars in OPERA.** Sardinia and the Tyrrhenian are nodata. The
  comparison covers Iberia, southern France, Corsica and the Balearics.
- **AEMET renamed its radars** (`esbar` → `esgld`, …). `node_alias` maps the
  old codes, and `extra_radars` places radars missing from the OPERA database
  (none needed so far: `essan`, despite
  the name, is Aguión in Asturias, `essls` in the database). The mask script logs any contributing
  radar it still cannot place.
- **Radar underestimates the extreme tail** (C-band attenuation, fixed Z-R).
  A model "too intense" at p99.99 is partly the reference. The cross-check
  against the gauges is built in: see the three-way comparisons in the
  rain-gauge section.
- **Radial streaks.** Over 2013–2020 the Spanish composite shows streaks along
  radials from the radars, at ±10–50% of their surroundings (beam blockage,
  interference). The smoothed-surroundings tests (0.5× / 2×) miss them, and
  tightening those would also remove real orographic gradients. So the mask
  has a **radial test** that detects them by shape:
  - each cell is assigned to its nearest radar;
  - cells are grouped into 1° × 10 km polar bins, and each bin is compared
    with the median of the azimuths within ±10° at the same range;
  - a bin is flagged when its ratio stays outside 0.85–1.18 for ≥ 30 km
    along the radial, on the mean rain or on the wet-hour frequency.

  Four settings were compared. This one removes 1.4% of the area, including
  the clear spokes around Murcia, Almería, south of Valencia and part of
  Madrid. Weaker streaks remain across the Spanish composite: no setting
  removes them without large losses. They partly cancel in regional
  statistics and are averaged at 10 km, so read the 2 km per-cell radar maps
  with them in mind.
- **EURADCLIM has no data for 1–22 January 2013.** Those files are empty
  placeholders: no valid pixel anywhere, and start = end in the ODIM
  attributes. The reader rejects them and the hours stay missing.
- **Some hours cover only 45 minutes** (a scan is missing). They are rejected
  too, since accepting them would record about 25% too little rain for the
  hour. Both kinds are logged as "not a 1-h accumulation".
- **Seasonal cycle.** `plot_radar_model_qq.py` also writes
  `radar_model_seasonal.png`: all 12 months, gridded and masked, with a row of
  EPICC/EURADCLIM ratios.


---

## Rain-gauge evaluation (combined AEMET dataset)

The third observational reference. It works at points rather than on a grid,
but it is independent of both the satellite and the radar data. Settings are in
`station_config.py`. Everything about the stations sits under
`/scratch3/dargueso/obs-mcs-tracking/STATIONS/`:

```
STATIONS/AEMET_combined/   the database: data files, stations.csv, README.md,
                           build_summary.txt (README and summary are written
                           by the build, with that build's numbers)
STATIONS/evaluation/       model (and EURADCLIM) at the stations, figures, numbers
```

```bash
python make_station_dataset.py   # the database: 10-min, 1-h, daily, Arnau    (~90 s)
python extract_at_stations.py    # EPICC (and EURADCLIM) at the stations      (~2 min)
python plot_station_model.py     # Q-Q, diurnal, duration maxima, daily scores (~20 s)
```

**Months.** All evaluation plots (radar and gauges) take the same options,
from `seasons.py`:

- `--season ASON|ANN|DJF|MAM|JJA|SON` (default ASON, the study season)
- `--all-months`, the same as `--season ANN`
- `--months 6 7 8` for any other set

The three are mutually exclusive. The tag goes into every output filename, so
runs for different months sit side by side. A custom list equal to a named
season gets that season's tag.

Whole-year gauge results (`--all-months`) match ASON: model/gauge hourly
p99.9 1.19 [1.13–1.25], 1-h seasonal maxima 1.12, daily correlation 0.63.

### The dataset: 10-min, clock-hour and daily products

Two sub-daily products share one set of stations (291) but follow different
rules. The **10-min** product is strict, for later use against the model's
10-min output. The **clock-hour** product follows rules designed for hourly
data, and is the one this study uses. A daily file in the manner of Arnau is
built from both.

| File (in `AEMET_combined/`) | Holds |
|---|---|
| `AEMET_10MIN_PREC_2011-2020.nc` | `prec(station, time)` in mm per 10 min, plus `day_tier` and `day_source` per station-day |
| `AEMET_01H_PREC_2011-2020.nc` | clock-hour totals, **hourly rules** (below) |
| `AEMET_DAILY_2011-2020.nc` | per day: `prec`, `max_1h` (clock hour) and `hour_max_1h` from the 1-h product; `max_10min`, `time_max_10min` and `max_60min_sliding` (Arnau's PMAX10/PMAX60 analogues) from the 10-min product; valid counts; tiers under both rule sets |
| `AEMET_ARNAU_DAILY_2011-2019.nc` | AEMET's validated daily record (P24, PMAX10..PMAX12H, flags), 403 stations: the reference |
| `stations.csv` | code, name, position, sources, fraction of valid hours, verified days and complete days |
| `README.md`, `build_summary.txt` | written by each build |

**Hourly rules.** The clock-hour product differs from the 10-min one in two
ways:

- **Verification ignores the 10-min maximum.** The daily total and the 60-min
  maximum must still match Arnau. A 10-min timing difference inside an hour
  does not make the hourly totals wrong.
- **In verified days, missing 10-min steps are 0.** The day reproduces AEMET's
  validated total, so the gaps held no rain, and every hour of the day is known.

No extra stations are available anywhere in these data, but the gain in data
is large:

- **Valid station-hours:** 15.1 M → 19.3 M (+27%).
- **2011–2014:** from 14–35% to 58–67% of possible hours valid; these years
  were mostly lost before.
- **Stations with ≥ 50% valid hours:** 245 → 252.

**Daily file.** Values are present only where the day is complete in the
product they come from; verified days are always complete. Checked against
Arnau on all 117,019 verified wet days: the daily total, max 10-min and max
sliding 60-min all agree within tolerance (100%). The clock-hour maximum
never exceeds the sliding 60-min one. Their median ratio is 1.074, the
sliding/clock factor.

**Three AEMET sources, one set of instruments.**

| Source | Period | Stations | Quality control? |
|---|---|---|---|
| HyMEX 10-min | 2011–2020 | 287 | **none** (real-time AWS feed) |
| Balearic 10-min archive | 2009–2024 | 44 | AEMET historical archive |
| Arnau daily + maxima | 2011–2019 | 403 | AEMET flags, manual/automatic |

Arnau is mostly *derived from the same 10-min record* (`ID_FLAG_P = 1`, 87%
of days). That is what makes it usable as the quality control for the raw
10-min data: where the two disagree, AEMET corrected the day and the raw copy
is wrong.

**Quality tiers per station-day** (`day_tier`):

| Tier | Share | Meaning |
|---|---|---|
| 1 verified | 69.5% | reproduces Arnau's validated daily total and max 60-min amount (plus max 10-min in the 10-min product) to 0.2 mm + 5% |
| 2 automatic | 28.9% | no usable Arnau record that day, and all automatic checks pass |
| 8 rejected | 1.5% | disagrees with Arnau when Arnau derives from the same record, or Arnau flags the day doubtful (20/21) |
| 9 rejected | 0.0% | failed an automatic check |

- **Automatic checks:** 10-min ≤ 50 mm, day ≤ 400 mm, no stuck sensor (6
  identical values ≥ 0.5 mm in a row), no isolated spike (≥ 10 mm with nothing
  within ±30 min).
- **Why the cap is 50 mm:** the largest AEMET-validated 10-min amount in these
  data is 44.7 mm (60-min: 159 mm). A lower cap would delete real extremes such
  as the September 2019 cut-off low.
- **Days with gaps can still be verified.** A partial day that reproduces
  Arnau is tier 1, since amounts cannot be negative. One that exceeds Arnau is
  rejected, and one below it stays tier 2.
- **Arnau from another instrument** (`ID_FLAG_P` 0/2): a match still verifies
  the day, but a mismatch proves nothing, so the day stays tier 2.
- **Why days are tier 2:**
  - station not in Arnau (48%)
  - 2020, after Arnau ends (30%)
  - partial day below Arnau's total (11%)
  - no Arnau record that day (10%)
- The evaluation uses tiers 1 and 2 (`eval_tiers`). Restrict to 1 for the
  strictest subset.

**Decisions that change the answer:**

- **Timestamps were measured, not assumed.** Both 10-min sources stamp the
  *end* of the interval. Summing the end-stamped steps (00:10..24:00)
  reproduces Arnau's daily total exactly on 99.6% (HyMEX) and 100% (Balearic)
  of rain days, against 85% and 75% for the start-stamped ones
  (00:00..23:50). The output is stamped at the **start**, like every other
  file here: the 10-min value at 13:00 covers 13:00–13:10, and the hourly
  value covers 13:00–14:00 UTC.
- **Station codes.** HyMEX names a station by its WMO synoptic code where it
  has one and by its climatological code otherwise, and 11 stations appear
  under both in the same year. Synoptic codes are mapped to climatological ones
  through the Arnau station lists (`readme.rar`) and the Balearic master file.
  Without this, 80 stations lost their Arnau quality control or coordinates.
  The two copies never disagreed (0 of 579 overlapping values). Seventeen codes
  have no coordinates in any source and are dropped; they are listed in the log.
- **Source priority.** The Balearic archive is used where it passes, otherwise
  HyMEX. On days both verify, they are identical (100% of 10-min values), so
  combining them does not mix two measurements.
- **Restricted data.** HyMEX data are restricted to HyMeX projects, so the
  combined files inherit that restriction.

### The evaluation (`plot_station_model.py`)

The model is sampled at each station's own 2 km cell, and also as the max and
mean over the 3×3 block around it. Cell and neighbourhood maximum bracket the
effect of small location errors. The model is counted only in hours when the
gauge is valid, and EURADCLIM is added automatically once its files exist.

| Output | What |
|---|---|
| `station_model_qq_<season>.png/.csv` | hourly all-hour quantiles per subregion, year-block bootstrap |
| `station_model_diurnal_<season>.png` | mean rain and wet-hour frequency by UTC hour |
| `station_model_durations_<season>.png/.csv` | seasonal max 1–24 h totals, model/gauge per station-season |
| `station_model_daily_<season>.png/.csv` | paired daily correlation and POD/FAR/ETS per station |
| `station_model_biasmaps_<season>.png` | per-station mean rain, P99 and P99.9: gauge values, then EPICC/gauges, EURADCLIM/gauges and EPICC/EURADCLIM |
| `station_model_seasonal.png/.csv` | monthly mean rain, wet-hour frequency and hourly P99 per region: gauges, EPICC, EURADCLIM; always all 12 months |
| `station_model_<season>_numbers.txt` | all numbers, as printed |

**Three-way comparisons.** Once `extract_at_stations.py --radar` has run,
every station figure compares gauges, EPICC and EURADCLIM on a *joint* sample:
the same hours at the same stations for all three. EURADCLIM is used only
where the station's cell passes the radar quality mask. That gives three pairs:

- **EPICC vs gauges:** the model error.
- **EURADCLIM vs gauges:** the radar's own error, e.g. an underestimated
  extreme tail.
- **EPICC vs EURADCLIM:** the comparison the gridded radar figures make
  everywhere, here checked at points where the truth is known.

Without the radar file the figures fall back to gauges and EPICC over
2011–2020. Each dataset keeps one colour in every figure: gauges blue,
EPICC orange, EURADCLIM aqua.

**Bias maps.** A station quantile is computed only where there are at least
10 exceedances' worth of data (P99.9 needs ≥ 10,000 valid hours). A station
mean needs ≥ 2,000 hours. First numbers (ASON, gauges and EPICC), median
ratio over stations [share of stations > 1]:

| | ALL | CAT | LEV | BAL |
|---|---|---|---|---|
| mean rain | 1.30 [79%] | 1.40 [97%] | 1.37 [90%] | 0.85 [26%] |
| P99 | 1.31 | 1.50 | 1.38 | 0.80 |
| P99.9 | 1.27 | 1.34 | 1.44 | 0.95 |

The model is wetter than the gauges almost everywhere on the mainland, and
drier over the Balearics. The seasonal cycle shows where this comes from: the
model has about the same number of wet hours as the gauges, but more rain in
them, and most of all in September–October in Catalonia.

Each comparison uses the part of the combined dataset that suits it:

- **Hourly and diurnal comparisons:** the 10-min stations, as clock hours.
- **Duration maxima:** the same stations, compared on clock-aligned hourly
  steps. The 10-min data also give the sliding/clock factor.
- **Daily scores:** the 10-min stations **plus the 151 Arnau-only stations**
  (validated daily totals), 413 stations in all.

**First results (ASON 2011–2020, EPICC_2km_ERA5, tiers 1–2):**

| | ALL | CAT | LEV | BAL |
|---|---|---|---|---|
| hourly p99.9, model cell / gauge | 1.18 [1.09–1.24] | 1.30 | 1.32 | 0.87 |
| hourly p99.99, model cell / gauge | 1.00 [0.90–1.07] | 1.03 | 1.05 | 0.80 |
| seasonal max 1 h, model cell / gauge (median) | 1.10 | 1.20 | 1.21 | 0.95 |
| seasonal max 6 h, model cell / gauge (median) | 1.19 | 1.28 | 1.28 | 0.92 |

- **The model is, if anything, too intense.** A gauge point should show *higher*
  extremes than a 4 km² average, yet the model cell matches or exceeds the
  gauges from p99 to p99.99 and for 1–24 h maxima. The Balearics are the
  exception.
- **Sliding vs clock-hour maxima.** A sliding 60-min gauge maximum (AEMET's
  PMAX60) is **×1.08** the clock-hour one (median of 1,547 station-seasons with
  a ≥ 5 mm hour). This is measured, not the textbook 1.13, and it is the
  factor to apply before comparing PMAX-type gauge extremes with hourly model
  output.
- **Diurnal cycle.** The model concentrates rain in a 15–17 UTC peak much
  stronger than the gauges show in Catalonia and Valencia, and it
  underestimates night and morning rain over the Balearics. This is consistent
  with the missing nocturnal initiation peak in the satellite comparison.
- **Daily skill (413 stations):**
  - correlation median 0.62 (IQR 0.51–0.72)
  - ETS ≥ 1 mm 0.42; ETS ≥ 20 mm 0.28
  - frequency bias ≥ 20 mm 1.14
  - total bias median 1.12

**Timing checks behind the diurnal comparison.** The model's diurnal cycle
differs strongly from the gauges' (a much stronger 15–17 UTC peak), so the
time handling was checked from the data:

| Check | Result |
|---|---|
| Gauge time stamps vs AEMET's own time of maximum intensity (`PHINT`) | `PHINT` is the centre of our maximum 10-min interval on 96% of 16,181 days: gauge stamps correct |
| Hourly correlation, EURADCLIM vs gauges, lags −3..+3 h | peaks sharply at 0 h (0.686, against 0.41 at −1 h and 0.34 at +1 h): the observations agree |
| Hourly correlation, EPICC vs gauges, lags −4..+4 h | peaks about +35 min (0.201 at 0 h, 0.202 at +1 h): no hour-scale shift, no UTC/local mix-up |
| Model hourly window | the ORIGINAL model hour HH covers HH−1:50 to HH:50, 10 min early (see below) |

**The original model hourly rain is 10 minutes early.** The EPICC 10-min rain
is WRF's `PREC_ACC_NC` from the `wrfprec` files: rain over the *last*
10 minutes, stamped at the interval end. `cdo hoursum` grouped the stamps
HH:00–HH:50, so the original `UIB_01H_RAIN` files (and `UIB_01H_RAIN.zarr`)
cover HH−1:50 to HH:50.

This was verified against `RAINNC` in `wrfout_d01_2020-01-10`, whose
differences give the exact rain from HH:00 to HH+1:00:

| 10-min stamps summed | median error against the exact `RAINNC` total |
|---|---|
| HH:00–HH:50 (original files) | 0.07 mm |
| HH:10–HH+1:00 | 0.000003 mm |

Corrected clock-hour files, built from `UIB_10MIN_RAIN.zarr` by
`EPICC_scripts/WRF_processing/fix_hourly_rain_clockhour.py`, are in
`/scratch3/dargueso/postprocessed/EPICC/<run>/RAIN/`, for both runs (first
written as `RAIN_clockhour/`). All postprocessed input now comes from
`/scratch3`; `/scratch1` keeps only the original rain, for manuscripts that
used it. The aggregation code is fixed in `EPICC_scripts` and
in the general `wrfprocessing`.

The whole evaluation was re-run on the corrected files (2026-09-29). The
previous outputs are kept as `*_before_clockhour_fix` and in
`figures_before_clockhour_fix/`.

- **Distributions:** every ratio (quantiles, bias maps, gridded radar
  comparison, daily scores) moved by ≤ 0.04.
- **Timing:** the model's lag behind the gauges fell from +37 to
  **+21 minutes**. That remainder is the model's own timing. Neither can produce the difference in
diurnal *shape*, which is therefore a model feature, not bookkeeping.

**Caveats:**
- **Points vs areas.** A gauge is a point; the model and radar cells are
  areas. Point extremes exceed area-mean extremes, which makes "model ≥ gauge"
  a conservative statement of over-intensity.
- **Station density** is highest in Catalonia, the Ebro basin and Valencia,
  so regional panels matter more than the pooled "ALL" panel.
- **Gauges are not fully independent of EURADCLIM.** Its gauge adjustment
  uses ECA&D, which includes AEMET stations, so gauge–radar agreement in
  daily totals is partly built in. The hourly structure and the extremes are
  much more independent.
- **Model hourly window.** The original model hourly files are 10 minutes
  early (HH−1:50 to HH:50); use the corrected files in `/scratch3/.../<run>/RAIN/`. The
  model's 10-min values are END-stamped, so model 10-min stamp *t* matches
  the start-stamped 10-min gauge value at *t* − 10 min.
