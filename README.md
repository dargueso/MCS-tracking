# Mesoscale Convective System and Convective storms algorithm.
Optimized version based on A. Prein MCS-tracking.

Adapted to ingest WRF model outputs, or any other netCDF with OLR and Precipitation at any frequency (hourly or 3-hourly is recommended). 
It facilitates computing statistics such as the size of the storm, total volume of precipitation and maximum precipititation rate for each system.

[![DOI](https://zenodo.org/badge/474651021.svg)](https://doi.org/10.5281/zenodo.15732413)
## Storm summaries: what is computed

`Plotting/plot_scatter_hist_storm_characteristics.py` with `calc_summary=True`
writes one pickle per climate,

    storms_<period>_summary_<syear>-<eyear>_<m0>-<m1>_<reg>_<exp>_<bt>_m3.pkl

holding one row per tracked storm. These are the input to every figure and
statistic downstream, so the definitions below are the ones the manuscript
inherits.

### Region convention: clipped, not whole-track

A storm enters the summary if any of its timesteps falls inside the region box,
but **only the timesteps inside the box contribute to its statistics** —
duration included. A system that forms over the Atlantic and crosses into the
domain is counted by the part of its life spent inside.

This is the project convention as of 2026-09-30 and `Plotting/check_bt_robustness.py`
clips the same way. It matters: keeping the whole track whenever any step fell
inside — which the robustness check used to do — raises the present-day median
area from 5,366 to 6,280 km² and the median rain volume from 299 to 434 ×10⁶ m³,
and changes the PGW/present volume ratio from 1.41 to 1.67. The two conventions
are not interchangeable and results from one must not be quoted beside the other.

### Columns

| column | meaning | unit |
|---|---|---|
| `storm_id` | index within the period, after the region filter | |
| `prmax` | peak instantaneous rain rate over the storm's in-region life | mm/h |
| `wspd_max` | max of the hourly-mean 10-m wind over the storm footprint — not a gust | m/s |
| `max_size` | largest instantaneous footprint | km² |
| `mean_size` | footprint averaged over the in-region lifetime | km² |
| `tot_vol` | rain volume integrated over the in-region life, area-weighted per cell | m³ |
| `duration` | number of in-region timesteps | h |
| `duration_track` | full track length, for reference | h |
| `mean_int` | volume-consistent mean rain rate, `tot_vol / (mean_size · duration)` | mm/h |
| `pw_mean`, `pw_sum` | precipitable water over the footprint, 3 h before each step | mm |
| `mcape_mean`, `mcape_max` | MCAPE over the footprint, 3 h before each step | J/kg |

`mean_int` is defined so that

    tot_vol == mean_size · mean_int · duration

holds exactly for every storm. That is what lets the manuscript's volume
decomposition

    Δln R = Δln N + Δln A + Δln I + Δln D + ε,    R = N·A·I·D

close with a residual ε that is purely the cross-storm covariance between the
four factors, rather than a definitional mismatch. Note that ε is not small —
storms that grow larger also intensify and last longer — so the decomposition
should be reported with ε shown rather than folded into the other terms.

Two cautions when using these columns:

- **Means, not medians, are what decompose.** R is a sum, so only means are
  additive in logs. The distribution tables elsewhere report medians, and the
  two differ because the distributions are right-skewed.
- **`mean_int` is not `prmax`.** The volume-consistent mean rain rate and the
  lifetime peak rate respond differently to warming, and conflating them
  overstates the intensity contribution to the volume change.

### Environmental fields

`WSPD10` is hourly; `PW` and `CAPE2D` are 3-hourly and are sampled **3 hours
before** each storm timestep, so they describe the pre-storm environment rather
than the storm's own perturbation of it. Storm times are floored to the clock
hour before pairing: rain is stamped at the middle of its accumulation window
(HH:30 in the clock-hour files) and `method='nearest'` would break the tie
upward, silently pairing every storm with the following hour's fields.

`CAPE2D` holds `lev = (mcape, mcin, lcl, lfc)`; MCAPE is `lev=0`. The summary
builder asserts `lev == 4` — files written before 2026-09-30 had the four
variables along `time` and the times of day along `lev`, and are kept in
`CAPE2D_oldlayout`.
