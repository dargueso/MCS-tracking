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
variables along `time` and the times of day along `lev`; they were converted
in place on 2026-09-30 (`EPICC_scripts/WRF_processing/fix_cape2d_layout.py`)
and the old copies deleted.

## Tracker configurations: exp1–exp11

`mcs_config.py` holds the reference configuration (exp1 = ST1 in the
manuscript) and is the one every analysis of the reference storms reads. Do
not edit it to run another configuration: the tracker, `MCS_tracking_WRF.py`
and `Observations/track_storms.py` import the settings module named by the
environment variable `MCS_CONFIG` (default `mcs_config`), and the other
configurations live in `mcs_config_sens.py`, selected with `SENS_EXP` and
`MCS_RUN`:

    MCS_CONFIG=mcs_config_sens SENS_EXP=exp6 MCS_RUN=EPICC_2km_ERA5 python MCS_tracking_WRF.py
    MCS_CONFIG=mcs_config_sens SENS_EXP=exp6 python Observations/track_storms.py obs mod0.1_YS_pres
    ./run_sens.sh exp6 exp7          # both, 0.1 deg pair then 2 km present and PGW

Every setting is written explicitly in `mcs_config_sens.py` (the exp1 values,
then the experiment's changes), so a run does not depend on the state of
`mcs_config.py`. `run_st2_st5.sh` is the older driver for exp2–exp5 that
rewrites `mcs_config.py` with sed and restores it; prefer the environment
mechanism.

| exp | kind | definition (changes from exp1) |
|---|---|---|
| exp1 | reference | rain ≥ 5 mm/h, ≥ 500 km², ≥ 3 h, lifetime peak ≥ 15 mm/h; cloud shield Tb ≤ 241 K, ≥ 1000 km², ≥ 5 h, core ≤ 225 K; ≥ 5 consecutive hours |
| exp2 | ST2 | rain ≥ 1000 km², shield ≥ 2000 km² |
| exp3 | ST3 | rain ≥ 1000 km², shield ≥ 10 000 km², peak ≥ 10 |
| exp4 | bar raised | rain ≥ 15 mm/h (object and MCS threshold), peak ≥ 30 |
| exp5 | bar raised | as exp4, rain ≥ 1000 km², shield ≥ 2000 km² |
| exp6 | definition | rain only: no cloud shield (`require_bt = False`) |
| exp7 | definition | loose: 3 mm/h, 250 km², 2 h, MCS threshold 3, peak 10 |
| exp8 | definition | persistence: 6 h rain object, 8 h cloud object, 8 h storm |
| exp9 | definition | Gaussian smoothing, σ = 1 cell, on rain and Tb |
| exp10 | definition | linking with ≥ 30% overlap (`min_overlap = 0.3`): 2-D components linked hour to hour, one successor and one predecessor per object, instead of the 3-D labelling that merges anything touching |
| exp11 | definition | intense rain only: no cloud shield, peak ≥ 30 mm/h, ≥ 1000 km² |

The two tracker options behind exp6, exp10 and exp11 (`require_bt`,
`min_overlap`; `label_objects()` in `tracking_functions_optimized.py`) default
to the reference behaviour and are recorded in the output netCDF attributes
together with `exp_label`. Output goes to
`<path_track_root>/<run>/ConvStormTracking_YS/<exp>/` at 2 km and to
`obs-mcs-tracking/tracking/{obs,mod0.1_YS_pres}/<exp>/` on the 0.1° grid. All
eleven configurations are tracked for both climates and on the 0.1° pair
(Yang & Slingo, clock-hour rain, as of 2026-10-01).

`Plotting/check_exp_robustness.py` compares the PGW/present ratios of all
configurations (paired year-block bootstrap, the format of
`check_bt_robustness.py`) and the present-day model against IMERG + MERGIR on
the 0.1° grid; `Plotting/plot_scatter_hist_obs_model.py <exp>` draws the
observed-vs-model storm characteristics. The findings are summarised in
`Analyses/EPICC/MCS-tracking/tracker_sensitivity_summary.md`: every
cloud-based definition at the reference intensity reproduces the climate
signal; without a cloud criterion the number and size of rain systems do not
change while peak rate and volume still rise; with a 30 mm/h peak bar storms
become more frequent in the warmer climate.
